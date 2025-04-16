#!/home/crellis/miniconda3/envs/transformers2/bin/python
"""
Prefix tuning script for large language models.
Evaluates model performance on multiple-choice questions.
"""
import torch
import datetime
import time
import gc
import numpy as np
from transformers import AutoModelForCausalLM, GenerationConfig, AutoConfig, AutoTokenizer
from utils import load_questions, Prompt, create_prompt_string, parse_for_index, INDEX_TO_LETTER

# Constants
torch.manual_seed(123)
DEFAULT_PADDING_TOKEN_ID = 128009

# ====================== UTILITY FUNCTIONS ======================

def get_current_date_formatted():
    """Returns the current date in 'DD MMM YYYY' format."""
    now = datetime.datetime.now()
    return now.strftime('%d %b %Y')

# ====================== TOKENIZATION FUNCTIONS ======================

def tokenize_prompt(prompt, device, tokenizer, system_prompt=''):
    """Tokenize a prompt with the chat template."""
    message = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
    input_ids = tokenizer.apply_chat_template(
        message, 
        add_generation_prompt=False, 
        return_tensors="pt", 
        date_string=get_current_date_formatted()
    )
    return input_ids[:,:-1].to(device)

def encode_raw(text, tokenizer):
    """Encode text without special tokens."""
    return tokenizer.encode(text, return_tensors="pt", add_special_tokens=False)

def batch_decode(token_ids, tokenizer):
    """
    Decode a batch of token IDs back into text.
    
    Args:
        token_ids (torch.Tensor): Tensor of shape [batch_size, seq_len] containing token IDs
        tokenizer: The tokenizer object with decode method
    
    Returns:
        list: List of decoded strings, one for each sequence in the batch
    """
    
    # Convert to list of token ID sequences and remove padding
    token_lists = [ids[ids != 0].tolist() for ids in token_ids]
    
    # Decode each sequence
    return [tokenizer.decode(tokens) for tokens in token_lists]

def batch_raw_tokenize(prompts, device, tokenizer, padding=None):
    """Tokenize a batch of prompts without the chat template."""
    tokenized_prompts = [encode_raw(prompt, tokenizer) for prompt in prompts]

    if padding is None:
        return torch.cat(tokenized_prompts, dim=0)
    
    # Pad each sequence
    padded = []
    for t in tokenized_prompts:
        # Calculate padding needed
        pad_len = padding - t.size(1)
        # Pad the prompt
        padded_prompt = torch.nn.functional.pad(t, (0, pad_len), value=DEFAULT_PADDING_TOKEN_ID)
        padded.append(padded_prompt)
    
    # Stack all tensors into a single batch
    batched = torch.cat(padded, dim=0)

    # Move to specified device
    return batched.to(device)

def batch_tokenize(prompts, device, tokenizer):
    """Tokenize a batch of prompts with padding and attention masks."""
    suffix = '<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nThe correct answer is ('
    tokenized_suffix = encode_raw(suffix, tokenizer)
    tokenized_prompts = [encode_raw(prompt, tokenizer) for prompt in prompts]
    
    # Find the maximum sequence length for prompts
    max_prompt_len = max(t.size(1) for t in tokenized_prompts)
    
    padded = []
    attention_masks = []
    
    for t in tokenized_prompts:
        # Calculate padding needed
        pad_len = max_prompt_len - t.size(1)
        # Pad the prompt
        padded_prompt = torch.nn.functional.pad(t, (0, pad_len), value=DEFAULT_PADDING_TOKEN_ID)
        # Concatenate with suffix
        padded_tensor = torch.cat([padded_prompt, tokenized_suffix], dim=1)
        padded.append(padded_tensor)
        
        # Create attention mask (1 for real tokens, 0 for padding tokens)
        prompt_mask = torch.ones((1, t.size(1)), dtype=torch.long)
        padding_mask = torch.zeros((1, pad_len), dtype=torch.long)
        suffix_mask = torch.ones((1, tokenized_suffix.size(1)), dtype=torch.long)
        # Combine all masks
        attention_mask = torch.cat([prompt_mask, padding_mask, suffix_mask], dim=1)
        attention_masks.append(attention_mask)
    
    # Stack all tensors into a single batch
    batched = torch.cat(padded, dim=0)
    attention_mask_batched = torch.cat(attention_masks, dim=0)

    # Move to specified device
    return batched.to(device), attention_mask_batched.to(device)

def pad_batch(batch, max_generation_length, tokenizer):
    """Pad a batch of sequences to the same length."""
    padded = []
    for t in batch:
        pad_len = max_generation_length - t.size(1)
        padded_prompt = torch.nn.functional.pad(t, (0, pad_len), value=DEFAULT_PADDING_TOKEN_ID)
        padded.append(padded_prompt)
    return torch.cat(padded, dim=0)

# ====================== EVALUATION FUNCTIONS ======================
def evaluatev2(model_responses, correct_indices):
    """Evaluate model responses against correct answers."""
    correct_count = 0
    refused_count = 0

    for i, response in enumerate(model_responses):
       #print(f"Response: {response[0]}, Correct: {INDEX_TO_LETTER[correct_indices[i]]}")
        if response[0] == INDEX_TO_LETTER[correct_indices[i]]:
            correct_count += 1
        else:
            refused_count += 1

    return correct_count, refused_count


def evaluate(model_responses, correct_indices):
    """Evaluate model responses against correct answers."""
    correct_count = 0
    incorrect_count = 0
    refused_count = 0
    correct_indices_responses = []

    for i, model_response in enumerate(model_responses):
        parsed_response = parse_for_index(model_response)
        if parsed_response is not None:
            if parsed_response == correct_indices[i]:
                correct_count += 1
                correct_indices_responses.append(i)
            else:
                incorrect_count += 1
        else:
            refused_count += 1

    return correct_count, refused_count, correct_indices_responses

def evaluate_prefix(prefix, model, tokenizer, device, temperature):
    """
    Evaluate model performance on the development set.
    
    Args:
        prefix: The trained prefix embeddings
        model: The language model
        tokenizer: The tokenizer
        device: Device to run evaluation on
        
    Returns:
        tuple: (accuracy, refusal_rate)
    """
    print("Evaluating on the test set...")
    model.eval()
    dev_config = GenerationConfig.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    dev_batch_size = 256
    dev_questions, dev_correct_indices, _ = load_questions(dataset='test')  # Using validation as dev set
    dev_max_generation_length = 1
    
    # Development set config
    dev_config.max_new_tokens = dev_max_generation_length
    dev_config.pad_token_id = tokenizer.eos_token_id
    if temperature == 0:
        dev_config.temperature = None
        dev_config.do_sample = False
    else:
        dev_config.temperature = temperature
        dev_config.do_sample = True

    dev_config.use_cache = True
    
    # Evaluation metrics
    correct_predictions = 0
    refusal_count = 0
    processed_count = 0
    
    # List to store correct questions and responses
    correct_data = []
    
    eval_start_time = time.time()
    for i in range(0, len(dev_questions), dev_batch_size):
        # Get current batch
        batch_end = min(i + dev_batch_size, len(dev_questions))
        batch_indices = list(range(i, batch_end))
        batch_size_actual = len(batch_indices)
        
        # Extract batch data
        batch_questions = [dev_questions[j] for j in batch_indices]
        batch_correct_indices = [dev_correct_indices[j] for j in batch_indices]
        
        # Tokenize batch
        batch_encoded_questions, attention_mask = batch_tokenize(batch_questions, device, tokenizer)
        attention_mask = torch.cat((torch.ones((batch_size_actual, prefix.size(1)), device=device), attention_mask), dim=1)
        
        # Generate responses for the batch
        batch_token_ids = model.generate(
            input_ids=batch_encoded_questions,
            prefix=prefix,
            generation_config=dev_config, 
            attention_mask=attention_mask
        )
        
        # Decode outputs
        batch_decoded_texts = batch_decode(batch_token_ids, tokenizer)
        
        # Evaluate accuracy
        batch_correct, batch_refusals = evaluatev2(batch_decoded_texts, batch_correct_indices)
        correct_predictions += batch_correct
        refusal_count += batch_refusals
        processed_count += batch_size_actual
        
    
        # Clean up
        del batch_token_ids, batch_encoded_questions
        gc.collect()
        torch.cuda.empty_cache()
        
    eval_time = time.time() - eval_start_time
    accuracy = correct_predictions / processed_count
    refusal_rate = refusal_count / processed_count
    
    print(f"\nTest Set Evaluation Results:")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Refusal Rate: {refusal_rate:.2%}")
    print(f"Total questions: {processed_count}")
    print(f"Correct predictions: {correct_predictions}")
    print(f"Evaluation time: {eval_time:.2f} seconds")
          
    return accuracy, refusal_rate

# ====================== TRAINING FUNCTIONS ======================

def setup_model_and_tokenizer():
    """Initialize and setup the model and tokenizer."""
    model_id = "meta-llama/Llama-3.2-1B-Instruct"
    print("Loading model and tokenizer...")
    
    #tokenizer = AutoTokenizer.from_pretrained(model_id)
    #tokenizer.pad_token_id = tokenizer.eos_token_id
    
    model_config = AutoConfig.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Load model in full precision for stability
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        config=model_config
    )
    print("Model loaded...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    return model, tokenizer

def create_prefix_embedding(prompt_prefix, model, tokenizer, device, system_prompt):
    """Create and initialize the prefix embedding."""
    tokenized_prefix = tokenize_prompt(prompt_prefix, device, tokenizer, system_prompt=system_prompt)
    with torch.no_grad():
        prefix_embed = model.model.embed_tokens(tokenized_prefix).squeeze(0)
    
    # Make prefix trainable (full precision)
    return torch.nn.Parameter(prefix_embed.detach().clone())

def main():
    """Evaluate model performance on the development set."""
    # Setup model and tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = setup_model_and_tokenizer()

    # Define the prompt template
    baseline_prompt = Prompt(
        persona="You are a helpful assistant.\n",
        task_instructions="Answer the following question.",
        output_format="\"The correct answer is (insert answer letter choice here)\"",
        examples=None
    )
    prompt_prefix = create_prompt_string(baseline_prompt)
    
    # Create prefix embedding
    prefix = create_prefix_embedding(
        prompt_prefix, model, tokenizer, device, baseline_prompt.persona
    )

    accuracy, refusal_rate = evaluate_prefix(prefix, model, tokenizer, device, temperature=0)
    print(f"\nFinal Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Refusal Rate: {refusal_rate:.4f}")
    # Evaluate on dev set
    for i in range(1,11):
        accuracy, refusal_rate = evaluate_prefix(prefix, model, tokenizer, device, temperature=i/10)
        print(f"\nFinal Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Refusal Rate: {refusal_rate:.4f}")

if __name__ == "__main__":
    main() 