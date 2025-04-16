from datasets import load_dataset
import re
LETTER_TO_INDEX = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
INDEX_TO_LETTER = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
import random
from itertools import product
import os
import torch
import logging
import datetime
from itertools import product
torch.manual_seed(123)

DEFAULT_PADDING_TOKEN_ID = 128009

class Question:
    def __init__(self,
                 question,
                 answer_choices,
                 correct_answer,
                 correct_index):
        self.question: str = question
        self.answer_choices = answer_choices
        self.correct_answer = correct_answer
        self.correct_index = correct_index
        self.answer_formats: list[str] = format_multiple_choice_variants(answer_choices) # converts answer choices into multiple variants

class Prompt:
    def __init__(self,
                 persona,             # role or persona of the assistant
                 task_instructions,    # Overview of the task including the primary goal
                 output_format,       # Format of the output
                 examples=None,       # Example outputs or reference formats to guide the answer
                ):
        
        self.persona: str = persona
        self.task_instructions: str = task_instructions
        self.output_format: str = output_format
        self.answer_format: int = 0
        self.all_examples: list[Example] = examples
        self.selected_examples: list[Example] = None
        
class Example:
    def __init__(self, question, choice1, choice2, choice3, choice4, correct_answer, correct_index):
        self.question = question
        self.choice1 = choice1
        self.choice2 = choice2
        self.choice3 = choice3
        self.choice4 = choice4
        self.correct_answer = correct_answer
        self.correct_index = correct_index
        self.cot = None
        self.answer_formats = format_simple_multiple_choice([choice1, choice2, choice3, choice4])



def format_examples(examples: list[Example], answer_format: int):
    formatted_examples = []
    for example in examples:
        # Split the formatted answer choices into lines
        answer_lines = example.answer_formats[answer_format].strip().split('\n')
        
        # Find the line containing the correct answer
        formatted_correct_answer = ""
        for line in answer_lines:
            if example.correct_answer in line:
                formatted_correct_answer = line
                break
        
        example_str = f"Question: {example.question}\n"
        example_str += f"Answer Choices: {example.answer_formats[answer_format]}\n"
        if example.cot:
            example_str += f"Chain of Thought: {example.cot}\n"
        example_str += f"Correct Answer: {formatted_correct_answer}\n"
        formatted_examples.append(example_str)
    return "\n".join(formatted_examples)

def create_prompt_string(prompt: Prompt): 
    """ Creates a prompt string from a Prompt object in the order: 
    "persona", 
    "task_instructions",  
    "output_format", 
    "examples"
    """

    prompt_str = ""
    
    #if prompt.persona:
    #    prompt_str += f"{prompt.persona}\n"
    if prompt.task_instructions:
        prompt_str += f"{prompt.task_instructions}\n"
    if prompt.output_format:
        prompt_str += f"Please output your response in the following format: {prompt.output_format}"
        prompt_str += f"\n\n"
    return prompt_str


def format_multiple_choice_variants(choices):
    """
    Formats a list of answer choices into multiple variants for both alphabetical
    and numerical multiple-choice styles, and returns a single list of all variants.

    For each style (alphabetical and numerical) the following is used:

    Base wrappers for labels:
      - Alphabetical: uses letters (A, B, C, …)
      - Numerical: uses numbers (1, 2, 3, …)

    Four base wrappers:
      1. No opening punctuation with a closing parenthesis, e.g., "A)" or "1)"
      2. Wrapped in parentheses, e.g., "(A)" or "(1)"
      3. No wrapper, e.g., "A" or "1"
      4. Wrapped in square brackets, e.g., "[A]" or "[1]"
      5. Wrapped in curly braces, e.g., "{A}" or "{1}"

    Modifications (applied in all combinations):
      - Colon insertion: If True, a colon is inserted between the label and the option.
      - Hyphen prefix: If True, a hyphen is prefixed to the label.
      - Lowercase conversion: For alphabetical labels, converts the letter to lowercase 
        (has no effect on numerical labels).

    Returns:
        list: A list of multi-line string variants. The first half of the list contains 
              alphabetical variants and the second half contains numerical variants.
    """
    variants = []
    
    # Define the base wrappers.
    base_wrappers = [
        {"open": "(",  "close": ")"},  # e.g., (A) or (1)
        {"open": "",   "close": ""},   # e.g., A or 1
        {"open": "",   "close": ")"},  # e.g., A) or 1)
        {"open": "[",  "close": "]"},  # e.g., [A] or [1]
        {"open": "{",  "close": "}"}   # e.g., {A} or {1}
    ]
    
    # Generate alphabetical variants.
    for wrapper in base_wrappers:
        for colon, hyphen, lowercase in product([False, True], repeat=3):
            formatted_variant = ""
            for i, choice in enumerate(choices):
                # Use letter labels (fallback to "Option" if beyond A-Z).
                letter = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[i] if i < 26 else f"Option{i+1}"
                if lowercase:
                    letter = letter.lower()
                label = f"{wrapper['open']}{letter}{wrapper['close']}"
                if hyphen:
                    label = "-" + label
                if colon:
                    formatted_line = f"{label}: {choice}"
                else:
                    formatted_line = f"{label} {choice}"
                formatted_variant += formatted_line + "\n"
            variants.append(formatted_variant)
    
    return variants


def parse_for_index(response):
    patterns = [
        r"the\s+correct\s+answer\s+is\s*\(?([abcd])\)?",
        # Match phrases like "final answer is: $\boxed{(B)}$" or similar variants.
        r"(?:the\s+)?final\s+answer\s+is[:\s]*\$?\\?boxed\{?\(?([abcd])\)?\}?\$?",
        # Match "the correct answer is (C)" and similar variants.
        r"(?:the\s+)?correct\s+answer\s+is[:\s]*\(?([abcd])\)?",
        # Match "answer is (B)" or "answer is: (B)".
        r"(?:the\s+)?answer\s+is[:\s]*\(?([abcd])\)?",
        # Match "answer: (A)".
        r"(?:the\s+)?answer[:\s]*\(?([abcd])\)?",
        # Match a single letter only on a line.
        r"^\s*([abcd])\s*$",
        # Match "option (D)".
        r"option[:\s]*\(?([abcd])\)?",
        # Match "choice (A)".
        r"choice[:\s]*\(?([abcd])\)?",
        # Match "I choose (B)" or "I select (B)"
        r"i (?:choose|select)[:\s]*\(?([abcd])\)?",
        # Match "The answer should be (C)"
        r"(?:the\s+)?answer\s+should\s+be[:\s]*\(?([abcd])\)?",
        # Match "(A) is correct" or "(A) is the correct answer"
        r"\(?([abcd])\)?\s+is\s+(?:the\s+)?correct(?:\s+answer)?",
        # Match LaTeX boxed answers like $\boxed{A}$ or \boxed{A}
        r"\$?\\boxed\{([abcd])\}\$?",
        # Match "correct answer is D)" or "correct answer: D"
        r"correct\s+answer(?:\s+is)?[:\s]*\(?([abcd])\)?",
        # Match "Therefore, ... D)" pattern
        r"[Tt]herefore,.*?\(?([abcd])\)",
        # Match "I believe the answer is (A)" or similar
        r"i\s+believe\s+(?:the\s+)?(?:answer|choice)\s+is[:\s]*\(?([abcd])\)?",
        
        # Match "Based on ... the answer is (A)"
        r"based\s+on.*?(?:the\s+)?answer\s+is[:\s]*\(?([abcd])\)?",
        
        # Match "Let's solve this step by step... answer is (A)"
        r"step\s+by\s+step.*?(?:the\s+)?answer\s+is[:\s]*\(?([abcd])\)?",
        
        # Match "After analyzing... answer is (A)"
        r"after\s+(?:analyzing|calculating).*?(?:the\s+)?answer\s+is[:\s]*\(?([abcd])\)?",
        
        # Match "This gives us... answer (A)"
        r"(?:this|which)\s+gives\s+us.*?(?:the\s+)?(?:answer|choice)[:\s]*\(?([abcd])\)?",
        
        # Match "The solution is (A)" or "The result is (A)"
        r"(?:the\s+)?(?:solution|result)\s+is[:\s]*\(?([abcd])\)?",
        
        # Match "We can conclude that (A)" or "We can conclude the answer is (A)"
        r"we\s+can\s+conclude\s+(?:that|the\s+answer\s+is)[:\s]*\(?([abcd])\)?",
        
        # Match "Final result: (A)" or "Final answer: (A)"
        r"final\s+(?:result|answer)[:\s]*\(?([abcd])\)?",
    ]
    
    # Convert response to lowercase for case-insensitive matching
    response_lower = response.lower()
    
    for pattern in patterns:
        match = re.search(pattern, response_lower)
        if match:
            return LETTER_TO_INDEX[match.group(1).upper()]
    
    return None


def evaluate_responses( model_responses, correct_index):

# GPQA / MMLU response analysis
    correct_indices = []
    incorrect_indices = []
    refused_indices = []

    for i, model_response in enumerate(model_responses):
        parsed_response = parse_for_index(model_response)
        if parsed_response is not None:
            if parsed_response == correct_index[i]:
                correct_indices.append(i)
            else:
                incorrect_indices.append(i)
        else:
            refused_indices.append(i)


    accuracy = len(correct_indices) / len(model_responses)
    refusal_rate = len(refused_indices) / len(model_responses)

    return accuracy, refusal_rate


def format_simple_multiple_choice(choices):

    formatted = ""
    for i, choice in enumerate(choices):
        letter = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[i] if i < 26 else f"Option{i+1}"
        formatted += f"({letter}) {choice}\n"
    return formatted

def get_mmlu_dataset(ds_split="train", sample_size=None, num_examples=None):
    """
    Load MMLU dataset for a given split with optional sampling.
    
    Args:
        sample_size: Number of examples to randomly sample (if None, returns all loaded examples)
        ds_split: Dataset split ("train", "validation", or "test")
        num_examples: Number of examples to initially load
    ß
    Returns:
        Tuple of (question_strings, correct_indices, correct_answers)
    """
    # If num_examples is provided, limit the dataset size
    if num_examples is not None:
        ds_range = f"[0:{num_examples}]"
    else:
        ds_range = ""
        
    # Load the dataset
    ds = load_dataset("cais/mmlu", "all", split=f"{ds_split}{ds_range}")
    
    # If sample_size is provided and smaller than the dataset, sample indices first
    total_size = len(ds)
    process_indices = list(range(total_size))
    
    if sample_size is not None and sample_size < total_size:
        process_indices = random.sample(process_indices, sample_size)
    
    # Process only the necessary examples
    question_strings = []
    correct_indices = []
    correct_answers = []
    
    for i in process_indices:
        question = ds[i]["question"]
        # Get the original answer list and correct answer index
        answer_list = ds[i]["choices"]
        answer_index = ds[i]["answer"]
        answer = answer_list[answer_index]
        
        # Shuffle the answer list and update the correct index accordingly
        random.shuffle(answer_list)
        new_index = answer_list.index(answer)
        
        q_str = f"\nQuestion: {question}\n"
        q_str += f"Answer Choices:\n{format_simple_multiple_choice(answer_list)}"
        question_strings.append(q_str)
        
        correct_indices.append(new_index)
        correct_answers.append(f'The correct answer is ({INDEX_TO_LETTER[new_index]})<|eot_id|>')
    
    return question_strings, correct_indices, correct_answers


def load_questions(dataset="train", sample_size=None, num_examples=None):
    try:
        if dataset == "train":
            question_strings, correct_indices, correct_answers = get_mmlu_dataset("auxiliary_train", sample_size=sample_size, num_examples=num_examples)
        elif dataset == "validation":
            question_strings, correct_indices, correct_answers = get_mmlu_dataset("validation", sample_size=sample_size, num_examples=num_examples)
        elif dataset == "test":
            question_strings, correct_indices, correct_answers = get_mmlu_dataset("test", sample_size=sample_size, num_examples=num_examples)
        elif dataset == "dev":
            question_strings, correct_indices, correct_answers = get_mmlu_dataset("dev", sample_size=sample_size, num_examples=num_examples)
        return question_strings, correct_indices, correct_answers
    except Exception as e:
        logging.error(f"Error loading questions: {e}")
        raise   

def get_current_date_formatted():
    """Returns the current date in 'DD MMM YYYY' format."""
    now = datetime.datetime.now()
    return now.strftime('%d %b %Y')

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

def generate_from_embeddings(model, tokenizer, embeddings, question_embeddings, max_tokens=300, temperature=0.0, device='cuda', top_k=40, top_p=0.9):
    """
    Generate a response using optimized prompt embeddings and question embeddings.
    Includes improvements like top-p sampling and proper caching.
    """
    model.eval()
    
    # Combine prompt embeddings and question embeddings
    context = torch.cat([embeddings, question_embeddings], dim=1)
    
    generated_ids = []
    past_key_values = None  # For caching attention computations
    
    # Generate tokens one by one
    with torch.no_grad():
        for _ in range(max_tokens):
            # Forward pass through the model with attention caching
            if past_key_values is None:
                logits = model.trf_blocks(context)
                # Here we would save key/values from attention if model supported it
            else:
                # Use cached key/values if model supported it
                # This would be model specific implementation
                logits = model.trf_blocks(context[:, -1:])
            
            logits = model.final_norm(logits)
            logits = model.out_head(logits.to(torch.bfloat16))
            
            # Get the logits for the last token
            next_token_logits = logits[:, -1, :]
            
            # Apply temperature and sampling techniques
            if temperature > 0:
                # Filter with top_k
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits = next_token_logits.masked_fill(indices_to_remove, -float('inf'))
                
                # Apply temperature
                probs = torch.nn.functional.softmax(next_token_logits / temperature, dim=-1)
                
                # Apply top_p (nucleus) sampling
                if top_p < 1.0:
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Shift the indices to the right to keep also the first token above the threshold
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = torch.zeros_like(probs, dtype=torch.bool).scatter_(
                        -1, sorted_indices, sorted_indices_to_remove
                    )
                    probs = probs.masked_fill(indices_to_remove, 0.0)
                    # Renormalize
                    probs = probs / probs.sum(dim=-1, keepdim=True)
                
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy decoding
                next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            
            generated_ids.append(next_token.item())
            
            # Check for EOS token or other stopping criteria
            if next_token.item() == tokenizer.tokenizer.special_tokens["<|eot_id|>"]:
                break
                
            # Update context with the new token's embedding
            with torch.no_grad():
                next_token_emb = model.tok_emb(next_token.to(device))
                context = torch.cat([context, next_token_emb], dim=1)
    
    # Decode the generated token IDs to text
    response = tokenizer.decode(generated_ids)
    
    return response

def setup_logging(log_dir="logs"):
    """
    Set up logging to both console and file
    """
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Get current timestamp for unique log file
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_{timestamp}.log")
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # This will output to console
        ]
    )
    
    # Return the timestamp for other file naming
    return timestamp


def generate_experiment_name():
    """Generate a descriptive experiment name with a timestamp"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"mmlu_prompt_opt_{timestamp}"



def setup_logging(log_dir="logs"):
    """
    Set up logging to both console and file
    """
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Get current timestamp for unique log file
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_{timestamp}.log")
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # This will output to console
        ]
    )
    
    # Return the timestamp for other file naming
    return timestamp


def decode_optimized_embeddings(model, embeddings, tokenizer, device=None):
    """
    Decode optimized embeddings with a dummy question appended to see the full prompt structure.
    Outputs only the highest probability token for each embedding.
    
    Args:
        model: The model containing the token embedding matrix
        embeddings: The optimized embeddings to decode
        tokenizer: The tokenizer for encoding/decoding
        device: The device to use (defaults to embeddings device)
    """
    # Use provided device or infer from embeddings
    if device is None:
        device = model.device
    embeddings = embeddings.to(torch.bfloat16)
    # Get the full embedding matrix
    with torch.no_grad():
        embedding_matrix = model.tok_emb.weight
        # Normalize embedding matrix once for efficiency
        embedding_matrix_norm = embedding_matrix / embedding_matrix.norm(dim=1, keepdim=True)
        
        # First decode the optimized embeddings
        decoded_tokens = []
        logging.info("Decoding optimized embeddings:")
        
        for i in range(embeddings.shape[1]):
            current_embedding = embeddings[0, i, :].unsqueeze(0)
            # Normalize embeddings for proper cosine similarity
            current_embedding_norm = current_embedding / current_embedding.norm(dim=1, keepdim=True)
            
            # Get similarity scores with all tokens in vocabulary
            similarities = torch.matmul(current_embedding_norm, embedding_matrix_norm.T).squeeze(0)
            
            # Get the most similar token ID and its similarity
            top_similarity, top_index = torch.max(similarities, dim=0)
            
            # Add to our decoded tokens list
            closest_token_id = top_index.item()
            decoded_tokens.append(closest_token_id)
            
            # Print just the best token and its similarity
            token_text = tokenizer.decode([closest_token_id])
            similarity = top_similarity.item()
            logging.info(f"Token {i}: ID {closest_token_id} ('{token_text}') - similarity: {similarity:.4f}")
    
    
    # Convert token IDs to text
    original_prompt = tokenizer.decode(decoded_tokens)
    
    logging.info("\nOriginal optimized prompt:")
    logging.info(original_prompt)

    
    # Return both the decoded prompts and the token IDs
    return {
        'original_prompt': original_prompt,
        'original_token_ids': decoded_tokens,
    }