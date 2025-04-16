#!/home/crellis/miniconda3/envs/transformers2/bin/python
import random
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
import os
import torch
from transformers import AutoTokenizer
import torch.nn as nn
torch.manual_seed(123)
import numpy as np
import logging
import gc
import json
import wandb  # Add wandb import
from raw_llama import Llama3Model, LLAMA32_CONFIG, Tokenizer, ChatFormat, load_weights_into_llama
from utils import Prompt, create_prompt_string, load_questions, setup_logging, generate_experiment_name, decode_optimized_embeddings, INDEX_TO_LETTER, get_current_date_formatted, tokenize_prompt
from evaluator import setup_model_and_tokenizer, evaluate_prefix

def evaluatev2(model_responses, correct_indices):
    """Evaluate model responses against correct answers."""
    correct_count = 0
    refused_count = 0

    for i, response in enumerate(model_responses):
        if response == INDEX_TO_LETTER[correct_indices[i]]:
            correct_count += 1
        else:
            refused_count += 1

    return correct_count, refused_count

def create_batches(indices, encoded_questions, batch_size):
    num_batches = (len(encoded_questions) + batch_size - 1) // batch_size
    batches = []
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(encoded_questions))
        batch = {
            'indices': indices[start_idx:end_idx],
            'encoded': encoded_questions[start_idx:end_idx],
        }
        # Add debug logging for batch creation
        logging.debug(f"Created batch {i+1}/{num_batches} with {len(batch['encoded'])} questions")
        batches.append(batch)
    return batches

def encode_raw(text, tokenizer):
    """Encode text without special tokens."""
    return tokenizer.encode(text, return_tensors="pt", add_special_tokens=False)

def create_prefix_embedding(prompt_prefix, model, tokenizer, device, system_prompt):
    """Create and initialize the prefix embedding."""
    tokenized_prefix = tokenize_prompt(prompt_prefix, device, tokenizer, system_prompt=system_prompt)
    with torch.no_grad():
        prefix_embed = model.tok_emb(tokenized_prefix).squeeze(0)
    
    # Make prefix trainable (full precision)
    return torch.nn.Parameter(prefix_embed.detach().clone())

def batch_tokenize(prompts, device, tokenizer):
    """Tokenize a batch of prompts with padding and attention masks."""
    suffix = '<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nThe correct answer is ('
    tokenized_prompts = [encode_raw(prompt+suffix, tokenizer).to(device) for prompt in prompts]


    # Move to specified device
    return tokenized_prompts

def optimize_prompt(
    model,
    tokenizer,
    best_embeddings,
    num_iterations=100,
    learning_rate=1e-4,
    device=None,
    batch_size=8,
    early_stopping_threshold=0.000001,
    patience=5,
    log_dir="logs",
    experiment_name=None,
    train_sample_size=20,
    max_tokens=300,
    weight_decay=0.01,  # Added weight decay parameter
):
    # Setup logging
    timestamp = setup_logging(log_dir)
    if experiment_name is None:
        experiment_name = f"prompt_opt_{timestamp}"
    
    
    # Log initial embeddings as table
    initial_embedding_data = best_embeddings.detach().cpu().flatten().tolist()[:10]  # Only log first 10 values
    wandb.log({"initial_embedding_preview": wandb.Table(
        data=[[i, val] for i, val in enumerate(initial_embedding_data)],
        columns=["index", "value"]
    )})
    
    # Improved optimizer with weight decay for regularization
    optimizer = torch.optim.AdamW([best_embeddings], lr=learning_rate, weight_decay=weight_decay)
    
    criterion = nn.CrossEntropyLoss()
 
    # Freeze model parameters
    for param in model.parameters():
        param.requires_grad = False
    
    model.eval()
    
    # Dictionary to track metrics
    metrics_history = {
        'train_loss': [],
        'learning_rate': [],
        'val_loss': [],
    }
    
    # Variables for early stopping
    best_loss = float('inf')
    patience_counter = 0
    
    # Directory for embeddings
    embeddings_dir = os.path.join(log_dir, f"{experiment_name}_embeddings")
    os.makedirs(embeddings_dir, exist_ok=True)
    
    # Save the initial embeddings
    initial_embeddings_file = os.path.join(embeddings_dir, "initial_embeddings.pt")
    torch.save(best_embeddings.detach().cpu(), initial_embeddings_file)
    
    logging.info("Starting training loop...")
    for iteration in range(num_iterations):
        # Load fresh training data for each iteration
        logging.info(f"Loading training data...")
        raw_questions, correct_index, correct_answer = load_questions(
            sample_size=train_sample_size
        )
                
        # Split into training and validation
        val_size = max(16, int(len(raw_questions) * 0.2))  # 20% for validation, at least 16 samples
        train_size = len(raw_questions) - val_size
        
        # Training data
        train_raw_questions = raw_questions[:train_size]
        train_correct_index = correct_index[:train_size]
        
        # Validation data
        val_raw_questions = raw_questions[train_size:]
        val_correct_index = correct_index[train_size:]
        
        # Convert to embeddings
        logging.info("Converting questions to embeddings...")
        tokenized_train = batch_tokenize(train_raw_questions, device, tokenizer)
        embed_train_questions = [model.tok_emb(question) for question in tokenized_train]
        tokenized_val = batch_tokenize(val_raw_questions, device, tokenizer)
        embed_val_questions = [model.tok_emb(question) for question in tokenized_val]

        logging.info(f"Successfully created {len(embed_train_questions)} training and {len(embed_val_questions)} validation embeddings")
        
        # Create batches for this iteration
        logging.info("Creating training and validation batches...")
        train_batches = create_batches(train_correct_index, embed_train_questions, batch_size)
        val_batches = create_batches( val_correct_index, embed_val_questions, batch_size)
        
        logging.info(f"Created {len(train_batches)} training and {len(val_batches)} validation batches")
        
        # Training phase
        train_losses = []
        correct_predictions = 0
        total_predictions = 0
        model.eval()
        
        for batch_idx, batch in enumerate(train_batches):
            # Process all samples in the batch
            batch_questions = batch['encoded']
            batch_indices = batch['indices']
            
            optimizer.zero_grad()
            batch_loss = 0.0
            
            # Process each question separately
            for q_idx, (question_emb, correct_idx) in enumerate(zip(batch_questions, batch_indices)):
                # Get the target answer as a letter (A, B, C, D)
                target_letter = INDEX_TO_LETTER[correct_idx]
                target_token_id = encode_raw(target_letter, tokenizer)[0]  # Get token ID for the letter
                
                # Forward pass with the prefix and question
                current_embeddings = torch.cat((best_embeddings.unsqueeze(0), question_emb), dim=1)
                logits = model.trf_blocks(current_embeddings)
                logits = model.final_norm(logits)
                logits = model.out_head(logits[:, -1, :].to(torch.bfloat16))
                
                # Calculate loss comparing to the target letter token
                target = torch.tensor([target_token_id], device=device)
                q_loss = criterion(logits, target)
                
                if q_loss.item() is not None and q_loss.item() > 0:
                    batch_loss += q_loss.item()
                    q_loss.backward()
                
                # Predict and log (for debugging)
                next_token = torch.argmax(logits, dim=-1).cpu().tolist()[0]
                pred_letter = tokenizer.decode([next_token])
                logging.debug(f'Prediction: {pred_letter}, Target: {target_letter}')
                
                # Track accuracy for wandb
                total_predictions += 1
                if pred_letter == target_letter:
                    correct_predictions += 1
                
                # Clean up to save memory
                del current_embeddings, q_loss, logits, target
                torch.cuda.empty_cache()
                
                # Update weights periodically within the batch
                if q_idx % 2 == 0 or q_idx == len(batch_questions) - 1:
                    optimizer.step()
                    optimizer.zero_grad()
            
            # Average loss for the batch
            batch_loss /= len(batch_questions)
            train_losses.append(batch_loss)
            logging.info(f"Batch {batch_idx} training loss: {batch_loss:.6f}")
        
        # Calculate training accuracy
        train_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        
        # Validation phase
        val_losses = []
        val_correct_predictions = 0
        val_total_predictions = 0
        model.eval()
        
        with torch.no_grad():
            if len(val_batches) == 0:
                logging.warning("No validation batches available. Skipping validation.")
            else:
                logging.info(f"Running validation on {len(val_batches)} batches")
                
                for batch_idx, batch in enumerate(val_batches):
                    batch_questions = batch['encoded']
                    batch_indices = batch['indices']
                    
                    batch_val_loss = 0.0
                    
                    # Process each question separately
                    for q_idx, (question_emb, correct_idx) in enumerate(zip(batch_questions, batch_indices)):
                        # Get the target answer as a letter (A, B, C, D)
                        target_letter = INDEX_TO_LETTER[correct_idx]
                        target_token_id = encode_raw(target_letter, tokenizer)[0]  # Get token ID for the letter
                        
                        # Forward pass with the prefix and question
                        current_embeddings = torch.cat((best_embeddings.unsqueeze(0), question_emb), dim=1)
                        logits = model.trf_blocks(current_embeddings)
                        logits = model.final_norm(logits)
                        logits = model.out_head(logits[:, -1, :].to(torch.bfloat16))
                        
                        # Calculate loss comparing to the target letter token
                        target = torch.tensor([target_token_id], device=device)
                        q_loss = criterion(logits, target)
                        
                        if q_loss.item() is not None and q_loss.item() > 0:
                            batch_val_loss += q_loss.item()
                        
                        # Track validation accuracy
                        next_token = torch.argmax(logits, dim=-1).cpu().tolist()[0]
                        pred_letter = tokenizer.decode([next_token])
                        val_total_predictions += 1
                        if pred_letter == target_letter:
                            val_correct_predictions += 1
                        
                        # Clean up to save memory
                        del current_embeddings, q_loss, logits, target
                        torch.cuda.empty_cache()
                    
                    # Average loss for the batch
                    batch_val_loss /= len(batch_questions)
                    val_losses.append(batch_val_loss)
                    logging.info(f"Batch {batch_idx} validation loss: {batch_val_loss:.6f}")
        
        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses) if val_losses else float('inf')
        val_accuracy = val_correct_predictions / val_total_predictions if val_total_predictions > 0 else 0
        
        logging.info(f"Validation loss calculation: {len(val_losses)} batches, avg_val_loss={avg_val_loss:.6f}")
        
        metrics_history['train_loss'].append(avg_train_loss)
        metrics_history['val_loss'].append(avg_val_loss)
        metrics_history['learning_rate'].append(optimizer.param_groups[0]['lr'])
        
        # Log embedding norm - useful for tracking divergence
        embedding_norm = torch.norm(best_embeddings).item()
        
        # Log to wandb with more metrics
        wandb.log({
            'loss/train': avg_train_loss,
            'loss/val': avg_val_loss,
            'accuracy/train': train_accuracy,
            'accuracy/val': val_accuracy,
            'learning_rate': optimizer.param_groups[0]['lr'],
            'embedding/norm': embedding_norm,
            'embedding/mean': best_embeddings.mean().item(),
            'embedding/std': best_embeddings.std().item(),
            'patience_counter': patience_counter
        }, step=iteration)
        
        # Early stopping check (using validation loss if available)
        monitor_loss = avg_val_loss if val_losses else avg_train_loss
        if monitor_loss < best_loss - early_stopping_threshold:
            best_loss = monitor_loss
            patience_counter = 0
            # Save best model
            best_emb_file = os.path.join(embeddings_dir, "best_embeddings.pt")
            best_embedding = best_embeddings.detach().clone()
            torch.save(best_embedding.cpu(), best_emb_file)
            logging.info(f"New best loss: {best_loss:.6f}, saved best model")
            
            # Log best embedding to wandb as artifact
            embedding_artifact = wandb.Artifact(
                name=f"best_embedding_{experiment_name}", 
                type="embedding",
                description=f"Best embedding at iteration {iteration}"
            )
            embedding_artifact.add_file(best_emb_file)
            wandb.log_artifact(embedding_artifact)
        else:
            patience_counter += 1
            logging.info(f"No improvement in loss, patience counter: {patience_counter}/{patience}")

        # Memory cleanup after each iteration
        torch.cuda.empty_cache()
        
        # Log metrics 
        log_message = f"\nIteration {iteration}\n"
        log_message += f"Training Loss: {avg_train_loss:.4f}\n"
        log_message += f"Validation Loss: {avg_val_loss:.4f}\n"
        log_message += f"Training Accuracy: {train_accuracy:.4f}\n"
        log_message += f"Validation Accuracy: {val_accuracy:.4f}\n"
        log_message += f"Current learning rate: {optimizer.param_groups[0]['lr']:.6f}"
        logging.info(log_message)
        
        # Early stopping check
        if patience_counter >= patience:
            logging.info(f"Early stopping at iteration {iteration}")
            break
    
    # Save metrics history
    metrics_file = os.path.join(log_dir, f"{experiment_name}_metrics.json")
    with open(metrics_file, 'w') as f:
        json.dump(metrics_history, f, indent=2)
    
    # Log final embedding preview
    final_embedding_data = best_embeddings.detach().cpu().flatten().tolist()[:10]  # Only log first 10 values
    wandb.log({"final_embedding_preview": wandb.Table(
        data=[[i, val] for i, val in enumerate(final_embedding_data)],
        columns=["index", "value"]
    )})
    
    
    # Return only the best embedding
    return best_embedding

def load_trainer_model(config):
    weights_file = "Llama-3.2-1B-Instruct/model.safetensors"
    if not os.path.exists(weights_file):
        raise FileNotFoundError(f"Model file {weights_file} not found. Please ensure the model is downloaded locally.")
    model = Llama3Model(config)
    
    combined_weights = load_file(weights_file)
    load_weights_into_llama(model, LLAMA32_CONFIG, combined_weights)

    del combined_weights

    return model

def main():
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    
    model = load_trainer_model(LLAMA32_CONFIG)
    model.to(device)

    # Define the prompt template
    baseline_prompt = Prompt(
        persona="You are an accomplished expert in interpreting and explaining topics covered in the MMLU dataset, ensuring accurate, in-depth answers.\n",
        task_instructions="Please answer the following question.",
        output_format="\"The correct answer is (insert answer letter choice here)\"",
        examples=None
    )
    prompt_prefix = create_prompt_string(baseline_prompt)
    
    # Create prefix embedding
    best_embedding = create_prefix_embedding(
        prompt_prefix, model, tokenizer, device, baseline_prompt.persona
    )

    del model


    evaluator_model, _ = setup_model_and_tokenizer()

    # Initialize wandb for initial evaluation
    wandb.init(
        project="prompt_sweep_training",
        name=f"initial_evaluation_{get_current_date_formatted()}",
        job_type="evaluation"
    )
    best_accuracy = 0

    for i in range(10):
        # Evaluate only the best embedding
        evaluator_model, _ = setup_model_and_tokenizer()
        accuracy, refusal_rate = evaluate_prefix(best_embedding, evaluator_model, tokenizer, device, 0.4)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
        # Log initial metrics
        wandb.log({
            "evaluated_accuracy": accuracy,
            "evaluated_refusal_rate": refusal_rate
        })
        
        del evaluator_model
        torch.cuda.empty_cache()
        gc.collect()


        model = load_trainer_model(LLAMA32_CONFIG)
        model.to(device)


        experiment_name = generate_experiment_name()
        optimized_embedding = optimize_prompt(
            model=model,
            tokenizer=tokenizer,
            device=device,
            best_embeddings = best_embedding,
            num_iterations=2,
            train_sample_size=128,
            batch_size=16,
            learning_rate=5e-4,
            early_stopping_threshold=0.00001,
            patience=5,
            weight_decay=0.01,
            log_dir="prompt_optimization_logs",
            experiment_name=experiment_name,
        )

        del model
        torch.cuda.empty_cache()
        gc.collect()
        print(f"Best accuracy: {best_accuracy}")
        # Initialize wandb for final evaluation
        with open(f"embedding_{experiment_name}_final.pt", "wb") as f:
            torch.save(optimized_embedding.cpu(), f)
    


if __name__ == "__main__":
    main()
