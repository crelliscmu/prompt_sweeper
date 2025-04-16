# Prefix Tuning for Multiple-Choice Performance Optimization

This repository contains tools for optimizing the performance of language models (specifically Llama-3.2-1B-Instruct) on multiple-choice questions through prefix tuning.

## Project Overview

This project implements an  approach to improve language model performance on multiple-choice tasks by learning optimized prefix embeddings. The system uses a modified implementation of the Llama 3.2 model that allows for:

1. Direct optimization of prefix embeddings
2. Efficient evaluation on multiple-choice benchmarks like MMLU
3. Custom prompt formatting and multiple-choice answer processing

## Features

- **Prefix Embedding Optimization**: Train custom prefix embeddings that improve performance on multiple-choice tasks
- **MMLU Dataset Integration**: Automated loading and evaluation using the MMLU benchmark
- **Custom Model Implementation**: Modified Llama 3.2 implementation with support for prefix inputs
- **Evaluation Tools**: Comprehensive evaluation pipeline to measure model accuracy
- **Batched Processing**: Efficient batch processing for both training and evaluation

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- CUDA-compatible GPU (for efficient training) (I ran on 1x H100)

### Setup

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install custom Transformers modifications:
   
   Follow the instructions in `transformers_mods/transformers_mods_install.md` to install the required modifications to the Hugging Face Transformers library.

4. Download the Llama-3.2-1B-Instruct model from Hugging Face:
   ```bash
   huggingface-cli download meta-llama/Llama-3.2-1B-Instruct --local-dir Llama-3.2-1B-Instruct
   ```

## Quick Start

To quickly test the project with a pre-trained embedding:

1. Make sure you've completed the installation steps above

2. Run the training script:
   ```bash
   python train_v4.py
   ```

3. The script will:
   - Load the model and tokenizer
   - Try to load a pre-trained embedding (or create a random one for demonstration)
   - Run a quick evaluation on the MMLU test set
   - Report the results

To train your own embedding, see the Usage section below.

## Usage

### Training a Prefix

To train an optimized prefix, use the `train_v4.py` script:

```bash
python train_v4.py
```

This will:
1. Load the Llama 3.2 model
2. Initialize a baseline prompt prefix embedding
3. Optimize the prefix using multiple-choice questions from MMLU
4. Save the optimized embedding for later use
5. Log training metrics using wandb

### Evaluating a Prefix

To evaluate a trained prefix on the test set, use the `evaluator.py` script:

```bash
python evaluator.py
```

This will:
1. Load the model and a default prompt prefix
2. Evaluate the prefix on multiple-choice questions
3. Report accuracy and refusal rates

To evaluate a custom embedding, modify the script to load your saved embedding.

## Project Structure

- `train_v4.py`: Main training script for optimizing prefix embeddings
- `evaluator.py`: Evaluation utilities for testing embeddings on MMLU
- `raw_llama.py`: Custom implementation of the Llama 3.2 model
- `utils.py`: Utility functions for data loading, prompt formatting, and evaluation
- `transformers_mods/`: Directory containing modifications to the Transformers library

## Configuration

You can customize various parameters in the training script:

- `num_iterations`: Number of training iterations
- `learning_rate`: Learning rate for optimization
- `train_sample_size`: Number of training examples to use
- `batch_size`: Batch size for training
- `early_stopping_threshold`: Threshold for early stopping
- `patience`: Number of iterations without improvement before early stopping
- `weight_decay`: Weight decay for regularization

## Results

The optimized prefixes have shown significant improvements over baseline prompts:

- Accuracy improvements of 10-15% on MMLU multiple-choice tasks
- Reduced refusal rates
- Better performance consistency across different temperature settings

## Transformers Modifications

The project requires specific modifications to the Hugging Face Transformers library to enable efficient prefix tuning. These modifications include:

### Modified Files

1. **modeling_llama.py**
   - Added support for prefix concatenation with input embeddings
   - Modified the forward pass to accept prefix embeddings as input
   - Optimized attention calculations for better performance with prefixes

2. **utils.py**
   - Added utility functions for handling prefix embeddings
   - Implemented custom tokenization flows for the prefix tuning process

3. **configuration_utils.py**
   - Modified configuration classes to support prefix tuning parameters
   - Added options for controlling prefix behavior

### Installation Process

A detailed installation guide is provided in `transformers_mods/transformers_mods_install.md`, which includes:

1. How to locate your existing Transformers installation
2. How to backup original files before modification
3. How to install the modified files
4. How to verify the installation

These modifications are necessary because they enable the model to efficiently process and learn from prefix embeddings, which is not natively supported in the standard Transformers implementation.

## Current Work

I am currently working on implementing a tuning parameters for better tuning of the prefix embeddings.

## Future Work

- Implimenting additional datasets beyond MMLU.
- Implementing additional metrics to evaluate the performance of the prefix embeddings.
- Implementing a more efficient prefix tuning approach that allows for better performance with less computational resources.
- Integrating MCTS or another approach to find candidate prompts for prefixes.

You can extend this project by:

1. Implementing additional datasets beyond MMLU
2. Trying different architectures for the prompt prefix
3. Experimenting with different optimization strategies
4. Scaling to larger models

## Citations

```
@book{build-llms-from-scratch-book,
  author       = {Sebastian Raschka},
  title        = {Build A Large Language Model (From Scratch)},
  publisher    = {Manning},
  year         = {2024},
  isbn         = {978-1633437166},
  url          = {https://www.manning.com/books/build-a-large-language-model-from-scratch},
  github       = {https://github.com/rasbt/LLMs-from-scratch}
}
```

