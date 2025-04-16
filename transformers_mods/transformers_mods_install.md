# Installing Custom Transformers Modifications

This guide explains how to integrate the custom modifications from the `transformers_mods` directory into your Hugging Face Transformers installation.

## Prerequisites

- An existing Python environment with Hugging Face Transformers installed
- The custom modification files from the `transformers_mods` directory

## Files to Install

The `transformers_mods` directory contains the following modified files:

- `modeling_llama.py`: Custom modifications to the LLaMA model implementation
- `utils.py`: Utility functions with custom modifications
- `configuration_utils.py`: Modified configuration utilities

## Installation Steps

1. **Locate your Transformers package installation**

   First, find where your Transformers package is installed in your Python environment:

   ```bash
   python -c "import transformers; print(transformers.__path__)"
   ```

2. **Back up original files**

   Before replacing any files, create backups of the original files:

   ```bash
   # Navigate to your transformers installation directory
   cd /path/to/transformers/package
   
   # Create backups
   cp models/llama/modeling_llama.py models/llama/modeling_llama.py.backup
   cp utils/utils.py utils/utils.py.backup  
   cp configuration_utils.py configuration_utils.py.backup
   ```

3. **Copy modified files**

   Copy the custom files to their respective locations in the Transformers package:

   ```bash
   # Copy modeling_llama.py to the llama models directory
   cp /path/to/transformers_mods/modeling_llama.py /path/to/transformers/package/models/llama/
   
   # Copy utils.py to the appropriate directory
   cp /path/to/transformers_mods/utils.py /path/to/transformers/package/utils/
   
   # Copy configuration_utils.py to the root directory
   cp /path/to/transformers_mods/configuration_utils.py /path/to/transformers/package/
   ```

   The exact paths may vary depending on your Transformers version and installation method.

4. **Alternative: Use development installation**

   For a cleaner approach, especially if you're developing or testing modifications:

   ```bash
   # Clone the transformers repository
   git clone https://github.com/huggingface/transformers.git
   cd transformers
   
   # Install in development mode
   pip install -e .
   
   # Now copy your modified files to the appropriate locations in this directory
   cp /path/to/transformers_mods/modeling_llama.py ./src/transformers/models/llama/
   cp /path/to/transformers_mods/utils.py ./src/transformers/
   cp /path/to/transformers_mods/configuration_utils.py ./src/transformers/
   ```

## Verification

To verify that your modifications have been applied correctly:

```python
import transformers
from transformers import LlamaModel

# Check if your custom model changes are loaded
model = LlamaModel.from_pretrained("meta-llama/Llama-2-7b")
```

## Restoring Original Files

If you need to revert to the original files:

```bash
# Navigate to your transformers installation
cd /path/to/transformers/package

# Restore from backups
cp models/llama/modeling_llama.py.backup models/llama/modeling_llama.py
cp utils/utils.py.backup utils/utils.py
cp configuration_utils.py.backup configuration_utils.py
```

## Troubleshooting

- If you encounter import errors, ensure the files were copied to the correct location
- For version conflicts, make sure your modifications are compatible with your installed Transformers version
- Consider installing Transformers in development mode for easier testing and reverting of changes 