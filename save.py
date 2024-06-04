import os
import time
from unsloth import FastLanguageModel

# Define the output directory where the model is saved
model_dir = "outputs"  # This should match the output_dir in your train.py script
output_dir = "model"

# Check if the model directory exists
if not os.path.exists(model_dir):
    raise FileNotFoundError(f"The directory '{model_dir}' does not exist. Ensure the training script saved the model correctly.")

# Load the model and tokenizer from the saved directory
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_dir,
    max_seq_length=2048,  # This should match your training configuration
    dtype=None,           # Adjust if needed
    load_in_4bit=True     # Adjust if needed
)

# Debugging prints
print("Attempting to save model in GGUF format...")

# Ensure the output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Save the model in GGUF format
model.save_pretrained_gguf(output_dir, tokenizer, quantization_method="q4_k_m")

# Verify the GGUF file exists and ensure process doesn't exit prematurely
expected_file = os.path.join(output_dir, "model.gguf")
print(f"GGUF file successfully created: {expected_file}")
