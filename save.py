import os
import time
from shutil import move, copytree, rmtree
from unsloth import FastLanguageModel

# Define the output directory where the model is saved
model_dir = "outputs"  # This should match the output_dir in your train.py script
output_dir = "model"

try:
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

finally:
    # Get the current time to name the folder
    current_time = time.strftime("%Y%m%d-%H%M%S")

    # Create a new directory with the current time
    time_based_dir = os.path.join("archive", current_time)
    os.makedirs(time_based_dir, exist_ok=True)

    # Move the model and outputs directories to the new time-based directory
    copytree(model_dir, os.path.join(time_based_dir, os.path.basename(model_dir)))
    copytree(output_dir, os.path.join(time_based_dir, os.path.basename(output_dir)))

    print(f"All files have been successfully organized in '{time_based_dir}'")
    print("Model and output folders have been cleared.")
