import os
from colorama import Fore, Style, init
from unsloth import FastLanguageModel, is_bfloat16_supported
import torch
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Load default values from .env
DEFAULT_MODEL_NAME = os.getenv("DEFAULT_MODEL_NAME", "model")
DEFAULT_DATASET_NAME = os.getenv("DEFAULT_DATASET_NAME", "TBLgGamin/sun_is_blue")
DEFAULT_TRAINING_STEPS = int(os.getenv("DEFAULT_TRAINING_STEPS", 60))

# Global variable to store the final model name
final_model_name = None

# Introduction styling with gradient text
def gradient_text(text):
    colors = [
        "\033[38;5;202m",  # Orange
        "\033[38;5;214m",  # Light Orange
        "\033[38;5;220m",  # Yellow
    ]
    reset = "\033[0m"

    gradient = ""
    color_index = 0
    for char in text:
        if char != ' ':
            gradient += colors[color_index] + char
            color_index = (color_index + 1) % len(colors)
        else:
            gradient += char  # Keep spaces uncolored

    gradient += reset
    return gradient

def display_header():
    header = "Welcome to the AI training program!"
    print(gradient_text(header))
    print("""
     _    _             ___  
    | |  | |           / _ \ 
    | |  | | ___ _ __ / /_\ \\
    | |/\| |/ _ \ '_ \|  _  |
    \  /\  /  __/ | | | | | |
     \/  \/ \___|_| |_\_| |_/
    """)
    print("Web-enabled neural applications\n")
    print(f"Created by {Fore.LIGHTCYAN_EX}\033]8;;https://www.linkedin.com/in/wessel-van-der-vlugt-351213254/\aWessel van der Vlugt\033]8;;\a{Style.RESET_ALL} with help from {Fore.LIGHTCYAN_EX}\033]8;;https://github.com/unsloth\aUnsloth\033]8;;\a{Style.RESET_ALL}")

def get_user_inputs():
    fourbit_models = [
        "unsloth/mistral-7b-v0.3-bnb-4bit",
        "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
        "unsloth/llama-3-8b-bnb-4bit",
        "unsloth/llama-3-8b-Instruct-bnb-4bit",
        "unsloth/llama-3-70b-bnb-4bit",
        "unsloth/Phi-3-mini-4k-instruct",
        "unsloth/Phi-3-medium-4k-instruct",
        "unsloth/mistral-7b-bnb-4bit",
        "unsloth/gemma-7b-bnb-4bit",
    ]

    print("\nAvailable models for finetuning:")
    for i, model in enumerate(fourbit_models):
        print(f"{i + 1}. {model}")

    model_choice = int(input(f"\nChoose a model (1-{len(fourbit_models)}): ")) - 1
    chosen_model = fourbit_models[model_choice]

    dataset_name = input(f"\nEnter the HuggingFace dataset name (default: {DEFAULT_DATASET_NAME}): ") or DEFAULT_DATASET_NAME

    global final_model_name
    final_model_name = input(f"\nEnter the final model name (default: {DEFAULT_MODEL_NAME}): ") or DEFAULT_MODEL_NAME

    print("\nRecommended training steps for use cases:")
    print("1. Testing: 50-60 steps")
    print("2. Proof of concept: 500-600 steps")
    print("3. Production: 2000+ steps")
    training_steps = int(input(f"Enter the number of training steps (default: {DEFAULT_TRAINING_STEPS}): ") or DEFAULT_TRAINING_STEPS)

    return chosen_model, dataset_name, final_model_name, training_steps

def load_and_prepare_model(model_name):
    max_seq_length = 2048
    dtype = None
    load_in_4bit = True

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )
    return model, tokenizer

def prepare_dataset(tokenizer, dataset_name):
    alpaca_prompt = """Below is a question and its corresponding answer. Write a response that appropriately completes the request.

    ### Question:
    {}

    ### Answer:
    {}"""

    EOS_TOKEN = tokenizer.eos_token

    def formatting_prompts_func(examples):
        questions = examples["question"]
        answers = examples["answer"]
        texts = [alpaca_prompt.format(q, a) + EOS_TOKEN for q, a in zip(questions, answers)]
        return {"text": texts}

    dataset = load_dataset(dataset_name, split="train")
    dataset = dataset.map(formatting_prompts_func, batched=True)
    return dataset

def train_model(model, tokenizer, dataset, max_seq_length, training_steps):
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            max_steps=training_steps,
            learning_rate=2e-4,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="outputs",
        ),
    )

    trainer_stats = trainer.train()
    return trainer_stats

def save_model_and_tokenizer(model, tokenizer, final_model_name):
    model.save_pretrained(final_model_name)
    tokenizer.save_pretrained(final_model_name)

def get_final_model_name():
    return final_model_name

def main():
    display_header()
    chosen_model, dataset_name, final_model_name, training_steps = get_user_inputs()
    model, tokenizer = load_and_prepare_model(chosen_model)
    dataset = prepare_dataset(tokenizer, dataset_name)
    train_model(model, tokenizer, dataset, max_seq_length=2048, training_steps=training_steps)
    save_model_and_tokenizer(model, tokenizer, final_model_name)

if __name__ == "__main__":
    main()
