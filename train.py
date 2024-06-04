from colorama import Fore, Style, init
# Introduction styling with gradient text
def gradient_text(text):
    """
    Styles the given text with a gradient from orange to yellow.
    """
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

from unsloth import FastLanguageModel
import torch
max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
fourbit_models = [
    "unsloth/mistral-7b-v0.3-bnb-4bit",      # New Mistral v3 2x faster!
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    "unsloth/llama-3-8b-bnb-4bit",           # Llama-3 15 trillion tokens model 2x faster!
    "unsloth/llama-3-8b-Instruct-bnb-4bit",
    "unsloth/llama-3-70b-bnb-4bit",
    "unsloth/Phi-3-mini-4k-instruct",        # Phi-3 2x faster!
    "unsloth/Phi-3-medium-4k-instruct",
    "unsloth/mistral-7b-bnb-4bit",
    "unsloth/gemma-7b-bnb-4bit",             # Gemma 2.2x faster!
] # More models at https://huggingface.co/unsloth

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3-8b-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)


#LorA adapters!
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)


#Data prep! #You can ChatGPT this to make it fit your dataset!
alpaca_prompt = """Below is a question and its corresponding answer. Write a response that appropriately completes the request.

### Question:
{}

### Answer:
{}"""

EOS_TOKEN = tokenizer.eos_token  # Must add EOS_TOKEN
def formatting_prompts_func(examples):
    questions = examples["question"]
    answers   = examples["answer"]
    texts = []
    for question, answer in zip(questions, answers):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = alpaca_prompt.format(question, answer) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }
pass

from datasets import load_dataset
dataset = load_dataset("TBLgGamin/sun_is_blue", split = "train")
dataset = dataset.map(formatting_prompts_func, batched = True,)


#Training!
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 60,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)

trainer_stats = trainer.train()

if True: model.save_pretrained_gguf("model", tokenizer, quantization_method = "q4_k_m")