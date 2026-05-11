import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer, SFTConfig

MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
DATASET_PATH = "combined_finetune_dataset.jsonl"
OUTPUT_DIR = "./phi3-cowrie-lora-adapter"

MAX_SEQ_LENGTH = 512
_USER   = "<|user|>"
_ASST   = "<|assistant|>"
_END    = "<|end|>"
_SYSTEM = "<|system|>"
SYSTEM_PROMPT = (
    "You are a Linux terminal emulator (Cowrie SSH honeypot). "
    "Respond ONLY with the exact terminal output that the given command "
    "would produce on a Debian-based system. Do not add explanations."
)

print("[*] Loading dataset...")
dataset = load_dataset("json", data_files=DATASET_PATH, split="train")

def format_example(example):
    return f"{_SYSTEM}\n{SYSTEM_PROMPT}{_END}\n{_USER}\n{example['instruction']}{_END}\n{_ASST}\n{example['output']}{_END}\n"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=False)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map={"": 0},
    trust_remote_code=False,
    attn_implementation="eager",
)
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)

num_epochs = 5
per_device_batch_size = 1
grad_acc_steps = 8
warmup_ratio_val = 0.03

total_training_steps = (len(dataset) // (per_device_batch_size * grad_acc_steps)) * num_epochs
warmup_steps_val = int(total_training_steps * warmup_ratio_val)

training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=num_epochs,
    per_device_train_batch_size=per_device_batch_size,
    gradient_accumulation_steps=grad_acc_steps,
    optim="paged_adamw_8bit",
    learning_rate=2e-4,
    weight_decay=0.01,
    warmup_steps=warmup_steps_val, 
    lr_scheduler_type="cosine",
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=2,
    report_to="none"
)


trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    formatting_func=format_example,
    args=training_args
)

print("[*] Starting training...")
trainer.train()

trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"[+] SUCCESS: LoRA adapter saved to {OUTPUT_DIR}")
