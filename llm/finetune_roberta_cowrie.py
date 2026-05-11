"""
finetune_roberta_cowrie.py  —  Phase 3: Domain-Aware RoBERTa Scorer
─────────────────────────────────────────────────────────────────────
Fine-tunes RoBERTa-large with LoRA adapters for Masked Language Modeling
on Cowrie terminal command/response pairs.  The resulting model is used
as the BERTScore reference encoder in evaluate_phi3_cowrie_v3.py under
the "lora_finetuned_domain_roberta" condition.

Pipeline:
  1. Load combined_finetune_dataset.jsonl
  2. Tokenize and apply dynamic 15% MLM masking
  3. Fine-tune RoBERTa-large with LoRA (r=8, α=16)
  4. Save LoRA adapter  → roberta-cowrie-lora/
  5. Merge + save full model → roberta-cowrie-merged/

Install (Colab cell):
  !pip install -q peft transformers accelerate bitsandbytes datasets
  from google.colab import drive; drive.mount('/content/drive')
"""

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

# ────────────────────────────────  PATHS  ────────────────────────────────────
BASE_DIR     = Path(__file__).resolve().parent
MODEL_NAME   = "roberta-large"
DATASET_PATH = str(BASE_DIR / "combined_finetune_dataset.jsonl")
ADAPTER_OUT  = str(BASE_DIR / "roberta-cowrie-lora")
MERGED_OUT   = str(BASE_DIR / "roberta-cowrie-merged")

# ──────────────────────────  HYPERPARAMETERS  ────────────────────────────────
LORA_R            = 8
LORA_ALPHA        = 16
LORA_DROPOUT      = 0.05
LORA_TARGET       = ["query", "value"]   # RoBERTa attention modules

MLM_PROBABILITY   = 0.15
MAX_SEQ_LENGTH    = 128
BATCH_SIZE        = 16
LEARNING_RATE     = 2e-4
NUM_EPOCHS        = 3
WARMUP_RATIO      = 0.06
WEIGHT_DECAY      = 0.01
SEED              = 42

DRIVE_CKPT = Path("/content/drive/MyDrive/roberta_cowrie_checkpoints")
LOCAL_CKPT = Path("/content/roberta_checkpoints")
CKPT_DIR   = DRIVE_CKPT if Path("/content/drive/MyDrive").exists() else LOCAL_CKPT


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA
# ═══════════════════════════════════════════════════════════════════════════════

def load_texts(jsonl_path: str) -> List[str]:
    """
    Extract free-form text from the finetune dataset.
    Each entry is expected to have 'instruction' and 'output' fields.
    Both are concatenated as: "<instruction> [SEP] <output>"
    """
    texts: List[str] = []
    with open(jsonl_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            instr  = str(entry.get("instruction", "")).strip()
            output = str(entry.get("output",      "")).strip()
            if instr or output:
                texts.append(f"{instr} [SEP] {output}")

    print(f"  Loaded {len(texts)} text samples from {jsonl_path}")
    return texts


class CowrieMLMDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    random.seed(SEED)
    torch.manual_seed(SEED)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n  Device: {device}")
    print(f"  Checkpoint dir: {CKPT_DIR}\n")

    # ── 1. Tokenizer ──────────────────────────────────────────────────────────
    print("  Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # ── 2. Data ───────────────────────────────────────────────────────────────
    texts = load_texts(DATASET_PATH)
    if not texts:
        raise RuntimeError(f"No samples found in {DATASET_PATH}")

    print("  Tokenizing...")
    encodings = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=MAX_SEQ_LENGTH,
        return_special_tokens_mask=True,
    )
    dataset = CowrieMLMDataset(encodings)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=MLM_PROBABILITY,
    )

    # ── 3. Model + LoRA ───────────────────────────────────────────────────────
    print("  Loading RoBERTa-large...")
    base_model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)

    lora_config = LoraConfig(
        task_type=TaskType.TOKEN_CLS,   # closest fit for MLM
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET,
        bias="none",
    )

    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    # ── 4. Training ───────────────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=str(CKPT_DIR),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        save_strategy="epoch",
        logging_steps=20,
        fp16=(device == "cuda"),
        dataloader_num_workers=0,
        report_to="none",
        seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    print("\n  Starting LoRA fine-tuning...")
    trainer.train()

    # ── 5. Save adapter ───────────────────────────────────────────────────────
    print(f"\n  Saving LoRA adapter → {ADAPTER_OUT}")
    os.makedirs(ADAPTER_OUT, exist_ok=True)
    model.save_pretrained(ADAPTER_OUT)
    tokenizer.save_pretrained(ADAPTER_OUT)

    # ── 6. Merge + save full model ────────────────────────────────────────────
    print(f"  Merging weights and saving → {MERGED_OUT}")
    merged = model.merge_and_unload()
    os.makedirs(MERGED_OUT, exist_ok=True)
    merged.save_pretrained(MERGED_OUT)
    tokenizer.save_pretrained(MERGED_OUT)

    print("\n  Done.  Use roberta-cowrie-merged/ as the BERTScore model_type.")
    print(f"  Example:\n    from bert_score import score")
    print(f"    P, R, F1 = score(hyps, refs, model_type='{MERGED_OUT}')")


if __name__ == "__main__":
    main()
