"""
finetune_roberta_cowrie.py
──────────────────────────────────────────────────────────────────────────────
Phase 3: RoBERTa Fine-tuning for Domain-Aware Scoring

Bu script:
- combined_finetune_dataset.jsonl dosyasını okur (instruction + output).
- Metinleri birleştirerek MLM (Masked Language Modeling) formatında hazırlar.
- Lokaldeki roberta-large modeline LoRA (query, value, r=8, alpha=16) uygular.
- HuggingFace Trainer ile eğitir.
- Adaptörü ./roberta-cowrie-lora/ içine kaydeder.
- Son olarak adaptörü baz modelle birleştirir ve ./roberta-cowrie-merged/ içine
  kaydeder ki BERTScore fonksiyonu direkt bu yoldan modeli yükleyebilsin.
──────────────────────────────────────────────────────────────────────────────
"""

import json
import torch
from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model
from datasets import Dataset

# ═══════════════════════════════════════════════════════════════════════════
#  PATHS
# ═══════════════════════════════════════════════════════════════════════════
BASE_DIR       = Path(__file__).resolve().parent
DATASET_PATH   = BASE_DIR / "combined_finetune_dataset.jsonl"
ROBERTA_PATH   = "/content/drive/MyDrive/roberta-large"
OUTPUT_ADAPTER = BASE_DIR / "roberta-cowrie-lora"
OUTPUT_MERGED  = BASE_DIR / "roberta-cowrie-merged"
OUTPUT_TRAIN   = BASE_DIR / "roberta-cowrie-training"

def load_data() -> Dataset:
    texts = []
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            instr = data.get("instruction", "").strip()
            out = data.get("output", "").strip()
            
            # MLM için instruction ve output'u arka arkaya ekliyoruz
            # Böylece model Cowrie terminal çıktılarına ve Linux komutlarına aşina olur
            text_pair = f"{instr}\n{out}"
            texts.append(text_pair)
            
    return Dataset.from_dict({"text": texts})

def main():
    print(f"\n[1/5] Loading tokenizer from {ROBERTA_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(ROBERTA_PATH)

    print("[2/5] Preparing dataset...")
    dataset = load_data()
    print(f"      Loaded {len(dataset)} samples.")
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=512,
            return_special_tokens_mask=True
        )

    tokenized_datasets = dataset.map(
        tokenize_function, 
        batched=True, 
        remove_columns=["text"],
        desc="Tokenizing dataset"
    )

    print(f"\n[3/5] Loading Base Model for MLM from {ROBERTA_PATH}...")
    model = AutoModelForMaskedLM.from_pretrained(ROBERTA_PATH)

    print("\n      Applying LoRA (target_modules: query, value; r=8; alpha=16)...")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["query", "value"],
        lora_dropout=0.05,
        bias="none",
        task_type=None  # MLM is standard feature extraction for PEFT
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # MLM Veri Toplayıcı (Data Collator) - %15 maskeleme oranı
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15
    )

    print("\n[4/5] Training...")
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_TRAIN),
        #overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        save_steps=500,
        save_total_limit=2,
        logging_steps=50,
        learning_rate=5e-4,
        fp16=torch.cuda.is_available(),
        report_to="none" # wandb vs kapatmak icin
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_datasets,
    )

    trainer.train()

    print(f"\n[5/5] Saving Models...")
    print(f"      Saving adapter only to: {OUTPUT_ADAPTER}")
    model.save_pretrained(str(OUTPUT_ADAPTER))
    tokenizer.save_pretrained(str(OUTPUT_ADAPTER))

    print(f"      Merging adapter and saving full model to: {OUTPUT_MERGED}")
    # Merge and unload, modeli baseline ile birleştirip kalıcı hale getirir
    # Bu sayede bert_score_fn doğrudan bu dizini model_type olarak kullanabilir
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(str(OUTPUT_MERGED))
    tokenizer.save_pretrained(str(OUTPUT_MERGED))

    print("\nDone! RoBERTa fine-tuning is complete.")

if __name__ == "__main__":
    main()