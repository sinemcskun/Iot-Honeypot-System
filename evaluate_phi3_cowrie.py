"""
evaluate_phi3_cowrie.py
=======================
Academic evaluation script for the fine-tuned Phi-3 Cowrie SSH honeypot model.
Runs local inference with 4-bit quantization and calculates:
  1. Exact Match Rate (%) and BLEU Score on ground-truth samples.
  2. AI Refusal Rate (%) on unseen commands.
"""

import json
import os
import random
import re
import warnings

import pandas as pd
import torch
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

warnings.filterwarnings("ignore", category=UserWarning)

MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"

ADAPTER_PATH = "phi3-cowrie-lora-adapter"
DATASET_PATH = "./combined_finetune_dataset.jsonl"
CSV_PATH = "Processed_Data.csv"

GROUND_TRUTH_SAMPLES = 20
UNSEEN_SAMPLES = 20 
SEED = 42

MAX_NEW_TOKENS = 128
TEMPERATURE = 0.1
DO_SAMPLE = True

_SYS  = "\x3c|system|\x3e"
_USR  = "\x3c|user|\x3e"
_ASST = "\x3c|assistant|\x3e"
_END  = "\x3c|end|\x3e"

SYSTEM_PROMPT = (
    "You are a Linux terminal emulator (Cowrie SSH honeypot). "
    "Respond ONLY with the exact terminal output that the given command "
    "would produce on a Debian-based system. Do not add explanations."
)

AI_REFUSAL_PHRASES = [
    "i am an ai", "as an ai", "i cannot", "i can't", "i'm sorry", "sorry",
    "i apologize", "as a language model", "as an assistant", "here is",
    "here's", "i don't have", "i do not have", "it seems like", "it looks like",
    "please note", "note that", "unfortunately", "i'm not able", "i am not able",
    "i'm unable", "i am unable", "however,", "sure!", "certainly!", "of course!"
]

def build_prompt(instruction: str) -> str:
    return (
        f"{_SYS}\n{SYSTEM_PROMPT}{_END}\n"
        f"{_USR}\n{instruction}{_END}\n"
        f"{_ASST}\n"
    )

def clean_output(raw_text: str, prompt: str) -> str:
    if raw_text.startswith(prompt):
        generated = raw_text[len(prompt):]
    else:
        generated = raw_text
    for token in [_END, _ASST, _USR, _SYS, "\x3c/s\x3e", "\x3cs\x3e"]:
        generated = generated.replace(token, "")
    return generated.strip()

def detect_ai_refusal(text: str) -> bool:
    lower = text.lower()
    return any(phrase in lower for phrase in AI_REFUSAL_PHRASES)

def load_jsonl(path: str) -> list:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

@torch.no_grad()
def generate_response(model, tokenizer, prompt: str, device: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        do_sample=DO_SAMPLE,
        pad_token_id=tokenizer.eos_token_id,
    )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=False)
    return clean_output(decoded, prompt)

LINE_WIDTH = 80
def hline(char="="): print(char * LINE_WIDTH)
def section(title: str): print(f"\n{char * LINE_WIDTH}\n  {title}\n{char * LINE_WIDTH}")
def kv(key: str, val, indent=2): print(" " * indent + f"{key:<40s} {val}")

# =============================================================================

def main():
    random.seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n================================================================================")
    print("  1 / 4  Loading Model (4-bit quantised)")
    print("================================================================================")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    print("  Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, quantization_config=bnb_config, device_map={"": 0},
        trust_remote_code=False, attn_implementation="eager"
    )

    print("  Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=False)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    print("  Applying LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()

    print("\n================================================================================")
    print("  2 / 4  Ground-Truth Evaluation (BLEU & Exact Match)")
    print("================================================================================")

    gt_data = load_jsonl(DATASET_PATH)
    gt_samples = random.sample(gt_data, min(GROUND_TRUTH_SAMPLES, len(gt_data)))

    exact_matches = 0
    references_corpus = []
    hypotheses_corpus = []
    smoother = SmoothingFunction()

    for i, sample in enumerate(gt_samples, 1):
        instruction = sample["instruction"]
        expected    = sample["output"].strip()
        prompt      = build_prompt(instruction)
        generated   = generate_response(model, tokenizer, prompt, device)

        if generated.strip() == expected.strip():
            exact_matches += 1

        ref_tokens = expected.split() or [""]
        hyp_tokens = generated.split() or [""]
        references_corpus.append([ref_tokens])
        hypotheses_corpus.append(hyp_tokens)

        tag = "MATCH" if generated.strip() == expected.strip() else "DIFF "
        print(f"  [{i:>2}/{len(gt_samples)}] [{tag}] $ {instruction[:60]}")

    exact_match_pct = (exact_matches / len(gt_samples)) * 100
    bleu = corpus_bleu(references_corpus, hypotheses_corpus, smoothing_function=smoother.method1)

    print("\n================================================================================")
    print("  3 / 4  Unseen Generalization (AI Refusal Test)")
    print("================================================================================")

    csv_cols = ["source_cowrie", "request_data"]
    df = pd.read_csv(CSV_PATH, usecols=csv_cols, low_memory=False)

    # Filtre gevşetildi: Sadece source_cowrie olanların isteklerini al
    mask = (df["source_cowrie"] == 1) & (df["event_type"].str.contains("command", case=False, na=False))
    cowrie_cmds = df.loc[mask, "request_data"].dropna().unique().tolist()
    del df

    known_instructions = set(s["instruction"] for s in gt_data)
    unseen_cmds = [c for c in cowrie_cmds if c.strip() and c not in known_instructions]

    print(f"  Total unique cowrie commands : {len(cowrie_cmds)}")
    print(f"  Truly unseen (not in JSONL)  : {len(unseen_cmds)}")

    if not unseen_cmds:
        print("  No unseen commands available for testing. Skipping.")
        return

    unseen_sample = random.sample(unseen_cmds, min(UNSEEN_SAMPLES, len(unseen_cmds)))

    refusals = 0
    unseen_results = []

    for i, cmd in enumerate(unseen_sample, 1):
        prompt    = build_prompt(cmd)
        generated = generate_response(model, tokenizer, prompt, device)
        refused   = detect_ai_refusal(generated)

        if refused: refusals += 1
        unseen_results.append({"cmd": cmd, "output": generated, "refused": refused})

        tag = "REFUSED" if refused else "OK     "
        print(f"  [{i:>2}/{len(unseen_sample)}] [{tag}] $ {cmd[:60]}")

    refusal_pct = (refusals / len(unseen_sample)) * 100 if unseen_sample else 0

    print("\n================================================================================")
    print("  4 / 4  EVALUATION REPORT")
    print("================================================================================")

    print("\n  A. Unseen Commands & Generated Outputs")
    print("  " + "-" * 76)
    for j, r in enumerate(unseen_results, 1):
        status = "REFUSED" if r["refused"] else "PASSED"
        print(f"\n  [{j}] Command : {r['cmd']}")
        print(f"      Status  : {status}")
        print(f"      Output  :")
        for line in r["output"].split("\n"):
            print(f"        {line}")

    print("\n  B. Summary Metrics")
    print("  " + "-" * 76)
    kv("Ground-Truth Samples", f"{len(gt_samples)}")
    kv("Exact Match Rate", f"{exact_match_pct:.1f}%")
    kv("Corpus BLEU Score", f"{bleu:.4f}")
    print()
    kv("Unseen Commands Tested", f"{len(unseen_sample)}")
    kv("AI Refusal Rate", f"{refusal_pct:.1f}%")

    print("\n================================================================================")
    print("  Evaluation complete.")
    print("================================================================================\n")

if __name__ == "__main__":
    main()