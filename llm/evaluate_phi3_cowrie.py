import json
import os
import random
import re
import warnings
from collections import Counter

import torch
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu, sentence_bleu
from peft import PeftModel
from rouge_score import rouge_scorer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

warnings.filterwarnings("ignore", category=UserWarning)

MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
ADAPTER_PATH = "phi3-cowrie-lora-adapter"
DATASET_PATH = "./combined_finetune_dataset.jsonl"

GROUND_TRUTH_SAMPLES = 30       
CONSISTENCY_SAMPLES = 10        
CONSISTENCY_RUNS = 3            
HALLUCINATION_THRESHOLD = 0.30  
SEED = 42

MAX_NEW_TOKENS = 128
TEMPERATURE = 0.1
DO_SAMPLE = True

# Phi-3 chat template tokens
_SYS  = "\x3c|system|\x3e"
_USR  = "\x3c|user|\x3e"
_ASST = "\x3c|assistant|\x3e"
_END  = "\x3c|end|\x3e"

SYSTEM_PROMPT = (
    "You are a Linux terminal emulator (Cowrie SSH honeypot). "
    "Respond ONLY with the exact terminal output that the given command "
    "would produce on a Debian-based system. Do not add explanations."
)


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


def hline(char="=", width=80):
    print(char * width)


def section(title: str):
    print()
    hline()
    print(f"  {title}")
    hline()


def kv(key: str, val, indent=2):
    print(" " * indent + f"{key:<40s} {val}")


def compute_rouge(references: list[str], hypotheses: list[str]) -> dict:
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )
    totals = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    per_sample_rougeL = []

    for ref, hyp in zip(references, hypotheses):
        scores = scorer.score(ref, hyp)
        for key in totals:
            totals[key] += scores[key].fmeasure
        per_sample_rougeL.append(scores["rougeL"].fmeasure)

    n = len(references)
    averages = {k: v / n for k, v in totals.items()}
    return averages, per_sample_rougeL


def compute_consistency(
    model, tokenizer, samples: list[dict], device: str, runs: int = 3
) -> float:
    consistent_count = 0

    for i, sample in enumerate(samples, 1):
        prompt = build_prompt(sample["instruction"])
        outputs = []
        for _ in range(runs):
            out = generate_response(model, tokenizer, prompt, device)
            outputs.append(out.strip())

        # All runs produced the same output?
        if len(set(outputs)) == 1:
            consistent_count += 1
            tag = "CONSISTENT"
        else:
            tag = "INCONSISTENT"

        print(f"  [{i:>2}/{len(samples)}] [{tag:<12}] $ {sample['instruction'][:55]}")

    return (consistent_count / len(samples)) * 100 if samples else 0.0


def compute_hallucination_rate(
    per_sample_rougeL: list[float], threshold: float
) -> tuple[float, int]:
    hallucinated = sum(1 for s in per_sample_rougeL if s < threshold)
    rate = (hallucinated / len(per_sample_rougeL)) * 100 if per_sample_rougeL else 0.0
    return rate, hallucinated

def main():
    random.seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    section("1 / 4  Loading Model (4-bit quantised)")

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
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("  Applying LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()

    section("2 / 4  Ground-Truth Evaluation (BLEU + ROUGE)")

    gt_data = load_jsonl(DATASET_PATH)
    gt_samples = random.sample(gt_data, min(GROUND_TRUTH_SAMPLES, len(gt_data)))

    references_text = []
    hypotheses_text = []
    references_corpus = []
    hypotheses_corpus = []
    smoother = SmoothingFunction()

    for i, sample in enumerate(gt_samples, 1):
        instruction = sample["instruction"]
        expected = sample["output"].strip()
        prompt = build_prompt(instruction)
        generated = generate_response(model, tokenizer, prompt, device)

        references_text.append(expected)
        hypotheses_text.append(generated)

        ref_tokens = expected.split() or [""]
        hyp_tokens = generated.split() or [""]
        references_corpus.append([ref_tokens])
        hypotheses_corpus.append(hyp_tokens)

        s_bleu = sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoother.method1)
        match_tag = "HIGH" if s_bleu > 0.5 else "LOW "
        print(f"  [{i:>2}/{len(gt_samples)}] [BLEU {s_bleu:.2f} {match_tag}] $ {instruction[:50]}")
    bleu_score = corpus_bleu(
        references_corpus, hypotheses_corpus,
        smoothing_function=smoother.method1
    )

    rouge_averages, per_sample_rougeL = compute_rouge(references_text, hypotheses_text)

    hallucination_rate, hallucinated_count = compute_hallucination_rate(
        per_sample_rougeL, HALLUCINATION_THRESHOLD
    )

    section("3 / 4  Consistency Test")

    consistency_samples = random.sample(
        gt_data, min(CONSISTENCY_SAMPLES, len(gt_data))
    )
    consistency_rate = compute_consistency(
        model, tokenizer, consistency_samples, device, runs=CONSISTENCY_RUNS
    )

    section("4 / 4  EVALUATION REPORT")

    print("\n  A. BLEU Score")
    print("  " + "-" * 76)
    kv("Corpus BLEU", f"{bleu_score:.4f}")

    print("\n  B. ROUGE Scores")
    print("  " + "-" * 76)
    kv("ROUGE-1 (F1)", f"{rouge_averages['rouge1']:.4f}")
    kv("ROUGE-2 (F1)", f"{rouge_averages['rouge2']:.4f}")
    kv("ROUGE-L (F1)", f"{rouge_averages['rougeL']:.4f}")

    print("\n  C. Consistency")
    print("  " + "-" * 76)
    kv("Samples tested", f"{len(consistency_samples)}")
    kv("Runs per sample", f"{CONSISTENCY_RUNS}")
    kv("Consistency Rate", f"{consistency_rate:.1f}%")

    print("\n  D. Hallucination Rate")
    print("  " + "-" * 76)
    kv("ROUGE-L threshold", f"< {HALLUCINATION_THRESHOLD}")
    kv("Hallucinated samples", f"{hallucinated_count} / {len(gt_samples)}")
    kv("Hallucination Rate", f"{hallucination_rate:.1f}%")

    print("\n  E. Summary")
    print("  " + "-" * 76)
    kv("BLEU", f"{bleu_score:.4f}")
    kv("ROUGE-1 / ROUGE-2 / ROUGE-L",
       f"{rouge_averages['rouge1']:.4f} / {rouge_averages['rouge2']:.4f} / {rouge_averages['rougeL']:.4f}")
    kv("Consistency Rate", f"{consistency_rate:.1f}%")
    kv("Hallucination Rate", f"{hallucination_rate:.1f}%")

    hline()
    print("  Evaluation complete.")
    hline()
    print()


if __name__ == "__main__":
    main()