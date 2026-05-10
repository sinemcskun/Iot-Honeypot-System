import sys
import json
import math
import sqlite3
import numpy as np
import joblib
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

from analysis.ml_preparation.preprocessor import MLPreprocessor

SCALER_PATH      = ROOT / "output" / "scaler.pkl"
BOT_MODEL_PATH   = ROOT / "output" / "model_results" / "bot_detection_best_model.joblib"
TUNNEL_MODEL_PATH= ROOT / "output" / "model_results" / "tunnel_detection_best_model.joblib"
MULTI_MODEL_PATH = ROOT / "output" / "model_results" / "multilabel_best_model.joblib"
FEATURE_COLS_PATH= ROOT / "output" / "ml_ready" / "feature_columns.json"

MULTILABEL_NAMES = [
    "Bruteforce", "Malware Dropper", "Reconnaissance", "Lateral Movement",
    "Credential Spray", "Tunneling", "Port Scan", "Service Interaction", "Network Probe",
]

SENSITIVE_PORTS = {21, 22, 23, 80, 443, 3389}


def _entropy(text: str) -> float:
    if not text:
        return 0.0
    n = len(text)
    freq: dict = {}
    for ch in text:
        freq[ch] = freq.get(ch, 0) + 1
    return -sum((c / n) * math.log2(c / n) for c in freq.values())


def _list_entropy(items: list) -> float:
    items = [i for i in items if i]
    if not items:
        return 0.0
    n = len(items)
    freq: dict = {}
    for i in items:
        freq[i] = freq.get(i, 0) + 1
    return -sum((c / n) * math.log2(c / n) for c in freq.values())


def compute_session_features(session_id: str) -> dict:
    db_path = ROOT.parent / "central" / "database" / "honeyiot_central.db"
    if not db_path.exists():
        db_path = ROOT.parent / "Processed_Data.db"
    if not db_path.exists():
        raise FileNotFoundError("Database not found")

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM Preprocessed_Log WHERE session_id = ? ORDER BY timestamp",
        (session_id,),
    ).fetchall()
    conn.close()

    if not rows:
        raise ValueError(f"Session '{session_id}' not found")

    rows = [dict(r) for r in rows]  # convert to plain dicts so .get() works

    timestamps   = [r["timestamp"]    for r in rows if r["timestamp"]]
    # Use UNIQUE commands — SessionBuilder joins with .unique() so cmd_count == cmd_unique_count in training
    commands_raw = [r["request_data"] for r in rows if r["request_data"]]
    commands     = list(dict.fromkeys(commands_raw))   # deduplicate, preserve order
    cmd_ts_raw   = [(r["timestamp"], r["request_data"]) for r in rows if r["request_data"] and r["timestamp"]]
    # Keep timestamps for unique commands only (first occurrence)
    seen: set = set()
    cmd_ts: list = []
    for ts_val, cmd in cmd_ts_raw:
        if cmd not in seen:
            seen.add(cmd)
            cmd_ts.append(ts_val)

    dest_ports   = list({str(r["dest_port"]) for r in rows if r["dest_port"]})
    usernames    = list({r["username"]  for r in rows if r["username"]})
    passwords    = list({r["password"]  for r in rows if r["password"]})
    # Use actual protocol column (TCP/UDP/SSH…) matching SessionBuilder aggregation
    protocols    = list({str(r["protocol"]) for r in rows if r.get("protocol") and str(r["protocol"]).strip()})
    dns_queries  = list({r["dns_query"] for r in rows if r.get("dns_query") and str(r["dns_query"]).strip()})
    http_uris    = list({r["http_uri"]  for r in rows if r.get("http_uri")  and str(r["http_uri"]).strip()})

    # Duration
    try:
        ts = pd.to_datetime(timestamps, errors="coerce", utc=True).dropna()
        duration_sec = float((ts.max() - ts.min()).total_seconds()) if len(ts) >= 2 else 0.0
    except Exception:
        duration_sec = 0.0

    # Inter-command delays (over unique-command timestamps)
    try:
        ct = pd.to_datetime(cmd_ts, errors="coerce", utc=True).dropna()
        if len(ct) >= 2:
            delays = np.diff(ct.values).astype("timedelta64[ms]").astype(float) / 1000.0
            min_delay = float(np.min(delays))
            max_delay = float(np.max(delays))
        else:
            min_delay = max_delay = 0.0
    except Exception:
        min_delay = max_delay = 0.0

    # cmd_count == cmd_unique_count (matches training — SessionBuilder stored unique commands only)
    cmd_count         = len(commands)
    cmd_unique_count  = len(commands)
    cmd_entropy       = _list_entropy(commands)
    cmd_avg_length    = float(np.mean([len(c) for c in commands])) if commands else 0.0
    cmd_max_length    = float(max((len(c) for c in commands), default=0))
    payload_entropy_avg = float(np.mean([_entropy(c) for c in commands])) if commands else 0.0
    payload_entropy_max = float(max((_entropy(c) for c in commands), default=0.0))
    avg_payload_length  = cmd_avg_length

    port_nums       = [int(p) for p in dest_ports if p.isdigit()]
    port_range_span = float(max(port_nums) - min(port_nums)) if len(port_nums) >= 2 else 0.0
    has_sensitive   = int(any(p in SENSITIVE_PORTS for p in port_nums))
    protocol_count  = float(len(protocols))

    dns_query_count      = float(len(dns_queries))
    dns_avg_query_length = float(np.mean([len(q) for q in dns_queries])) if dns_queries else 0.0
    http_unique_uris     = float(len(http_uris))  # already unique

    unique_usernames     = float(len(usernames))   # already unique
    password_entropy_avg = float(np.mean([_entropy(p) for p in passwords])) if passwords else 0.0

    event_count      = float(len(rows))
    avg_cmd_per_event= cmd_count / event_count if event_count > 0 else 0.0
    log_duration     = math.log1p(duration_sec)
    log_event_count  = math.log1p(event_count)
    entropy_ratio    = cmd_entropy / payload_entropy_avg if payload_entropy_avg > 0 else 0.0

    return {
        "duration_sec":         duration_sec,
        "event_count":          event_count,
        "min_inter_cmd_delay":  min_delay,
        "max_inter_cmd_delay":  max_delay,
        "cmd_count":            float(cmd_count),
        "cmd_unique_count":     float(cmd_unique_count),
        "cmd_entropy":          cmd_entropy,
        "cmd_avg_length":       cmd_avg_length,
        "cmd_max_length":       cmd_max_length,
        "port_range_span":      port_range_span,
        "has_sensitive_ports":  float(has_sensitive),
        "protocol_count":       protocol_count,
        "dns_query_count":      dns_query_count,
        "dns_avg_query_length": dns_avg_query_length,
        "http_unique_uris":     http_unique_uris,
        "unique_usernames":     unique_usernames,
        "password_entropy_avg": password_entropy_avg,
        "payload_entropy_avg":  payload_entropy_avg,
        "payload_entropy_max":  payload_entropy_max,
        "avg_payload_length":   avg_payload_length,
        "avg_cmd_per_event":    avg_cmd_per_event,
        "log_duration":         log_duration,
        "log_event_count":      log_event_count,
        "entropy_ratio":        entropy_ratio,
    }


def predict(features: dict) -> dict:
    with open(FEATURE_COLS_PATH) as f:
        feature_cols = json.load(f)

    preprocessor = MLPreprocessor.load_scaler(str(SCALER_PATH))
    df = pd.DataFrame([{col: float(features.get(col, 0.0)) for col in feature_cols}])
    df_scaled = preprocessor.transform(df, feature_cols)
    X = df_scaled[feature_cols].values

    bot_model    = joblib.load(BOT_MODEL_PATH)
    tunnel_model = joblib.load(TUNNEL_MODEL_PATH)
    multi_model  = joblib.load(MULTI_MODEL_PATH)

    bot_prob    = float(bot_model.predict_proba(X)[0][1])
    tunnel_prob = float(tunnel_model.predict_proba(X)[0][1])

    try:
        raw = multi_model.predict_proba(X)
        if isinstance(raw, list):
            attack_probs = [float(p[0][1]) for p in raw]
        else:
            attack_probs = [float(v) for v in raw[0]]
    except Exception:
        pred = multi_model.predict(X)
        attack_probs = [float(v) for v in pred[0]]

    return {
        "bot":    {"probability": bot_prob,    "label": "Bot"    if bot_prob    >= 0.5 else "Human"},
        "tunnel": {"probability": tunnel_prob, "label": "Tunnel" if tunnel_prob >= 0.5 else "Normal"},
        "attack_types": [
            {"name": name, "probability": prob}
            for name, prob in zip(MULTILABEL_NAMES, attack_probs)
        ],
    }


def main():
    args = sys.argv[1:]
    try:
        if "--session" in args:
            session_id = args[args.index("--session") + 1]
            features   = compute_session_features(session_id)
            predictions = predict(features)
            print(json.dumps({"ok": True, "features": features, "predictions": predictions}))

        elif "--features" in args:
            features    = json.loads(args[args.index("--features") + 1])
            predictions = predict(features)
            print(json.dumps({"ok": True, "predictions": predictions}))

        else:
            print(json.dumps({"ok": False, "error": "Usage: infer.py --session <id> | --features <json>"}))

    except Exception as e:
        print(json.dumps({"ok": False, "error": str(e)}))


if __name__ == "__main__":
    main()
