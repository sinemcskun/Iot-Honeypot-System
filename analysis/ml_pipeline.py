import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from analysis.ml_preparation.feature_selector import (
    get_safe_features, get_heuristic_features, get_label_cols,
    SAFE_FEATURES, HEURISTIC_FEATURES, LABEL_COLS,
)
from analysis.ml_preparation.preprocessor import MLPreprocessor
from analysis.ml_preparation import imbalance_report

INPUT_PARQUET = Path(__file__).resolve().parent / "output" / "session_intelligence.parquet"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
ML_DIR = OUTPUT_DIR / "ml_ready"


def standardize_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    print("[labels] Standardization starting...")

    rename_map = {"label_brute_force": "label_bruteforce"}
    for old, new in rename_map.items():
        if old in df.columns and new not in df.columns:
            df[new] = df[old].astype(int)

    auto = df.get("automation_likelihood", pd.Series("mixed", index=df.index)).str.lower()
    df["bot_label"] = np.where(auto == "manual", 0, 1).astype(int)
    df["tunnel_label"] = df.get("label_tunneling", pd.Series(0, index=df.index)).astype(int)

    from analysis.ml_preparation.feature_selector import MULTI_LABEL_COLS
    existing_ml = [c for c in MULTI_LABEL_COLS if c in df.columns]
    df["attack_any"] = (df[existing_ml].sum(axis=1) > 0).astype(int)

    print(f"[labels] bot_label: {dict(df['bot_label'].value_counts())}")
    print(f"[labels] tunnel_label: {dict(df['tunnel_label'].value_counts())}")
    print(f"[labels] attack_any: {dict(df['attack_any'].value_counts())}")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    print("[features] Adding derived features...")

    df["avg_cmd_per_event"] = np.where(
        df["event_count"] > 0, df["cmd_count"] / df["event_count"], 0.0,
    )
    df["log_duration"] = np.log1p(df["duration_sec"])
    df["log_event_count"] = np.log1p(df["event_count"])
    df["entropy_ratio"] = np.where(
        df["payload_entropy_avg"] > 0,
        df["cmd_entropy"] / df["payload_entropy_avg"], 0.0,
    )
    print("[features] 4 derived features added.")
    return df


def check_leakage(feature_cols: list, label_cols: list, df: pd.DataFrame) -> None:
    print("[leakage] Checking...")

    leak = set(feature_cols) & set(label_cols)
    assert not leak, f"LEAKAGE: {leak} in both features and labels!"

    forbidden = ["attack_label_count", "attack_labels", "behavior_labels",
                 "attack_type", "src_ip", "dest_ip", "session_id"]
    for col in forbidden:
        assert col not in feature_cols, f"LEAKAGE: {col} in features!"

    heuristic_leak = set(feature_cols) & set(HEURISTIC_FEATURES)
    assert not heuristic_leak, f"LEAKAGE: heuristic features in ML set: {heuristic_leak}"

    print("[leakage] PASSED -- no leakage detected.")


def stratified_split(df: pd.DataFrame):
    print("[split] Stratified split (bot_label)...")

    strat = df["bot_label"]

    train, temp = train_test_split(df, test_size=0.30, stratify=strat, random_state=42)
    val, test = train_test_split(temp, test_size=0.50, stratify=temp["bot_label"], random_state=42)

    print(f"[split] train={len(train):,}  val={len(val):,}  test={len(test):,}")
    return train, val, test


def export_splits(train, val, test, feature_cols, label_cols, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cols = feature_cols + label_cols
    for name, split_df in [("train", train), ("val", val), ("test", test)]:
        split_df[cols].to_parquet(output_dir / f"{name}.parquet", index=False, engine="pyarrow")
        print(f"[export] {name}.parquet: {len(split_df):,} rows, {len(cols)} columns")

    with open(output_dir / "feature_columns.json", "w") as f:
        json.dump(feature_cols, f, indent=2)

    with open(output_dir / "label_columns.json", "w") as f:
        json.dump(label_cols, f, indent=2)


def main():
    print("=" * 60)
    print("  ML-Ready Dataset Pipeline")
    print("=" * 60)

    df = pd.read_parquet(INPUT_PARQUET)
    print(f"[input] {len(df):,} sessions ({INPUT_PARQUET.name})")

    df = standardize_labels(df)
    df = engineer_features(df)

    feature_cols = get_safe_features(df.columns)
    heuristic_cols = get_heuristic_features(df.columns)
    label_cols_final = get_label_cols(df.columns)

    print(f"[selector] {len(feature_cols)} SAFE features (used for ML)")
    print(f"[selector] {len(heuristic_cols)} HEURISTIC features (excluded)")
    print(f"[selector] {len(label_cols_final)} labels")

    check_leakage(feature_cols, label_cols_final, df)

    imbalance_report.generate_report(
        df, label_cols_final, str(OUTPUT_DIR / "class_distribution.json"),
    )

    train_raw, val_raw, test_raw = stratified_split(df)

    preprocessor = MLPreprocessor()
    train_scaled = preprocessor.fit_transform(train_raw, feature_cols)
    val_scaled = preprocessor.transform(val_raw, feature_cols)
    test_scaled = preprocessor.transform(test_raw, feature_cols)
    preprocessor.save_scaler(str(OUTPUT_DIR / "scaler.pkl"))

    export_splits(train_scaled, val_scaled, test_scaled, feature_cols, label_cols_final, ML_DIR)

    print("\n" + "=" * 60)
    print("  ML-Ready pipeline complete!")
    print(f"  SAFE features: {len(feature_cols)}")
    print(f"  Heuristic excluded: {len(heuristic_cols)}")
    print(f"  Labels: {len(label_cols_final)}")
    print(f"  Scaler: {OUTPUT_DIR / 'scaler.pkl'}")
    print(f"  Splits: {ML_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
