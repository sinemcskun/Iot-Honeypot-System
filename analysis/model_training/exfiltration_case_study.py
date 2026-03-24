import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import IsolationForest

RANDOM_STATE = 42
MIN_POSITIVE = 50


def run_case_study(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list,
    label_col: str = "label_data_exfiltration",
    output_path: str = None,
) -> dict:
    print("\n" + "=" * 60)
    print("  DATA EXFILTRATION -- Anomaly Detection Case Study")
    print("=" * 60)

    all_df = pd.concat([train_df, test_df], ignore_index=True)
    total_pos = int(all_df[label_col].sum())
    print(f"[exfiltration] Total positive: {total_pos}")

    if total_pos >= MIN_POSITIVE:
        print(f"[exfiltration] {total_pos} >= {MIN_POSITIVE}, supervised training is viable.")
        return {"status": "skipped", "reason": "enough positive samples"}

    # Train IsolationForest on negative (non-exfiltration) samples only
    neg_train = train_df[train_df[label_col] == 0][feature_cols].values
    print(f"[exfiltration] IsolationForest: training on {len(neg_train):,} negative samples...")

    iso = IsolationForest(
        n_estimators=200,
        contamination=0.01,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    iso.fit(neg_train)

    pos_all = all_df[all_df[label_col] == 1][feature_cols].values
    neg_test = test_df[test_df[label_col] == 0][feature_cols].values

    # predict: -1 = anomaly, 1 = normal
    pos_pred = iso.predict(pos_all)
    pos_scores = iso.decision_function(pos_all)
    neg_pred = iso.predict(neg_test)
    neg_scores = iso.decision_function(neg_test)

    pos_detected = int(np.sum(pos_pred == -1))
    pos_total = len(pos_all)
    detection_rate = pos_detected / pos_total if pos_total > 0 else 0.0

    neg_flagged = int(np.sum(neg_pred == -1))
    neg_total = len(neg_test)
    fpr = neg_flagged / neg_total if neg_total > 0 else 0.0

    result = {
        "model": "IsolationForest",
        "approach": "anomaly_detection",
        "reason": f"Only {total_pos} positive samples (< {MIN_POSITIVE} threshold)",
        "training_negatives": len(neg_train),
        "total_positives_evaluated": pos_total,
        "positives_detected_as_anomaly": pos_detected,
        "detection_rate": round(detection_rate, 4),
        "test_negatives": neg_total,
        "negatives_flagged_as_anomaly": neg_flagged,
        "false_positive_rate": round(fpr, 4),
        "positive_anomaly_scores": {
            "mean": round(float(np.mean(pos_scores)), 4),
            "min": round(float(np.min(pos_scores)), 4),
            "max": round(float(np.max(pos_scores)), 4),
        },
        "negative_anomaly_scores": {
            "mean": round(float(np.mean(neg_scores)), 4),
            "min": round(float(np.min(neg_scores)), 4),
            "max": round(float(np.max(neg_scores)), 4),
        },
    }

    print(f"[exfiltration] Detection rate: {pos_detected}/{pos_total} = {detection_rate:.1%}")
    print(f"[exfiltration] False positive rate: {neg_flagged}/{neg_total} = {fpr:.1%}")
    print(f"[exfiltration] Positive score: mean={np.mean(pos_scores):.4f}")
    print(f"[exfiltration] Negative score: mean={np.mean(neg_scores):.4f}")

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"[exfiltration] Report -> {output_path}")

    print("=" * 60)
    return result
