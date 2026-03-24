import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from analysis.model_training.trainer import train_binary, train_multilabel, MODEL_FACTORY
from analysis.model_training.metrics import (compute_binary_metrics, compute_multilabel_metrics, dummy_accuracy_demo)
from analysis.model_training.exfiltration_case_study import run_case_study
from analysis.ml_preparation.feature_selector import MULTI_LABEL_COLS
from analysis.model_training.kfold_validation import run_kfold_all

ML_DIR = Path(__file__).resolve().parents[1] / "output" / "ml_ready"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "output" / "model_results"

with open(ML_DIR / "feature_columns.json") as f:
    FEATURE_COLS = json.load(f)
with open(ML_DIR / "label_columns.json") as f:
    LABEL_COLS = json.load(f)

MIN_SUPERVISED_POSITIVE = 50
EXFILTRATION_LABEL = "label_data_exfiltration"


def _load_splits():
    train = pd.read_parquet(ML_DIR / "train.parquet")
    val = pd.read_parquet(ML_DIR / "val.parquet")
    test = pd.read_parquet(ML_DIR / "test.parquet")
    return train, val, test


def _save_json(data, filename):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


def run_binary_task(train, val, test, label_col: str, task_name: str) -> dict:
    print(f"\n{'='*60}")
    print(f"  {task_name} -- Binary Classification")
    print(f"{'='*60}")

    X_train = train[FEATURE_COLS].values
    y_train = train[label_col].values
    X_val = val[FEATURE_COLS].values
    y_val = val[label_col].values
    X_test = test[FEATURE_COLS].values
    y_test = test[label_col].values

    dummy = dummy_accuracy_demo(y_val)
    print(f"\n  [DUMMY] {dummy['explanation']}")

    results = {"task": task_name, "label": label_col, "dummy_demo": dummy, "models": {}}
    best_model = None
    best_f1 = -1

    for model_name in MODEL_FACTORY:
        print(f"\n  -- {model_name} --")

        out = train_binary(X_train, y_train, X_val, y_val, model_name)
        val_metrics = compute_binary_metrics(y_val, out["y_pred"], out["y_prob"])
        print(f"  VAL  F1={val_metrics['f1_macro']:.4f}  "
              f"PR-AUC={val_metrics['pr_auc']}  ROC-AUC={val_metrics['roc_auc']}")

        out_test = train_binary(X_train, y_train, X_test, y_test, model_name)
        test_metrics = compute_binary_metrics(y_test, out_test["y_pred"], out_test["y_prob"])
        print(f"  TEST F1={test_metrics['f1_macro']:.4f}  "
              f"PR-AUC={test_metrics['pr_auc']}  ROC-AUC={test_metrics['roc_auc']}")

        results["models"][model_name] = {
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
        }

        if val_metrics["f1_macro"] > best_f1:
            best_f1 = val_metrics["f1_macro"]
            best_model = model_name

    results["best_model"] = best_model
    results["best_val_f1"] = best_f1
    print(f"\n  BEST: {best_model} (val F1={best_f1:.4f})")
    return results


def run_multilabel_task(train, val, test) -> dict:
    print(f"\n{'='*60}")
    print(f"  Multi-Label Attack Classification")
    print(f"{'='*60}")

    eligible = []
    excluded = []
    for col in MULTI_LABEL_COLS:
        if col not in train.columns:
            continue
        pos = int(train[col].sum())
        if pos < MIN_SUPERVISED_POSITIVE:
            excluded.append((col, pos))
        else:
            eligible.append(col)

    if excluded:
        for col, pos in excluded:
            print(f"  [EXCLUDED] {col}: {pos} positive (<{MIN_SUPERVISED_POSITIVE})")

    print(f"  {len(eligible)} labels to train, {len(excluded)} excluded")

    X_train = train[FEATURE_COLS].values
    Y_train = train[eligible].values
    X_val = val[FEATURE_COLS].values
    Y_val = val[eligible].values
    X_test = test[FEATURE_COLS].values
    Y_test = test[eligible].values

    results = {
        "task": "multi_label_attack",
        "eligible_labels": eligible,
        "excluded_labels": [e[0] for e in excluded],
        "models": {},
    }
    best_model = None
    best_f1 = -1

    for model_name in MODEL_FACTORY:
        print(f"\n  -- {model_name} (OneVsRest) --")

        out = train_multilabel(X_train, Y_train, X_val, Y_val, model_name)
        val_metrics = compute_multilabel_metrics(
            Y_val, out["Y_pred"], out["Y_prob"], eligible,
        )
        print(f"  VAL  macro-F1={val_metrics['macro']['f1_macro']:.4f}")
        for lbl, m in val_metrics["per_label"].items():
            short = lbl.replace("label_", "")
            print(f"    {short}: F1={m['f1_macro']:.3f}  Recall={m['recall_macro']:.3f}")

        out_t = train_multilabel(X_train, Y_train, X_test, Y_test, model_name)
        test_metrics = compute_multilabel_metrics(
            Y_test, out_t["Y_pred"], out_t["Y_prob"], eligible,
        )
        print(f"  TEST macro-F1={test_metrics['macro']['f1_macro']:.4f}")

        results["models"][model_name] = {
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
        }

        if val_metrics["macro"]["f1_macro"] > best_f1:
            best_f1 = val_metrics["macro"]["f1_macro"]
            best_model = model_name

    results["best_model"] = best_model
    results["best_val_macro_f1"] = best_f1
    print(f"\n  BEST: {best_model} (val macro-F1={best_f1:.4f})")
    return results


def main():
    print("=" * 60)
    print("  Model Training Pipeline")
    print("=" * 60)
    print("  Accuracy is NOT used as the primary metric")
    print("  due to extreme class imbalance (e.g., data_exfiltration 1:3000).")
    print("  A dummy classifier achieves high accuracy by always predicting 0,")
    print("  but recall = 0. Therefore F1/PR-AUC are used instead.")
    print("=" * 60)

    train, val, test = _load_splits()
    print(f"[data] train={len(train):,}  val={len(val):,}  test={len(test):,}")
    print(f"[data] features={len(FEATURE_COLS)}  labels={len(LABEL_COLS)}")

    bot_results = run_binary_task(train, val, test, "bot_label", "Bot Detection")
    _save_json(bot_results, "bot_results.json")

    tunnel_results = run_binary_task(train, val, test, "tunnel_label", "Tunnel Detection")
    _save_json(tunnel_results, "tunnel_results.json")

    ml_results = run_multilabel_task(train, val, test)
    _save_json(ml_results, "multilabel_results.json")

    exfil_results = run_case_study(
        train, test, FEATURE_COLS,
        label_col=EXFILTRATION_LABEL,
        output_path=str(RESULTS_DIR / "exfiltration_case_study.json"),
    )

    print("\n" + "=" * 60)
    print("  FINAL SUMMARY")
    print("=" * 60)

    print(f"\n  Bot Detection:")
    print(f"    Best model: {bot_results['best_model']}")
    print(f"    Val F1: {bot_results['best_val_f1']:.4f}")

    print(f"\n  Tunnel Detection:")
    print(f"    Best model: {tunnel_results['best_model']}")
    print(f"    Val F1: {tunnel_results['best_val_f1']:.4f}")

    print(f"\n  Multi-Label Attack:")
    print(f"    Best model: {ml_results['best_model']}")
    print(f"    Val macro-F1: {ml_results['best_val_macro_f1']:.4f}")

    print(f"\n  Exfiltration Anomaly Detection:")
    if "detection_rate" in exfil_results:
        dr = exfil_results["detection_rate"]
        print(f"    Detection rate: {dr:.1%}")
    else:
        print(f"    Status: {exfil_results.get('status', 'N/A')}")

    print(f"\n  Accuracy is NOT used as the primary metric")
    print(f"  due to extreme class imbalance.")
    print("=" * 60)

    run_kfold_all()


if __name__ == "__main__":
    main()
