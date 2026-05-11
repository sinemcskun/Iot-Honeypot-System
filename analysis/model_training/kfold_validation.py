import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from analysis.model_training.trainer import train_binary, train_multilabel, MODEL_FACTORY
from analysis.model_training.metrics import compute_binary_metrics, compute_multilabel_metrics
from analysis.ml_preparation.feature_selector import MULTI_LABEL_COLS

ML_DIR = Path(__file__).resolve().parents[1] / "output" / "ml_ready"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "output" / "model_results"

N_SPLITS = 5
RANDOM_STATE = 42
MIN_SUPERVISED_POSITIVE = 50


def _load_all_data():
    with open(ML_DIR / "feature_columns.json") as f:
        feature_cols = json.load(f)
    with open(ML_DIR / "label_columns.json") as f:
        label_cols = json.load(f)

    train = pd.read_parquet(ML_DIR / "train.parquet")
    val = pd.read_parquet(ML_DIR / "val.parquet")
    test = pd.read_parquet(ML_DIR / "test.parquet")

    all_data = pd.concat([train, val, test], ignore_index=True)
    print(f"[kfold] Combined data: {len(all_data):,} samples")
    return all_data, feature_cols, label_cols


def _aggregate_metrics(fold_metrics_list):
    keys = [k for k in fold_metrics_list[0] if fold_metrics_list[0][k] is not None]
    agg = {}
    for key in keys:
        values = [m[key] for m in fold_metrics_list if m[key] is not None]
        if values:
            agg[key] = {
                "mean": round(float(np.mean(values)), 4),
                "std": round(float(np.std(values)), 4),
                "per_fold": [round(float(v), 4) for v in values],
            }
    return agg


def kfold_binary(all_data, feature_cols, label_col, task_name):
    print(f"\n{'='*60}")
    print(f"  K-FOLD ({N_SPLITS}) -- {task_name}")
    print(f"{'='*60}")

    X = all_data[feature_cols].values
    y = all_data[label_col].values

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    results = {"task": task_name, "label": label_col, "n_splits": N_SPLITS, "models": {}}

    for model_name in MODEL_FACTORY:
        fold_metrics = []
        print(f"\n  -- {model_name} --")

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            out = train_binary(X_train, y_train, X_test, y_test, model_name)
            metrics = compute_binary_metrics(y_test, out["y_pred"], out["y_prob"])
            fold_metrics.append(metrics)

            print(f"    Fold {fold_idx}: F1={metrics['f1_macro']:.4f}  "
                  f"PR-AUC={metrics['pr_auc']}  ROC-AUC={metrics['roc_auc']}")

        agg = _aggregate_metrics(fold_metrics)
        results["models"][model_name] = {
            "fold_metrics": fold_metrics,
            "aggregated": agg,
        }

        f1_info = agg.get("f1_macro", {})
        pr_info = agg.get("pr_auc", {})
        roc_info = agg.get("roc_auc", {})
        print(f"    MEAN:  F1={f1_info.get('mean', 'N/A')} ± {f1_info.get('std', 'N/A')}  "
              f"PR-AUC={pr_info.get('mean', 'N/A')} ± {pr_info.get('std', 'N/A')}  "
              f"ROC-AUC={roc_info.get('mean', 'N/A')} ± {roc_info.get('std', 'N/A')}")

    return results


def kfold_multilabel(all_data, feature_cols):
    print(f"\n{'='*60}")
    print(f"  K-FOLD ({N_SPLITS}) -- Multi-Label Attack Classification")
    print(f"{'='*60}")

    eligible = []
    for col in MULTI_LABEL_COLS:
        if col not in all_data.columns:
            continue
        pos = int(all_data[col].sum())
        if pos >= MIN_SUPERVISED_POSITIVE:
            eligible.append(col)
        else:
            print(f"  [EXCLUDED] {col}: {pos} positive (<{MIN_SUPERVISED_POSITIVE})")

    print(f"  {len(eligible)} labels used for K-Fold")

    X = all_data[feature_cols].values
    Y = all_data[eligible].values

    strat_key = all_data[eligible[0]].values if eligible else np.zeros(len(all_data))

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    results = {
        "task": "multi_label_attack_kfold",
        "eligible_labels": eligible,
        "n_splits": N_SPLITS,
        "models": {},
    }

    for model_name in MODEL_FACTORY:
        fold_macro_metrics = []
        print(f"\n  -- {model_name} (OneVsRest) --")

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, strat_key), 1):
            X_train, X_test = X[train_idx], X[test_idx]
            Y_train, Y_test = Y[train_idx], Y[test_idx]

            out = train_multilabel(X_train, Y_train, X_test, Y_test, model_name)
            metrics = compute_multilabel_metrics(
                Y_test, out["Y_pred"], out["Y_prob"], eligible,
            )
            fold_macro_metrics.append(metrics["macro"])

            print(f"    Fold {fold_idx}: macro-F1={metrics['macro']['f1_macro']:.4f}")

        agg = _aggregate_metrics(fold_macro_metrics)
        results["models"][model_name] = {
            "fold_macro_metrics": fold_macro_metrics,
            "aggregated": agg,
        }

        f1_info = agg.get("f1_macro", {})
        print(f"    MEAN:  macro-F1={f1_info.get('mean', 'N/A')} ± {f1_info.get('std', 'N/A')}")

    return results


def run_kfold_all():
    print("\n" + "=" * 60)
    print("  K-FOLD CROSS-VALIDATION (Metric Reliability Check)")
    print("=" * 60)

    all_data, feature_cols, label_cols = _load_all_data()

    all_results = {}

    bot_kfold = kfold_binary(all_data, feature_cols, "bot_label", "Bot Detection")
    all_results["bot_detection"] = bot_kfold

    tunnel_kfold = kfold_binary(all_data, feature_cols, "tunnel_label", "Tunnel Detection")
    all_results["tunnel_detection"] = tunnel_kfold

    ml_kfold = kfold_multilabel(all_data, feature_cols)
    all_results["multilabel_attack"] = ml_kfold

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "kfold_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n[kfold] Results saved -> {output_path}")

    print("\n" + "=" * 60)
    print("  K-FOLD SUMMARY (mean ± std)")
    print("=" * 60)

    for task_key, task_data in all_results.items():
        task_name = task_data.get("task", task_key)
        print(f"\n  {task_name}:")
        for model_name, model_data in task_data.get("models", {}).items():
            agg = model_data.get("aggregated", {})
            f1 = agg.get("f1_macro", {})
            parts = [f"F1={f1.get('mean', 'N/A')} ± {f1.get('std', 'N/A')}"]

            if "pr_auc" in agg:
                pr = agg["pr_auc"]
                parts.append(f"PR-AUC={pr.get('mean', 'N/A')} ± {pr.get('std', 'N/A')}")
            if "roc_auc" in agg:
                roc = agg["roc_auc"]
                parts.append(f"ROC-AUC={roc.get('mean', 'N/A')} ± {roc.get('std', 'N/A')}")

            print(f"    {model_name}: {'  '.join(parts)}")

    print("\n" + "=" * 60)
    return all_results


if __name__ == "__main__":
    run_kfold_all()
