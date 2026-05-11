import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from analysis.model_training.trainer import train_binary, train_multilabel, MODEL_FACTORY
from analysis.ml_preparation.feature_selector import MULTI_LABEL_COLS

ML_DIR = Path(__file__).resolve().parents[1] / "output" / "ml_ready"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "output" / "model_results"

with open(ML_DIR / "feature_columns.json") as f:
    FEATURE_COLS = json.load(f)

MIN_SUPERVISED_POSITIVE = 50


def _get_importances(model, feature_cols):
    if hasattr(model, "feature_importances_"):
        return dict(zip(feature_cols, model.feature_importances_.tolist()))
    elif hasattr(model, "coef_"):
        coefs = np.abs(model.coef_[0]) if model.coef_.ndim > 1 else np.abs(model.coef_)
        return dict(zip(feature_cols, coefs.tolist()))
    return {}


def _sorted_importances(imp_dict):
    return dict(sorted(imp_dict.items(), key=lambda x: x[1], reverse=True))


def binary_importances(train, label_col, task_name):
    print(f"\n{'='*60}")
    print(f"  Feature Importance: {task_name}")
    print(f"{'='*60}")

    X_train = train[FEATURE_COLS].values
    y_train = train[label_col].values

    results = {}

    for model_name in MODEL_FACTORY:
        out = train_binary(X_train, y_train, X_train, y_train, model_name)
        model = out["model"]
        imp = _get_importances(model, FEATURE_COLS)
        imp_sorted = _sorted_importances(imp)
        results[model_name] = imp_sorted

        print(f"\n  -- {model_name} --")
        print(f"  {'Feature':<30} {'Importance':>12}")
        print(f"  {'-'*30} {'-'*12}")
        for i, (feat, val) in enumerate(imp_sorted.items()):
            if i >= 24:
                break
            print(f"  {feat:<30} {val:>12.4f}")

    return results


def multilabel_importances(train):
    print(f"\n{'='*60}")
    print(f"  Feature Importance: Multi-Label Attack")
    print(f"{'='*60}")

    eligible = []
    for col in MULTI_LABEL_COLS:
        if col not in train.columns:
            continue
        pos = int(train[col].sum())
        if pos >= MIN_SUPERVISED_POSITIVE:
            eligible.append(col)

    X_train = train[FEATURE_COLS].values
    Y_train = train[eligible].values

    results = {}

    for model_name in MODEL_FACTORY:
        out = train_multilabel(X_train, Y_train, X_train, Y_train, model_name)
        clf = out["model"]

        label_importances = {}
        for i, label in enumerate(eligible):
            estimator = clf.estimators_[i]
            imp = _get_importances(estimator, FEATURE_COLS)
            label_importances[label] = _sorted_importances(imp)

        avg_imp = {}
        for feat in FEATURE_COLS:
            vals = [label_importances[lbl].get(feat, 0) for lbl in eligible]
            avg_imp[feat] = float(np.mean(vals))
        avg_sorted = _sorted_importances(avg_imp)

        results[model_name] = {
            "average": avg_sorted,
            "per_label": label_importances,
        }

        short_labels = [l.replace("label_", "") for l in eligible]
        print(f"\n  -- {model_name} (Average across {len(eligible)} labels) --")
        print(f"  {'Feature':<30} {'Avg Importance':>14}")
        print(f"  {'-'*30} {'-'*14}")
        for i, (feat, val) in enumerate(avg_sorted.items()):
            if i >= 24:
                break
            print(f"  {feat:<30} {val:>14.4f}")

        print(f"\n  Per-label Top 3:")
        for lbl in eligible:
            short = lbl.replace("label_", "")
            top3 = list(label_importances[lbl].items())[:3]
            top3_str = ", ".join([f"{f}={v:.3f}" for f, v in top3])
            print(f"    {short:<22} {top3_str}")

    return results


def main():
    print("=" * 60)
    print("  Feature Importance Analysis")
    print("=" * 60)

    train = pd.read_parquet(ML_DIR / "train.parquet")
    print(f"[data] train={len(train):,}  features={len(FEATURE_COLS)}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    bot_imp = binary_importances(train, "bot_label", "Bot Detection")
    tunnel_imp = binary_importances(train, "tunnel_label", "Tunnel Detection")
    ml_imp = multilabel_importances(train)

    all_results = {
        "feature_columns": FEATURE_COLS,
        "bot_detection": bot_imp,
        "tunnel_detection": tunnel_imp,
        "multilabel_attack": {
            model: data["average"] for model, data in ml_imp.items()
        },
        "multilabel_per_label": {
            model: data["per_label"] for model, data in ml_imp.items()
        },
    }

    output_path = RESULTS_DIR / "feature_importance.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n[saved] {output_path}")

    print(f"\n{'='*60}")
    print(f"  DOMINANT FEATURE CHECK")
    print(f"{'='*60}")
    for task_name, task_data in [("Bot Detection", bot_imp), ("Tunnel Detection", tunnel_imp)]:
        for model_name, imp in task_data.items():
            top_feat, top_val = list(imp.items())[0]
            total = sum(imp.values())
            pct = (top_val / total * 100) if total > 0 else 0
            flag = " [!] DOMINANT" if pct > 50 else ""
            print(f"  {task_name} / {model_name}: top={top_feat} ({pct:.1f}%){flag}")

    print("=" * 60)


if __name__ == "__main__":
    main()
