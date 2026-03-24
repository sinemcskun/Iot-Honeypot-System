import json
from pathlib import Path
import pandas as pd


IMBALANCE_THRESHOLD = 20


def generate_report(df: pd.DataFrame, label_cols: list, output_path: str) -> dict:
    print("\n" + "=" * 60)
    print("  CLASS DISTRIBUTION REPORT")
    print("=" * 60)

    report = {}
    warnings = []

    for col in label_cols:
        if col not in df.columns:
            continue

        vals = df[col]

        if vals.dtype == "object":
            dist = vals.value_counts().to_dict()
            report[col] = {"type": "categorical", "distribution": {str(k): int(v) for k, v in dist.items()}}
            print(f"\n  {col} (categorical):")
            for k, v in sorted(dist.items(), key=lambda x: -x[1]):
                print(f"    {k}: {v:,}")
            continue

        pos = int(vals.sum())
        neg = int(len(vals) - pos)
        ratio = neg / pos if pos > 0 else float("inf")

        report[col] = {
            "type": "binary",
            "positive": pos,
            "negative": neg,
            "total": len(vals),
            "positive_pct": round(100 * pos / len(vals), 2),
            "ratio": round(ratio, 1),
        }

        status = ""
        if ratio > IMBALANCE_THRESHOLD:
            status = " [!] IMBALANCED"
            warnings.append(col)

        print(f"\n  {col}:")
        print(f"    positive={pos:,}  negative={neg:,}  ratio=1:{ratio:.0f}{status}")

    report["_warnings"] = warnings
    report["_total_samples"] = len(df)

    if warnings:
        print(f"\n  [!] Severe imbalance ({IMBALANCE_THRESHOLD}x+): {warnings}")
    else:
        print(f"\n  No severe imbalance detected.")
    print("=" * 60)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    print(f"[imbalance] Report -> {output_path}")

    return report
