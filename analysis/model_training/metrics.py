import numpy as np
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score,
)


def compute_binary_metrics(y_true, y_pred, y_prob=None) -> dict:
    metrics = {
        "f1_macro": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "precision_macro": round(precision_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "recall_macro": round(recall_score(y_true, y_pred, average="macro", zero_division=0), 4),
    }

    if y_prob is not None:
        try:
            metrics["roc_auc"] = round(roc_auc_score(y_true, y_prob), 4)
        except ValueError:
            metrics["roc_auc"] = None
        try:
            metrics["pr_auc"] = round(average_precision_score(y_true, y_prob), 4)
        except ValueError:
            metrics["pr_auc"] = None
    else:
        metrics["roc_auc"] = None
        metrics["pr_auc"] = None

    return metrics


def compute_multilabel_metrics(y_true, y_pred, y_prob=None, label_names=None) -> dict:
    n_labels = y_true.shape[1]
    if label_names is None:
        label_names = [f"label_{i}" for i in range(n_labels)]

    per_label = {}
    for i, name in enumerate(label_names):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        yprob = y_prob[:, i] if y_prob is not None else None
        per_label[name] = compute_binary_metrics(yt, yp, yprob)

    macro = {
        "f1_macro": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "precision_macro": round(precision_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "recall_macro": round(recall_score(y_true, y_pred, average="macro", zero_division=0), 4),
    }

    if y_prob is not None:
        try:
            macro["roc_auc"] = round(roc_auc_score(y_true, y_prob, average="macro"), 4)
        except ValueError:
            macro["roc_auc"] = None
        try:
            macro["pr_auc"] = round(average_precision_score(y_true, y_prob, average="macro"), 4)
        except ValueError:
            macro["pr_auc"] = None
    else:
        macro["roc_auc"] = None
        macro["pr_auc"] = None

    return {"per_label": per_label, "macro": macro}


def dummy_accuracy_demo(y_true) -> dict:
    n = len(y_true)
    pos = int(np.sum(y_true))
    neg = n - pos

    dummy_pred = np.zeros_like(y_true)
    dummy_accuracy = float(np.mean(dummy_pred == y_true))

    return {
        "total": n,
        "positive": pos,
        "negative": neg,
        "dummy_accuracy": round(dummy_accuracy, 4),
        "dummy_recall": 0.0,
        "explanation": (
            f"Dummy classifier (predict all 0): accuracy={dummy_accuracy:.2%} "
            f"but recall=0.0. With {pos} positive out of {n}, predicting all "
            f"negative gives misleadingly high accuracy. This is why F1/PR-AUC "
            f"are used instead."
        ),
    }
