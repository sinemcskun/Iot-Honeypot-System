import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from xgboost import XGBClassifier


RANDOM_STATE = 42


def _make_lr():
    return LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=RANDOM_STATE,
        solver="lbfgs",
    )


def _make_rf():
    return RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def _make_xgb():
    return XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        tree_method="hist",
        device="cpu",
        random_state=RANDOM_STATE,
        eval_metric="logloss",
        verbosity=0,
    )


MODEL_FACTORY = {
    "LogisticRegression": _make_lr,
    "RandomForest": _make_rf,
    "XGBoost": _make_xgb,
}


def train_binary(X_train, y_train, X_val, y_val, model_name: str) -> dict:
    model = MODEL_FACTORY[model_name]()

    # Dynamic scale_pos_weight for XGBoost (balanced class handling)
    if isinstance(model, XGBClassifier):
        neg = int(np.sum(y_train == 0))
        pos = int(np.sum(y_train == 1))
        if pos > 0:
            model.set_params(scale_pos_weight=neg / pos)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_val)[:, 1]
    elif hasattr(model, "decision_function"):
        y_prob = model.decision_function(X_val)
    else:
        y_prob = None

    return {
        "model": model,
        "y_pred": y_pred,
        "y_prob": y_prob,
    }


def train_multilabel(X_train, Y_train, X_val, Y_val, model_name: str) -> dict:
    base = MODEL_FACTORY[model_name]()
    clf = OneVsRestClassifier(base)
    clf.fit(X_train, Y_train)

    Y_pred = clf.predict(X_val)

    if hasattr(clf, "predict_proba"):
        try:
            Y_prob = clf.predict_proba(X_val)
            if hasattr(Y_prob, "toarray"):
                Y_prob = Y_prob.toarray()
        except AttributeError:
            Y_prob = None
    else:
        Y_prob = None

    return {
        "model": clf,
        "Y_pred": Y_pred,
        "Y_prob": Y_prob,
    }
