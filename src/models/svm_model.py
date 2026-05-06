"""
SVM Classification with kernel selection via Optuna.

Uses sklearn's SVC with probability=True for predict_proba support.
Training-set subsampling keeps runtime feasible (SVM is O(n^2)–O(n^3)).
Optuna selects kernel, regularisation (C), and kernel-specific params.

Requires NaN-free, scaled numeric input (handled by preprocessing.py).
"""
import logging

import numpy as np
import optuna
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

logger = logging.getLogger(__name__)

RANDOM_STATE = 42
OVERFIT_PENALTY = 0.5

KERNEL_NAMES = ["rbf", "poly", "sigmoid", "linear"]


def get_svm_params(trial: optuna.Trial) -> dict:
    """Return Optuna-suggested SVM hyperparameters."""
    kernel = trial.suggest_categorical("kernel", KERNEL_NAMES)
    params = {
        "kernel":          kernel,
        "C":               trial.suggest_float("C", 1e-2, 100.0, log=True),
        "n_train_samples": trial.suggest_int("n_train_samples", 1000, 3000, step=500),
    }
    if kernel in ("rbf", "poly", "sigmoid"):
        params["gamma"] = trial.suggest_categorical("gamma", ["scale", "auto"])
    if kernel == "poly":
        params["degree"] = trial.suggest_int("degree", 2, 5)
    return params


def _subsample(X, y, n_samples, rng):
    """Stratified subsample to keep class balance."""
    if n_samples >= len(X):
        return X, y
    y_arr = np.asarray(y)
    pos_idx = np.where(y_arr == 1)[0]
    neg_idx = np.where(y_arr == 0)[0]
    pos_rate = len(pos_idx) / len(y_arr)
    n_pos = max(1, int(n_samples * pos_rate))
    n_neg = n_samples - n_pos
    chosen = np.concatenate([
        rng.choice(pos_idx, size=min(n_pos, len(pos_idx)), replace=False),
        rng.choice(neg_idx, size=min(n_neg, len(neg_idx)), replace=False),
    ])
    rng.shuffle(chosen)
    if hasattr(X, "iloc"):
        return X.iloc[chosen], y_arr[chosen]
    return X[chosen], y_arr[chosen]


def _fit_svm(params: dict, X_train, y_train):
    """Fit a SVM classifier with the given params on a subsample."""
    rng = np.random.default_rng(RANDOM_STATE)
    X_sub, y_sub = _subsample(X_train, y_train, params["n_train_samples"], rng)

    svm_params = {k: v for k, v in params.items() if k != "n_train_samples"}
    model = SVC(
        **svm_params,
        probability=True,
        random_state=RANDOM_STATE,
        cache_size=500,
    )
    logger.info(
        "  [SVM] Fitting on %d samples (kernel=%s, C=%.4f)...",
        len(X_sub), params["kernel"], params["C"],
    )
    model.fit(X_sub, y_sub)
    return model


def objective(trial, X_train, y_train, X_val, y_val) -> float:
    """Optuna objective: combined val_auc - penalty * overfit_gap."""
    params = get_svm_params(trial)
    model = _fit_svm(params, X_train, y_train)

    train_proba = model.predict_proba(X_train)[:, 1]
    val_proba   = model.predict_proba(X_val)[:, 1]

    train_auc = roc_auc_score(y_train, train_proba)
    val_auc   = roc_auc_score(y_val,   val_proba)
    overfit_gap = max(0.0, train_auc - val_auc)
    return val_auc - OVERFIT_PENALTY * overfit_gap


def cv_objective(trial, X, y, n_splits=5) -> float:
    """Optuna objective with Stratified K-Fold CV for SVM."""
    params = get_svm_params(trial)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    y_arr = np.asarray(y)

    val_aucs, train_aucs = [], []
    for train_idx, val_idx in skf.split(X, y):
        model = _fit_svm(params, X.iloc[train_idx], y_arr[train_idx])
        train_aucs.append(roc_auc_score(y_arr[train_idx], model.predict_proba(X.iloc[train_idx])[:, 1]))
        val_aucs.append(roc_auc_score(y_arr[val_idx], model.predict_proba(X.iloc[val_idx])[:, 1]))

    mean_val = np.mean(val_aucs)
    mean_gap = np.mean([t - v for t, v in zip(train_aucs, val_aucs)])
    return mean_val - OVERFIT_PENALTY * max(0.0, mean_gap)


def train_final(params: dict, X_train, y_train, X_val, y_val):  # pylint: disable=unused-argument
    """Train with best params and return (model, history).

    SVM doesn't have iterative loss history, so history is minimal.
    """
    model = _fit_svm(params, X_train, y_train)

    n_sv = model.n_support_.sum()
    logger.info("  [SVM] Support vectors: %d / %d", n_sv, params["n_train_samples"])
    logger.info("  [SVM] Kernel: %s", model.kernel)

    history = {
        "train": [float(n_sv)],
        "val":   [float(n_sv)],
    }
    return model, history
