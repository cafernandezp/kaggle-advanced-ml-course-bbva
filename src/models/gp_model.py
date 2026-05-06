"""
Gaussian Process Classification with kernel selection via Optuna.

Uses sklearn's GaussianProcessClassifier with training-set subsampling
to keep runtime feasible (full GP is O(n³)).  Optuna selects the kernel
structure; the GP optimises kernel hyperparameters via marginal likelihood.

Requires NaN-free numeric input (handled by build_features_numeric in preprocessing.py).
"""
import logging

import numpy as np
import optuna
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.gaussian_process.kernels import (
    ConstantKernel,
    Matern,
    RationalQuadratic,
    RBF,
    WhiteKernel,
)
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)

RANDOM_STATE = 42
OVERFIT_PENALTY = 0.5


# ── Kernel registry ──────────────────────────────────────────────────────────

KERNEL_NAMES = [
    "rbf", "matern_1.5", "matern_2.5",
    "rbf+white", "matern_2.5+white", "rational_quadratic",
]


def _build_kernel(name: str):
    """Return a kernel object by name."""
    kernels = {
        "rbf":                ConstantKernel() * RBF(),
        "matern_1.5":         ConstantKernel() * Matern(nu=1.5),
        "matern_2.5":         ConstantKernel() * Matern(nu=2.5),
        "rbf+white":          ConstantKernel() * RBF() + WhiteKernel(),
        "matern_2.5+white":   ConstantKernel() * Matern(nu=2.5) + WhiteKernel(),
        "rational_quadratic": ConstantKernel() * RationalQuadratic(),
    }
    return kernels[name]


# ── Optuna params ────────────────────────────────────────────────────────────

def get_gp_params(trial: optuna.Trial) -> dict:
    """Return Optuna-suggested GP hyperparameters."""
    return {
        "kernel_name":      trial.suggest_categorical("kernel_name", KERNEL_NAMES),
        "n_train_samples":  trial.suggest_int("n_train_samples", 1000, 3000, step=500),
        "n_restarts":       trial.suggest_int("n_restarts", 0, 3),
        "max_iter_predict": trial.suggest_int("max_iter_predict", 50, 200, step=50),
    }


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


def _fit_gp(params: dict, X_train, y_train):
    """Fit a GP classifier with the given params on a subsample."""
    rng = np.random.default_rng(RANDOM_STATE)
    X_sub, y_sub = _subsample(X_train, y_train, params["n_train_samples"], rng)

    kernel = _build_kernel(params["kernel_name"])
    model = GaussianProcessClassifier(
        kernel=kernel,
        n_restarts_optimizer=params["n_restarts"],
        max_iter_predict=params["max_iter_predict"],
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    logger.info(
        "  [GP] Fitting on %d samples (kernel=%s, restarts=%d)...",
        len(X_sub), params["kernel_name"], params["n_restarts"],
    )
    model.fit(X_sub, y_sub)
    return model


# ── Optuna objective ─────────────────────────────────────────────────────────

def objective(trial, X_train, y_train, X_val, y_val) -> float:
    """Optuna objective: combined val_auc − penalty * overfit_gap."""
    params = get_gp_params(trial)
    model = _fit_gp(params, X_train, y_train)

    train_proba = model.predict_proba(X_train)[:, 1]
    val_proba   = model.predict_proba(X_val)[:, 1]

    train_auc = roc_auc_score(y_train, train_proba)
    val_auc   = roc_auc_score(y_val,   val_proba)
    overfit_gap = max(0.0, train_auc - val_auc)
    return val_auc - OVERFIT_PENALTY * overfit_gap


# ── Final training ───────────────────────────────────────────────────────────

def cv_objective(trial, X, y, n_splits=5) -> float:
    """Optuna objective with Stratified K-Fold CV for GP."""
    params = get_gp_params(trial)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    y_arr = np.asarray(y)

    val_aucs, train_aucs = [], []
    for train_idx, val_idx in skf.split(X, y):
        model = _fit_gp(params, X.iloc[train_idx], y_arr[train_idx])
        train_aucs.append(roc_auc_score(y_arr[train_idx], model.predict_proba(X.iloc[train_idx])[:, 1]))
        val_aucs.append(roc_auc_score(y_arr[val_idx], model.predict_proba(X.iloc[val_idx])[:, 1]))

    mean_val = np.mean(val_aucs)
    mean_gap = np.mean([t - v for t, v in zip(train_aucs, val_aucs)])
    return mean_val - OVERFIT_PENALTY * max(0.0, mean_gap)


def train_final(params: dict, X_train, y_train, X_val, y_val):  # pylint: disable=unused-argument
    """Train with best params and return (model, loss_history).

    GP doesn't have iterative loss, so history contains kernel log-marginal-likelihood.
    """
    model = _fit_gp(params, X_train, y_train)

    lml = model.log_marginal_likelihood_value_
    logger.info("  [GP] Log-marginal-likelihood: %.4f", lml)
    logger.info("  [GP] Optimised kernel: %s", model.kernel_)

    # "History" for compatibility — single-point, not iterative
    history = {
        "train": [lml],
        "val":   [lml],
    }
    return model, history
