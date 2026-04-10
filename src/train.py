"""
Model training module — Optuna HPO + final fit for each model type.

Each function returns a standardised dict so the pipeline can treat all models uniformly.
This module does NO evaluation, plotting, or artifact saving — that's the pipeline's job.
"""
import logging
from functools import partial

import optuna
import pandas as pd
import torch

from src.metrics import find_best_threshold, threshold_sweep
from src.models import gp_model, lgbm_model, mlp_model, svm_model, xgb_model

logger = logging.getLogger(__name__)

RANDOM_STATE = 42
DEFAULT_N_TRIALS = 30
CV_SPLITS = 5


def run_study(objective_fn, model_name: str, n_trials: int = DEFAULT_N_TRIALS, use_cv: bool = False):
    """Run an Optuna TPE study; return (best_params, study)."""
    mode = f"CV {CV_SPLITS}-fold" if use_cv else "single split"
    logger.info("=" * 55)
    logger.info("  Optimizing %s (%d trials, TPE, %s)", model_name, n_trials, mode)
    logger.info("=" * 55)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    study.optimize(objective_fn, n_trials=n_trials, show_progress_bar=True)
    logger.info("  Best combined score : %.4f", study.best_value)
    logger.info("  Best params         : %s", study.best_params)
    return study.best_params, study


def train_lgbm(X_train, y_train, X_val, y_val, X_test,
               n_trials=DEFAULT_N_TRIALS, use_cv=False) -> dict:
    """HPO + final training for LightGBM. Returns standardised result dict."""
    if use_cv:
        X_full = pd.concat([X_train, X_val])
        y_full = pd.concat([y_train, y_val])
        obj_fn = partial(lgbm_model.cv_objective, X=X_full, y=y_full, n_splits=CV_SPLITS)
    else:
        obj_fn = partial(lgbm_model.objective, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)
    best, study = run_study(obj_fn, "LightGBM", n_trials, use_cv)
    logger.info("Training final LightGBM with best params...")
    model, history = lgbm_model.train_final(best, X_train, y_train, X_val, y_val)

    val_proba   = model.predict_proba(X_val)[:, 1]
    train_proba = model.predict_proba(X_train)[:, 1]
    sweep       = threshold_sweep(y_val, val_proba)
    th_info     = find_best_threshold(y_val, val_proba, secondary_metric="accuracy")
    threshold   = th_info["threshold"]
    test_preds  = (model.predict_proba(X_test)[:, 1] >= threshold).astype(int)

    fi = pd.Series(model.feature_importances_, index=X_train.columns)
    fi_pct = (fi / fi.sum() * 100).sort_values(ascending=False).round(2)

    return {
        "name": "lgbm", "label": "LightGBM",
        "model": model, "history": history, "study": study, "params": best,
        "val_proba": val_proba, "train_proba": train_proba,
        "test_preds": test_preds, "test_index": X_test.index,
        "sweep": sweep, "threshold_info": th_info, "threshold": threshold,
        "feature_importance_pct": fi_pct,
    }


def train_xgb(X_train, y_train, X_val, y_val, X_test,
              n_trials=DEFAULT_N_TRIALS, use_cv=False) -> dict:
    """HPO + final training for XGBoost. Returns standardised result dict."""
    if use_cv:
        X_full = pd.concat([X_train, X_val])
        y_full = pd.concat([y_train, y_val])
        obj_fn = partial(xgb_model.cv_objective, X=X_full, y=y_full, n_splits=CV_SPLITS)
    else:
        obj_fn = partial(xgb_model.objective, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)
    best, study = run_study(obj_fn, "XGBoost", n_trials, use_cv)
    logger.info("Training final XGBoost with best params...")
    model, history = xgb_model.train_final(best, X_train, y_train, X_val, y_val)

    val_proba   = model.predict_proba(X_val)[:, 1]
    train_proba = model.predict_proba(X_train)[:, 1]
    sweep       = threshold_sweep(y_val, val_proba)
    th_info     = find_best_threshold(y_val, val_proba, secondary_metric="accuracy")
    threshold   = th_info["threshold"]
    test_preds  = (model.predict_proba(X_test)[:, 1] >= threshold).astype(int)

    fi = pd.Series(model.feature_importances_, index=X_train.columns)
    fi_pct = (fi / fi.sum() * 100).sort_values(ascending=False).round(2)

    return {
        "name": "xgb", "label": "XGBoost",
        "model": model, "history": history, "study": study, "params": best,
        "val_proba": val_proba, "train_proba": train_proba,
        "test_preds": test_preds, "test_index": X_test.index,
        "sweep": sweep, "threshold_info": th_info, "threshold": threshold,
        "feature_importance_pct": fi_pct,
    }


def train_mlp(X_train, y_train, X_val, y_val, X_test,
              n_trials=DEFAULT_N_TRIALS, use_cv=False) -> dict:
    """HPO + final training for MLP. Returns standardised result dict."""
    if use_cv:
        X_full = pd.concat([X_train, X_val])
        y_full = pd.concat([y_train, y_val])
        obj_fn = partial(mlp_model.cv_objective, X=X_full, y=y_full, n_splits=CV_SPLITS)
    else:
        obj_fn = partial(mlp_model.objective, X_train=X_train, y_train=y_train,
                         X_val=X_val, y_val=y_val)
    best, study = run_study(obj_fn, "MLP", n_trials, use_cv)
    logger.info("Training final MLP with best params...")
    model_raw, wrapper, history = mlp_model.train_final(
        best, X_train, y_train, X_val, y_val,
    )

    model_raw.eval()
    with torch.no_grad():
        val_t   = torch.tensor(X_val.values, dtype=torch.float32).to(mlp_model.DEVICE)
        val_proba = torch.sigmoid(model_raw(val_t)).cpu().numpy()
        train_t = torch.tensor(X_train.values, dtype=torch.float32).to(mlp_model.DEVICE)
        train_proba = torch.sigmoid(model_raw(train_t)).cpu().numpy()

    sweep   = threshold_sweep(y_val, val_proba)
    th_info = find_best_threshold(y_val, val_proba, secondary_metric="accuracy")
    threshold = th_info["threshold"]

    with torch.no_grad():
        test_t = torch.tensor(X_test.values, dtype=torch.float32).to(mlp_model.DEVICE)
        test_preds = (torch.sigmoid(model_raw(test_t)).cpu().numpy() >= threshold).astype(int)

    return {
        "name": "mlp", "label": "MLP",
        "model": wrapper, "history": history, "study": study, "params": best,
        "val_proba": val_proba, "train_proba": train_proba,
        "test_preds": test_preds, "test_index": X_test.index,
        "sweep": sweep, "threshold_info": th_info, "threshold": threshold,
        "feature_importance_pct": None,
    }


def train_gp(X_train, y_train, X_val, y_val, X_test,
             n_trials=DEFAULT_N_TRIALS, use_cv=False) -> dict:
    """HPO + final training for Gaussian Process. Returns standardised result dict."""
    if use_cv:
        X_full = pd.concat([X_train, X_val])
        y_full = pd.concat([y_train, y_val])
        obj_fn = partial(gp_model.cv_objective, X=X_full, y=y_full, n_splits=CV_SPLITS)
    else:
        obj_fn = partial(gp_model.objective, X_train=X_train, y_train=y_train,
                         X_val=X_val, y_val=y_val)
    best, study = run_study(obj_fn, "GP", n_trials, use_cv)
    logger.info("Training final GP with best params...")
    model, history = gp_model.train_final(best, X_train, y_train, X_val, y_val)

    val_proba   = model.predict_proba(X_val)[:, 1]
    train_proba = model.predict_proba(X_train)[:, 1]
    sweep       = threshold_sweep(y_val, val_proba)
    th_info     = find_best_threshold(y_val, val_proba, secondary_metric="accuracy")
    threshold   = th_info["threshold"]
    test_preds  = (model.predict_proba(X_test)[:, 1] >= threshold).astype(int)

    return {
        "name": "gp", "label": "GaussianProcess",
        "model": model, "history": history, "study": study, "params": best,
        "val_proba": val_proba, "train_proba": train_proba,
        "test_preds": test_preds, "test_index": X_test.index,
        "sweep": sweep, "threshold_info": th_info, "threshold": threshold,
        "feature_importance_pct": None,
    }


def train_svm(X_train, y_train, X_val, y_val, X_test,
              n_trials=DEFAULT_N_TRIALS, use_cv=False) -> dict:
    """HPO + final training for SVM. Returns standardised result dict."""
    if use_cv:
        X_full = pd.concat([X_train, X_val])
        y_full = pd.concat([y_train, y_val])
        obj_fn = partial(svm_model.cv_objective, X=X_full, y=y_full, n_splits=CV_SPLITS)
    else:
        obj_fn = partial(svm_model.objective, X_train=X_train, y_train=y_train,
                         X_val=X_val, y_val=y_val)
    best, study = run_study(obj_fn, "SVM", n_trials, use_cv)
    logger.info("Training final SVM with best params...")
    model, history = svm_model.train_final(best, X_train, y_train, X_val, y_val)

    val_proba   = model.predict_proba(X_val)[:, 1]
    train_proba = model.predict_proba(X_train)[:, 1]
    sweep       = threshold_sweep(y_val, val_proba)
    th_info     = find_best_threshold(y_val, val_proba, secondary_metric="accuracy")
    threshold   = th_info["threshold"]
    test_preds  = (model.predict_proba(X_test)[:, 1] >= threshold).astype(int)

    return {
        "name": "svm", "label": "SVM",
        "model": model, "history": history, "study": study, "params": best,
        "val_proba": val_proba, "train_proba": train_proba,
        "test_preds": test_preds, "test_index": X_test.index,
        "sweep": sweep, "threshold_info": th_info, "threshold": threshold,
        "feature_importance_pct": None,
    }


# Registry: maps CLI model names to train functions and their required data keys.
# Order defines default execution order in the pipeline.
TRAINERS = {
    "lgbm": {"fn": train_lgbm, "data": "tree"},
    "xgb":  {"fn": train_xgb,  "data": "tree"},
    "gp":   {"fn": train_gp,   "data": "scaled"},
    "svm":  {"fn": train_svm,  "data": "scaled"},
    "mlp":  {"fn": train_mlp,  "data": "scaled"},
}
ALL_MODELS = list(TRAINERS.keys())
