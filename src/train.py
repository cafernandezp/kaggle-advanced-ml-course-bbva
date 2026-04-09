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
from src.models import gp_model, lgbm_model, mlp_model, xgb_model
from src.preprocessing import load_test, load_test_numeric

logger = logging.getLogger(__name__)

RANDOM_STATE = 42
DEFAULT_N_TRIALS = 30


def run_study(objective_fn, model_name: str, n_trials: int = DEFAULT_N_TRIALS):
    """Run an Optuna TPE study; return (best_params, study)."""
    logger.info("=" * 55)
    logger.info("  Optimizing %s (%d trials, TPE sampler)", model_name, n_trials)
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


def train_lgbm(X_train, y_train, X_val, y_val, top_features, n_trials=DEFAULT_N_TRIALS) -> dict:
    """HPO + final training for LightGBM. Returns standardised result dict."""
    best, study = run_study(
        partial(lgbm_model.objective, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val),
        "LightGBM", n_trials,
    )
    logger.info("Training final LightGBM with best params...")
    model, history = lgbm_model.train_final(best, X_train, y_train, X_val, y_val)

    val_proba   = model.predict_proba(X_val)[:, 1]
    train_proba = model.predict_proba(X_train)[:, 1]
    sweep       = threshold_sweep(y_val, val_proba)
    th_info     = find_best_threshold(y_val, val_proba, secondary_metric="accuracy")
    threshold   = th_info["threshold"]
    test_proba  = model.predict_proba(load_test()[top_features])[:, 1]
    test_preds  = (test_proba >= threshold).astype(int)
    test_index  = load_test()[top_features].index

    fi = pd.Series(model.feature_importances_, index=X_train.columns)
    fi_pct = (fi / fi.sum() * 100).sort_values(ascending=False).round(2)

    return {
        "name": "lgbm", "label": "LightGBM",
        "model": model, "history": history, "study": study, "params": best,
        "val_proba": val_proba, "train_proba": train_proba,
        "test_preds": test_preds, "test_index": test_index,
        "sweep": sweep, "threshold_info": th_info, "threshold": threshold,
        "feature_importance_pct": fi_pct,
    }


def train_xgb(X_train, y_train, X_val, y_val, top_features, n_trials=DEFAULT_N_TRIALS) -> dict:
    """HPO + final training for XGBoost. Returns standardised result dict."""
    best, study = run_study(
        partial(xgb_model.objective, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val),
        "XGBoost", n_trials,
    )
    logger.info("Training final XGBoost with best params...")
    model, history = xgb_model.train_final(best, X_train, y_train, X_val, y_val)

    val_proba   = model.predict_proba(X_val)[:, 1]
    train_proba = model.predict_proba(X_train)[:, 1]
    sweep       = threshold_sweep(y_val, val_proba)
    th_info     = find_best_threshold(y_val, val_proba, secondary_metric="accuracy")
    threshold   = th_info["threshold"]
    test_proba  = model.predict_proba(load_test()[top_features])[:, 1]
    test_preds  = (test_proba >= threshold).astype(int)
    test_index  = load_test()[top_features].index

    fi = pd.Series(model.feature_importances_, index=X_train.columns)
    fi_pct = (fi / fi.sum() * 100).sort_values(ascending=False).round(2)

    return {
        "name": "xgb", "label": "XGBoost",
        "model": model, "history": history, "study": study, "params": best,
        "val_proba": val_proba, "train_proba": train_proba,
        "test_preds": test_preds, "test_index": test_index,
        "sweep": sweep, "threshold_info": th_info, "threshold": threshold,
        "feature_importance_pct": fi_pct,
    }


def train_mlp(X_train_num, y_train, X_val_num, y_val, mlp_top_cols,
              n_trials=DEFAULT_N_TRIALS) -> dict:
    """HPO + final training for MLP. Returns standardised result dict."""
    best, study = run_study(
        partial(mlp_model.objective, X_train=X_train_num, y_train=y_train,
                X_val=X_val_num, y_val=y_val),
        "MLP", n_trials,
    )
    logger.info("Training final MLP with best params...")
    model_raw, wrapper, history = mlp_model.train_final(
        best, X_train_num, y_train, X_val_num, y_val,
    )

    model_raw.eval()
    with torch.no_grad():
        val_t   = torch.tensor(X_val_num.values, dtype=torch.float32).to(mlp_model.DEVICE)
        val_proba = torch.sigmoid(model_raw(val_t)).cpu().numpy()

        train_t = torch.tensor(X_train_num.values, dtype=torch.float32).to(mlp_model.DEVICE)
        train_proba = torch.sigmoid(model_raw(train_t)).cpu().numpy()

    sweep   = threshold_sweep(y_val, val_proba)
    th_info = find_best_threshold(y_val, val_proba, secondary_metric="accuracy")
    threshold = th_info["threshold"]

    X_test_num = load_test_numeric()[mlp_top_cols]
    with torch.no_grad():
        test_t = torch.tensor(X_test_num.values, dtype=torch.float32).to(mlp_model.DEVICE)
        test_preds = (torch.sigmoid(model_raw(test_t)).cpu().numpy() >= threshold).astype(int)

    return {
        "name": "mlp", "label": "MLP",
        "model": wrapper, "history": history, "study": study, "params": best,
        "val_proba": val_proba, "train_proba": train_proba,
        "test_preds": test_preds, "test_index": X_test_num.index,
        "sweep": sweep, "threshold_info": th_info, "threshold": threshold,
        "feature_importance_pct": None,
    }


def train_gp(X_train_num, y_train, X_val_num, y_val, mlp_top_cols,
             n_trials=DEFAULT_N_TRIALS) -> dict:
    """HPO + final training for Gaussian Process. Returns standardised result dict."""
    best, study = run_study(
        partial(gp_model.objective, X_train=X_train_num, y_train=y_train,
                X_val=X_val_num, y_val=y_val),
        "GP", n_trials,
    )
    logger.info("Training final GP with best params...")
    model, history = gp_model.train_final(best, X_train_num, y_train, X_val_num, y_val)

    val_proba   = model.predict_proba(X_val_num)[:, 1]
    train_proba = model.predict_proba(X_train_num)[:, 1]
    sweep       = threshold_sweep(y_val, val_proba)
    th_info     = find_best_threshold(y_val, val_proba, secondary_metric="accuracy")
    threshold   = th_info["threshold"]

    X_test_num = load_test_numeric()[mlp_top_cols]
    test_preds = (model.predict_proba(X_test_num)[:, 1] >= threshold).astype(int)

    return {
        "name": "gp", "label": "GaussianProcess",
        "model": model, "history": history, "study": study, "params": best,
        "val_proba": val_proba, "train_proba": train_proba,
        "test_preds": test_preds, "test_index": X_test_num.index,
        "sweep": sweep, "threshold_info": th_info, "threshold": threshold,
        "feature_importance_pct": None,
    }


# Registry: maps CLI model names to train functions and their required data keys
TRAINERS = {
    "lgbm": {"fn": train_lgbm, "data": "tree"},
    "xgb":  {"fn": train_xgb,  "data": "tree"},
    "mlp":  {"fn": train_mlp,  "data": "mlp"},
    "gp":   {"fn": train_gp,   "data": "mlp"},  # uses numeric features like MLP
}
ALL_MODELS = list(TRAINERS.keys())
