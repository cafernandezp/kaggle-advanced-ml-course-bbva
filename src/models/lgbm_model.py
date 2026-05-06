"""
LightGBM model: Optuna objective (single-split + CV) + final training with loss history.
"""
import numpy as np
import lightgbm as lgb
import optuna
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

RANDOM_STATE = 42
OVERFIT_PENALTY = 0.5
EARLY_STOPPING = 50

# NOTE: LightGBM uses the FIRST metric in the list for early stopping.
# "binary_logloss" is first because it's the most stable metric to stop on;
# "auc" and "binary_error" are tracked for monitoring only.
EVAL_METRICS = ["binary_logloss", "auc", "binary_error"]


def get_lgbm_params(trial: optuna.Trial) -> dict:
    """Return Optuna-suggested LightGBM hyperparameters merged with fixed params."""
    return {
        "n_estimators":      trial.suggest_int(   "n_estimators",      100,  2000, step=50),
        "learning_rate":     trial.suggest_float(  "learning_rate",     1e-3, 0.3,  log=True),
        "num_leaves":        trial.suggest_int(    "num_leaves",         20,  300,  step=10),
        "min_child_samples": trial.suggest_int(    "min_child_samples",  10,  200,  step=10),
        "feature_fraction":  trial.suggest_float(  "feature_fraction",  0.4,  1.0,  log=False),
        "bagging_fraction":  trial.suggest_float(  "bagging_fraction",  0.4,  1.0,  log=False),
        "bagging_freq":      trial.suggest_int(    "bagging_freq",        1,   10,  step=1),
        "reg_alpha":         trial.suggest_float(  "reg_alpha",         1e-8, 10.0, log=True),
        "reg_lambda":        trial.suggest_float(  "reg_lambda",        1e-8, 10.0, log=True),
        # fixed
        "objective": "binary",
        "metric": EVAL_METRICS,   # first metric (binary_logloss) triggers early stopping
        "verbose": -1,
        "random_state": RANDOM_STATE,
    }


def objective(trial, X_train, y_train, X_val, y_val) -> float:
    """Optuna objective: combined val_auc − penalty * overfit_gap."""
    params = get_lgbm_params(trial)
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING, verbose=False),
            lgb.log_evaluation(-1),
        ],
    )
    train_auc = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])
    val_auc   = roc_auc_score(y_val,   model.predict_proba(X_val)[:, 1])
    overfit_gap = max(0.0, train_auc - val_auc)
    return val_auc - OVERFIT_PENALTY * overfit_gap


def cv_objective(trial, X, y, n_splits=5) -> float:
    """Optuna objective with Stratified K-Fold CV. More robust than single split."""
    params = get_lgbm_params(trial)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    val_aucs, train_aucs = [], []
    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_vl = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_vl = y.iloc[train_idx], y.iloc[val_idx]

        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_vl, y_vl)],
            callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False), lgb.log_evaluation(-1)],
        )
        train_aucs.append(roc_auc_score(y_tr, model.predict_proba(X_tr)[:, 1]))
        val_aucs.append(roc_auc_score(y_vl, model.predict_proba(X_vl)[:, 1]))

    mean_val  = np.mean(val_aucs)
    mean_gap  = np.mean([t - v for t, v in zip(train_aucs, val_aucs)])
    overfit_gap = max(0.0, mean_gap)
    return mean_val - OVERFIT_PENALTY * overfit_gap


def train_final(params: dict, X_train, y_train, X_val, y_val):
    """Train with best params and return (model, history).

    history = {
        "train": {"binary_logloss": [...], "auc": [...], "binary_error": [...]},
        "val":   {"binary_logloss": [...], "auc": [...], "binary_error": [...]},
    }
    """
    evals_result: dict = {}
    # Merge trial params with fixed params (best_params only has trial-suggested ones)
    final_params = {
        **params,
        "objective": "binary",
        "metric": EVAL_METRICS,  # multi-metric; first (binary_logloss) triggers early stopping
        "verbose": -1,
        "random_state": RANDOM_STATE,
    }
    model = lgb.LGBMClassifier(**final_params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        eval_names=["train", "val"],
        callbacks=[
            lgb.record_evaluation(evals_result),
            lgb.early_stopping(EARLY_STOPPING, verbose=False),
            lgb.log_evaluation(-1),
        ],
    )
    history = {
        "train": {m: list(evals_result["train"][m]) for m in EVAL_METRICS},
        "val":   {m: list(evals_result["val"][m])   for m in EVAL_METRICS},
    }
    return model, history
