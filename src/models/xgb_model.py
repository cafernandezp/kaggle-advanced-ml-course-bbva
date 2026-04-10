"""
XGBoost model: Optuna objective (single-split + CV) + final training with loss history.
"""
import numpy as np
import optuna
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

RANDOM_STATE = 42
OVERFIT_PENALTY = 0.5
EARLY_STOPPING = 50


def get_xgb_params(trial: optuna.Trial) -> dict:
    """Return Optuna-suggested XGBoost hyperparameters merged with fixed params."""
    return {
        "n_estimators":     trial.suggest_int(   "n_estimators",    100,  2000, step=50),
        "max_depth":        trial.suggest_int(   "max_depth",          3,    10, step=1),
        "eta":              trial.suggest_float(  "eta",             1e-3,  0.3, log=True),
        "subsample":        trial.suggest_float(  "subsample",        0.4,  1.0, log=False),
        "colsample_bytree": trial.suggest_float(  "colsample_bytree", 0.3,  1.0, log=False),
        "min_child_weight": trial.suggest_int(   "min_child_weight",    1,   50, step=1),
        "reg_alpha":        trial.suggest_float(  "reg_alpha",       1e-8, 10.0, log=True),
        "reg_lambda":       trial.suggest_float(  "reg_lambda",      1e-8, 10.0, log=True),
        # fixed
        "eval_metric": "logloss",
        "enable_categorical": True,   # required for pandas category dtype
        "seed": RANDOM_STATE,
        "n_jobs": -1,
    }


def objective(trial, X_train, y_train, X_val, y_val) -> float:
    """Optuna objective: combined val_auc − penalty * overfit_gap."""
    params = get_xgb_params(trial)
    model = xgb.XGBClassifier(**params, early_stopping_rounds=EARLY_STOPPING, verbosity=0)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    train_auc = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])
    val_auc   = roc_auc_score(y_val,   model.predict_proba(X_val)[:, 1])
    overfit_gap = max(0.0, train_auc - val_auc)
    return val_auc - OVERFIT_PENALTY * overfit_gap


def cv_objective(trial, X, y, n_splits=5) -> float:
    """Optuna objective with Stratified K-Fold CV."""
    params = get_xgb_params(trial)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    val_aucs, train_aucs = [], []
    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_vl = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_vl = y.iloc[train_idx], y.iloc[val_idx]

        model = xgb.XGBClassifier(**params, early_stopping_rounds=EARLY_STOPPING, verbosity=0)
        model.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)], verbose=False)
        train_aucs.append(roc_auc_score(y_tr, model.predict_proba(X_tr)[:, 1]))
        val_aucs.append(roc_auc_score(y_vl, model.predict_proba(X_vl)[:, 1]))

    mean_val = np.mean(val_aucs)
    mean_gap = np.mean([t - v for t, v in zip(train_aucs, val_aucs)])
    return mean_val - OVERFIT_PENALTY * max(0.0, mean_gap)


def train_final(params: dict, X_train, y_train, X_val, y_val):
    """Train with best params and return (model, loss_history).

    loss_history = {"train": [logloss per round], "val": [logloss per round]}
    """
    # enable_categorical is a fixed param not returned by study.best_params
    model = xgb.XGBClassifier(
        **params,
        enable_categorical=True,
        eval_metric="logloss",
        seed=RANDOM_STATE,
        n_jobs=-1,
        early_stopping_rounds=EARLY_STOPPING,
        verbosity=0,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=False,
    )
    res = model.evals_result()
    history = {
        "train": res["validation_0"]["logloss"],
        "val":   res["validation_1"]["logloss"],
    }
    return model, history
