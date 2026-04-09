"""
LightGBM model: Optuna objective + final training with loss history.
"""
import lightgbm as lgb
import optuna
from sklearn.metrics import roc_auc_score

RANDOM_STATE = 42
OVERFIT_PENALTY = 0.5
EARLY_STOPPING = 50


def get_lgbm_params(trial: optuna.Trial) -> dict:
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
        "metric": "binary_logloss",
        "verbose": -1,
        "random_state": RANDOM_STATE,
    }


def objective(trial, X_train, y_train, X_val, y_val) -> float:
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


def train_final(params: dict, X_train, y_train, X_val, y_val):
    """Train with best params and return (model, loss_history).

    loss_history = {"train": [logloss per round], "val": [logloss per round]}
    """
    evals_result: dict = {}
    # Merge trial params with fixed params (best_params only has trial-suggested ones)
    final_params = {
        **params,
        "objective": "binary",
        "metric": "binary_logloss",
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
        "train": evals_result["train"]["binary_logloss"],
        "val":   evals_result["val"]["binary_logloss"],
    }
    return model, history
