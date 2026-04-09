"""
Model comparison pipeline: LightGBM, XGBoost, MLP with Optuna TPE optimization.

Each model is tuned with a combined objective:
    score = val_auc - 0.5 * max(0, train_auc - val_auc)   (maximize)

Run from project root:
    python -m src.train           # full pipeline
    python -m src.train --report  # compare tracked runs
"""
import sys
import tempfile
from functools import partial
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

from src.models import lgbm_model, mlp_model, xgb_model
from src.plots import plot_loss_curves, plot_roc_curves
from src.preprocessing import load_splits, load_splits_numeric, load_test, load_test_numeric
from src.tracking import ExperimentTracker

ROOT = Path(__file__).parent.parent
SUBMISSION_DIR = ROOT / "data/processed"
SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_TRIALS = 50
EXPERIMENT = "banking-marketing-classification"

optuna.logging.set_verbosity(optuna.logging.WARNING)
tracker = ExperimentTracker(EXPERIMENT)

# ── Report-only mode ──────────────────────────────────────────────────────────
if "--report" in sys.argv:
    tracker.generate_report()
    sys.exit(0)

# ── Data ──────────────────────────────────────────────────────────────────────
print("Loading data...")
X_train, X_val, y_train, y_val = load_splits()
X_train_num, X_val_num, _, _ = load_splits_numeric()
print(f"Tree splits : train={X_train.shape}, val={X_val.shape}")
print(f"MLP splits  : train={X_train_num.shape}, val={X_val_num.shape}\n")

# ── Optuna helper ─────────────────────────────────────────────────────────────
def run_study(objective_fn, name: str) -> dict:
    print(f"{'='*55}")
    print(f"  Optimizing {name} ({N_TRIALS} trials, TPE sampler)")
    print(f"{'='*55}")
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    study.optimize(objective_fn, n_trials=N_TRIALS, show_progress_bar=True)
    print(f"  Best combined score : {study.best_value:.4f}")
    print(f"  Best params         : {study.best_params}\n")
    return study.best_params


# ── Results containers ────────────────────────────────────────────────────────
histories: dict[str, dict] = {}
probas:    dict[str, np.ndarray] = {}
results:   dict[str, dict] = {}

# ══════════════════════════════════════════════════════════════════════════════
# 1. LightGBM
# ══════════════════════════════════════════════════════════════════════════════
lgbm_best = run_study(
    partial(lgbm_model.objective, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val),
    "LightGBM",
)
print("Training final LightGBM with best params...")
lgbm_model_fit, lgbm_history = lgbm_model.train_final(lgbm_best, X_train, y_train, X_val, y_val)
lgbm_proba = lgbm_model_fit.predict_proba(X_val)[:, 1]
lgbm_preds = (lgbm_proba >= 0.5).astype(int)

histories["lgbm"] = lgbm_history
probas["lgbm"]    = lgbm_proba
results["lgbm"]   = {
    "val_auc":      roc_auc_score(y_val, lgbm_proba),
    "train_auc":    roc_auc_score(y_train, lgbm_model_fit.predict_proba(X_train)[:, 1]),
    "val_accuracy": accuracy_score(y_val, lgbm_preds),
    "model":        lgbm_model_fit,
    "params":       lgbm_best,
}
results["lgbm"]["overfit_gap"] = results["lgbm"]["train_auc"] - results["lgbm"]["val_auc"]
print(f"  LGBM val AUC={results['lgbm']['val_auc']:.4f}  overfit_gap={results['lgbm']['overfit_gap']:+.4f}\n")

with tracker.start_run("lgbm_optuna"):
    tracker.log_params({**lgbm_best, "model": "LightGBM", "n_trials": N_TRIALS})
    tracker.log_metrics({k: v for k, v in results["lgbm"].items() if isinstance(v, float)})

# ══════════════════════════════════════════════════════════════════════════════
# 2. XGBoost
# ══════════════════════════════════════════════════════════════════════════════
xgb_best = run_study(
    partial(xgb_model.objective, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val),
    "XGBoost",
)
print("Training final XGBoost with best params...")
xgb_model_fit, xgb_history = xgb_model.train_final(xgb_best, X_train, y_train, X_val, y_val)
xgb_proba = xgb_model_fit.predict_proba(X_val)[:, 1]
xgb_preds = (xgb_proba >= 0.5).astype(int)

histories["xgb"] = xgb_history
probas["xgb"]    = xgb_proba
results["xgb"]   = {
    "val_auc":      roc_auc_score(y_val, xgb_proba),
    "train_auc":    roc_auc_score(y_train, xgb_model_fit.predict_proba(X_train)[:, 1]),
    "val_accuracy": accuracy_score(y_val, xgb_preds),
    "model":        xgb_model_fit,
    "params":       xgb_best,
}
results["xgb"]["overfit_gap"] = results["xgb"]["train_auc"] - results["xgb"]["val_auc"]
print(f"  XGB val AUC={results['xgb']['val_auc']:.4f}  overfit_gap={results['xgb']['overfit_gap']:+.4f}\n")

with tracker.start_run("xgb_optuna"):
    tracker.log_params({**xgb_best, "model": "XGBoost", "n_trials": N_TRIALS})
    tracker.log_metrics({k: v for k, v in results["xgb"].items() if isinstance(v, float)})

# ══════════════════════════════════════════════════════════════════════════════
# 3. MLP
# ══════════════════════════════════════════════════════════════════════════════
mlp_best = run_study(
    partial(mlp_model.objective, X_train=X_train_num, y_train=y_train, X_val=X_val_num, y_val=y_val),
    "MLP",
)
print("Training final MLP with best params...")
mlp_model_fit, mlp_history = mlp_model.train_final(mlp_best, X_train_num, y_train, X_val_num, y_val)

import torch
mlp_model_fit.eval()
X_val_t = torch.tensor(X_val_num.values, dtype=torch.float32).to(mlp_model.DEVICE)
with torch.no_grad():
    mlp_proba = torch.sigmoid(mlp_model_fit(X_val_t)).cpu().numpy()
mlp_preds = (mlp_proba >= 0.5).astype(int)

X_train_t = torch.tensor(X_train_num.values, dtype=torch.float32).to(mlp_model.DEVICE)
with torch.no_grad():
    mlp_train_proba = torch.sigmoid(mlp_model_fit(X_train_t)).cpu().numpy()

histories["mlp"] = mlp_history
probas["mlp"]    = mlp_proba
results["mlp"]   = {
    "val_auc":      roc_auc_score(y_val, mlp_proba),
    "train_auc":    roc_auc_score(y_train, mlp_train_proba),
    "val_accuracy": accuracy_score(y_val, mlp_preds),
    "model":        mlp_model_fit,
    "params":       mlp_best,
}
results["mlp"]["overfit_gap"] = results["mlp"]["train_auc"] - results["mlp"]["val_auc"]
print(f"  MLP val AUC={results['mlp']['val_auc']:.4f}  overfit_gap={results['mlp']['overfit_gap']:+.4f}\n")

with tracker.start_run("mlp_optuna"):
    tracker.log_params({**mlp_best, "model": "MLP", "n_trials": N_TRIALS})
    tracker.log_metrics({k: v for k, v in results["mlp"].items() if isinstance(v, float)})

# ══════════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*55}")
print(f"  {'Model':<8} {'Val AUC':>10} {'Val Acc':>10} {'Overfit Δ':>12}")
print(f"  {'-'*44}")
for name, r in results.items():
    flag = "⚠" if r["overfit_gap"] > 0.02 else "✓"
    print(f"  {name.upper():<8} {r['val_auc']:>10.4f} {r['val_accuracy']:>10.4f} {r['overfit_gap']:>+11.4f} {flag}")
print(f"{'='*55}\n")

# ── Plots ─────────────────────────────────────────────────────────────────────
plot_roc_curves(probas, y_val)
plot_loss_curves(histories)

# ── Submission: best model by val AUC ────────────────────────────────────────
best_name = max(results, key=lambda k: results[k]["val_auc"])
print(f"Best model: {best_name.upper()} (val AUC = {results[best_name]['val_auc']:.4f})")

if best_name == "mlp":
    X_test_num = load_test_numeric()
    X_test_t = torch.tensor(X_test_num.values, dtype=torch.float32).to(mlp_model.DEVICE)
    mlp_model_fit.eval()
    with torch.no_grad():
        test_preds = (torch.sigmoid(mlp_model_fit(X_test_t)).cpu().numpy() >= 0.5).astype(int)
    index = X_test_num.index
else:
    X_test = load_test()
    best_model = results[best_name]["model"]
    test_preds = best_model.predict(X_test)
    index = X_test.index

submission = pd.DataFrame({"Id": index, "subscribed": test_preds})
submission_path = SUBMISSION_DIR / "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission saved → {submission_path}  ({len(submission)} rows)")
print(f"Predicted positive rate: {submission['subscribed'].mean():.2%}")
print("\nTo compare all runs:  python -m src.train --report")
