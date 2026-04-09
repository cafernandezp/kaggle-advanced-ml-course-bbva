"""
Pipeline orchestrator — CLI entry point for the full ML pipeline.

Usage:
    uv run python -m src.pipeline                        # all models
    uv run python -m src.pipeline --models lgbm xgb      # subset
    uv run python -m src.pipeline --report               # compare runs
    uv run python -m src.pipeline --n-trials 50           # override trials
"""
import logging
import sys
from datetime import datetime
from pathlib import Path

import click
import lightgbm as lgb
import pandas as pd
from sklearn.inspection import permutation_importance

from src.evaluations import evaluate_all, evaluate_model, merge_evaluation_summary
from src.plots import (
    plot_feature_importance, plot_loss_curves,
    plot_permutation_importance, plot_roc_curves,
)
from src.preprocessing import (
    expand_features_for_mlp, load_splits, load_splits_numeric,
)
from src.train import ALL_MODELS, TRAINERS
from src.tracking import ExperimentTracker

ROOT = Path(__file__).parent.parent
SUBMISSION_DIR = ROOT / "data/processed"
SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
RUNS_DIR = ROOT / "reports/runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TOP_N_FEATURES = 15
EXPERIMENT = "banking-marketing-classification"


def setup_logging() -> Path:
    """Configure root logger → console + timestamped log file. Return log path."""
    run_ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = RUNS_DIR / f"pipeline_{run_ts}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
    )
    return log_path


def select_features(X_train, y_train, X_val, y_val, top_n=TOP_N_FEATURES):
    """PFI-based feature selection on a quick LightGBM baseline."""
    logger = logging.getLogger(__name__)
    logger.info("Selecting top %d features via PFI ...", top_n)
    baseline = lgb.LGBMClassifier(n_estimators=200, random_state=RANDOM_STATE, verbose=-1)
    baseline.fit(X_train, y_train)
    pfi = permutation_importance(
        baseline, X_val, y_val,
        scoring="roc_auc", n_repeats=5, random_state=RANDOM_STATE, n_jobs=1,
    )
    top_features = (
        pd.Series(pfi.importances_mean, index=X_train.columns)
        .nlargest(top_n)
        .index.tolist()
    )
    logger.info("Selected features: %s", top_features)
    return top_features


def save_artifacts(trained, tracker, y_train, y_val):
    """Save per-model artifacts: run.json, evaluation, sweep, model, submission, FI."""
    logger = logging.getLogger(__name__)
    for res in trained:
        name = res["name"]
        model_eval = pd.DataFrame(evaluate_model(
            name, res["threshold"],
            res["train_proba"], res["val_proba"],
            res["test_preds"], y_train, y_val,
        ))

        val_row   = model_eval[model_eval["split"] == "val"].iloc[0]
        train_row = model_eval[model_eval["split"] == "train"].iloc[0]
        run_metrics = {
            "val_auc":          val_row["roc_auc"],
            "train_auc":        train_row["roc_auc"],
            "val_accuracy":     val_row["accuracy"],
            "train_accuracy":   train_row["accuracy"],
            "val_precision":    val_row["precision"],
            "train_precision":  train_row["precision"],
            "val_recall":       val_row["recall"],
            "train_recall":     train_row["recall"],
            "val_ks":           val_row["ks_statistic"],
            "train_ks":         train_row["ks_statistic"],
            "val_gini":         val_row["gini"],
            "train_gini":       train_row["gini"],
            "threshold":        round(res["threshold"], 4),
            "youden_index":     round(res["threshold_info"]["youden_index"], 4),
            "best_youden":      round(res["threshold_info"]["best_youden"], 4),
            "overfit_gap_auc":  round(train_row["roc_auc"] - val_row["roc_auc"], 4),
        }

        submission = pd.DataFrame({
            "Id": res["test_index"],
            "subscribed": res["test_preds"],
        })

        with tracker.start_run(f"{name}_optuna"):
            tracker.log_params({
                **res["params"], "model": res["label"],
                "n_trials": len(res["study"].trials),
            })
            tracker.log_metrics(run_metrics)
            tracker.log_study(res["study"])
            tracker.log_model(res["model"])
            tracker.log_dataframe(res["sweep"], "threshold_sweep.csv")
            tracker.log_dataframe(model_eval, "evaluation_results.csv")
            tracker.log_dataframe(submission, "submission.csv")
            if res["feature_importance_pct"] is not None:
                fi = res["feature_importance_pct"]
                tracker.log_dataframe(
                    fi.reset_index().rename(columns={"index": "feature", 0: "importance_pct"}),
                    "feature_importance_pct.csv",
                )

    # Global evaluation summary — built from all run folders
    merge_evaluation_summary()
    logger.info("Evaluation summary → %s", RUNS_DIR / "evaluation_summary.csv")


def print_summary(trained, eval_df):
    """Log the summary comparison table."""
    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info(
        "  %-8s %9s %9s %9s %10s %8s %8s %12s",
        "Model", "Val AUC", "Val Acc", "Val KS", "Threshold", "Youden", "Gini", "Overfit AUC",
    )
    logger.info("  %s", "-" * 70)
    for res in trained:
        name = res["name"]
        val = eval_df[(eval_df["model"] == name) & (eval_df["split"] == "val")].iloc[0]
        trn = eval_df[(eval_df["model"] == name) & (eval_df["split"] == "train")].iloc[0]
        gap = trn["roc_auc"] - val["roc_auc"]
        flag = "⚠" if gap > 0.02 else "✓"
        logger.info(
            "  %-8s %9.4f %9.4f %9.4f %10.2f %8.4f %8.4f %+11.4f %s",
            name.upper(), val["roc_auc"], val["accuracy"], val["ks_statistic"],
            res["threshold"], res["threshold_info"]["youden_index"],
            val["gini"], gap, flag,
        )
    logger.info("=" * 80)


def load_and_select_features():
    """Load data, run PFI feature selection, return filtered splits and metadata."""
    logger = logging.getLogger(__name__)
    logger.info("Loading data...")
    X_train, X_val, y_train, y_val = load_splits()
    X_train_num, X_val_num, _, _ = load_splits_numeric()
    logger.info("Tree splits : train=%s  val=%s", X_train.shape, X_val.shape)
    logger.info("MLP splits  : train=%s  val=%s", X_train_num.shape, X_val_num.shape)

    top_features = select_features(X_train, y_train, X_val, y_val)
    X_train = X_train[top_features]
    X_val   = X_val[top_features]

    mlp_top_cols = expand_features_for_mlp(top_features, X_train_num)
    X_train_num = X_train_num[mlp_top_cols]
    X_val_num   = X_val_num[mlp_top_cols]
    logger.info("Tree features: %d  |  MLP features: %d", len(top_features), len(mlp_top_cols))

    return {
        "X_train": X_train, "X_val": X_val, "y_train": y_train, "y_val": y_val,
        "X_train_num": X_train_num, "X_val_num": X_val_num,
        "top_features": top_features, "mlp_top_cols": mlp_top_cols,
    }


def generate_plots(trained, X_val, y_val):
    """Generate comparison plots: ROC, loss curves, feature importance, PFI."""
    plot_roc_curves({r["name"]: r["val_proba"] for r in trained}, y_val)
    plot_loss_curves({r["name"]: r["history"] for r in trained})

    tree_trained = [r for r in trained if r["feature_importance_pct"] is not None]
    if tree_trained:
        feat_importances = {
            r["name"]: pd.Series(r["model"].feature_importances_, index=X_val.columns)
            for r in tree_trained
        }
        plot_feature_importance(feat_importances, top_n=15)
        plot_permutation_importance(
            estimators={r["name"]: r["model"] for r in tree_trained},
            Xs={r["name"]: X_val for r in tree_trained},
            y_val=y_val,
            top_n=15,
            n_repeats=5,
        )


def save_best_submission(trained, eval_df):
    """Copy the best model's submission to data/processed/submission.csv."""
    logger = logging.getLogger(__name__)
    val_aucs = eval_df[eval_df["split"] == "val"].set_index("model")["roc_auc"]
    best_name = val_aucs.idxmax()
    best_res = next(r for r in trained if r["name"] == best_name)
    logger.info("Best model: %s (val AUC = %.4f)", best_name.upper(), val_aucs[best_name])

    submission = pd.DataFrame({
        "Id": best_res["test_index"],
        "subscribed": best_res["test_preds"],
    })
    submission_path = SUBMISSION_DIR / "submission.csv"
    submission.to_csv(submission_path, index=False)
    logger.info("Submission saved → %s  (%d rows)", submission_path, len(submission))
    logger.info("Predicted positive rate: %.2f%%", submission["subscribed"].mean() * 100)


@click.command()
@click.option(
    "--models", "-m", multiple=True, default=ALL_MODELS,
    type=click.Choice(ALL_MODELS, case_sensitive=False),
    help="Models to train. Default: all.",
)
@click.option("--n-trials", default=30, show_default=True, help="Optuna trials per model.")
@click.option("--report", is_flag=True, help="Compare all tracked runs and exit.")
def main(models, n_trials, report):
    """ML pipeline: preprocess → feature selection → HPO → train → evaluate → submit."""
    log_path = setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Log → %s", log_path)

    tracker = ExperimentTracker(EXPERIMENT)

    if report:
        tracker.generate_report()
        return

    models = list(models)
    logger.info("Models to train: %s", models)

    # ── 1–2. Load data + feature selection ────────────────────────────────────
    data = load_and_select_features()

    # ── 3. Train models ───────────────────────────────────────────────────────
    trained: list[dict] = []
    for name in models:
        spec = TRAINERS[name]
        if spec["data"] == "tree":
            result = spec["fn"](
                data["X_train"], data["y_train"], data["X_val"], data["y_val"],
                data["top_features"], n_trials,
            )
        else:
            result = spec["fn"](
                data["X_train_num"], data["y_train"], data["X_val_num"], data["y_val"],
                data["mlp_top_cols"], n_trials,
            )
        trained.append(result)

    # ── 4. Evaluate ───────────────────────────────────────────────────────────
    models_data = {
        r["name"]: {
            "threshold": r["threshold"], "train_proba": r["train_proba"],
            "val_proba": r["val_proba"], "test_preds": r["test_preds"],
        }
        for r in trained
    }
    eval_df = evaluate_all(models_data, data["y_train"], data["y_val"])

    # ── 5–8. Save, summarise, plot, submit ────────────────────────────────────
    save_artifacts(trained, tracker, data["y_train"], data["y_val"])
    print_summary(trained, eval_df)
    generate_plots(trained, data["X_val"], data["y_val"])
    save_best_submission(trained, eval_df)
    logger.info("Full log → %s", log_path)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
