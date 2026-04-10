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
import warnings
from datetime import datetime
from pathlib import Path

import click
import mlflow
import mlflow.sklearn
import pandas as pd

from src.evaluations import (
    build_run_metrics, evaluate_all, evaluate_model, merge_evaluation_summary,
)
from src.feature_selection import (
    build_feature_selection_report,
    drop_correlated_features,
    drop_high_missing,
    select_top_features_lgbm_pfi_based,
    select_top_mutual_information,
)
from src.plots import (
    plot_feature_importance, plot_loss_curves,
    plot_permutation_importance, plot_roc_curves, plot_threshold_selection,
)
from src.preprocessing import (
    ProcessedData, expand_features_for_mlp, preprocess_data,
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


# ── Setup helpers ─────────────────────────────────────────────────────────────

def setup_logging() -> Path:
    """Configure root logger → console + timestamped log file. Return log path."""
    run_ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
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


def setup_mlflow():
    """Configure MLflow: suppress noisy loggers, set tracking URI and experiment."""
    for name in ("mlflow", "alembic", "sqlalchemy"):
        logging.getLogger(name).setLevel(logging.WARNING)
    warnings.filterwarnings("ignore", category=FutureWarning, module=r"mlflow\.models\.model")
    logging.getLogger("mlflow.utils.environment").setLevel(logging.ERROR)

    mlflow.set_tracking_uri(str(RUNS_DIR / "mlruns"))
    mlflow.set_experiment(EXPERIMENT)


# ── Stage functions ───────────────────────────────────────────────────────────

def load_data() -> ProcessedData:
    """Single call: read raw CSVs, build all variants, return ProcessedData container."""
    logger = logging.getLogger(__name__)
    logger.info("Loading and preprocessing data...")
    data = preprocess_data()
    logger.info("Tree splits  : train=%s  val=%s  test=%s",
                data.X_train.shape, data.X_val.shape, data.X_test.shape)
    logger.info("Numeric splits: train=%s  val=%s  test=%s",
                data.X_train_num.shape, data.X_val_num.shape, data.X_test_num.shape)
    logger.info("Scaled splits : train=%s  val=%s  test=%s",
                data.X_train_scaled.shape, data.X_val_scaled.shape, data.X_test_scaled.shape)
    return data


def _onehot_to_original(col_name: str, original_cols: list[str]) -> str:
    """Map a one-hot column name back to its original feature.

    Examples:
        "job_blue-collar" → "job"
        "duration"        → "duration"
    """
    for orig in original_cols:
        if col_name == orig or col_name.startswith(orig + "_"):
            return orig
    return col_name


def apply_feature_selection(data: ProcessedData) -> tuple[ProcessedData, list[str], list[str]]:
    """Run feature selection on the *numeric variant* (already imputed and one-hot encoded),
    then map the selected columns back to original feature names for the tree variant.

    Each stage is explicit and skippable — comment out any line to skip it.

    Returns:
        (filtered_data, top_features, mlp_top_cols)
    """
    logger = logging.getLogger(__name__)
    # Feature selection runs on the numeric variant (no NaN, no category dtypes)
    X_train_num, X_val_num = data.X_train_num, data.X_val_num
    y_train = data.y_train
    n_start = X_train_num.shape[1]
    logger.info("Feature selection: starting with %d numeric features", n_start)
    survivors: dict[str, list[str]] = {}

    # Stage 1: drop features with too many missing values (none expected post-imputation,
    # but kept for safety / future raw inputs)
    kept = drop_high_missing(X_train_num, threshold=0.5)
    survivors["stage1_missing"] = kept.copy()
    X_train_num, X_val_num = X_train_num[kept], X_val_num[kept]

    # Stage 2: drop highly correlated features (Spearman, keep the one with higher MI)
    kept = drop_correlated_features(X_train_num, y_train, threshold=0.9)
    survivors["stage2_correlation"] = kept.copy()
    X_train_num, X_val_num = X_train_num[kept], X_val_num[kept]

    # Stage 3: keep top K features by Mutual Information with target
    kept = select_top_mutual_information(X_train_num, y_train, top_k=20)
    survivors["stage3_mi"] = kept.copy()
    X_train_num, X_val_num = X_train_num[kept], X_val_num[kept]

    # Stage 4: keep top N features by LightGBM PFI
    selected_num = select_top_features_lgbm_pfi_based(
        X_train_num, y_train, X_val_num, data.y_val, top_n=15,
    )
    survivors["stage4_pfi"] = selected_num.copy()

    # Map one-hot columns back to original feature names (preserve order, dedupe)
    original_cols = list(data.X_train.columns)
    seen: set[str] = set()
    top_features: list[str] = []
    for col in selected_num:
        orig = _onehot_to_original(col, original_cols)
        if orig not in seen:
            seen.add(orig)
            top_features.append(orig)

    # Final scaled/numeric column list = expand selected tree features back to one-hot
    mlp_top_cols = expand_features_for_mlp(top_features, data.X_train_num)

    logger.info(
        "Feature selection: %d numeric → %d numeric (top PFI) → %d original features",
        n_start, len(selected_num), len(top_features),
    )
    logger.info("Final features (tree names): %s", top_features)

    # Build and save the per-feature report (using numeric variant for consistency)
    fs_report = build_feature_selection_report(
        data.X_train_num, data.y_train, survivors, selected_num,
    )
    fs_report.to_csv(RUNS_DIR / "feature_selection_report.csv", index=False)
    logger.info("Feature selection report → %s", RUNS_DIR / "feature_selection_report.csv")

    filtered = ProcessedData(
        X_train=data.X_train[top_features],
        X_val=data.X_val[top_features],
        X_test=data.X_test[top_features],
        X_train_num=data.X_train_num[mlp_top_cols],
        X_val_num=data.X_val_num[mlp_top_cols],
        X_test_num=data.X_test_num[mlp_top_cols],
        X_train_scaled=data.X_train_scaled[mlp_top_cols],
        X_val_scaled=data.X_val_scaled[mlp_top_cols],
        X_test_scaled=data.X_test_scaled[mlp_top_cols],
        y_train=data.y_train,
        y_val=data.y_val,
        scaler=data.scaler,
        median_imputer=data.median_imputer,
    )
    logger.info("Tree features: %d  |  Scaled features: %d", len(top_features), len(mlp_top_cols))
    return filtered, top_features, mlp_top_cols


def train_and_save_models(
    models: list[str],
    data: ProcessedData,
    tracker: ExperimentTracker,
    n_trials: int,
    use_cv: bool = False,
) -> list[dict]:
    """Train each model and save its artifacts IMMEDIATELY after training finishes.

    This means: if model 3 of 5 crashes, models 1 and 2 are already on disk.
    You can also `tail -f` the run folder while later models are still training.

    Saves per model:
    - Custom tracker: run.json, evaluation_results.csv, threshold_sweep.csv,
      submission.csv, model.pkl, val_proba.csv, threshold_selection.png,
      optuna_trials.csv, optuna_study.pkl, feature_importance_pct.csv (tree only)
    - MLflow: params, metrics, model artifact

    Returns the full list of trained results (used afterwards for cross-model
    plots and the best-submission selection).
    """
    logger = logging.getLogger(__name__)
    setup_mlflow()
    _data_map = {
        "tree":   (data.X_train,        data.X_val,        data.X_test),
        "scaled": (data.X_train_scaled, data.X_val_scaled, data.X_test_scaled),
    }
    trained: list[dict] = []
    for name in models:
        spec = TRAINERS[name]
        x_tr, x_vl, x_te = _data_map[spec["data"]]

        # Train
        result = spec["fn"](x_tr, data.y_train, x_vl, data.y_val, x_te, n_trials, use_cv=use_cv)

        # Evaluate
        model_eval = pd.DataFrame(evaluate_model(
            result["name"], result["threshold"],
            result["train_proba"], result["val_proba"],
            result["test_preds"], data.y_train, data.y_val,
        ))
        run_metrics = build_run_metrics(model_eval, result["threshold_info"], result["threshold"])

        # Save IMMEDIATELY — before training the next model
        save_to_tracker(result, run_metrics, model_eval, tracker)
        save_to_mlflow(result, run_metrics)
        logger.info("[%s] artifacts saved — moving to next model", name)

        trained.append(result)

    return trained


def save_to_tracker(res: dict, run_metrics: dict, model_eval, tracker: ExperimentTracker):
    """Save one model's artifacts via the custom tracker."""
    name = res["name"]
    submission = pd.DataFrame({"Id": res["test_index"], "subscribed": res["test_preds"]})

    val_proba_df = pd.DataFrame({"y_proba": res["val_proba"]})

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
        tracker.log_dataframe(val_proba_df, "val_proba.csv")
        if res["feature_importance_pct"] is not None:
            fi = res["feature_importance_pct"]
            tracker.log_dataframe(
                fi.reset_index().rename(columns={"index": "feature", 0: "importance_pct"}),
                "feature_importance_pct.csv",
            )
        # Threshold selection plot (accuracy vs Youden Index)
        plot_threshold_selection(
            res["sweep"], res["threshold_info"], name,
            save_path=tracker.run_dir / "threshold_selection.png",
        )


def save_to_mlflow(res: dict, run_metrics: dict):
    """Save one model's params, metrics, and model object to MLflow."""
    logger = logging.getLogger(__name__)
    mlflow_params = {k: str(v) for k, v in res["params"].items()}
    mlflow_params["model"] = res["label"]
    mlflow_params["n_trials"] = len(res["study"].trials)

    with mlflow.start_run(run_name=f"{res['name']}_optuna"):
        mlflow.log_params(mlflow_params)
        mlflow.log_metrics(run_metrics)
        mlflow.set_tag("model_type", res["label"])
        mlflow.sklearn.log_model(res["model"], name="model")
        logger.info("[mlflow] Run logged: %s", res["name"])


def merge_summary_and_log():
    """Merge per-model evaluation_results.csv into evaluation_summary.csv (also generates combined ROC)."""
    merge_evaluation_summary()
    logging.getLogger(__name__).info("Evaluation summary → %s", RUNS_DIR / "evaluation_summary.csv")


def print_summary(trained: list[dict], eval_df: pd.DataFrame):
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


def generate_plots(trained: list[dict], data: ProcessedData):
    """Generate comparison plots: ROC, loss curves, feature importance, PFI."""
    plot_roc_curves({r["name"]: r["val_proba"] for r in trained}, data.y_val)
    plot_loss_curves({r["name"]: r["history"] for r in trained})

    tree_trained = [r for r in trained if r["feature_importance_pct"] is not None]
    if tree_trained:
        feat_importances = {
            r["name"]: pd.Series(r["model"].feature_importances_, index=data.X_val.columns)
            for r in tree_trained
        }
        plot_feature_importance(feat_importances, top_n=15)
        plot_permutation_importance(
            estimators={r["name"]: r["model"] for r in tree_trained},
            Xs={r["name"]: data.X_val for r in tree_trained},
            y_val=data.y_val,
            top_n=15,
            n_repeats=5,
        )


def save_best_submission(trained: list[dict], eval_df: pd.DataFrame):
    """Copy the best model's submission to data/processed/submission.csv."""
    logger = logging.getLogger(__name__)
    val_aucs = eval_df[eval_df["split"] == "val"].set_index("model")["roc_auc"]
    best_name = val_aucs.idxmax()
    best_res = next(r for r in trained if r["name"] == best_name)
    logger.info("Best model: %s (val AUC = %.4f)", best_name.upper(), val_aucs[best_name])

    submission = pd.DataFrame({"Id": best_res["test_index"], "subscribed": best_res["test_preds"]})
    submission_path = SUBMISSION_DIR / "submission.csv"
    submission.to_csv(submission_path, index=False)
    logger.info("Submission saved → %s  (%d rows)", submission_path, len(submission))
    logger.info("Predicted positive rate: %.2f%%", submission["subscribed"].mean() * 100)


# ── CLI entry point ───────────────────────────────────────────────────────────

@click.command()
@click.option(
    "--models", "-m", multiple=True, default=ALL_MODELS,
    type=click.Choice(ALL_MODELS, case_sensitive=False),
    help="Models to train. Default: all.",
)
@click.option("--n-trials", default=30, show_default=True, help="Optuna trials per model.")
@click.option("--cv", is_flag=True, help="Use Stratified 5-fold CV in Optuna instead of single split.")
@click.option("--report", is_flag=True, help="Compare all tracked runs and exit.")
def main(models, n_trials, cv, report):
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

    data = load_data()
    data, _, _ = apply_feature_selection(data)

    # Train + save each model incrementally — artifacts on disk after every model
    trained = train_and_save_models(models, data, tracker, n_trials, use_cv=cv)

    # Build the cross-model evaluation table (used for summary table + best-model selection)
    eval_df = evaluate_all(
        {r["name"]: {"threshold": r["threshold"], "train_proba": r["train_proba"],
                     "val_proba": r["val_proba"], "test_preds": r["test_preds"]}
         for r in trained},
        data.y_train, data.y_val,
    )

    # Final cross-model artifacts (need all models)
    merge_summary_and_log()
    print_summary(trained, eval_df)
    generate_plots(trained, data)
    save_best_submission(trained, eval_df)
    logger.info("Full log → %s", log_path)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
