"""
Pipeline orchestrator — CLI entry point for the full ML pipeline.

Usage:
    uv run python -m src.pipeline                        # all models
    uv run python -m src.pipeline --models lgbm xgb      # subset
    uv run python -m src.pipeline --report               # compare runs
    uv run python -m src.pipeline --n-trials 50           # override trials
"""
import json
import logging
import sys
import warnings
from datetime import datetime
from pathlib import Path

import click
import mlflow
import mlflow.sklearn
import pandas as pd

from src.config import DEFAULT_CONFIG, PipelineConfig
from src.evaluations import (
    build_run_metrics, evaluate_all, evaluate_model, merge_evaluation_summary,
    write_run_report_md,
)
from src.feature_selection import (
    FEATURE_SELECTION_DIR_NAME,
    build_feature_selection_report,
    drop_correlated_features,
    drop_high_missing,
    load_feature_selection,
    save_feature_selection,
    select_top_features_lgbm_pfi_based,
    select_top_mutual_information,
)
from src.plots import (
    plot_feature_importance, plot_loss_curves, plot_model_feature_importance,
    plot_permutation_importance, plot_reliability_diagram, plot_roc_curves,
    plot_threshold_selection, plot_training_curves,
)
from src.preprocessing import (
    PREPROCESSING_DIR_NAME, ProcessedData, expand_features_for_mlp,
    load_processed_data, preprocess_data, save_preprocessing_artifacts,
    save_processed_dataframes,
)
from src.train import ALL_MODELS, TRAINERS
from src.tracking import ExperimentTracker

ROOT = Path(__file__).parent.parent
SUBMISSION_DIR = ROOT / "data/processed"
SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
RUNS_DIR = ROOT / "reports/runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

# Module-level config — set by main(). Other functions read via _get_config().
# To override for tests or a different project:
#     from src import pipeline
#     pipeline.CONFIG = PipelineConfig(experiment="other", id_col="id", target_col="label")
CONFIG: PipelineConfig = DEFAULT_CONFIG


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
    mlflow.set_experiment(CONFIG.experiment)


# ── Stage functions ───────────────────────────────────────────────────────────

def load_data() -> ProcessedData:
    """Single call: read raw CSVs, build all variants, return ProcessedData container."""
    logger = logging.getLogger(__name__)
    logger.info("Loading and preprocessing data...")
    data = preprocess_data(target_col=CONFIG.target_col, id_col=CONFIG.id_col)
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
    kept = drop_high_missing(X_train_num, threshold=CONFIG.missing_threshold)
    survivors["stage1_missing"] = kept.copy()
    X_train_num, X_val_num = X_train_num[kept], X_val_num[kept]

    # Stage 2: drop highly correlated features (Spearman, keep the one with higher MI)
    kept = drop_correlated_features(X_train_num, y_train, threshold=CONFIG.correlation_threshold)
    survivors["stage2_correlation"] = kept.copy()
    X_train_num, X_val_num = X_train_num[kept], X_val_num[kept]

    # Stage 3: keep top K features by Mutual Information with target
    kept = select_top_mutual_information(X_train_num, y_train, top_k=CONFIG.mi_top_k)
    survivors["stage3_mi"] = kept.copy()
    X_train_num, X_val_num = X_train_num[kept], X_val_num[kept]

    # Stage 4: keep top N features by LightGBM PFI
    selected_num = select_top_features_lgbm_pfi_based(
        X_train_num, y_train, X_val_num, data.y_val, top_n=CONFIG.top_n_features,
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

    filtered = _slice_data_by_features(data, top_features, mlp_top_cols)
    logger.info("Tree features: %d  |  Scaled features: %d", len(top_features), len(mlp_top_cols))
    return filtered, top_features, mlp_top_cols


def _slice_data_by_features(
    data: ProcessedData,
    top_features: list[str],
    mlp_top_cols: list[str],
) -> ProcessedData:
    """Apply pre-computed feature lists to every variant of a ProcessedData container.

    Used both at the end of apply_feature_selection() and by the `train`
    subcommand when it loads feature lists from disk.
    """
    return ProcessedData(
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


def train_and_save_models(
    models: list[str],
    data: ProcessedData,
    tracker: ExperimentTracker,
    n_trials: int,
    top_features: list[str],
    mlp_top_cols: list[str],
    use_cv: bool = False,
    cv_splits: int = 5,
) -> list[dict]:
    """Train each model and save its artifacts IMMEDIATELY after training finishes.

    This means: if model 3 of 5 crashes, models 1 and 2 are already on disk.
    You can also `tail -f` the run folder while later models are still training.

    Saves per model:
    - Custom tracker: run.json, evaluation_results.csv, threshold_sweep.csv,
      submission.csv, model.pkl, val_proba.csv, threshold_selection.png,
      reliability_diagram.png, features.json, optuna_trials.csv,
      optuna_study.pkl, feature_importance_pct.csv (tree only)
    - MLflow: params, metrics, model artifact, all per-run artifacts

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
        result = spec["fn"](
            x_tr, data.y_train, x_vl, data.y_val, x_te, n_trials,
            use_cv=use_cv, cv_splits=cv_splits,
        )

        # Evaluate
        model_eval = pd.DataFrame(evaluate_model(
            result["name"], result["threshold"],
            result["train_proba"], result["val_proba"],
            result["test_preds"], data.y_train, data.y_val,
        ))
        run_metrics = build_run_metrics(model_eval, result["threshold_info"], result["threshold"])
        # Add best_iteration from tree models (None for MLP/GP/SVM) so it lands in run.json
        if result.get("best_iteration") is not None:
            run_metrics["best_iteration"] = int(result["best_iteration"])

        # Features for this run (needed by predict.py to re-slice input data)
        features_meta = {
            "tree_features":   top_features,
            "scaled_features": mlp_top_cols,
            "data_variant":    spec["data"],   # "tree" or "scaled"
        }

        # Save IMMEDIATELY — before training the next model
        save_to_tracker(result, run_metrics, model_eval, tracker, features_meta, data)
        save_to_mlflow(result, run_metrics, tracker)
        logger.info("[%s] artifacts saved — moving to next model", name)

        trained.append(result)

    return trained


def _history_to_long_df(history: dict) -> pd.DataFrame:
    """Convert a training history dict into a long-format DataFrame.

    Handles 3 formats:
    - Flat:    {"train": [...], "val": [...]}            → single metric "value"
    - Nested:  {"train": {metric: [...]}, "val": {...}}  → multi-metric
    - MLP:     {"train_loss": [...], "val_loss": [...]}  → single metric "loss"

    Output columns: iteration, split, metric, value
    """
    rows = []
    # MLP format
    if "train_loss" in history:
        for i, v in enumerate(history["train_loss"]):
            rows.append({"iteration": i, "split": "train", "metric": "loss", "value": v})
        for i, v in enumerate(history["val_loss"]):
            rows.append({"iteration": i, "split": "val", "metric": "loss", "value": v})
        return pd.DataFrame(rows)

    train_entry = history["train"]
    val_entry   = history["val"]

    # Nested multi-metric (xgb, lgbm)
    if isinstance(train_entry, dict):
        for metric, values in train_entry.items():
            for i, v in enumerate(values):
                rows.append({"iteration": i, "split": "train", "metric": metric, "value": v})
        for metric, values in val_entry.items():
            for i, v in enumerate(values):
                rows.append({"iteration": i, "split": "val", "metric": metric, "value": v})
        return pd.DataFrame(rows)

    # Flat single-metric (GP, SVM — single point or small list)
    for i, v in enumerate(train_entry):
        rows.append({"iteration": i, "split": "train", "metric": "value", "value": v})
    for i, v in enumerate(val_entry):
        rows.append({"iteration": i, "split": "val", "metric": "value", "value": v})
    return pd.DataFrame(rows)


def save_to_tracker(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    res: dict,
    run_metrics: dict,
    model_eval,
    tracker: ExperimentTracker,
    features_meta: dict,
    data: ProcessedData,
):
    """Save one model's artifacts via the custom tracker."""
    name = res["name"]
    submission = pd.DataFrame({
        CONFIG.id_col: res["test_index"],
        CONFIG.target_col: res["test_preds"],
    })
    val_proba_df = pd.DataFrame({"y_proba": res["val_proba"]})
    history_df = _history_to_long_df(res["history"])

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
        tracker.log_dataframe(history_df, "training_history.csv")
        if res["feature_importance_pct"] is not None:
            fi = res["feature_importance_pct"]
            tracker.log_dataframe(
                fi.reset_index().rename(columns={"index": "feature", 0: "importance_pct"}),
                "feature_importance_pct.csv",
            )

        # Features manifest — consumed by predict.py to re-slice new data
        (tracker.run_dir / "features.json").write_text(json.dumps(features_meta, indent=2))

        # Threshold selection plot (secondary metric vs Youden Index)
        plot_threshold_selection(
            res["sweep"], res["threshold_info"], name,
            secondary_metric=CONFIG.secondary_metric,
            save_path=tracker.run_dir / "threshold_selection.png",
        )
        # Reliability diagram (probability calibration diagnostic, on val set)
        plot_reliability_diagram(
            data.y_val, res["val_proba"], name,
            split="val",
            save_path=tracker.run_dir / "reliability_diagram.png",
        )
        # Per-model training curves (only if history is nested multi-metric)
        if isinstance(res["history"].get("train"), dict):
            plot_training_curves(
                res["history"], name,
                save_path=tracker.run_dir / "training_curves.png",
            )
        # Per-model feature importance plot (tree models only)
        if res["feature_importance_pct"] is not None:
            plot_model_feature_importance(
                res["feature_importance_pct"], name,
                save_path=tracker.run_dir / "feature_importance.png",
            )


def save_to_mlflow(res: dict, run_metrics: dict, tracker: ExperimentTracker):
    """Save one model's params, metrics, model, AND all run-folder artifacts to MLflow."""
    logger = logging.getLogger(__name__)
    mlflow_params = {k: str(v) for k, v in res["params"].items()}
    mlflow_params["model"] = res["label"]
    mlflow_params["n_trials"] = len(res["study"].trials)

    with mlflow.start_run(run_name=f"{res['name']}_optuna"):
        mlflow.log_params(mlflow_params)
        mlflow.log_metrics(run_metrics)
        mlflow.set_tag("model_type", res["label"])
        mlflow.sklearn.log_model(res["model"], name="model")

        # Log all run-folder artifacts for UI browsing
        run_dir = tracker.run_dir
        if run_dir is not None:
            artifact_files = [
                "evaluation_results.csv", "threshold_sweep.csv", "submission.csv",
                "val_proba.csv", "training_history.csv",
                "threshold_selection.png", "training_curves.png",
                "reliability_diagram.png",
                "feature_importance.png", "feature_importance_pct.csv",
                "features.json",
                "optuna_trials.csv",
            ]
            for fname in artifact_files:
                fpath = run_dir / fname
                if fpath.exists():
                    mlflow.log_artifact(str(fpath))

        # Global artifacts (shared across runs): feature selection report + preprocessing state
        fs_report_path = RUNS_DIR / "feature_selection_report.csv"
        if fs_report_path.exists():
            mlflow.log_artifact(str(fs_report_path))

        preprocessing_dir = RUNS_DIR / PREPROCESSING_DIR_NAME
        if preprocessing_dir.exists():
            for fname in ("imputer.pkl", "scaler.pkl", "numeric_columns.json", "scaled_columns.json"):
                fpath = preprocessing_dir / fname
                if fpath.exists():
                    mlflow.log_artifact(str(fpath), artifact_path="preprocessing")

        logger.info("[mlflow] Run logged: %s", res["name"])


def merge_summary_and_log():
    """Merge per-model evaluation_results.csv into evaluation_summary.csv (also generates combined ROC)."""
    merge_evaluation_summary()
    logging.getLogger(__name__).info("Evaluation summary → %s", RUNS_DIR / "evaluation_summary.csv")
    write_run_report_md()


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
        flag = "⚠" if gap > CONFIG.overfit_warning_gap else "✓"
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
        plot_feature_importance(feat_importances, top_n=CONFIG.top_n_features)
        plot_permutation_importance(
            estimators={r["name"]: r["model"] for r in tree_trained},
            Xs={r["name"]: data.X_val for r in tree_trained},
            y_val=data.y_val,
            top_n=CONFIG.top_n_features,
            n_repeats=5,
        )


def save_best_submission(trained: list[dict], eval_df: pd.DataFrame):
    """Copy the best model's submission to data/processed/submission.csv."""
    logger = logging.getLogger(__name__)
    val_aucs = eval_df[eval_df["split"] == "val"].set_index("model")["roc_auc"]
    best_name = val_aucs.idxmax()
    best_res = next(r for r in trained if r["name"] == best_name)
    logger.info("Best model: %s (val AUC = %.4f)", best_name.upper(), val_aucs[best_name])

    submission = pd.DataFrame({
        CONFIG.id_col: best_res["test_index"],
        CONFIG.target_col: best_res["test_preds"],
    })
    submission_path = SUBMISSION_DIR / "submission.csv"
    submission.to_csv(submission_path, index=False)
    logger.info("Submission saved → %s  (%d rows)", submission_path, len(submission))
    logger.info("Predicted positive rate: %.2f%%", submission[CONFIG.target_col].mean() * 100)


# ── CLI entry point ───────────────────────────────────────────────────────────

@click.group()
def cli():
    """ML pipeline — run all stages (`run`) or each one independently.

    Stages in order:

    \b
        1. preprocess       → read CSVs, build variants, save to reports/runs/preprocessing/
        2. feature-select   → load preprocessed data, run 4-stage filtering
        3. train            → load preprocessed + features, train models
        4. report           → cross-run comparison (doesn't train anything)

    Or run everything end-to-end:

    \b
        run                 → preprocess + feature-select + train in one go
    """


def _persist_preprocessing(data: ProcessedData) -> None:
    """Save both the fitted transformers and the DataFrames to reports/runs/preprocessing/."""
    logger = logging.getLogger(__name__)
    out_dir = RUNS_DIR / PREPROCESSING_DIR_NAME
    save_preprocessing_artifacts(data, out_dir)
    save_processed_dataframes(data, out_dir)
    logger.info("Preprocessing artifacts → %s/", out_dir)


@cli.command("preprocess")
def preprocess_cmd():
    """Stage 1: read raw CSVs, build variants, save to reports/runs/preprocessing/."""
    log_path = setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Log → %s", log_path)

    data = load_data()
    _persist_preprocessing(data)

    logger.info("✓ Stage 1 (preprocess) complete.")
    logger.info("  Next step: `make pipeline-feature-select`")


@cli.command("feature-select")
def feature_select_cmd():
    """Stage 2: load preprocessed data, run 4-stage feature selection."""
    log_path = setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Log → %s", log_path)

    data = load_processed_data(RUNS_DIR / PREPROCESSING_DIR_NAME)
    _, top_features, mlp_top_cols = apply_feature_selection(data)
    save_feature_selection(
        top_features, mlp_top_cols, RUNS_DIR / FEATURE_SELECTION_DIR_NAME,
    )

    logger.info("✓ Stage 2 (feature-select) complete.")
    logger.info("  Feature lists → %s/", RUNS_DIR / FEATURE_SELECTION_DIR_NAME)
    logger.info("  Next step: `make pipeline-train` (optionally with MODELS=...)")


@cli.command("train")
@click.option(
    "--models", "-m", multiple=True, default=ALL_MODELS,
    type=click.Choice(ALL_MODELS, case_sensitive=False),
    help="Models to train. Default: all.",
)
@click.option("--n-trials", default=30, show_default=True, help="Optuna trials per model.")
@click.option("--cv", is_flag=True, help="Use Stratified K-fold CV in Optuna instead of single split.")
@click.option(
    "--cv-folds", default=CONFIG.cv_splits, show_default=True, type=int,
    help="Number of folds when --cv is enabled. Overrides CONFIG.cv_splits.",
)
def train_cmd(models, n_trials, cv, cv_folds):
    """Stage 3: load preprocessed + selected features, train models, save artifacts."""
    log_path = setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Log → %s", log_path)

    tracker = ExperimentTracker(CONFIG.experiment)
    models = list(models)
    logger.info("Models to train: %s", models)

    data = load_processed_data(RUNS_DIR / PREPROCESSING_DIR_NAME)
    top_features, mlp_top_cols = load_feature_selection(RUNS_DIR / FEATURE_SELECTION_DIR_NAME)
    data = _slice_data_by_features(data, top_features, mlp_top_cols)
    logger.info(
        "Loaded preprocessed data + selected features (%d tree / %d scaled)",
        len(top_features), len(mlp_top_cols),
    )

    trained = train_and_save_models(
        models, data, tracker, n_trials, top_features, mlp_top_cols,
        use_cv=cv, cv_splits=cv_folds,
    )
    eval_df = _build_eval_df(trained, data)
    merge_summary_and_log()
    print_summary(trained, eval_df)
    generate_plots(trained, data)
    save_best_submission(trained, eval_df)
    logger.info("✓ Stage 3 (train) complete.")
    logger.info("Full log → %s", log_path)


@cli.command("report")
def report_cmd():
    """Cross-run comparison: generate reports/runs/comparison.{csv,png}."""
    setup_logging()
    tracker = ExperimentTracker(CONFIG.experiment)
    tracker.generate_report()


@cli.command("run")
@click.option(
    "--models", "-m", multiple=True, default=ALL_MODELS,
    type=click.Choice(ALL_MODELS, case_sensitive=False),
    help="Models to train. Default: all.",
)
@click.option("--n-trials", default=30, show_default=True, help="Optuna trials per model.")
@click.option("--cv", is_flag=True, help="Use Stratified K-fold CV in Optuna instead of single split.")
@click.option(
    "--cv-folds", default=CONFIG.cv_splits, show_default=True, type=int,
    help="Number of folds when --cv is enabled. Overrides CONFIG.cv_splits.",
)
def run_cmd(models, n_trials, cv, cv_folds):
    """Run ALL stages end-to-end (preprocess + feature-select + train)."""
    log_path = setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Log → %s", log_path)

    tracker = ExperimentTracker(CONFIG.experiment)
    models = list(models)
    logger.info("Models to train: %s", models)

    # Stage 1: preprocess
    data = load_data()
    _persist_preprocessing(data)

    # Stage 2: feature selection
    data, top_features, mlp_top_cols = apply_feature_selection(data)
    save_feature_selection(
        top_features, mlp_top_cols, RUNS_DIR / FEATURE_SELECTION_DIR_NAME,
    )

    # Stage 3: train + save each model incrementally
    trained = train_and_save_models(
        models, data, tracker, n_trials, top_features, mlp_top_cols,
        use_cv=cv, cv_splits=cv_folds,
    )
    eval_df = _build_eval_df(trained, data)

    # Cross-model aggregation
    merge_summary_and_log()
    print_summary(trained, eval_df)
    generate_plots(trained, data)
    save_best_submission(trained, eval_df)
    logger.info("Full log → %s", log_path)


def _build_eval_df(trained: list[dict], data: ProcessedData) -> pd.DataFrame:
    """Build the cross-model evaluation table used for summary + submission selection."""
    return evaluate_all(
        {r["name"]: {"threshold": r["threshold"], "train_proba": r["train_proba"],
                     "val_proba": r["val_proba"], "test_preds": r["test_preds"]}
         for r in trained},
        data.y_train, data.y_val,
    )


if __name__ == "__main__":
    cli()  # pylint: disable=no-value-for-parameter
