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
from dataclasses import dataclass, field
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
    expand_features_for_mlp, load_splits, load_splits_scaled,
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


# ── Data container ────────────────────────────────────────────────────────────

@dataclass
class PipelineData:  # pylint: disable=too-many-instance-attributes,invalid-name
    """Typed container for all data splits produced by load_and_select_features.

    Attribute names use ML conventions (X_train, X_val) rather than snake_case.
    """

    X_train: pd.DataFrame
    X_val: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    X_train_scaled: pd.DataFrame
    X_val_scaled: pd.DataFrame
    top_features: list = field(default_factory=list)
    mlp_top_cols: list = field(default_factory=list)


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

def load_data() -> PipelineData:
    """Load all data splits. No feature selection — returns the full feature set."""
    logger = logging.getLogger(__name__)
    logger.info("Loading data...")
    X_train, X_val, y_train, y_val = load_splits()
    X_train_scaled, X_val_scaled, _, _ = load_splits_scaled()
    logger.info("Tree splits  : train=%s  val=%s", X_train.shape, X_val.shape)
    logger.info("Scaled splits: train=%s  val=%s", X_train_scaled.shape, X_val_scaled.shape)
    return PipelineData(
        X_train=X_train, X_val=X_val, y_train=y_train, y_val=y_val,
        X_train_scaled=X_train_scaled, X_val_scaled=X_val_scaled,
    )


def apply_feature_selection(data: PipelineData) -> PipelineData:
    """Run feature selection stages on the data. Each stage is explicit and skippable.

    Comment out any stage you don't need. The order matters:
    missing → correlation → MI → PFI
    """
    logger = logging.getLogger(__name__)
    X_train, X_val = data.X_train, data.X_val
    y_train = data.y_train
    n_start = X_train.shape[1]
    logger.info("Feature selection: starting with %d features", n_start)
    survivors: dict[str, list[str]] = {}

    # Stage 1: drop features with too many missing values
    kept = drop_high_missing(X_train, threshold=0.5)
    survivors["stage1_missing"] = kept.copy()
    X_train, X_val = X_train[kept], X_val[kept]

    # Stage 2: drop highly correlated features (Spearman, keep the one with higher MI)
    numeric_cols = X_train.select_dtypes(include="number").columns.tolist()
    if numeric_cols:
        kept_num = drop_correlated_features(X_train[numeric_cols], y_train, threshold=0.9)
        non_numeric = [c for c in X_train.columns if c not in numeric_cols]
        kept = non_numeric + kept_num
        X_train, X_val = X_train[kept], X_val[kept]
    survivors["stage2_correlation"] = list(X_train.columns)

    # Stage 3: keep top K features by Mutual Information with target
    kept = select_top_mutual_information(X_train, y_train, top_k=20)
    survivors["stage3_mi"] = kept.copy()
    X_train, X_val = X_train[kept], X_val[kept]

    # Stage 4: keep top N features by LightGBM PFI
    top_features = select_top_features_lgbm_pfi_based(X_train, y_train, X_val, data.y_val, top_n=15)
    survivors["stage4_pfi"] = top_features.copy()
    X_train, X_val = X_train[top_features], X_val[top_features]

    logger.info("Feature selection: %d → %d features", n_start, len(top_features))
    logger.info("Final features: %s", top_features)

    # Build and save the per-feature report (with stage-by-stage verdicts)
    fs_report = build_feature_selection_report(
        data.X_train, data.y_train, survivors, top_features,
    )
    fs_report.to_csv(RUNS_DIR / "feature_selection_report.csv", index=False)
    logger.info("Feature selection report → %s", RUNS_DIR / "feature_selection_report.csv")

    # Apply to scaled data
    mlp_top_cols = expand_features_for_mlp(top_features, data.X_train_scaled)
    X_train_scaled = data.X_train_scaled[mlp_top_cols]
    X_val_scaled   = data.X_val_scaled[mlp_top_cols]
    logger.info("Tree features: %d  |  Scaled features: %d", len(top_features), len(mlp_top_cols))

    return PipelineData(
        X_train=X_train, X_val=X_val, y_train=data.y_train, y_val=data.y_val,
        X_train_scaled=X_train_scaled, X_val_scaled=X_val_scaled,
        top_features=top_features, mlp_top_cols=mlp_top_cols,
    )


def train_models(models: list[str], data: PipelineData, n_trials: int, use_cv: bool = False) -> list[dict]:
    """Train each requested model and return a list of standardised result dicts."""
    _data_map = {
        "tree":   (data.X_train,        data.X_val,        data.top_features),
        "scaled": (data.X_train_scaled, data.X_val_scaled, data.mlp_top_cols),
    }
    trained = []
    for name in models:
        spec = TRAINERS[name]
        x_tr, x_vl, feat_key = _data_map[spec["data"]]
        result = spec["fn"](x_tr, data.y_train, x_vl, data.y_val, feat_key, n_trials, use_cv=use_cv)
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


def save_all_artifacts(trained: list[dict], data: PipelineData, tracker: ExperimentTracker):
    """Save artifacts for all models via custom tracker + MLflow, then merge summary."""
    setup_mlflow()
    for res in trained:
        model_eval = pd.DataFrame(evaluate_model(
            res["name"], res["threshold"],
            res["train_proba"], res["val_proba"],
            res["test_preds"], data.y_train, data.y_val,
        ))
        run_metrics = build_run_metrics(model_eval, res["threshold_info"], res["threshold"])
        save_to_tracker(res, run_metrics, model_eval, tracker)
        save_to_mlflow(res, run_metrics)

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


def generate_plots(trained: list[dict], data: PipelineData):
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
    data = apply_feature_selection(data)
    trained = train_models(models, data, n_trials, use_cv=cv)

    eval_df = evaluate_all(
        {r["name"]: {"threshold": r["threshold"], "train_proba": r["train_proba"],
                     "val_proba": r["val_proba"], "test_preds": r["test_preds"]}
         for r in trained},
        data.y_train, data.y_val,
    )

    save_all_artifacts(trained, data, tracker)
    print_summary(trained, eval_df)
    generate_plots(trained, data)
    save_best_submission(trained, eval_df)
    logger.info("Full log → %s", log_path)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
