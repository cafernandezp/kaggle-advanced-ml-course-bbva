"""
Evaluation pipeline — compute metrics for any model across train / val / test splits.

Reusable functions:
    evaluate_model(...)  → 3-row list (train, val, test) for one model
    evaluate_all(...)    → DataFrame with all models × all splits

Output files:
    Per model:  reports/runs/<ts>_<model>/evaluation_results.csv  (3 rows)
    Global:     reports/runs/evaluation_summary.csv               (all models × splits)
"""
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve

from src.metrics import perf_row, test_row

ROOT = Path(__file__).parent.parent
RUNS_DIR = ROOT / "reports/runs"

logger = logging.getLogger(__name__)


def evaluate_split(
    model_name: str,
    y_true,
    y_proba: np.ndarray,
    threshold: float,
    split: str,
) -> dict:
    """Evaluate one model on one labelled split (train or val).

    Returns a flat dict with model name, split, threshold, and all metrics
    from perf_row (roc_auc, gini, ks_statistic, pr_auc, accuracy, precision,
    recall, f1, n_samples, n_positive, positive_rate).
    """
    y_pred = (y_proba >= threshold).astype(int)
    row = perf_row(split, y_true, y_proba, y_pred)
    return {"model": model_name, "threshold": round(threshold, 4), **row}


def evaluate_test_split(
    model_name: str,
    y_pred: np.ndarray,
    threshold: float,
    split: str = "test",
) -> dict:
    """Evaluate one model on the test split (no labels available).

    Returns a flat dict with model name, split, threshold, and prediction
    statistics (n_samples, n_positive, positive_rate). Metric fields are None.
    """
    row = test_row(y_pred)
    row["split"] = split
    return {"model": model_name, "threshold": round(threshold, 4), **row}


def evaluate_model(
    model_name: str,
    threshold: float,
    train_proba: np.ndarray,
    val_proba: np.ndarray,
    test_preds: np.ndarray,
    y_train,
    y_val,
) -> list[dict]:
    """Evaluate one model on all 3 splits.

    Returns:
        [train_row, val_row, test_row] — each is a flat dict.
    """
    return [
        evaluate_split(model_name, y_train, train_proba, threshold, "train"),
        evaluate_split(model_name, y_val,   val_proba,   threshold, "val"),
        evaluate_test_split(model_name, test_preds, threshold),
    ]


def evaluate_all(
    models_data: dict,
    y_train,
    y_val,
) -> pd.DataFrame:
    """Evaluate all models on all splits and return a unified DataFrame.

    Args:
        models_data: {model_name: {"threshold", "train_proba", "val_proba", "test_preds"}}
        y_train, y_val: ground-truth labels

    Returns:
        DataFrame with one row per (model × split).
        Columns: model, split, threshold, roc_auc, gini, ks_statistic, pr_auc,
                 accuracy, precision, recall, f1, n_samples, n_positive, positive_rate.
    """
    rows = []
    for name, data in models_data.items():
        model_rows = evaluate_model(
            model_name=name,
            threshold=data["threshold"],
            train_proba=data["train_proba"],
            val_proba=data["val_proba"],
            test_preds=data["test_preds"],
            y_train=y_train,
            y_val=y_val,
        )
        rows.extend(model_rows)
        logger.info(
            "[eval] %s  threshold=%.2f  val_auc=%.4f  val_acc=%.4f",
            name.upper(), data["threshold"],
            model_rows[1].get("roc_auc", 0), model_rows[1].get("accuracy", 0),
        )

    return pd.DataFrame(rows)


def build_run_metrics(model_eval: pd.DataFrame, threshold_info: dict, threshold: float) -> dict:
    """Build a flat metrics dict for run.json from a model's evaluation DataFrame.

    Args:
        model_eval: DataFrame with train/val/test rows (from evaluate_model)
        threshold_info: dict from find_best_threshold (youden_index, best_youden)
        threshold: the chosen threshold value

    Returns:
        dict with all metrics rounded to 4 decimals, ready for tracker.log_metrics()
    """
    val_row   = model_eval[model_eval["split"] == "val"].iloc[0]
    train_row = model_eval[model_eval["split"] == "train"].iloc[0]
    return {
        "val_auc":         round(val_row["roc_auc"], 4),
        "train_auc":       round(train_row["roc_auc"], 4),
        "val_accuracy":    round(val_row["accuracy"], 4),
        "train_accuracy":  round(train_row["accuracy"], 4),
        "val_precision":   round(val_row["precision"], 4),
        "train_precision": round(train_row["precision"], 4),
        "val_recall":      round(val_row["recall"], 4),
        "train_recall":    round(train_row["recall"], 4),
        "val_ks":          round(val_row["ks_statistic"], 4),
        "train_ks":        round(train_row["ks_statistic"], 4),
        "val_gini":        round(val_row["gini"], 4),
        "train_gini":      round(train_row["gini"], 4),
        "val_brier":       round(val_row["brier_score"], 4),
        "train_brier":     round(train_row["brier_score"], 4),
        "val_ece":         round(val_row["ece"], 4),
        "train_ece":       round(train_row["ece"], 4),
        "threshold":       round(threshold, 4),
        "youden_index":    round(threshold_info["youden_index"], 4),
        "best_youden":     round(threshold_info["best_youden"], 4),
        "overfit_gap_auc": round(train_row["roc_auc"] - val_row["roc_auc"], 4),
    }


def _pivot_long_to_wide(long_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot a long evaluation DataFrame (1 row per model×split) into wide format
    with one row per model: val metrics first, then train metrics, then test stats.

    Column ordering: model, threshold, val_*, train_*, test_*.
    All column names are lower_snake_case.
    """
    # Columns that are shared across splits (and we keep once per model)
    shared_cols = ["model", "threshold"]
    # Metric columns (those that make sense for labelled splits — val/train)
    labelled_metrics = [
        "roc_auc", "gini", "ks_statistic", "pr_auc",
        "accuracy", "precision", "recall", "f1",
    ]
    # Test-only columns (no labels → only prediction stats are meaningful)
    test_only = ["n_samples", "n_positive", "positive_rate"]

    rows: list[dict] = []
    for model in long_df["model"].unique():
        model_df = long_df[long_df["model"] == model]
        val   = model_df[model_df["split"] == "val"].iloc[0]
        train = model_df[model_df["split"] == "train"].iloc[0]
        test  = model_df[model_df["split"] == "test"].iloc[0]

        row: dict = {c: val[c] for c in shared_cols}
        # val metrics first
        for m in labelled_metrics:
            row[f"val_{m}"] = val[m]
        row["val_n_samples"]    = val["n_samples"]
        row["val_n_positive"]   = val["n_positive"]
        row["val_positive_rate"] = val["positive_rate"]
        # train metrics next
        for m in labelled_metrics:
            row[f"train_{m}"] = train[m]
        row["train_n_samples"]    = train["n_samples"]
        row["train_n_positive"]   = train["n_positive"]
        row["train_positive_rate"] = train["positive_rate"]
        # test prediction stats last (no labelled metrics)
        for c in test_only:
            row[f"test_{c}"] = test[c]

        rows.append(row)

    return pd.DataFrame(rows)


def merge_evaluation_summary(runs_dir: Path = RUNS_DIR) -> pd.DataFrame:
    """Merge all per-model evaluation_results.csv into a single evaluation_summary.csv.

    Scans reports/runs/*/evaluation_results.csv, keeps the latest run per model
    (by folder timestamp), and writes a wide-format summary (one row per model)
    with val metrics, then train metrics, then test prediction stats.

    Standalone: make eval-summary
    """
    files = sorted(runs_dir.glob("*/evaluation_results.csv"))
    if not files:
        logger.info("No evaluation_results.csv found in %s", runs_dir)
        return pd.DataFrame()

    frames = []
    for f in files:
        df = pd.read_csv(f)
        df["run_folder"] = f.parent.name
        frames.append(df)

    long_merged = pd.concat(frames, ignore_index=True)

    # Keep only the latest run per model (last folder alphabetically = latest timestamp)
    long_merged = long_merged.sort_values("run_folder")
    long_merged = long_merged.drop_duplicates(subset=["model", "split"], keep="last")
    long_merged = long_merged.drop(columns=["run_folder"])

    # Pivot to wide format: one row per model, val→train→test column order
    wide = _pivot_long_to_wide(long_merged).sort_values("model").reset_index(drop=True)

    out = runs_dir / "evaluation_summary.csv"
    wide.to_csv(out, index=False)
    logger.info(
        "Merged %d run(s) → %s (%d models: %s)",
        len(files), out.name, wide["model"].nunique(),
        sorted(wide["model"].unique()),
    )

    # Generate combined ROC from all historical runs
    generate_combined_roc(runs_dir)

    return wide


def generate_combined_roc(runs_dir: Path = RUNS_DIR) -> None:
    """Generate a combined ROC curve from all historical runs.

    Reads val_proba.csv + run.json from each run folder, loads y_val from
    preprocessing, plots all models on one chart.

    Standalone: make eval-summary (calls this after merging).
    """
    from src.preprocessing import preprocess_data  # pylint: disable=import-outside-toplevel

    proba_files = sorted(runs_dir.glob("*/val_proba.csv"))
    if not proba_files:
        logger.info("No val_proba.csv found — skipping combined ROC")
        return

    # Load y_val once (re-applies preprocessing — same train/val split)
    y_val = preprocess_data().y_val

    # Collect latest run per model
    model_probas: dict[str, np.ndarray] = {}
    model_folders: dict[str, str] = {}
    for f in proba_files:
        run_json = f.parent / "run.json"
        if not run_json.exists():
            continue
        run_data = json.loads(run_json.read_text())
        model_name = run_data.get("params", {}).get("model", f.parent.name)
        # Use the short name from the folder (e.g., "lgbm" from "20260409_103119_lgbm_optuna")
        parts = f.parent.name.split("_")
        short_name = parts[2] if len(parts) >= 3 else model_name
        model_folders[short_name] = f.parent.name
        model_probas[short_name] = pd.read_csv(f)["y_proba"].values

    if not model_probas:
        logger.info("No valid runs found for combined ROC")
        return

    colors = {"lgbm": "#4878d0", "xgb": "#ee854a", "mlp": "#6acc65",
              "gp": "#d65f5f", "svm": "#956cb4"}

    _, ax = plt.subplots(figsize=(8, 7))
    for name, proba in sorted(model_probas.items()):
        fpr, tpr, _ = roc_curve(y_val, proba)
        auc_score = auc(fpr, tpr)
        color = colors.get(name, None)
        ax.plot(fpr, tpr, label=f"{name.upper()} (AUC={auc_score:.4f})", color=color, lw=2)

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random classifier")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Combined ROC — all models (latest runs)")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    plt.tight_layout()

    out = runs_dir / "combined_roc.png"
    plt.savefig(out, dpi=130)
    plt.close()
    logger.info("Combined ROC → %s (%d models)", out, len(model_probas))
