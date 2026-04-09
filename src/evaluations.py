"""
Evaluation pipeline — compute metrics for any model across train / val / test splits.

Reusable functions:
    evaluate_model(...)  → 3-row list (train, val, test) for one model
    evaluate_all(...)    → DataFrame with all models × all splits

Output files:
    Per model:  reports/runs/<ts>_<model>/evaluation_results.csv  (3 rows)
    Global:     reports/runs/evaluation_summary.csv               (all models × splits)
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd

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


def merge_evaluation_summary(runs_dir: Path = RUNS_DIR) -> pd.DataFrame:
    """Merge all per-model evaluation_results.csv into a single evaluation_summary.csv.

    Scans reports/runs/*/evaluation_results.csv, keeps the latest run per model
    (by folder timestamp), and writes the merged result.

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

    merged = pd.concat(frames, ignore_index=True)

    # Keep only the latest run per model (last folder alphabetically = latest timestamp)
    merged = merged.sort_values("run_folder")
    merged = merged.drop_duplicates(subset=["model", "split"], keep="last")
    merged = merged.drop(columns=["run_folder"])

    out = runs_dir / "evaluation_summary.csv"
    merged.to_csv(out, index=False)
    logger.info(
        "Merged %d run(s) → %s (%d models: %s)",
        len(files), out.name, merged["model"].nunique(),
        sorted(merged["model"].unique()),
    )
    return merged
