"""
Performance metric helpers for train/val/test summaries and threshold optimisation.

Key functions
-------------
threshold_sweep(y_true, y_proba)
    Evaluate every metric at every threshold → DataFrame.

find_best_threshold(y_true, y_proba, secondary_metric="accuracy", tolerance=0.02)
    1. Find the threshold that maximises Youden Index.
    2. Among all thresholds within `tolerance` of that best Youden, pick the
       one that maximises `secondary_metric` (accuracy, precision, recall …).

perf_row / test_row
    Single-row dicts for the per-model performance_summary.csv.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score, roc_curve,
)

# Default threshold grid: 0.10 → 0.90 in 0.01 steps (rounded to avoid float drift)
THRESHOLDS = np.round(np.arange(0.10, 0.91, 0.01), 2)

# Default Youden tolerance for threshold selection
DEFAULT_TOLERANCE = 0.02


# ── KS & Gini (probability-based, threshold-free) ────────────────────────────

def ks_statistic(y_true, y_proba: np.ndarray) -> float:
    """Kolmogorov–Smirnov statistic: max|TPR − FPR| across all thresholds."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    return float(np.max(tpr - fpr))


def gini_coefficient(y_true, y_proba: np.ndarray) -> float:
    """Gini = 2 × AUC − 1.  Ranges from 0 (random) to 1 (perfect)."""
    return 2.0 * roc_auc_score(y_true, y_proba) - 1.0


# ── Threshold sweep ──────────────────────────────────────────────────────────

def threshold_sweep(
    y_true,
    y_proba: np.ndarray,
    thresholds: np.ndarray = THRESHOLDS,
) -> pd.DataFrame:
    """Evaluate accuracy, precision, recall, F1 and Youden Index at every threshold.

    Returns a DataFrame with one row per threshold.  All metrics are computed on
    the validation set so the result can be used to pick any operating point
    without touching the test set.

    Youden Index = TPR + TNR − 1  (class-imbalance agnostic; range −1 … 1).
    """
    y_true = np.asarray(y_true)
    rows = []
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        rows.append({
            "threshold":            round(float(t), 2),
            "accuracy":             round(accuracy_score(y_true, y_pred), 4),
            "precision":            round(precision_score(y_true, y_pred, zero_division=0), 4),
            "recall":               round(sensitivity, 4),
            "specificity":          round(specificity, 4),
            "f1":                   round(f1_score(y_true, y_pred, zero_division=0), 4),
            "youden_index":         round(sensitivity + specificity - 1, 4),
            "n_predicted_positive": int(y_pred.sum()),
            "positive_rate":        round(float(y_pred.mean()), 4),
        })
    return pd.DataFrame(rows)


# ── Best threshold selection ─────────────────────────────────────────────────

def find_best_threshold(
    y_true,
    y_proba: np.ndarray,
    secondary_metric: str = "accuracy",
    tolerance: float = DEFAULT_TOLERANCE,
    thresholds: np.ndarray = THRESHOLDS,
) -> dict:
    """Pick the best operating threshold using Youden + secondary metric.

    Algorithm
    ---------
    1. Compute the full threshold_sweep.
    2. Find the maximum Youden Index across all thresholds.
    3. Keep only thresholds whose Youden is within `tolerance` of that max.
    4. Among those, choose the one that maximises `secondary_metric`.

    This ensures the classifier maintains near-optimal discrimination (Youden)
    while squeezing out the best competition metric (e.g. accuracy) within that
    near-optimal band.

    Args:
        secondary_metric: column in the sweep table to optimise within the
                          Youden-tolerance band ("accuracy", "precision",
                          "recall", "f1", etc.)
        tolerance:        how far below max Youden a threshold can be and
                          still be considered (default 0.02).

    Returns:
        dict with keys: threshold, youden_index, <secondary_metric>,
        best_youden (the global max), n_candidates (how many thresholds
        were in the tolerance band).
    """
    sweep = threshold_sweep(y_true, y_proba, thresholds)
    best_youden = float(sweep["youden_index"].max())

    # Filter to tolerance band around the best Youden
    candidates = sweep[sweep["youden_index"] >= best_youden - tolerance]
    idx = candidates[secondary_metric].idxmax()

    return {
        "threshold":       float(candidates.loc[idx, "threshold"]),
        "youden_index":    float(candidates.loc[idx, "youden_index"]),
        secondary_metric:  float(candidates.loc[idx, secondary_metric]),
        "best_youden":     best_youden,
        "n_candidates":    len(candidates),
    }


# ── Per-split metric rows ───────────────────────────────────────────────────

def perf_row(split: str, y_true, y_proba: np.ndarray, y_pred: np.ndarray) -> dict:
    """Return a metrics dict for one labelled split (train or val).

    Includes ROC-AUC, PR-AUC, accuracy, precision, recall, F1,
    KS statistic, and Gini coefficient.
    """
    auc_val = roc_auc_score(y_true, y_proba)
    return {
        "split":         split,
        "roc_auc":       round(auc_val, 4),
        "gini":          round(2.0 * auc_val - 1.0, 4),
        "ks_statistic":  round(ks_statistic(y_true, y_proba), 4),
        "pr_auc":        round(average_precision_score(y_true, y_proba), 4),
        "accuracy":      round(accuracy_score(y_true, y_pred), 4),
        "precision":     round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":        round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1":            round(f1_score(y_true, y_pred, zero_division=0), 4),
        "n_samples":     int(len(y_true)),
        "n_positive":    int(y_pred.sum()),
        "positive_rate": round(float(y_pred.mean()), 4),
    }


def test_row(y_pred: np.ndarray) -> dict:
    """Return a metrics dict for the test split (no labels available)."""
    return {
        "split":         "test",
        "roc_auc":       None,
        "gini":          None,
        "ks_statistic":  None,
        "pr_auc":        None,
        "accuracy":      None,
        "precision":     None,
        "recall":        None,
        "f1":            None,
        "n_samples":     int(len(y_pred)),
        "n_positive":    int(y_pred.sum()),
        "positive_rate": round(float(y_pred.mean()), 4),
    }
