"""
Plotting utilities: ROC-AUC comparison, loss curves, feature importance, PFI.
"""
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import auc, brier_score_loss, roc_curve

from src.metrics import expected_calibration_error

logger = logging.getLogger(__name__)

FIGURES_DIR = Path(__file__).parent.parent / "reports/figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {
    "lgbm": "#4878d0", "xgb": "#ee854a", "mlp": "#6acc65",
    "gp": "#d65f5f", "svm": "#956cb4",
}


def plot_roc_curves(
    probas: dict[str, np.ndarray],
    y_val,
    save_path: Path | None = None,
) -> None:
    """Plot ROC curves for all models on a single chart.

    Args:
        probas: {model_name: predict_proba array for val set}
        y_val:  true labels for validation set
    """
    _, ax = plt.subplots(figsize=(7, 6))

    for name, proba in probas.items():
        fpr, tpr, _ = roc_curve(y_val, proba)
        auc_score = auc(fpr, tpr)
        color = COLORS.get(name, None)
        ax.plot(fpr, tpr, label=f"{name.upper()}  (AUC = {auc_score:.4f})", color=color, lw=2)

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random classifier")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC-AUC comparison")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    plt.tight_layout()

    out = save_path or FIGURES_DIR / "roc_curves.png"
    plt.savefig(out, dpi=130)
    plt.close()
    logger.info("ROC curves → %s", out)


def _extract_loss_curve(hist: dict) -> tuple[list, list, str]:
    """Extract a single train/val curve from a history dict, tolerating multiple formats.

    Supports:
    - Flat:    {"train": [...], "val": [...]}                                  (tree legacy / GP / SVM)
    - Nested:  {"train": {"logloss": [...], "auc": [...]}, "val": {...}}       (tree multi-metric)
    - MLP:     {"train_loss": [...], "val_loss": [...]}                         (MLP)

    Picks a canonical metric (logloss/binary_logloss preferred, else first available).
    Returns (train_values, val_values, metric_label).
    """
    # MLP case
    if "train_loss" in hist:
        return list(hist["train_loss"]), list(hist["val_loss"]), "loss"

    train_entry = hist["train"]
    val_entry   = hist["val"]

    # Nested (multi-metric) case
    if isinstance(train_entry, dict):
        # Prefer logloss variants, else first metric
        metric = next(
            (m for m in ("binary_logloss", "logloss") if m in train_entry),
            next(iter(train_entry)),
        )
        return list(train_entry[metric]), list(val_entry[metric]), metric

    # Flat case
    return list(train_entry), list(val_entry), "loss"


def plot_loss_curves(
    histories: dict[str, dict],
    save_path: Path | None = None,
) -> None:
    """Plot train vs validation loss curves for each model (one canonical metric per model).

    Args:
        histories: {model_name: history dict} — flat, nested multi-metric, or MLP format
    """
    n = len(histories)
    _, axes = plt.subplots(1, n, figsize=(6 * n, 5), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, (name, hist) in zip(axes, histories.items()):
        color = COLORS.get(name, "#4878d0")
        train_vals, val_vals, metric_label = _extract_loss_curve(hist)
        x = range(1, len(train_vals) + 1)

        ax.plot(x, train_vals, label="train", color=color, lw=2)
        ax.plot(x, val_vals,   label="val",   color=color, lw=2, linestyle="--")
        ax.set_title(f"{name.upper()} — {metric_label} curve")
        ax.set_xlabel("Iteration / Epoch")
        ax.set_ylabel(metric_label)
        ax.legend()

        # Annotate overfit gap at the last point
        gap = train_vals[-1] - val_vals[-1]
        ax.annotate(
            f"Δ={gap:+.4f}",
            xy=(len(train_vals), val_vals[-1]),
            fontsize=8,
            color="gray",
        )

    plt.suptitle("Train vs Validation loss", fontsize=12)
    plt.tight_layout()

    out = save_path or FIGURES_DIR / "loss_curves.png"
    plt.savefig(out, dpi=130)
    plt.close()
    logger.info("Loss curves → %s", out)


def _best_iteration(val_vals: list, metric: str) -> int:
    """Return the 1-indexed iteration where val_vals reached its best value.

    For loss/error metrics, best = argmin. For auc/accuracy, best = argmax.
    """
    lower_is_better = any(k in metric for k in ("loss", "error"))
    if lower_is_better:
        return int(np.argmin(val_vals)) + 1
    return int(np.argmax(val_vals)) + 1


def plot_training_curves(
    history: dict,
    model_name: str,
    save_path: Path | None = None,
) -> None:
    """Plot train vs val curves for EVERY metric in a nested history dict.

    Expects history in the format:
        {"train": {"metric1": [...], "metric2": [...]}, "val": {"metric1": [...], ...}}

    Each subplot has:
    - train (blue) and val (red) lines
    - A vertical green dashed line at the early-stopping / best-iteration point
    - 'error' / 'binary_error' is automatically flipped to 'accuracy' (y = 1 - error)

    Args:
        history: nested history dict (multi-metric, tree models only)
        model_name: model label for the plot title
        save_path: output path; default is `reports/figures/<model>_training_curves.png`
    """
    train_hist = history["train"]
    val_hist   = history["val"]
    if not isinstance(train_hist, dict):
        logger.warning("plot_training_curves: history is not multi-metric for %s — skipping", model_name)
        return

    metrics = list(train_hist.keys())
    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4.5), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        train_vals = list(train_hist[metric])
        val_vals   = list(val_hist[metric])

        # Find best iteration on the RAW metric (before any flip for display)
        best_iter = _best_iteration(val_vals, metric)

        # Flip error → accuracy for clarity
        if metric in ("error", "binary_error"):
            display_metric = "accuracy"
            train_vals = [1 - v for v in train_vals]
            val_vals   = [1 - v for v in val_vals]
        else:
            display_metric = metric

        x = list(range(1, len(train_vals) + 1))
        ax.plot(x, train_vals, label="train", color="#4878d0", lw=2)
        ax.plot(x, val_vals,   label="val",   color="#d62728", lw=2)

        # Mark the best / early-stopping point.
        # Draw BEFORE setting xlim so the line is fully visible; nudge x-range a bit
        # on both sides so best_iter at position 1 or len(x) doesn't get clipped.
        ax.axvline(
            best_iter, color="#2ca02c", lw=2.0, linestyle="--",
            alpha=0.85, zorder=5, label=f"best iter={best_iter}",
        )
        # Mark the best point with a green dot on the val curve
        ax.scatter(
            [best_iter], [val_vals[best_iter - 1]],
            color="#2ca02c", s=70, zorder=6, edgecolors="black", linewidths=0.8,
        )

        ax.set_xlim(0, len(x) + 1)
        ax.set_xlabel("Iteration")
        ax.set_ylabel(display_metric)
        ax.set_title(display_metric.replace("_", " ").title())
        ax.legend(loc="best", fontsize=9)
        ax.grid(alpha=0.3)

    fig.suptitle(f"{model_name.upper()} — training curves (train vs val)", fontsize=12)
    plt.tight_layout()

    out = save_path or FIGURES_DIR / f"{model_name}_training_curves.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    logger.info("Training curves → %s", out)


def plot_model_feature_importance(
    feature_importances: pd.Series,
    model_name: str,
    top_n: int = 15,
    save_path: Path | None = None,
) -> None:
    """Plot a single model's feature importance as a horizontal bar chart.

    Used for per-run artifacts (vs. `plot_feature_importance` which compares multiple models).

    Args:
        feature_importances: pd.Series indexed by feature name, values = raw importance
        model_name: model label for the title
        top_n: number of top features to display
        save_path: output path; default is `reports/figures/<model>_feature_importance.png`
    """
    pct = (feature_importances / feature_importances.sum() * 100).nlargest(top_n).sort_values()
    color = COLORS.get(model_name, "#4878d0")

    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.35)))
    bars = ax.barh(pct.index, pct.values, color=color, edgecolor="white")
    ax.set_xlabel("% of total importance")
    ax.set_title(f"{model_name.upper()} — top {top_n} features (% importance)")
    ax.set_xlim(0, pct.max() * 1.18)

    for rect, val in zip(bars, pct.values):
        ax.text(val + 0.2, rect.get_y() + rect.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=8)

    plt.tight_layout()
    out = save_path or FIGURES_DIR / f"{model_name}_feature_importance.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    logger.info("Feature importance (single model) → %s", out)


def plot_feature_importance(
    importances: dict[str, pd.Series],
    top_n: int = 15,
    save_path: Path | None = None,
) -> None:
    """Plot feature importance as % of total for each tree model side by side.

    Args:
        importances: {model_name: pd.Series(importance, index=feature_name)}
                     Raw gain values — normalised to % internally.
        top_n: number of top features to display per model.
    """
    n = len(importances)
    _, axes = plt.subplots(1, n, figsize=(7 * n, max(5, top_n * 0.4)))
    if n == 1:
        axes = [axes]

    for ax, (name, imp) in zip(axes, importances.items()):
        # Normalise to percentage
        pct = imp / imp.sum() * 100
        pct = pct.nlargest(top_n).sort_values(ascending=True)

        color = COLORS.get(name, "#4878d0")
        bars = ax.barh(pct.index, pct.values, color=color, edgecolor="white")
        ax.set_title(f"{name.upper()} — feature importance (%)", fontsize=11)
        ax.set_xlabel("% of total importance (gain)")
        ax.set_xlim(0, pct.max() * 1.18)

        # Annotate each bar with its percentage
        for rect, val in zip(bars, pct.values):
            ax.text(
                val + 0.2, rect.get_y() + rect.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=8,
            )

    plt.suptitle(f"Top-{top_n} features by importance", fontsize=12)
    plt.tight_layout()

    out = save_path or FIGURES_DIR / "feature_importance_pct.png"
    plt.savefig(out, dpi=130)
    plt.close()
    logger.info("Feature importance → %s", out)


def plot_permutation_importance(
    estimators: dict[str, object],
    Xs: dict[str, pd.DataFrame],
    y_val,
    top_n: int = 15,
    n_repeats: int = 10,
    random_state: int = 42,
    save_path: Path | None = None,
) -> None:
    """Compute and plot Permutation Feature Importance (PFI) for all models.

    PFI is model-agnostic: it shuffles each feature n_repeats times and
    measures the mean drop in ROC-AUC, so results are comparable across
    LightGBM, XGBoost, and MLP.

    Args:
        estimators: {model_name: sklearn-compatible estimator}
        Xs:         {model_name: validation DataFrame} (may differ for MLP)
        y_val:      true labels for the validation set
        top_n:      number of top features to display per model
        n_repeats:  number of shuffle repetitions per feature
    """
    n = len(estimators)
    _, axes = plt.subplots(1, n, figsize=(7 * n, max(5, top_n * 0.4)))
    if n == 1:
        axes = [axes]

    for ax, name in zip(axes, estimators):
        estimator = estimators[name]
        X = Xs[name]
        color = COLORS.get(name, "#4878d0")

        logger.info("Computing PFI for %s (%d repeats)...", name.upper(), n_repeats)
        result = permutation_importance(
            estimator, X, y_val,
            scoring="roc_auc",
            n_repeats=n_repeats,
            random_state=random_state,
            n_jobs=1,
        )

        imp_mean = pd.Series(result.importances_mean, index=X.columns)
        imp_std  = pd.Series(result.importances_std,  index=X.columns)

        # Keep top_n by mean drop in AUC
        top = imp_mean.nlargest(top_n).sort_values(ascending=True)
        top_std = imp_std[top.index]

        ax.barh(
            top.index, top.values,
            xerr=top_std.values,
            color=color, edgecolor="white", capsize=3,
        )
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_title(f"{name.upper()} — Permutation Importance", fontsize=11)
        ax.set_xlabel("Mean drop in ROC-AUC (±std)")

        for i, (val, std) in enumerate(zip(top.values, top_std.values)):
            ax.text(val + std + 0.001, i, f"{val:.4f}", va="center", fontsize=7.5)

    plt.suptitle(
        f"Permutation Feature Importance (n_repeats={n_repeats}, scoring=ROC-AUC)",
        fontsize=12,
    )
    plt.tight_layout()

    out = save_path or FIGURES_DIR / "permutation_importance.png"
    plt.savefig(out, dpi=130)
    plt.close()
    logger.info("Permutation importance → %s", out)


def plot_threshold_selection(
    sweep: pd.DataFrame,
    threshold_info: dict,
    model_name: str,
    secondary_metric: str = "accuracy",
    save_path: Path | None = None,
) -> None:
    """Plot the secondary metric vs Youden Index across thresholds.

    Args:
        sweep:            DataFrame from threshold_sweep() (must contain `secondary_metric`
                          and `youden_index` columns)
        threshold_info:   dict from find_best_threshold() with keys:
                          threshold, youden_index, best_youden, <secondary_metric>
        model_name:       model name for the title
        secondary_metric: which metric to show on the left axis (default: "accuracy").
                          Must be a column in `sweep`; valid values are anything
                          produced by threshold_sweep() — "accuracy", "precision",
                          "recall", "f1", "specificity".
        save_path:        where to save the plot (default: FIGURES_DIR)
    """
    if secondary_metric not in sweep.columns:
        raise ValueError(
            f"secondary_metric={secondary_metric!r} not found in sweep columns "
            f"(available: {list(sweep.columns)})"
        )

    thresholds  = sweep["threshold"]
    selected_t  = threshold_info["threshold"]
    best_youden = threshold_info["best_youden"]
    tolerance   = best_youden - threshold_info["youden_index"]
    metric_label = secondary_metric.replace("_", " ").title()

    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Secondary metric on left axis — fixed [0, 1] range for comparability across models
    color_left = "#4878d0"
    ax1.plot(thresholds, sweep[secondary_metric], color=color_left, lw=2, label=metric_label)
    ax1.set_xlabel("Threshold")
    ax1.set_ylabel(metric_label, color=color_left)
    ax1.tick_params(axis="y", labelcolor=color_left)
    ax1.set_xlim(thresholds.min(), thresholds.max())
    ax1.set_ylim(0, 1)  # fixed range so all models look comparable

    # Youden Index on right axis
    ax2 = ax1.twinx()
    color_right = "#ee854a"
    ax2.plot(thresholds, sweep["youden_index"], color=color_right, lw=2, linestyle="--", label="Youden Index")
    ax2.set_ylabel("Youden Index", color=color_right)
    ax2.tick_params(axis="y", labelcolor=color_right)
    ax2.set_ylim(0, 1)  # Youden range is [-1, 1] but for well-separated classes [0, 1] suffices

    # Youden tolerance band (shaded)
    youden_min = best_youden - max(tolerance, 0.02)
    ax2.axhspan(youden_min, best_youden, alpha=0.12, color=color_right, label="Youden tolerance band")
    ax2.axhline(best_youden, color=color_right, lw=0.8, linestyle=":", alpha=0.6)

    # Selected threshold (vertical line)
    ax1.axvline(
        selected_t, color="#2ca02c", lw=2.5, linestyle="-",
        alpha=0.8, label=f"Selected t={selected_t:.2f}",
    )

    # Mark selected point on both curves
    sel_left  = sweep.loc[sweep["threshold"] == selected_t, secondary_metric].values
    sel_right = sweep.loc[sweep["threshold"] == selected_t, "youden_index"].values
    if len(sel_left) > 0:
        ax1.scatter([selected_t], sel_left,  color="#2ca02c", s=100, zorder=5, edgecolors="black")
        ax2.scatter([selected_t], sel_right, color="#2ca02c", s=100, zorder=5, edgecolors="black", marker="D")

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower center", fontsize=9)

    ax1.set_title(
        f"{model_name.upper()} — Threshold Selection  (metric: {secondary_metric})\n"
        f"Selected t={selected_t:.2f}  |  {metric_label}={sel_left[0]:.4f}  |  "
        f"Youden={sel_right[0]:.4f}"
        if len(sel_left) > 0 else f"{model_name.upper()} — Threshold Selection",
        fontsize=11,
    )
    ax1.grid(axis="x", alpha=0.3)
    fig.tight_layout()

    out = save_path or FIGURES_DIR / f"{model_name}_threshold_selection.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    logger.info("Threshold selection plot → %s", out)


def plot_reliability_diagram(
    y_true,
    y_proba: np.ndarray,
    model_name: str,
    split: str = "val",
    save_path: Path | None = None,
    n_bins: int = 10,
) -> None:
    """Plot a reliability diagram (calibration curve) with Brier + ECE in the title.

    A well-calibrated model produces points close to the diagonal: when the model
    predicts 0.7, roughly 70% of those samples should actually be positive.

    Args:
        y_true:     binary ground-truth labels (0/1)
        y_proba:    predicted probabilities for the positive class
        model_name: model label for the title
        split:      which split these probabilities came from ("train" / "val" / "test");
                    shown in the title so the diagnostic is unambiguous
        save_path:  output path; default is `reports/figures/<model>_reliability.png`
        n_bins:     number of bins for the calibration curve
    """
    frac_pos, mean_pred = calibration_curve(y_true, y_proba, n_bins=n_bins, strategy="uniform")
    brier = brier_score_loss(y_true, y_proba)
    ece = expected_calibration_error(y_true, y_proba, n_bins=n_bins)
    color = COLORS.get(model_name, "#4878d0")

    fig, (ax_cal, ax_hist) = plt.subplots(
        2, 1, figsize=(7, 7), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    # Top: reliability diagram
    ax_cal.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    ax_cal.plot(mean_pred, frac_pos, "o-", color=color, lw=2, markersize=8,
                label=f"{model_name.upper()}")
    ax_cal.set_ylabel("Fraction of positives (actual)")
    ax_cal.set_xlim(0, 1)
    ax_cal.set_ylim(0, 1)
    ax_cal.legend(loc="upper left")
    ax_cal.grid(alpha=0.3)
    ax_cal.set_title(
        f"{model_name.upper()} — Reliability Diagram  (on {split} set, n={len(y_true)})\n"
        f"Brier = {brier:.4f}  |  ECE = {ece:.4f}  |  {n_bins} bins",
        fontsize=11,
    )

    # Bottom: histogram of predicted probabilities (distribution)
    ax_hist.hist(y_proba, bins=n_bins, range=(0, 1), color=color, edgecolor="white", alpha=0.85)
    ax_hist.set_xlabel("Mean predicted probability")
    ax_hist.set_ylabel("Count")
    ax_hist.grid(alpha=0.3)

    plt.tight_layout()
    out = save_path or FIGURES_DIR / f"{model_name}_reliability_diagram.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    logger.info("Reliability diagram → %s", out)
