"""
Plotting utilities: ROC-AUC comparison, loss curves, feature importance, PFI.
"""
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import auc, roc_curve

logger = logging.getLogger(__name__)

FIGURES_DIR = Path(__file__).parent.parent / "reports/figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {"lgbm": "#4878d0", "xgb": "#ee854a", "mlp": "#6acc65"}


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


def plot_loss_curves(
    histories: dict[str, dict],
    save_path: Path | None = None,
) -> None:
    """Plot train vs validation loss curves for each model.

    Args:
        histories: {model_name: {"train": [...], "val": [...]}}
                   MLP uses keys "train_loss" / "val_loss"
    """
    n = len(histories)
    _, axes = plt.subplots(1, n, figsize=(6 * n, 5), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, (name, hist) in zip(axes, histories.items()):
        color = COLORS.get(name, "#4878d0")
        # Normalise key names (MLP uses train_loss/val_loss, others use train/val)
        train_key = "train_loss" if "train_loss" in hist else "train"
        val_key   = "val_loss"   if "val_loss"   in hist else "val"

        train_vals = hist[train_key]
        val_vals   = hist[val_key]
        x = range(1, len(train_vals) + 1)

        ax.plot(x, train_vals, label="train", color=color, lw=2)
        ax.plot(x, val_vals,   label="val",   color=color, lw=2, linestyle="--")
        ax.set_title(f"{name.upper()} — loss curve")
        ax.set_xlabel("Iteration / Epoch")
        ax.set_ylabel("Log Loss")
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
