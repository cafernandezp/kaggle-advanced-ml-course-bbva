"""
Plotting utilities: ROC-AUC comparison and loss curves.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, roc_curve

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
    fig, ax = plt.subplots(figsize=(7, 6))

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
    print(f"[plots] ROC curves → {out}")


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
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), sharey=False)
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
    print(f"[plots] Loss curves  → {out}")
