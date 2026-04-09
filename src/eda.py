"""
Exploratory Data Analysis — banking marketing campaign dataset.
Run from project root:  python -m src.eda
Plots are saved to reports/figures/.
"""
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).parent.parent
DATA_PATH = ROOT / "data/raw/train_set.csv"
FIGURES_DIR = ROOT / "reports/figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH, index_col="Id")
print(f"Shape: {df.shape}")
print(f"\nDtypes:\n{df.dtypes.to_string()}")

# ── Missing values ────────────────────────────────────────────────────────────
missing = df.isnull().sum()
print(f"\nMissing values:\n{missing[missing > 0] if missing.any() else 'none'}")

# ── Target distribution ───────────────────────────────────────────────────────
counts = df["subscribed"].value_counts()
pos_rate = counts[1] / len(df)
print(f"\nTarget distribution:\n{counts.to_string()}")
print(f"Positive rate: {pos_rate:.2%}")

fig, ax = plt.subplots(figsize=(5, 4))
counts.plot(kind="bar", ax=ax, color=["#4878d0", "#ee854a"], edgecolor="white")
ax.set_title("Target distribution (subscribed)")
ax.set_xlabel("subscribed")
ax.set_ylabel("count")
ax.tick_params(axis="x", rotation=0)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "target_distribution.png", dpi=120)
plt.close()

# ── Numerical features ────────────────────────────────────────────────────────
num_cols = df.select_dtypes(include="number").columns.drop("subscribed").tolist()
print(f"\nNumerical features: {num_cols}")
print(df[num_cols].describe().T.to_string())

ncols = 3
nrows = (len(num_cols) + ncols - 1) // ncols
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, nrows * 3))
for ax, col in zip(axes.flatten(), num_cols):
    df[col].hist(ax=ax, bins=40, edgecolor="white", color="#4878d0")
    ax.set_title(col, fontsize=9)
for ax in axes.flatten()[len(num_cols):]:
    ax.set_visible(False)
plt.suptitle("Numerical feature distributions", fontsize=12)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "numerical_distributions.png", dpi=120)
plt.close()

# ── Correlation with target ───────────────────────────────────────────────────
corr = (
    df[num_cols + ["subscribed"]]
    .corr()["subscribed"]
    .drop("subscribed")
    .sort_values()
)
print(f"\nCorrelation with target:\n{corr.to_string()}")

fig, ax = plt.subplots(figsize=(8, 5))
corr.plot(kind="barh", ax=ax, color="#4878d0")
ax.set_title("Correlation of numerical features with target")
ax.axvline(0, color="black", linewidth=0.8)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "correlation_with_target.png", dpi=120)
plt.close()

# ── Categorical features: subscription rate per category ─────────────────────
cat_cols = df.select_dtypes(include="object").columns.tolist()
print(f"\nCategorical features: {cat_cols}")

for col in cat_cols:
    rate = df.groupby(col)["subscribed"].mean().sort_values(ascending=False)
    count = df[col].value_counts()
    print(f"\n{col}:\n{pd.DataFrame({'rate': rate, 'count': count}).to_string()}")

    fig, ax = plt.subplots(figsize=(max(6, len(rate) * 0.8), 4))
    rate.plot(kind="bar", ax=ax, color="#4878d0", edgecolor="white")
    ax.set_title(f"Subscription rate by {col}")
    ax.set_ylabel("subscription rate")
    ax.tick_params(axis="x", rotation=45)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"rate_by_{col}.png", dpi=120)
    plt.close()

# ── pdays note ────────────────────────────────────────────────────────────────
pdays_999 = (df["pdays"] == 999).sum()
print(f"\npdays=999 (never contacted before): {pdays_999} ({pdays_999/len(df):.1%})")

print(f"\nAll plots saved to {FIGURES_DIR}/")
