"""
Exploratory Data Analysis — reusable for any tabular dataset.

Run from project root:  python -m src.eda
Plots → reports/figures/    Tables → reports/eda/

Output files are named after the source CSV (e.g. train_set.csv → train_set_profile.csv).
"""
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
FIGURES_DIR = ROOT / "reports/figures"
EDA_DIR = ROOT / "reports/eda"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
EDA_DIR.mkdir(parents=True, exist_ok=True)


# ── Reusable profiling functions ─────────────────────────────────────────────

def build_profile(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Build a full column-level profile and save to reports/eda/<name>_profile.csv.

    Includes: dtype, n_rows, n_unique, n_missing, pct_missing, n_zeros,
    mean/std/min/q25/median/q75/max/skew (numeric),
    top_value/top_count/top_pct (categorical).
    """
    num = df.select_dtypes(include="number")
    cat = df.select_dtypes(include="object")

    profile = pd.DataFrame({"dtype": df.dtypes})
    profile["n_rows"]      = len(df)
    profile["n_unique"]    = df.nunique()
    profile["n_missing"]   = df.isnull().sum()
    profile["pct_missing"] = (df.isnull().mean() * 100).round(2)
    profile["n_zeros"]     = (df == 0).sum()

    profile["mean"]   = num.mean()
    profile["std"]    = num.std()
    profile["min"]    = num.min()
    profile["q25"]    = num.quantile(0.25)
    profile["median"] = num.median()
    profile["q75"]    = num.quantile(0.75)
    profile["max"]    = num.max()
    profile["skew"]   = num.skew()

    for col in cat.columns:
        vc = df[col].value_counts()
        profile.loc[col, "top_value"] = vc.index[0]
        profile.loc[col, "top_count"] = vc.iloc[0]
        profile.loc[col, "top_pct"]   = round(vc.iloc[0] / len(df) * 100, 2)

    out = EDA_DIR / f"{name}_profile.csv"
    profile.to_csv(out)
    logger.info("[%s] Profile (%d rows × %d cols) → %s", name, len(df), len(df.columns), out.name)
    return profile


def save_describe(df: pd.DataFrame, name: str) -> None:
    """Save describe(include='all').T to reports/eda/<name>_describe.csv."""
    out = EDA_DIR / f"{name}_describe.csv"
    df.describe(include="all").T.to_csv(out)
    logger.info("[%s] Describe → %s", name, out.name)


def save_missing(df: pd.DataFrame, name: str) -> None:
    """Log and save missing-value summary to reports/eda/<name>_missing.csv."""
    missing = pd.DataFrame({
        "n_missing":   df.isnull().sum(),
        "pct_missing": (df.isnull().mean() * 100).round(2),
    }).sort_values("n_missing", ascending=False)
    out = EDA_DIR / f"{name}_missing.csv"
    missing.to_csv(out)
    has_missing = missing[missing["n_missing"] > 0]
    if has_missing.empty:
        logger.info("[%s] No missing values", name)
    else:
        logger.info("[%s] Missing values:\n%s", name, has_missing.to_string())


def save_value_counts(df: pd.DataFrame, name: str) -> None:
    """Save value_counts for every categorical column to reports/eda/<name>_value_counts.csv."""
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    if not cat_cols:
        return
    frames = []
    for col in cat_cols:
        vc = df[col].value_counts()
        tmp = pd.DataFrame({
            "feature": col,
            "value": vc.index,
            "count": vc.values,
            "pct": (vc.values / len(df) * 100).round(2),
        })
        frames.append(tmp)
    out = EDA_DIR / f"{name}_value_counts.csv"
    pd.concat(frames, ignore_index=True).to_csv(out, index=False)
    logger.info("[%s] Value counts (%d features) → %s", name, len(cat_cols), out.name)


def save_numeric_histograms(df: pd.DataFrame, name: str) -> None:
    """Plot histograms of all numeric columns and save to reports/figures/."""
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if not num_cols:
        return
    n_plot_cols = 3
    nrows = (len(num_cols) + n_plot_cols - 1) // n_plot_cols
    _, axes = plt.subplots(nrows=nrows, ncols=n_plot_cols, figsize=(15, nrows * 3))
    for ax, col in zip(axes.flatten(), num_cols):
        df[col].hist(ax=ax, bins=40, edgecolor="white", color="#4878d0")
        ax.set_title(col, fontsize=9)
    for ax in axes.flatten()[len(num_cols):]:
        ax.set_visible(False)
    plt.suptitle(f"Numerical distributions — {name}", fontsize=12)
    plt.tight_layout()
    out = FIGURES_DIR / f"{name}_numerical_distributions.png"
    plt.savefig(out, dpi=120)
    plt.close()
    logger.info("[%s] Numeric histograms → %s", name, out.name)


def run_dataset_eda(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Run the full EDA pipeline for a single dataset. Returns the profile DataFrame."""
    logger.info("=" * 55)
    logger.info("  EDA: %s  (%d rows × %d cols)", name, *df.shape)
    logger.info("=" * 55)

    profile = build_profile(df, name)
    save_describe(df, name)
    save_missing(df, name)
    save_value_counts(df, name)
    save_numeric_histograms(df, name)

    logger.info("[%s] Profile:\n%s", name, profile.to_string())
    return profile


# ── Target-specific analysis (only for datasets with a target column) ────────

def analyse_target(df: pd.DataFrame, name: str, target_col: str) -> None:
    """Plot target distribution, correlation, and subscription rate per category."""
    if target_col not in df.columns:
        logger.info("[%s] No target column '%s' — skipping target analysis", name, target_col)
        return

    logger.info("=" * 55)
    logger.info("  TARGET ANALYSIS: %s.%s", name, target_col)
    logger.info("=" * 55)

    # Distribution
    counts = df[target_col].value_counts()
    pos_rate = counts.get(1, 0) / len(df)
    logger.info("[%s] Target distribution:\n%s", name, counts.to_string())
    logger.info("[%s] Positive rate: %.2f%%", name, pos_rate * 100)

    _, ax = plt.subplots(figsize=(5, 4))
    counts.plot(kind="bar", ax=ax, color=["#4878d0", "#ee854a"], edgecolor="white")
    ax.set_title(f"Target distribution — {name}")
    ax.set_xlabel(target_col)
    ax.set_ylabel("count")
    ax.tick_params(axis="x", rotation=0)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"{name}_target_distribution.png", dpi=120)
    plt.close()

    # Correlation with target (numeric features only)
    num_cols = df.select_dtypes(include="number").columns.drop(target_col, errors="ignore").tolist()
    if num_cols:
        corr = df[num_cols + [target_col]].corr()[target_col].drop(target_col).sort_values()
        corr.to_csv(EDA_DIR / f"{name}_correlation_with_{target_col}.csv")
        logger.info("[%s] Correlation with %s:\n%s", name, target_col, corr.to_string())

        _, ax = plt.subplots(figsize=(8, 5))
        corr.plot(kind="barh", ax=ax, color="#4878d0")
        ax.set_title(f"Correlation with {target_col} — {name}")
        ax.axvline(0, color="black", linewidth=0.8)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"{name}_correlation_with_{target_col}.png", dpi=120)
        plt.close()

    # Subscription rate per categorical feature
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    if not cat_cols:
        return
    cat_frames = []
    for col in cat_cols:
        rate  = df.groupby(col)[target_col].mean().sort_values(ascending=False)
        count = df[col].value_counts()
        summary = pd.DataFrame({
            "feature": col,
            "value": rate.index,
            "target_rate": rate.values.round(4),
            "count": count.reindex(rate.index).values,
        })
        cat_frames.append(summary)

        _, ax = plt.subplots(figsize=(max(6, len(rate) * 0.8), 4))
        rate.plot(kind="bar", ax=ax, color="#4878d0", edgecolor="white")
        ax.set_title(f"{target_col} rate by {col} — {name}")
        ax.set_ylabel(f"{target_col} rate")
        ax.tick_params(axis="x", rotation=45)
        ax.set_ylim(0, 1)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"{name}_rate_by_{col}.png", dpi=120)
        plt.close()

    out = EDA_DIR / f"{name}_categorical_target_rates.csv"
    pd.concat(cat_frames, ignore_index=True).to_csv(out, index=False)
    logger.info("[%s] Categorical target rates → %s", name, out.name)


# ══════════════════════════════════════════════════════════════════════════════
# Main — run EDA on train_set and test_set
# ══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    """Run EDA on train_set and test_set."""
    datasets = {
        "train_set": ROOT / "data/raw/train_set.csv",
        "test_set":  ROOT / "data/raw/test_set.csv",
    }
    target_col = "subscribed"

    for ds_name, ds_path in datasets.items():
        df = pd.read_csv(ds_path, index_col="Id")
        run_dataset_eda(df, ds_name)
        analyse_target(df, ds_name, target_col)

    # Domain-specific note
    for ds_name, ds_path in datasets.items():
        df = pd.read_csv(ds_path, index_col="Id")
        pdays_999 = (df["pdays"] == 999).sum()
        logger.info(
            "[%s] pdays=999 (never contacted): %d (%.1f%%)",
            ds_name, pdays_999, pdays_999 / len(df) * 100,
        )

    logger.info("Plots → %s/", FIGURES_DIR)
    logger.info("Tables → %s/", EDA_DIR)
    logger.info("Files: %s", sorted(f.name for f in EDA_DIR.glob("*.csv")))


if __name__ == "__main__":
    main()
