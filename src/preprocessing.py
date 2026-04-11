"""
Feature engineering, train/val splits, imputation, and scaling.

Single entry point: preprocess_data() returns a ProcessedData container with
all variants (tree, numeric, scaled) for train, val, and test — built once
from a single CSV read.
"""
import json
import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
TRAIN_PATH = ROOT / "data/raw/train_set.csv"
TEST_PATH  = ROOT / "data/raw/test_set.csv"

RANDOM_STATE = 42
VAL_SIZE = 0.2

CAT_COLS = [
    "job", "marital", "default", "housing",
    "loan", "contact", "month", "day_of_week", "poutcome",
]

# education has a natural order — encoded as integers rather than unordered categorical
EDUCATION_ORDER = {
    "illiterate": 0,
    "basic.4y": 1,
    "basic.6y": 2,
    "basic.9y": 3,
    "high.school": 4,
    "professional.course": 5,
    "university.degree": 6,
    "unknown": np.nan,
}
NUM_COLS = [
    "age", "duration", "campaign", "pdays", "previous",
    "emp.var.rate", "cons.price.idx", "cons.conf.idx",
    "euribor3m", "nr.employed",
]
DEFAULT_TARGET = "subscribed"  # project-specific default; override via preprocess_data(target_col=...)


# ── Container ────────────────────────────────────────────────────────────────

@dataclass
class ProcessedData:  # pylint: disable=too-many-instance-attributes,invalid-name
    """All preprocessed data variants and fitted transformers, built in one pass."""

    # Tree variant — category dtypes, NaN preserved (LightGBM/XGBoost native)
    X_train: pd.DataFrame
    X_val:   pd.DataFrame
    X_test:  pd.DataFrame

    # Numeric variant — one-hot encoded, NaN imputed, float32 (feature selection / MI)
    X_train_num: pd.DataFrame
    X_val_num:   pd.DataFrame
    X_test_num:  pd.DataFrame

    # Scaled variant — numeric + StandardScaler (MLP, SVM, GP)
    X_train_scaled: pd.DataFrame
    X_val_scaled:   pd.DataFrame
    X_test_scaled:  pd.DataFrame

    # Targets
    y_train: pd.Series
    y_val:   pd.Series

    # Fitted transformers (kept for reproducibility / debugging)
    scaler: StandardScaler
    median_imputer: pd.Series  # the train medians used for NaN imputation

    # Per-column preprocessing metadata — describes *what* was done to *which*
    # variable and *why*. Populated by preprocess_data(), saved as JSON by
    # save_preprocessing_artifacts(). Empty dict after load_processed_data()
    # since the report is a run-time artifact, not part of the fitted state.
    report: dict = field(default_factory=dict)


# ── Building blocks ──────────────────────────────────────────────────────────

def build_features(
    df: pd.DataFrame,
    report: dict | None = None,
    split_label: str = "input",
) -> pd.DataFrame:
    """Apply feature engineering. Input must not contain the target column.

    Produces the *tree variant*: category dtypes, NaN preserved.

    If `report` is provided, populates it with per-column metadata describing
    the transformation applied, the rationale, and before/after stats. The
    `split_label` is used only in log messages (e.g. "train", "test").
    """
    df = df.copy()
    logger.info("  [build_features | %s] input shape=%s", split_label, df.shape)

    # WARNING — 'duration' is leaky: it equals the call duration in seconds, which is
    # only known after the call ends. Kept because the competition test set includes it.
    if report is not None and "duration" in df.columns:
        report.setdefault("duration", {}).update({
            "transformation": "passthrough",
            "rationale": (
                "LEAKY feature — call duration is only known after the call ends. "
                "Kept because the competition test set includes it."
            ),
            "dtype_before": str(df["duration"].dtype),
            "n_missing": int(df["duration"].isna().sum()),
            "min": float(df["duration"].min()),
            "max": float(df["duration"].max()),
        })

    # pdays=999 is a sentinel meaning 'client was never previously contacted'.
    n_sentinel = int((df["pdays"] == 999).sum())
    df["was_contacted"] = (df["pdays"] != 999).astype(np.int8)
    df["pdays"] = df["pdays"].replace(999, np.nan)
    logger.info(
        "  [build_features | %s] pdays: replaced %d sentinel values (999) with NaN, "
        "added `was_contacted` binary flag (positive rate=%.2f%%)",
        split_label, n_sentinel, (df["was_contacted"] == 1).mean() * 100,
    )
    if report is not None:
        report.setdefault("pdays", {}).update({
            "transformation": "sentinel_to_nan",
            "rationale": "pdays=999 means 'never contacted'. Split into NaN + binary flag.",
            "dtype_after": str(df["pdays"].dtype),
            "n_sentinel_replaced": n_sentinel,
            "n_missing_after": int(df["pdays"].isna().sum()),
        })
        report.setdefault("was_contacted", {}).update({
            "transformation": "derived_binary_flag",
            "rationale": "Derived from pdays sentinel (999 → 0, else 1).",
            "dtype_after": str(df["was_contacted"].dtype),
            "positive_rate": float((df["was_contacted"] == 1).mean()),
        })

    # education: ordinal encoding (more education = higher integer); unknown → NaN
    edu_value_counts_before = df["education"].value_counts(dropna=False).to_dict()
    df["education"] = df["education"].map(EDUCATION_ORDER)
    n_unknown = int(df["education"].isna().sum())
    logger.info(
        "  [build_features | %s] education: ordinal encoded (0..6); %d 'unknown' → NaN",
        split_label, n_unknown,
    )
    if report is not None:
        report.setdefault("education", {}).update({
            "transformation": "ordinal_encode",
            "rationale": (
                "education has a natural order; encoded as int 0..6. "
                "'unknown' → NaN (imputed later for numeric/scaled variants)."
            ),
            "ordering": {k: (None if pd.isna(v) else int(v))
                         for k, v in EDUCATION_ORDER.items()},
            "value_counts_before": {str(k): int(v) for k, v in edu_value_counts_before.items()},
            "n_unknown_to_nan": n_unknown,
            "dtype_after": str(df["education"].dtype),
        })

    # Encode remaining categoricals as pandas Categorical — LightGBM handles natively.
    for col in CAT_COLS:
        n_unique = int(df[col].nunique(dropna=True))
        n_missing = int(df[col].isna().sum())
        df[col] = df[col].astype("category")
        if report is not None:
            report.setdefault(col, {}).update({
                "transformation": "category_cast",
                "rationale": (
                    "Unordered categorical; kept as pandas Categorical so LightGBM/XGBoost "
                    "can use it natively, and one-hot encoded for numeric/scaled variants."
                ),
                "n_unique": n_unique,
                "n_missing": n_missing,
                "categories": [str(c) for c in df[col].cat.categories.tolist()],
                "dtype_after": "category",
            })
    logger.info(
        "  [build_features | %s] cast %d columns to category dtype: %s",
        split_label, len(CAT_COLS), CAT_COLS,
    )

    # Record pure numeric columns as passthrough (skip ones already described above)
    _already_described = {"duration", "pdays"}
    if report is not None:
        for col in NUM_COLS:
            if col in _already_described or col not in df.columns:
                continue
            report.setdefault(col, {}).update({
                "transformation": "passthrough",
                "rationale": "Numeric feature used as-is in the tree variant.",
                "dtype_before": str(df[col].dtype),
                "n_missing": int(df[col].isna().sum()),
                "min": float(df[col].min()),
                "max": float(df[col].max()),
            })

    return df


def _to_numeric(
    df_tree: pd.DataFrame,
    median_imputer: pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Convert tree variant → numeric variant: one-hot encode + median impute.

    Args:
        df_tree: DataFrame in tree variant (category dtypes, NaN present)
        median_imputer: pre-computed medians from train (use None to fit on this df)

    Returns:
        (numeric DataFrame, fitted median series)
    """
    df_num = pd.get_dummies(df_tree, columns=CAT_COLS, drop_first=True).astype("float32")
    if median_imputer is None:
        median_imputer = df_num.median()
    df_num = df_num.fillna(median_imputer)
    return df_num, median_imputer


def _scale(
    df_num: pd.DataFrame,
    scaler: StandardScaler,
) -> pd.DataFrame:
    """Apply a fitted StandardScaler to a numeric DataFrame, preserving index/columns."""
    return pd.DataFrame(
        scaler.transform(df_num),
        index=df_num.index,
        columns=df_num.columns,
    ).astype("float32")


# ── Single entry point ───────────────────────────────────────────────────────

def preprocess_data(  # pylint: disable=too-many-locals,too-many-statements
    target_col: str = DEFAULT_TARGET,
    id_col: str = "Id",
) -> ProcessedData:
    """Read raw CSVs once and build all preprocessed variants.

    Args:
        target_col: name of the target column in the train set (default: project default).
        id_col: name of the ID column used as the DataFrame index.

    Steps:
        1. Read train_set.csv and test_set.csv (indexed by `id_col`)
        2. Apply build_features() → tree variant
        3. Stratified train/val split
        4. Build numeric variant (one-hot + imputed) using TRAIN medians
        5. Build scaled variant (StandardScaler fit on TRAIN only)
        6. Apply same transformations to test set (no leakage)

    Returns:
        ProcessedData with all 9 DataFrames + targets + fitted transformers.
    """
    report: dict = {
        "target_col": target_col,
        "id_col": id_col,
        "random_state": RANDOM_STATE,
        "val_size": VAL_SIZE,
        "columns": {},
        "stages": {},
    }
    col_report: dict = report["columns"]

    # 1. Read raw CSVs ────────────────────────────────────────────────────────
    logger.info("=" * 70)
    logger.info("PREPROCESSING — stage 1: read raw CSVs")
    logger.info("=" * 70)
    train_raw = pd.read_csv(TRAIN_PATH, index_col=id_col)
    test_raw  = pd.read_csv(TEST_PATH,  index_col=id_col)
    logger.info("Train CSV → %s  shape=%s", TRAIN_PATH.name, train_raw.shape)
    logger.info("Test  CSV → %s  shape=%s", TEST_PATH.name,  test_raw.shape)
    logger.info("Target    : %s  (positive rate=%.2f%%)",
                target_col, train_raw[target_col].mean() * 100)
    logger.info("Columns   : %s", list(train_raw.columns))
    report["stages"]["1_read"] = {
        "train_shape": list(train_raw.shape),
        "test_shape": list(test_raw.shape),
        "train_positive_rate": float(train_raw[target_col].mean()),
        "columns": list(train_raw.columns),
    }

    # 2. Tree variant (category dtypes, NaN preserved) ───────────────────────
    logger.info("")
    logger.info("=" * 70)
    logger.info("PREPROCESSING — stage 2: build tree variant (feature engineering)")
    logger.info("=" * 70)
    train_tree = build_features(
        train_raw.drop(columns=[target_col]), report=col_report, split_label="train",
    )
    # Test split: pass report=None to avoid overwriting train-derived stats
    test_tree  = build_features(test_raw, report=None, split_label="test")
    y          = train_raw[target_col]
    logger.info("Tree variant columns (%d): %s", train_tree.shape[1], list(train_tree.columns))
    report["stages"]["2_tree"] = {
        "n_features": int(train_tree.shape[1]),
        "features": list(train_tree.columns),
    }

    # 3. Stratified train/val split ──────────────────────────────────────────
    logger.info("")
    logger.info("=" * 70)
    logger.info("PREPROCESSING — stage 3: stratified train/val split (val_size=%.0f%%)",
                VAL_SIZE * 100)
    logger.info("=" * 70)
    X_train, X_val, y_train, y_val = train_test_split(
        train_tree, y, test_size=VAL_SIZE, stratify=y, random_state=RANDOM_STATE,
    )
    logger.info("Train: %s  positive rate=%.2f%%", X_train.shape, y_train.mean() * 100)
    logger.info("Val  : %s  positive rate=%.2f%%", X_val.shape,   y_val.mean() * 100)
    report["stages"]["3_split"] = {
        "train_shape": list(X_train.shape),
        "val_shape": list(X_val.shape),
        "train_positive_rate": float(y_train.mean()),
        "val_positive_rate": float(y_val.mean()),
    }

    # 4. Numeric variant — one-hot + median impute (fit on TRAIN only) ──────
    logger.info("")
    logger.info("=" * 70)
    logger.info("PREPROCESSING — stage 4: numeric variant (one-hot + median impute)")
    logger.info("=" * 70)
    nan_before = {c: int(X_train[c].isna().sum()) for c in X_train.columns if X_train[c].isna().any()}
    X_train_num, median_imputer = _to_numeric(X_train, median_imputer=None)
    X_val_num,  _ = _to_numeric(X_val,  median_imputer=median_imputer)
    X_test_num, _ = _to_numeric(test_tree, median_imputer=median_imputer)

    # Align test columns with train (in case test has fewer dummy levels)
    X_val_num  = X_val_num.reindex(columns=X_train_num.columns, fill_value=0)
    X_test_num = X_test_num.reindex(columns=X_train_num.columns, fill_value=0)

    # Per-category dummy expansion counts
    dummy_counts = {}
    for cat in CAT_COLS:
        expanded = [c for c in X_train_num.columns if c == cat or c.startswith(cat + "_")]
        dummy_counts[cat] = len(expanded)
        # Update per-column report: each categorical now expands to N one-hot cols
        if cat in col_report:
            col_report[cat]["one_hot_columns"] = expanded
            col_report[cat]["n_one_hot_columns"] = len(expanded)

    logger.info(
        "One-hot encoding: %d tree columns → %d numeric columns  (drop_first=True)",
        X_train.shape[1], X_train_num.shape[1],
    )
    for cat, n in dummy_counts.items():
        logger.info("  %-12s → %d dummy columns", cat, n)
    logger.info(
        "Median imputation: fitted on TRAIN, applied to train/val/test "
        "(columns with NaN in train: %d, values filled: %d)",
        len(nan_before), sum(nan_before.values()),
    )
    for col, n in sorted(nan_before.items(), key=lambda kv: -kv[1]):
        median_val = median_imputer.get(col, float("nan"))
        logger.info("  %-25s  %d NaN → %s", col, n, f"{median_val:.4f}")
        if col in col_report:
            col_report[col]["median_imputed_value"] = (
                None if pd.isna(median_val) else float(median_val)
            )
            col_report[col]["n_imputed_train"] = int(n)

    report["stages"]["4_numeric"] = {
        "n_features_tree": int(X_train.shape[1]),
        "n_features_numeric": int(X_train_num.shape[1]),
        "one_hot_expansion": dummy_counts,
        "nan_counts_train": nan_before,
        "median_imputer": {k: (None if pd.isna(v) else float(v))
                           for k, v in median_imputer.items()},
    }

    # 5. Scaled variant — StandardScaler (fit on TRAIN only) ────────────────
    logger.info("")
    logger.info("=" * 70)
    logger.info("PREPROCESSING — stage 5: scaled variant (StandardScaler, fit on TRAIN)")
    logger.info("=" * 70)
    scaler = StandardScaler()
    scaler.fit(X_train_num)
    X_train_scaled = _scale(X_train_num, scaler)
    X_val_scaled   = _scale(X_val_num,   scaler)
    X_test_scaled  = _scale(X_test_num,  scaler)
    # Sanity check: train_scaled must be mean≈0, std≈1
    sanity_mean = float(X_train_scaled.mean().abs().max())
    sanity_std  = float((X_train_scaled.std() - 1).abs().max())
    logger.info(
        "Fitted on %d columns. Sanity check on train_scaled: max|mean|=%.2e, max|std-1|=%.2e",
        X_train_num.shape[1], sanity_mean, sanity_std,
    )
    report["stages"]["5_scaled"] = {
        "n_features": int(X_train_scaled.shape[1]),
        "train_max_abs_mean": sanity_mean,
        "train_max_std_deviation_from_1": sanity_std,
        "scaler_mean": {c: float(m) for c, m in zip(X_train_num.columns, scaler.mean_)},
        "scaler_scale": {c: float(s) for c, s in zip(X_train_num.columns, scaler.scale_)},
    }

    logger.info("")
    logger.info("✓ Preprocessing complete.  tree=%s  num=%s  scaled=%s",
                X_train.shape, X_train_num.shape, X_train_scaled.shape)

    return ProcessedData(
        X_train=X_train, X_val=X_val, X_test=test_tree,
        X_train_num=X_train_num, X_val_num=X_val_num, X_test_num=X_test_num,
        X_train_scaled=X_train_scaled, X_val_scaled=X_val_scaled, X_test_scaled=X_test_scaled,
        y_train=y_train, y_val=y_val,
        scaler=scaler, median_imputer=median_imputer,
        report=report,
    )


# ── Persistence: save/load fitted preprocessing state ───────────────────────

PREPROCESSING_DIR_NAME = "preprocessing"


def save_preprocessing_artifacts(state: ProcessedData, out_dir: Path) -> None:
    """Persist each fitted preprocessing step as its own inspectable artifact.

    Writes (creates out_dir if missing):
        out_dir/imputer.pkl                — pd.Series of train medians
        out_dir/scaler.pkl                 — fitted StandardScaler
        out_dir/numeric_columns.json       — column order after one-hot encoding
        out_dir/scaled_columns.json        — column order after scaling
        out_dir/preprocessing_report.json  — per-column transformation log
                                             (what/why/before/after stats)

    Separate files (rather than one bundle) so each step is independently
    inspectable, versionable, and replaceable.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "imputer.pkl", "wb") as fh:
        pickle.dump(state.median_imputer, fh)
    with open(out_dir / "scaler.pkl", "wb") as fh:
        pickle.dump(state.scaler, fh)
    (out_dir / "numeric_columns.json").write_text(
        json.dumps(list(state.X_train_num.columns), indent=2),
    )
    (out_dir / "scaled_columns.json").write_text(
        json.dumps(list(state.X_train_scaled.columns), indent=2),
    )
    if state.report:
        (out_dir / "preprocessing_report.json").write_text(
            json.dumps(state.report, indent=2, default=str, allow_nan=False),
        )


def load_preprocessing_artifacts(in_dir: Path) -> dict:
    """Load all fitted preprocessing artifacts from disk.

    Returns a dict with keys: imputer, scaler, numeric_columns, scaled_columns.
    Raises FileNotFoundError with a clear message if artifacts are missing.
    """
    if not in_dir.exists():
        raise FileNotFoundError(
            f"Preprocessing artifacts not found at {in_dir}. "
            f"Run `make pipeline` first to generate them."
        )
    with open(in_dir / "imputer.pkl", "rb") as fh:
        imputer = pickle.load(fh)
    with open(in_dir / "scaler.pkl", "rb") as fh:
        scaler = pickle.load(fh)
    numeric_columns = json.loads((in_dir / "numeric_columns.json").read_text())
    scaled_columns  = json.loads((in_dir / "scaled_columns.json").read_text())
    return {
        "imputer":         imputer,
        "scaler":          scaler,
        "numeric_columns": numeric_columns,
        "scaled_columns":  scaled_columns,
    }


# ── Full-state persistence (for stage-by-stage pipeline runs) ───────────────

_DATAFRAME_NAMES = (
    "X_train", "X_val", "X_test",
    "X_train_num", "X_val_num", "X_test_num",
    "X_train_scaled", "X_val_scaled", "X_test_scaled",
)


def save_processed_dataframes(state: ProcessedData, out_dir: Path) -> None:
    """Persist all DataFrames from a ProcessedData container as parquet files.

    Called after preprocess_data() to enable running pipeline stages
    independently (preprocess → feature-select → train).

    Writes 9 feature DataFrames + 2 target Series (as single-column DataFrames).
    Parquet preserves category dtypes and NaN values faithfully.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in _DATAFRAME_NAMES:
        df = getattr(state, name)
        df.to_parquet(out_dir / f"{name}.parquet")
    state.y_train.to_frame("target").to_parquet(out_dir / "y_train.parquet")
    state.y_val.to_frame("target").to_parquet(out_dir / "y_val.parquet")


def load_processed_data(in_dir: Path) -> ProcessedData:
    """Load a full ProcessedData container from disk.

    Inverse of save_preprocessing_artifacts + save_processed_dataframes.
    Raises FileNotFoundError with a clear next-step hint if any file is missing.
    """
    if not in_dir.exists():
        raise FileNotFoundError(
            f"Processed data not found at {in_dir}. "
            f"Run `make pipeline-preprocess` first."
        )
    artifacts = load_preprocessing_artifacts(in_dir)
    frames = {name: pd.read_parquet(in_dir / f"{name}.parquet") for name in _DATAFRAME_NAMES}
    y_train = pd.read_parquet(in_dir / "y_train.parquet")["target"]
    y_val   = pd.read_parquet(in_dir / "y_val.parquet")["target"]
    return ProcessedData(
        X_train=frames["X_train"],
        X_val=frames["X_val"],
        X_test=frames["X_test"],
        X_train_num=frames["X_train_num"],
        X_val_num=frames["X_val_num"],
        X_test_num=frames["X_test_num"],
        X_train_scaled=frames["X_train_scaled"],
        X_val_scaled=frames["X_val_scaled"],
        X_test_scaled=frames["X_test_scaled"],
        y_train=y_train,
        y_val=y_val,
        scaler=artifacts["scaler"],
        median_imputer=artifacts["imputer"],
    )


# ── Helper for tree → numeric column name mapping ────────────────────────────

def expand_features_for_mlp(selected_features: list[str], X_num: pd.DataFrame) -> list[str]:
    """Map tree feature names to their corresponding columns in the one-hot DataFrame.

    Categorical features (in CAT_COLS) expand to all derived dummy columns;
    numeric / ordinal features map 1:1.

    Example: "job" → ["job_blue-collar", "job_entrepreneur", ...]
             "duration" → ["duration"]
    """
    mlp_cols: list[str] = []
    for feat in selected_features:
        if feat in CAT_COLS:
            matching = [c for c in X_num.columns if c.startswith(feat + "_")]
            mlp_cols.extend(matching)
        elif feat in X_num.columns:
            mlp_cols.append(feat)
    return mlp_cols


if __name__ == "__main__":
    data = preprocess_data()
    print(f"Train tree   : {data.X_train.shape}  positive rate: {data.y_train.mean():.2%}")
    print(f"Val   tree   : {data.X_val.shape}    positive rate: {data.y_val.mean():.2%}")
    print(f"Train num    : {data.X_train_num.shape}")
    print(f"Train scaled : {data.X_train_scaled.shape}")
    print(f"Test  tree   : {data.X_test.shape}")
    print(f"Test  scaled : {data.X_test_scaled.shape}")
