"""
Feature engineering, train/val splits, imputation, and scaling.

Single entry point: preprocess_data() returns a ProcessedData container with
all variants (tree, numeric, scaled) for train, val, and test — built once
from a single CSV read.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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
TARGET = "subscribed"


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


# ── Building blocks ──────────────────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply feature engineering. Input must not contain the target column.

    Produces the *tree variant*: category dtypes, NaN preserved.
    """
    df = df.copy()

    # WARNING — 'duration' is leaky: it equals the call duration in seconds, which is
    # only known after the call ends. Kept because the competition test set includes it.

    # pdays=999 is a sentinel meaning 'client was never previously contacted'.
    df["was_contacted"] = (df["pdays"] != 999).astype(np.int8)
    df["pdays"] = df["pdays"].replace(999, np.nan)

    # education: ordinal encoding (more education = higher integer); unknown → NaN
    df["education"] = df["education"].map(EDUCATION_ORDER)

    # Encode remaining categoricals as pandas Categorical — LightGBM handles natively.
    for col in CAT_COLS:
        df[col] = df[col].astype("category")

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

def preprocess_data() -> ProcessedData:
    """Read raw CSVs once and build all preprocessed variants.

    Steps:
        1. Read train_set.csv and test_set.csv
        2. Apply build_features() → tree variant
        3. Stratified train/val split
        4. Build numeric variant (one-hot + imputed) using TRAIN medians
        5. Build scaled variant (StandardScaler fit on TRAIN only)
        6. Apply same transformations to test set (no leakage)

    Returns:
        ProcessedData with all 9 DataFrames + targets + fitted transformers.
    """
    # 1. Read raw CSVs
    train_raw = pd.read_csv(TRAIN_PATH, index_col="Id")
    test_raw  = pd.read_csv(TEST_PATH,  index_col="Id")

    # 2. Tree variant (category dtypes, NaN preserved)
    train_tree = build_features(train_raw.drop(columns=[TARGET]))
    test_tree  = build_features(test_raw)
    y          = train_raw[TARGET]

    # 3. Stratified train/val split — same indices for all variants
    X_train, X_val, y_train, y_val = train_test_split(
        train_tree, y, test_size=VAL_SIZE, stratify=y, random_state=RANDOM_STATE,
    )

    # 4. Numeric variant — fit imputer on TRAIN only, apply to val + test
    X_train_num, median_imputer = _to_numeric(X_train, median_imputer=None)
    X_val_num,  _ = _to_numeric(X_val,  median_imputer=median_imputer)
    X_test_num, _ = _to_numeric(test_tree, median_imputer=median_imputer)

    # Align test columns with train (in case test has fewer dummy levels)
    X_val_num  = X_val_num.reindex(columns=X_train_num.columns, fill_value=0)
    X_test_num = X_test_num.reindex(columns=X_train_num.columns, fill_value=0)

    # 5. Scaled variant — fit StandardScaler on TRAIN only
    scaler = StandardScaler()
    scaler.fit(X_train_num)
    X_train_scaled = _scale(X_train_num, scaler)
    X_val_scaled   = _scale(X_val_num,   scaler)
    X_test_scaled  = _scale(X_test_num,  scaler)

    return ProcessedData(
        X_train=X_train, X_val=X_val, X_test=test_tree,
        X_train_num=X_train_num, X_val_num=X_val_num, X_test_num=X_test_num,
        X_train_scaled=X_train_scaled, X_val_scaled=X_val_scaled, X_test_scaled=X_test_scaled,
        y_train=y_train, y_val=y_val,
        scaler=scaler, median_imputer=median_imputer,
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
