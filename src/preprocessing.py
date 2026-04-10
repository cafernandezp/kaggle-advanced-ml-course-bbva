"""
Feature engineering and train/validation split for the banking marketing dataset.

Importable module used by train.py.
Run standalone to verify split sizes:  python -m src.preprocessing
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent.parent
TRAIN_PATH = ROOT / "data/raw/train_set.csv"
TEST_PATH = ROOT / "data/raw/test_set.csv"

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


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply feature engineering. Input must not contain the target column."""
    df = df.copy()

    # WARNING — 'duration' is leaky: it equals the call duration in seconds, which is
    # only known after the call ends (i.e., after the outcome is determined). Kept because
    # the competition test set includes it, but a real production model must exclude it.

    # pdays=999 is a sentinel meaning 'client was never previously contacted'.
    # Split into a binary flag + replace 999 with NaN for the numeric value.
    df["was_contacted"] = (df["pdays"] != 999).astype(np.int8)
    df["pdays"] = df["pdays"].replace(999, np.nan)

    # education: ordinal encoding (more education = higher integer); unknown → NaN
    df["education"] = df["education"].map(EDUCATION_ORDER)

    # Encode remaining categoricals as pandas Categorical — LightGBM handles these natively.
    for col in CAT_COLS:
        df[col] = df[col].astype("category")

    return df


def load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Return stratified train/validation splits ready for model training."""
    raw = pd.read_csv(TRAIN_PATH, index_col="Id")
    X = build_features(raw.drop(columns=[TARGET]))
    y = raw[TARGET]
    return train_test_split(
        X, y, test_size=VAL_SIZE, stratify=y, random_state=RANDOM_STATE
    )


def load_test() -> pd.DataFrame:
    """Return the processed test set (no target column)."""
    raw = pd.read_csv(TEST_PATH, index_col="Id")
    return build_features(raw)


def build_features_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Same as build_features() but one-hot encodes categoricals and imputes NaN.

    Returns a fully float32 DataFrame with no category dtypes and no missing values.
    NaN sources (education=unknown → NaN, pdays=999 → NaN) are filled with column medians.
    """
    df = build_features(df)
    df = pd.get_dummies(df, columns=CAT_COLS, drop_first=True)
    df = df.astype("float32")
    df = df.fillna(df.median())
    return df


def load_splits_numeric() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train/val split with numeric (one-hot) features for MLP."""
    raw = pd.read_csv(TRAIN_PATH, index_col="Id")
    X = build_features_numeric(raw.drop(columns=[TARGET]))
    y = raw[TARGET]
    return train_test_split(
        X, y, test_size=VAL_SIZE, stratify=y, random_state=RANDOM_STATE
    )


def load_test_numeric() -> pd.DataFrame:
    """Processed test set with numeric (one-hot) features for MLP."""
    raw = pd.read_csv(TEST_PATH, index_col="Id")
    return build_features_numeric(raw)


# ── Scaled numeric variants (for MLP, SVM, GP, and other scale-sensitive models) ──

_scaler: StandardScaler | None = None  # pylint: disable=invalid-name


def load_splits_scaled() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train/val split with scaled numeric features.

    Fits StandardScaler on train, transforms both train and val.
    The fitted scaler is cached so load_test_scaled() uses the same transform.
    """
    global _scaler  # pylint: disable=global-statement
    x_tr, x_vl, y_tr, y_vl = load_splits_numeric()
    _scaler = StandardScaler()
    x_tr_sc = pd.DataFrame(
        _scaler.fit_transform(x_tr), index=x_tr.index, columns=x_tr.columns,
    ).astype("float32")
    x_vl_sc = pd.DataFrame(
        _scaler.transform(x_vl), index=x_vl.index, columns=x_vl.columns,
    ).astype("float32")
    return x_tr_sc, x_vl_sc, y_tr, y_vl


def load_test_scaled() -> pd.DataFrame:
    """Processed test set with scaled numeric features.

    Uses the scaler fitted by load_splits_scaled(). Must call load_splits_scaled() first.
    """
    if _scaler is None:
        raise RuntimeError("Call load_splits_scaled() before load_test_scaled().")
    X_test = load_test_numeric()
    return pd.DataFrame(
        _scaler.transform(X_test), index=X_test.index, columns=X_test.columns,
    ).astype("float32")


def expand_features_for_mlp(selected_features: list[str], X_num: pd.DataFrame) -> list[str]:
    """Map tree feature names to their corresponding columns in the one-hot numeric DataFrame.

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
    X_train, X_val, y_train, y_val = load_splits()
    print(f"Train : {X_train.shape}  |  positive rate: {y_train.mean():.2%}")
    print(f"Val   : {X_val.shape}  |  positive rate: {y_val.mean():.2%}")
    print(f"Features ({len(X_train.columns)}): {X_train.columns.tolist()}")
