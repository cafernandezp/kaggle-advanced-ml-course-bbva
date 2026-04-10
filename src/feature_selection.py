"""
Feature selection pipeline — sequential filtering stages.

Stages (applied in order):
    1. drop_high_missing()                  → remove features with > X% NaN
    2. drop_correlated_features()           → remove redundant features (Spearman)
    3. select_top_mutual_information()       → rank by MI with target, keep top K
    4. select_top_features_lgbm_pfi_based() → PFI on LightGBM baseline, final cut

Each function is independent and reusable from notebooks.
run_feature_selection_pipeline() chains them all with configurable thresholds.
"""
import logging

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.inspection import permutation_importance

logger = logging.getLogger(__name__)

RANDOM_STATE = 42


def _prepare_for_mi(X: pd.DataFrame) -> pd.DataFrame:
    """Convert a mixed-dtype DataFrame to numeric for MI / correlation computation.

    - Category columns → integer codes (NaN → -1)
    - Numeric columns → fillna(median)
    """
    out = pd.DataFrame(index=X.index)
    for col in X.columns:
        if hasattr(X[col], "cat"):
            out[col] = X[col].cat.codes.astype(float)  # NaN → -1
        else:
            out[col] = X[col].fillna(X[col].median())
    return out


# ── Stage 1: Missing data filter ─────────────────────────────────────────────

def drop_high_missing(
    X: pd.DataFrame,
    threshold: float = 0.5,
) -> list[str]:
    """Return feature names with missing rate <= threshold.

    Args:
        X: feature DataFrame (train set — use train stats to decide)
        threshold: max allowed fraction of NaN (e.g. 0.5 = 50%)

    Returns:
        list of column names that pass the filter
    """
    missing_rate = X.isnull().mean()
    dropped = missing_rate[missing_rate > threshold].index.tolist()
    kept = [c for c in X.columns if c not in dropped]
    if dropped:
        logger.info(
            "[feature_selection] Dropped %d features with >%.0f%% missing: %s",
            len(dropped), threshold * 100, dropped,
        )
    else:
        logger.info(
            "[feature_selection] No features dropped by missing filter (threshold=%.0f%%)",
            threshold * 100,
        )
    logger.info("[feature_selection] %d features remaining after missing filter", len(kept))
    return kept


# ── Stage 2: Correlated feature removal (Spearman) ──────────────────────────

def drop_correlated_features(
    X: pd.DataFrame,
    y: pd.Series,
    threshold: float = 0.9,
    method: str = "spearman",
) -> list[str]:
    """Remove redundant features whose pairwise correlation exceeds threshold.

    When two features are highly correlated, the one with LOWER mutual
    information with the target is dropped. This ensures we keep the more
    informative feature from each correlated pair.

    Args:
        X: feature DataFrame (numeric columns only)
        y: target series (used to compute MI for tie-breaking)
        threshold: max allowed absolute correlation between features
        method: "spearman" (default, captures monotonic non-linear) or "pearson"

    Returns:
        list of column names that pass the filter
    """
    corr_matrix = X.corr(method=method).abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape, dtype=bool), k=1))

    # Compute MI with target for tie-breaking
    mi_scores = pd.Series(
        mutual_info_classif(_prepare_for_mi(X), y, random_state=RANDOM_STATE),
        index=X.columns,
    )

    to_drop = set()
    for col in upper.columns:
        correlated_with = upper.index[upper[col] > threshold].tolist()
        for corr_col in correlated_with:
            if corr_col in to_drop or col in to_drop:
                continue
            # Drop the one with lower MI
            if mi_scores[col] >= mi_scores[corr_col]:
                to_drop.add(corr_col)
            else:
                to_drop.add(col)

    kept = [c for c in X.columns if c not in to_drop]
    if to_drop:
        logger.info(
            "[feature_selection] Dropped %d correlated features (%s, threshold=%.2f): %s",
            len(to_drop), method, threshold, sorted(to_drop),
        )
    else:
        logger.info(
            "[feature_selection] No features dropped by correlation filter (%s, threshold=%.2f)",
            method, threshold,
        )
    logger.info("[feature_selection] %d features remaining after correlation filter", len(kept))
    return kept


# ── Stage 3: Mutual Information with target ──────────────────────────────────

def select_top_mutual_information(
    X: pd.DataFrame,
    y: pd.Series,
    top_k: int = 20,
) -> list[str]:
    """Rank features by Mutual Information with the target and keep the top K.

    MI is non-parametric and captures non-linear dependencies — better than
    Pearson/Spearman for a binary target with mixed feature types.

    Args:
        X: feature DataFrame (numeric, NaN-free or will be median-imputed)
        y: binary target
        top_k: number of features to keep

    Returns:
        list of top-K feature names ordered by MI (highest first)
    """
    X_clean = _prepare_for_mi(X)
    mi = pd.Series(
        mutual_info_classif(X_clean, y, random_state=RANDOM_STATE),
        index=X.columns,
    ).sort_values(ascending=False)

    top_k = min(top_k, len(mi))
    kept = mi.nlargest(top_k).index.tolist()

    logger.info("[feature_selection] MI with target (top %d):", top_k)
    for feat in kept[:10]:
        logger.info("  %-30s MI=%.4f", feat, mi[feat])
    if top_k > 10:
        logger.info("  ... (%d more)", top_k - 10)
    logger.info("[feature_selection] %d features remaining after MI filter", len(kept))
    return kept


# ── Stage 4: LightGBM PFI ────────────────────────────────────────────────────

def select_top_features_lgbm_pfi_based(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    top_n: int = 15,
) -> list[str]:
    """Train a quick LightGBM baseline and return the top-N features by PFI.

    PFI is computed on the validation set (n_repeats=5, scoring=roc_auc).
    """
    logger.info("[feature_selection] Running LightGBM PFI (top %d)...", top_n)
    baseline = lgb.LGBMClassifier(
        n_estimators=200, random_state=RANDOM_STATE, verbose=-1,
    )
    baseline.fit(X_train, y_train)
    pfi = permutation_importance(
        baseline, X_val, y_val,
        scoring="roc_auc", n_repeats=5, random_state=RANDOM_STATE, n_jobs=1,
    )
    pfi_scores = pd.Series(pfi.importances_mean, index=X_train.columns).sort_values(ascending=False)
    top_n = min(top_n, len(pfi_scores))
    kept = pfi_scores.nlargest(top_n).index.tolist()

    logger.info("[feature_selection] PFI scores (top %d):", top_n)
    for feat in kept[:10]:
        logger.info("  %-30s PFI=%.4f", feat, pfi_scores[feat])
    if top_n > 10:
        logger.info("  ... (%d more)", top_n - 10)
    logger.info("[feature_selection] %d features selected (final)", len(kept))
    return kept


# ── Full pipeline ─────────────────────────────────────────────────────────────

def build_feature_selection_report(
    X_original: pd.DataFrame,
    final_features: list[str],
) -> pd.DataFrame:
    """Build a summary report comparing original features vs final selection.

    Args:
        X_original: the full feature DataFrame before any selection
        final_features: list of feature names that survived all stages

    Returns:
        DataFrame with columns: feature, missing_pct, selected
    """
    final_set = set(final_features)
    rows = []
    for col in X_original.columns:
        rows.append({
            "feature":     col,
            "missing_pct": round(X_original[col].isnull().mean() * 100, 2),
            "selected":    col in final_set,
        })
    return pd.DataFrame(rows).sort_values("selected", ascending=False).reset_index(drop=True)


def _run_stage2(X_train, X_val, y_train, report, threshold, method):
    """Stage 2: drop correlated features and update the report dict."""
    numeric_cols = X_train.select_dtypes(include="number").columns.tolist()
    if not numeric_cols:
        return X_train, X_val

    corr_matrix = X_train[numeric_cols].corr(method=method).abs()
    kept_numeric = set(drop_correlated_features(
        X_train[numeric_cols], y_train, threshold=threshold, method=method,
    ))
    for f in numeric_cols:
        if f not in kept_numeric and "dropped_at" not in report[f]:
            corr_with = corr_matrix[f].drop(f).idxmax()
            corr_val  = corr_matrix[f].drop(f).max()
            report[f]["dropped_at"] = "stage_2_correlation"
            report[f]["drop_reason"] = f"{method} corr={corr_val:.3f} with {corr_with}"

    non_numeric = [c for c in X_train.columns if c not in numeric_cols]
    kept = non_numeric + [c for c in numeric_cols if c in kept_numeric]
    return X_train[kept], X_val[kept]


def _run_stage3(X_train, X_val, y_train, report, mi_top_k):
    """Stage 3: MI filter and update the report dict."""
    mi_scores = pd.Series(
        mutual_info_classif(_prepare_for_mi(X_train), y_train, random_state=RANDOM_STATE),
        index=X_train.columns,
    )
    for f in X_train.columns:
        report[f]["mi_score"] = round(mi_scores[f], 4)

    kept_s3 = set(select_top_mutual_information(X_train, y_train, top_k=mi_top_k))
    for f in X_train.columns:
        if f not in kept_s3 and "dropped_at" not in report[f]:
            report[f]["dropped_at"] = "stage_3_mi"
            report[f]["drop_reason"] = f"MI={mi_scores[f]:.4f} (below top {mi_top_k})"

    kept_cols = [f for f in X_train.columns if f in kept_s3]
    return X_train[kept_cols], X_val[kept_cols]


def _run_stage4(X_train, y_train, X_val, y_val, report, pfi_top_n):
    """Stage 4: PFI filter and update the report dict. Returns final feature list."""
    final_list = select_top_features_lgbm_pfi_based(
        X_train, y_train, X_val, y_val, top_n=pfi_top_n,
    )
    final_set = set(final_list)

    baseline = lgb.LGBMClassifier(n_estimators=200, random_state=RANDOM_STATE, verbose=-1)
    baseline.fit(X_train, y_train)
    pfi = permutation_importance(
        baseline, X_val, y_val,
        scoring="roc_auc", n_repeats=5, random_state=RANDOM_STATE, n_jobs=1,
    )
    pfi_scores = pd.Series(pfi.importances_mean, index=X_train.columns)

    for f in X_train.columns:
        report[f]["pfi_score"] = round(pfi_scores[f], 4)
        if f not in final_set and "dropped_at" not in report[f]:
            report[f]["dropped_at"] = "stage_4_pfi"
            report[f]["drop_reason"] = f"PFI={pfi_scores[f]:.4f} (below top {pfi_top_n})"
    return final_list


def _build_report(report: dict, final_list: list[str]) -> pd.DataFrame:
    """Build the feature selection report DataFrame from the tracking dict."""
    for f in final_list:
        report[f]["dropped_at"] = "selected"
        report[f]["drop_reason"] = ""

    report_df = pd.DataFrame(report.values())
    return report_df.sort_values(
        by=["dropped_at", "feature"],
        key=lambda col: col.map(lambda x: 0 if x == "selected" else 1) if col.name == "dropped_at" else col,
    ).reset_index(drop=True)


def _log_report(report_df: pd.DataFrame):
    """Log the feature selection report to the logger."""
    for _, row in report_df.iterrows():
        status = "✓" if row["dropped_at"] == "selected" else "✗"
        logger.info(
            "  %s %-25s  %s  %s",
            status, row["feature"], row["dropped_at"], row.get("drop_reason", ""),
        )


def run_feature_selection_pipeline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    *,
    missing_threshold: float = 0.5,
    correlation_threshold: float = 0.9,
    correlation_method: str = "spearman",
    mi_top_k: int = 20,
    pfi_top_n: int = 15,
) -> tuple[list[str], pd.DataFrame]:
    """Run the full feature selection pipeline: missing → correlation → MI → PFI.

    Args:
        X_train, y_train: training data (from load_splits — category dtypes, NaN present)
        X_val, y_val: validation data (used by PFI stage)
        missing_threshold: max fraction of NaN allowed (stage 1)
        correlation_threshold: max absolute feature-feature correlation (stage 2)
        correlation_method: "spearman" or "pearson" (stage 2)
        mi_top_k: keep top K features by MI with target (stage 3)
        pfi_top_n: keep top N features by PFI (stage 4, final)

    Returns:
        (final_features, report_df) — selected feature names + detailed report DataFrame
    """
    all_features = X_train.columns.tolist()
    n_start = len(all_features)
    logger.info("=" * 60)
    logger.info("  FEATURE SELECTION PIPELINE")
    logger.info("  Starting with %d features", n_start)
    logger.info("=" * 60)

    # Initialise report: every feature starts as "kept"
    report = {f: {"feature": f, "missing_pct": round(X_train[f].isnull().mean() * 100, 2)}
              for f in all_features}

    # ── Stage 1: missing data ────────────────────────────────────────────────
    kept_s1 = set(drop_high_missing(X_train, threshold=missing_threshold))
    for f in all_features:
        if f not in kept_s1:
            report[f]["dropped_at"] = "stage_1_missing"
            report[f]["drop_reason"] = f"{report[f]['missing_pct']}% missing"
    X_train = X_train[[f for f in X_train.columns if f in kept_s1]]
    X_val   = X_val[[f for f in X_val.columns if f in kept_s1]]

    # ── Stage 2: correlated features ─────────────────────────────────────────
    X_train, X_val = _run_stage2(
        X_train, X_val, y_train, report, correlation_threshold, correlation_method,
    )

    # ── Stage 3: MI with target ──────────────────────────────────────────────
    X_train, X_val = _run_stage3(X_train, X_val, y_train, report, mi_top_k)

    # ── Stage 4: PFI ─────────────────────────────────────────────────────────
    final_list = _run_stage4(X_train, y_train, X_val, y_val, report, pfi_top_n)

    report_df = _build_report(report, final_list)

    logger.info("=" * 60)
    logger.info("  FEATURE SELECTION COMPLETE: %d → %d features", n_start, len(final_list))
    logger.info("  Final features: %s", final_list)
    logger.info("=" * 60)
    _log_report(report_df)

    return final_list, report_df
