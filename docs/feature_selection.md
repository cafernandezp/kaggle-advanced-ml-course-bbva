# Feature Selection Pipeline

> Last updated: 10-04-2026

## Overview

The feature selection pipeline runs 4 sequential stages, each reducing the feature set.
All stages are implemented in `src/feature_selection.py` and orchestrated by
`run_feature_selection_pipeline()`.

The output is a structured report (`feature_selection_report.csv`) saved to `reports/runs/`
after each pipeline run, documenting every feature's fate at each stage.

---

## Data state at each stage

Understanding **which data** enters each stage is critical:

```
raw CSV (21 features)
    │
    ▼  build_features() in preprocessing.py
    │  - pdays=999 → was_contacted=0/1 + pdays=NaN
    │  - education → ordinal 0–6, unknown=NaN
    │  - 9 categoricals → pandas Categorical dtype
    │
    ▼  load_splits() → stratified 80/20 split
    │  Result: X_train, X_val (21 features, category dtypes, NaN present)
    │
    ▼  ════════════════════════════════════════════
    │  FEATURE SELECTION PIPELINE (stages 1–4)
    │  Input: X_train from load_splits() — raw category dtypes, NaN present
    │  ════════════════════════════════════════════
    │
    ▼  Stage 1: drop_high_missing
    │  - Data: as-is (NaN present, that's the point)
    │  - Removes features with > X% NaN
    │  - Default threshold: 50%
    │  - In practice: education (~2% NaN) and pdays (~96% NaN → was_contacted)
    │    are the NaN sources. pdays will be dropped if threshold < 96%.
    │
    ▼  Stage 2: drop_correlated_features
    │  - Data: numeric columns only (category cols are excluded)
    │  - NaN: median-imputed TEMPORARILY for Spearman correlation and MI computation
    │  - Tie-breaking: when two features are correlated > threshold,
    │    the one with LOWER mutual information with target is dropped
    │  - Default: Spearman, threshold=0.9
    │  - Note: the 5 macro-economic indicators are highly correlated
    │    (emp.var.rate, cons.price.idx, cons.conf.idx, euribor3m, nr.employed)
    │
    ▼  Stage 3: select_top_mutual_information
    │  - Data: median-imputed TEMPORARILY for MI computation
    │  - Ranks all remaining features by MI with binary target
    │  - Keeps top K (default: 20)
    │  - MI is non-parametric — captures non-linear dependencies
    │
    ▼  Stage 4: select_top_features_lgbm_pfi_based
    │  - Data: as-is (LightGBM handles NaN and categoricals natively)
    │  - Trains a quick LightGBM (200 trees) and computes PFI on val set
    │  - Keeps top N (default: 15)
    │
    ▼  ════════════════════════════════════════════
    │  OUTPUT: list of final feature names
    │  ════════════════════════════════════════════
    │
    ▼  After selection, pipeline applies features to all data variants:
       - X_train[top_features], X_val[top_features] → tree models
       - X_train_scaled[mlp_top_cols], X_val_scaled[mlp_top_cols] → MLP, SVM, GP
         (mlp_top_cols = one-hot expansion of selected categorical features)
```

---

## Report format

Each pipeline run saves `reports/runs/feature_selection_report.csv`. All column names
are lower-case for consistency.

| Column | Type | Description |
|---|---|---|
| `feature` | str | Original feature name |
| `missing_pct` | float | Percentage of missing values in the train set |
| `mi_score` | float | Mutual Information with the target (computed on the original feature set) |
| `max_corr_with_other` | float | Maximum absolute Spearman correlation with any other feature |
| `max_corr_partner` | str | Name of the feature with which `max_corr_with_other` was achieved |
| `stage1_missing` | bool | `True` if the feature passed stage 1 (missing-rate filter) |
| `stage2_correlation` | bool | `True` if it passed stage 2 (correlation redundancy filter) |
| `stage3_mi` | bool | `True` if it passed stage 3 (top-K by MI) |
| `stage4_pfi` | bool | `True` if it passed stage 4 (top-N by LightGBM PFI) — final stage |
| `dropped_at` | str | First stage where it was dropped, or `selected` if it survived all stages |
| `drop_reason` | str | Human-readable explanation, e.g. `spearman corr=0.95 with euribor3m` |
| `selected` | bool | `True` if the feature is in the final selection (= passed all 4 stages) |

### Example rows

```
feature       missing_pct  mi_score  max_corr_with_other  max_corr_partner  stage1_missing  stage2_correlation  stage3_mi  stage4_pfi  dropped_at         drop_reason                                  selected
duration      0.00         0.0823    0.142                campaign          True            True                True       True        selected                                                        True
nr.employed   0.00         0.0651    0.971                euribor3m         True            False               False      False       stage2_correlation spearman corr=0.971 with euribor3m            False
pdays         96.47        0.0042    0.118                previous          False           False               False      False       stage1_missing     96.47% missing                                False
job           0.00         0.0089    0.087                marital           True            True                False      False       stage3_mi          mi=0.0089 (below top-k)                       False
```

### How to read the report

- **`dropped_at = selected`**: feature survived all stages and is in the final model
- **`dropped_at = stage1_missing`**: too many missing values (e.g. `pdays` with 96% NaN)
- **`dropped_at = stage2_correlation`**: redundant with another feature; check `drop_reason` for the partner
- **`dropped_at = stage3_mi`**: low mutual information with the target (kept only top-K)
- **`dropped_at = stage4_pfi`**: ranked low in LightGBM permutation importance (kept only top-N)

The boolean stage columns (`stage1_*` … `stage4_*`) let you reproduce the cascade:
a feature reaches stage N only if it was `True` in all previous stages.

---

## Configuration

All thresholds are configurable via `run_feature_selection_pipeline()`:

```python
run_feature_selection_pipeline(
    X_train, y_train, X_val, y_val,
    missing_threshold=0.5,        # Stage 1: max fraction of NaN
    correlation_threshold=0.9,    # Stage 2: max abs Spearman between features
    correlation_method="spearman",# Stage 2: "spearman" or "pearson"
    mi_top_k=20,                  # Stage 3: keep top K by MI
    pfi_top_n=15,                 # Stage 4: keep top N by PFI
)
```

---

## Why this order?

1. **Missing first**: no point computing correlations or MI on features that are mostly NaN.
2. **Correlation second**: remove redundant features before ranking by informativeness —
   otherwise MI/PFI might keep both members of a correlated pair.
3. **MI third**: broad filter that captures non-linear relationships. Reduces dimensionality
   before the expensive PFI step.
4. **PFI last**: model-based validation. Uses LightGBM which handles categoricals natively,
   so this stage sees the features as the tree models will.

---

## Design decisions

- **Spearman over Pearson for feature-feature correlation**: Spearman captures monotonic
  non-linear relationships. Pearson only catches linear ones. For tabular data with mixed
  distributions, Spearman is safer.
- **MI over correlation for feature-target ranking**: MI is non-parametric and works for
  binary targets. Point-biserial correlation (equivalent to Pearson for binary) assumes
  linearity, which doesn't hold for many features (e.g., `age`, `campaign`).
- **Temporary median imputation in stages 2-3**: these stages need complete data for
  computation. The imputation is local — the original NaN-containing data is passed to
  stage 4 (LightGBM handles NaN natively).
