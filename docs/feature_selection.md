# Feature Selection Pipeline

> Last updated: 09-04-2026

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

Each pipeline run saves `reports/runs/feature_selection_report.csv` with columns:

| Column | Description |
|---|---|
| `feature` | Original feature name |
| `stage_1_missing` | `kept` or `dropped (X% missing)` |
| `stage_2_correlation` | `kept` or `dropped (corr=0.95 with <other_feature>)` |
| `stage_3_mi` | `kept (MI=0.1234)` or `dropped (MI=0.0012, rank 25/21)` |
| `stage_4_pfi` | `kept (PFI=0.0456)` or `dropped (PFI=0.0001, rank 18/20)` |
| `final_status` | `selected` or `dropped_stage_N` |

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
