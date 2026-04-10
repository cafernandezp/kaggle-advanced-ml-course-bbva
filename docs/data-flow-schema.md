# Data Flow Schema

> Last updated: 10-04-2026

End-to-end data flow for a single `make pipeline` execution. Each stage is numbered;
the table below documents what runs, what comes in, what goes out, and what files
land on disk.

---

## Mermaid diagram

```mermaid
flowchart TD
    %% ── Stage 1: Raw data ──
    S1[1. Read raw CSVs<br/>data/raw/]:::stage --> S1out[(train_set.csv<br/>27595 × 22<br/>test_set.csv<br/>13593 × 21)]:::data

    %% ── Stage 2: Preprocessing ──
    S1out --> S2[2. preprocess_data<br/>src/preprocessing.py]:::stage
    S2 --> S2a[(X_train tree<br/>22076 × 21<br/>category dtypes, NaN)]:::data
    S2 --> S2b[(X_train_num<br/>22076 × 48<br/>one-hot, imputed)]:::data
    S2 --> S2c[(X_train_scaled<br/>22076 × 48<br/>StandardScaler)]:::data
    S2 --> S2d[(X_val / X_test variants<br/>same transformations)]:::data
    S2 --> S2e[(scaler + median_imputer<br/>fitted on TRAIN only)]:::data

    %% ── Stage 3: Feature selection ──
    S2a --> S3[3. apply_feature_selection<br/>src/feature_selection.py]:::stage
    S3 --> S3a[3a. drop_high_missing<br/>threshold=0.5]:::substage
    S3a --> S3b[3b. drop_correlated_features<br/>Spearman > 0.9]:::substage
    S3b --> S3c[3c. select_top_mutual_information<br/>top_k=20]:::substage
    S3c --> S3d[3d. select_top_features_lgbm_pfi_based<br/>top_n=15]:::substage
    S3d --> S3out[(top_features<br/>15 names)]:::data
    S3 --> F1[/feature_selection_report.csv/]:::file

    %% ── Stage 4: Train + save per model ──
    S3out --> S4[4. train_and_save_models<br/>src/pipeline.py]:::stage
    S2e --> S4
    S4 --> S4a[for each model: lgbm, xgb, gp, svm, mlp]:::substage
    S4a --> S4b[Optuna HPO 30 trials<br/>train_final<br/>threshold optimisation]:::substage
    S4b --> S4c[evaluate train / val / test]:::substage
    S4c --> S4d[save_to_tracker + save_to_mlflow<br/>IMMEDIATELY after each model]:::substage
    S4d --> F2[/reports/runs/ts_model/<br/>run.json, model.pkl,<br/>evaluation_results.csv,<br/>submission.csv,<br/>threshold_sweep.csv,<br/>threshold_selection.png,<br/>val_proba.csv,<br/>optuna_trials.csv,<br/>optuna_study.pkl,<br/>feature_importance_pct.csv/]:::file
    S4d --> F3[/reports/runs/mlruns/<br/>MLflow artifacts/]:::file

    %% ── Stage 5: Cross-model aggregation ──
    S4 --> S5[5. merge_summary_and_log<br/>generate_plots<br/>save_best_submission]:::stage
    S5 --> F4[/evaluation_summary.csv<br/>combined_roc.png/]:::file
    S5 --> F5[/reports/figures/<br/>roc_curves.png<br/>loss_curves.png<br/>feature_importance_pct.png<br/>permutation_importance.png/]:::file
    S5 --> F6[/data/processed/submission.csv<br/>best model by val AUC/]:::file

    %% ── Stage 6: Submission ──
    F6 --> S6[6. make submit<br/>kaggle CLI]:::stage
    S6 --> KAG[Kaggle leaderboard]:::external

    classDef stage fill:#4878d0,stroke:#2c5aa0,color:#fff,stroke-width:2px
    classDef substage fill:#a4c8f0,stroke:#4878d0,color:#000
    classDef data fill:#f3e9d2,stroke:#c5a572,color:#000
    classDef file fill:#c5e8b7,stroke:#5c8a4a,color:#000
    classDef external fill:#ee854a,stroke:#a85a30,color:#fff
```

---

## Stage table

| #   | Stage                            | Module / function                                                         | Input                                                                         | Output                                                                                                                          | Files written                                                                                                                                                                                                                                                            |
| --- | -------------------------------- | ------------------------------------------------------------------------- | ----------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 1   | Read raw CSVs                    | `pd.read_csv` inside `preprocessing.py`                                   | `data/raw/train_set.csv` (27,595 × 22), `data/raw/test_set.csv` (13,593 × 21) | Two raw `DataFrame` objects in memory                                                                                           | —                                                                                                                                                                                                                                                                        |
| 2   | Preprocess (single entry)        | `preprocess_data()` in `src/preprocessing.py`                             | The two raw DataFrames                                                        | `ProcessedData` container with **9 DataFrames** (3 variants × 3 splits) + `y_train`, `y_val`, fitted `scaler`, `median_imputer` | —                                                                                                                                                                                                                                                                        |
| 2a  | Tree variant                     | `build_features()`                                                        | raw DataFrame                                                                 | `X_train` (22,076 × 21), `X_val` (5,519 × 21), `X_test` (13,593 × 21) — **category dtypes, NaN preserved**                      | —                                                                                                                                                                                                                                                                        |
| 2b  | Numeric variant                  | `_to_numeric()` (one-hot + median impute, fit on TRAIN)                   | tree variant                                                                  | `X_train_num` (22,076 × 48), `X_val_num`, `X_test_num` — **float32, no NaN**                                                    | —                                                                                                                                                                                                                                                                        |
| 2c  | Scaled variant                   | `_scale()` with `StandardScaler` (fit on TRAIN only)                      | numeric variant                                                               | `X_train_scaled` (22,076 × 48), `X_val_scaled`, `X_test_scaled` — **mean 0, std 1**                                             | —                                                                                                                                                                                                                                                                        |
| 3   | Feature selection (4 stages)     | `apply_feature_selection()` in `src/pipeline.py`                          | `ProcessedData` (uses `X_train` tree variant)                                 | filtered `ProcessedData` (all variants pruned) + `top_features` list                                                            | `reports/runs/feature_selection_report.csv`                                                                                                                                                                                                                              |
| 3a  | Drop high-missing features       | `drop_high_missing(threshold=0.5)`                                        | 21 features                                                                   | ~20 features (drops `pdays` at 96% NaN)                                                                                         | —                                                                                                                                                                                                                                                                        |
| 3b  | Drop correlated features         | `drop_correlated_features(threshold=0.9, method="spearman")`              | output of 3a                                                                  | ~17 features (drops macro-economic dupes like `nr.employed` correlated with `euribor3m`)                                        | —                                                                                                                                                                                                                                                                        |
| 3c  | Top-K by Mutual Information      | `select_top_mutual_information(top_k=20)`                                 | output of 3b                                                                  | min(K, remaining) features                                                                                                      | —                                                                                                                                                                                                                                                                        |
| 3d  | Top-N by LightGBM PFI            | `select_top_features_lgbm_pfi_based(top_n=15)`                            | output of 3c                                                                  | **15 features** (final list)                                                                                                    | —                                                                                                                                                                                                                                                                        |
| 4   | Train + save per model           | `train_and_save_models()` in `src/pipeline.py`                            | filtered `ProcessedData` + tracker                                            | List of 5 result dicts (one per model)                                                                                          | `reports/runs/<ts>_<model>/` (×5, see 4d)                                                                                                                                                                                                                                |
| 4a  | Pick model + select data variant | `_data_map[spec["data"]]`                                                 | model name + `ProcessedData`                                                  | `(X_train, X_val, X_test)` of the right variant (tree or scaled)                                                                | —                                                                                                                                                                                                                                                                        |
| 4b  | HPO + final fit                  | `train_lgbm` / `train_xgb` / `train_gp` / `train_svm` / `train_mlp`       | train + val splits                                                            | fitted model, `val_proba`, `train_proba`, `test_preds`, optuna `study`, threshold info                                          | —                                                                                                                                                                                                                                                                        |
| 4c  | Evaluate model                   | `evaluate_model()` + `build_run_metrics()`                                | `result` dict + `y_train`, `y_val`                                            | `model_eval` DataFrame (3 rows) + `run_metrics` dict (16 metrics)                                                               | —                                                                                                                                                                                                                                                                        |
| 4d  | Save IMMEDIATELY after training  | `save_to_tracker()` + `save_to_mlflow()`                                  | `result`, `run_metrics`, `model_eval`                                         | written to disk in `reports/runs/<ts>_<model>/`                                                                                 | `run.json`, `evaluation_results.csv`, `submission.csv`, `model.pkl`, `threshold_sweep.csv`, `threshold_selection.png`, `val_proba.csv`, `optuna_trials.csv`, `optuna_study.pkl`, `feature_importance_pct.csv` (tree only)                                                |
| 5   | Cross-model aggregation          | `merge_summary_and_log()` + `generate_plots()` + `save_best_submission()` | list of all `trained` results                                                 | unified summary, plots, best model copy                                                                                         | `reports/runs/evaluation_summary.csv`, `reports/runs/combined_roc.png`, `reports/figures/roc_curves.png`, `reports/figures/loss_curves.png`, `reports/figures/feature_importance_pct.png`, `reports/figures/permutation_importance.png`, `data/processed/submission.csv` |
| 6   | Submit to Kaggle                 | `make submit` (or `make submit RUN=<run_dir>`)                            | `data/processed/submission.csv` (or per-model run)                            | Kaggle leaderboard score                                                                                                        | —                                                                                                                                                                                                                                                                        |

---

## Key design properties

1. **Single source of truth for transformations**: all imputation, encoding, and scaling
   happens in `preprocess_data()`. Models receive ready-to-use DataFrames.
2. **No data leakage**: medians and the StandardScaler are fit on **TRAIN only** then
   applied to val and test.
3. **Incremental saving**: each model's artifacts land on disk **immediately** after that
   model finishes (stage 4d). If model 5 of 5 crashes, models 1–4 are already safe.
4. **Multiple variants from one CSV read**: tree, numeric, and scaled variants are built
   in one pass — no redundant disk I/O.
5. **Feature selection report**: stage 3 produces a per-feature CSV showing which
   filter killed each rejected feature and why (see `docs/feature_selection.md`).

---

## Where to find what

| You want…                                                           | Look at                                                    |
| ------------------------------------------------------------------- | ---------------------------------------------------------- |
| Final selected features and why others were dropped                 | `reports/runs/feature_selection_report.csv`                |
| One model's metrics (run.json)                                      | `reports/runs/<ts>_<model>/run.json`                       |
| One model's threshold sweep (acc, recall, F1, Youden per threshold) | `reports/runs/<ts>_<model>/threshold_sweep.csv`            |
| Visual: where the threshold was selected                            | `reports/runs/<ts>_<model>/threshold_selection.png`        |
| All models × all splits in one table                                | `reports/runs/evaluation_summary.csv`                      |
| ROC of all historical runs in one plot                              | `reports/runs/combined_roc.png`                            |
| MLflow UI for cross-run exploration                                 | `uv run mlflow ui --backend-store-uri reports/runs/mlruns` |
| Best model's submission for Kaggle                                  | `data/processed/submission.csv`                            |
| Per-model submission (incl. non-best)                               | `reports/runs/<ts>_<model>/submission.csv`                 |
