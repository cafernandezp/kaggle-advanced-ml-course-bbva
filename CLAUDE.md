# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Purpose

This repository contains the solution for a Kaggle competition developed as part of an advanced ML course for BBVA.

**Problem**: Binary classification — predict whether a bank client will contract a product offered in a marketing campaign.  
**Competition**: `aprendizaje-automatico-avanzado-febrero-2026`  

### Target & metric
- Metric: **Accuracy**
- Target values: `0` (client does not contract) / `1` (client contracts)

### Submission format
Two-column CSV: `Id` (observation index) + `subscribed` (prediction). Column names live in `src/config.py` (`id_col`, `target_col`).

```
Id,subscribed
0,1
1,0
...
```

Work is organized in class groups and delivered primarily as Jupyter notebooks.

## Dataset domain knowledge

| Group | Variables |
|---|---|
| Client demographics | `age`, `job`, `marital`, `education` |
| Financial status | `default` (credit in default), `housing` (housing loan), `loan` (personal loan) |
| Last contact | `contact`, `month`, `day_of_week`, `duration` |
| Campaign history | `campaign` (contacts this campaign), `pdays`, `previous`, `poutcome` |
| Macro-economic indicators | `emp.var.rate`, `cons.price.idx`, `cons.conf.idx`, `euribor3m`, `nr.employed` |

**Key modeling notes**:
- `duration` is **leaky** — call duration is only known after the outcome. Kept because the competition test set includes it, but must be excluded in any real deployment.
- `pdays = 999` means the client was never previously contacted. Encoded as `was_contacted = 0` + `pdays = NaN` in `src/preprocessing.py`.
- `education` is **ordinal** (`illiterate=0` → `university.degree=6`, `unknown=NaN`) — encoded as integers in `src/preprocessing.py`.
- The 5 macro-economic indicators are externally sourced and identical for all clients contacted in the same period — high multicollinearity expected.

## Stack

- **Language**: Python
- **Notebooks**: Jupyter (`.ipynb`), checkpoints excluded via `.gitignore`
- **Experiment tracking**: custom tracker in `src/tracking.py` (`reports/runs/<ts>_<model>_optuna/`) **and** MLflow in parallel (`reports/runs/mlruns/`, gitignored). Both are written by `src/pipeline.py`.
- **Linting**: Ruff (`.ruff_cache/` is gitignored)
- **Environment**: uv (`pyproject.toml` + `uv.lock`)

See [`STACK.md`](STACK.md) for library versions and compatibility notes (pandas<3 for MLflow, GPy on Py 3.12+, dual-tracking fallback).

## Pipeline & models

Entry point is `src/pipeline.py` (Click CLI), driven via the Makefile.

```bash
make pipeline                              # all 5 models
make pipeline MODELS='lgbm xgb svm gp'     # skip MLP (slow) — preferred for quick e2e tests
make pipeline-preprocess                   # stage 1 only
make pipeline-feature-select               # stage 2 only
make pipeline-train MODELS='lgbm'          # stage 3 only, single model
make eval-summary                          # rebuild reports/runs/evaluation_summary.csv
```

Available model names (from `src/train.py:ALL_MODELS`): `lgbm`, `xgb`, `mlp`, `gp`, `svm`.

**Data convention**: tree models (`lgbm`, `xgb`) consume the categorical-dtype splits (`load_splits`); scale-sensitive models (`mlp`, `gp`, `svm`) consume the imputed + StandardScaler splits (`load_splits_scaled`). Both are produced by `src/preprocessing.py` and cached in `reports/runs/preprocessing/`.

## Public-repo policy

- `*.pkl` and `*.parquet` under `reports/runs/` are gitignored — regenerate via `make pipeline`, never commit.
- `reports/runs/mlruns/` (MLflow store) and `reports/runs/.trash/` are also gitignored.
- `.env` is gitignored — never commit Kaggle tokens or other secrets. Rotate any token that touches `.env`.

## Common commands

```bash
ruff check .
ruff format .

jupyter nbconvert --to script notebook.ipynb
python notebook.py
```

## Branching

- `main` — stable, course-ready work
- `develop` — active development branch; PRs merge into `main` (avoid direct merges to `main`)
