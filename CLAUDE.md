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
Two-column CSV: observation index + prediction.

```
id,target
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
- **Experiment tracking**: custom tracker in `src/tracking.py` — runs saved to `reports/runs/` as JSON
- **Linting**: Ruff (`.ruff_cache/` is gitignored)
- **Environment**: uv (`pyproject.toml` + `uv.lock`)

## Common commands

```bash
# Lint with Ruff
ruff check .
ruff format .

# Run a notebook as a script (if converted)
jupyter nbconvert --to script notebook.ipynb
python notebook.py
```

## Branching

- `main` — stable, course-ready work
- `develop` — active development branch; PRs merge into `main`
