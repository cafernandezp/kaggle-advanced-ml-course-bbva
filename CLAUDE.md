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

## Stack

- **Language**: Python
- **Notebooks**: Jupyter (`.ipynb`), checkpoints excluded via `.gitignore`
- **Experiment tracking**: MLflow (run dirs, DB, and artifacts are gitignored)
- **Linting**: Ruff (`.ruff_cache/` is gitignored)
- **Environment**: supports pyenv, uv, pixi, or venv — none locked in yet

## Common commands

```bash
# Lint with Ruff
ruff check .
ruff format .

# Run a notebook as a script (if converted)
jupyter nbconvert --to script notebook.ipynb
python notebook.py

# Start MLflow UI (if tracking experiments locally)
mlflow ui
```

## Branching

- `main` — stable, course-ready work
- `develop` — active development branch; PRs merge into `main`
