# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Purpose

This repository contains solutions to Kaggle competition problems, developed as part of an advanced ML course for BBVA. Work is organized around individual competition problems, typically as Jupyter notebooks.

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
