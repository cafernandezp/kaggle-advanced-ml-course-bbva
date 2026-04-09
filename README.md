Repository for kaggle resolution problems

## Setup

```bash
# Create folders
make init

# Initialize uv
uv init --no-workspace
uv add numpy pandas scikit-learn xgboost lightgbm matplotlib seaborn kaggle ipykernel
uv sync

# Activate the env
source .venv/bin/activate
```

## Download competition data

```bash
kaggle competitions download -c aprendizaje-automatico-avanzado-febrero-2026
unzip aprendizaje-automatico-avanzado-febrero-2026.zip -d data/raw/
```

## Run the ML pipeline

All scripts are run as modules from the **project root** with the virtual environment active.

```bash
# 1. Exploratory Data Analysis
#    → prints stats to terminal, saves plots to reports/figures/
python -m src.eda

# 2. Verify feature engineering and split sizes
#    → prints train/val shapes and feature list
python -m src.preprocessing

# 3. Train model and generate submission
#    → prints train/val accuracy, overfit gap, classification report, feature importances
#    → saves run JSON + artifacts to reports/runs/<timestamp>_<run_name>/
#    → saves submission to data/processed/submission.csv
python -m src.train

# 4. Compare all tracked runs
#    → saves reports/runs/comparison.csv  (metrics table)
#    →       reports/runs/comparison.png  (accuracy & overfit bar charts)
#    →       reports/runs/feature_importance_comparison.png  (if >1 run)
#    Each individual run is saved under reports/runs/<timestamp>_<run_name>/
#      run.json               — all params and metrics
#      feature_importance.csv — feature importances for that run
#      feature_importance.png — feature importance bar chart
#      submission.csv         — the Kaggle submission file
python -m src.train --report
```
