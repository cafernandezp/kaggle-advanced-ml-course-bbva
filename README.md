Repository for kaggle resolution problems


# Create folders

make init

# Initialize uv
uv init --no-workspace
uv add numpy pandas scikit-learn xgboost lightgbm matplotlib seaborn kaggle ipykernel
uv sync

# Activate the env
source .venv/bin/activate
