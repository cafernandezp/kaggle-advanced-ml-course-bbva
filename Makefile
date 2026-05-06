# ============================================================
# Makefile — kaggle-advanced-ml-course-bbva
# ============================================================

SRC_DIR  := src
DATA_DIRS := data/raw data/processed
NB_DIR   := notebooks

COMPETITION := aprendizaje-automatico-avanzado-febrero-2026
RUN         ?=
SUBMISSION  := $(if $(RUN),$(RUN)/submission.csv,data/processed/submission.csv)
MSG         ?= auto submission

.PHONY: init clean clean-results lint pipeline pipeline-preprocess pipeline-feature-select pipeline-train pipeline-report eval-summary predict submit help

help:
	@echo ""
	@echo "Available commands:"
	@echo "  make pipeline                    — Run full ML pipeline end-to-end (all models)"
	@echo "  make pipeline MODELS='lgbm xgb'  — Run specific models end-to-end"
	@echo ""
	@echo "  Stage-by-stage (inspect outputs between stages):"
	@echo "  make pipeline-preprocess         — Stage 1: preprocess raw CSVs → reports/runs/preprocessing/"
	@echo "  make pipeline-feature-select     — Stage 2: feature selection → reports/runs/feature_selection/"
	@echo "  make pipeline-train              — Stage 3: train models (uses stages 1+2 outputs)"
	@echo "  make pipeline-train MODELS='lgbm'— Stage 3 with a specific model"
	@echo "  make pipeline-report             — Cross-run comparison (no training)"
	@echo ""
	@echo "  make init                        — Create project folder structure"
	@echo "  make lint                        — Run pylint on src/"
	@echo "  make eval-summary                — Merge all evaluation_results.csv into one summary"
	@echo "  make predict RUN=<run> INPUT=<csv> OUTPUT=<csv>"
	@echo "                                    Score a new CSV using a trained run"
	@echo "  make submit                      — Submit best model to Kaggle"
	@echo "  make submit RUN=<run_dir>        — Submit a specific model's run"
	@echo "  make submit MSG=\"...\"            — Submit with a custom message"
	@echo "  make clean-results               — Delete all model runs, logs, and mlruns"
	@echo "  make clean                       — Remove __pycache__ and .pyc files"
	@echo ""

MODELS ?=
_MODEL_ARGS := $(if $(MODELS),$(foreach m,$(MODELS),--models $(m)),)

pipeline:
	uv run python -m src.pipeline run $(_MODEL_ARGS)

pipeline-preprocess:
	uv run python -m src.pipeline preprocess

pipeline-feature-select:
	uv run python -m src.pipeline feature-select

pipeline-train:
	uv run python -m src.pipeline train $(_MODEL_ARGS)

pipeline-report:
	uv run python -m src.pipeline report

init:
	@echo "→ Creating project structure..."
	@mkdir -p $(DATA_DIRS) $(NB_DIR) $(SRC_DIR)
	@touch data/raw/.gitkeep data/processed/.gitkeep
	@touch $(NB_DIR)/.gitkeep
	@touch $(SRC_DIR)/__init__.py
	@echo "✓ Done"

lint:
	uv run pylint $(SRC_DIR) --recursive=y

eval-summary:
	uv run python -c "from src.evaluations import merge_evaluation_summary; merge_evaluation_summary()"

INPUT  ?=
OUTPUT ?= data/processed/predictions.csv
predict:
	@test -n "$(RUN)" || { echo "Error: RUN is required. Usage: make predict RUN=reports/runs/<ts>_<model>_optuna INPUT=data/raw/new_data.csv"; exit 1; }
	@test -n "$(INPUT)" || { echo "Error: INPUT is required. Usage: make predict RUN=<run> INPUT=<csv>"; exit 1; }
	uv run python -m src.predict --run $(RUN) --input $(INPUT) --output $(OUTPUT)

submit:
	@test -f $(SUBMISSION) || { echo "Error: $(SUBMISSION) not found. Run the pipeline first."; exit 1; }
	kaggle competitions submit -c $(COMPETITION) -f $(SUBMISSION) -m "$(MSG)"
	@echo "✓ Submitted $(SUBMISSION) — message: $(MSG)"

clean-results:
	@rm -rf reports/runs/*
	@echo "✓ All runs, logs, and mlruns deleted from reports/runs/"

clean:
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "✓ Clean done"