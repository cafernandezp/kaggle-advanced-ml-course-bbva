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

.PHONY: init clean lint pipeline eval-summary submit help

help:
	@echo ""
	@echo "Available commands:"
	@echo "  make pipeline                    — Run full ML pipeline (all models)"
	@echo "  make pipeline MODELS='lgbm xgb'  — Run specific models"
	@echo "  make init                        — Create project folder structure"
	@echo "  make lint                        — Run pylint on src/"
	@echo "  make eval-summary                — Merge all evaluation_results.csv into one summary"
	@echo "  make submit                      — Submit best model to Kaggle"
	@echo "  make submit RUN=<run_dir>        — Submit a specific model's run"
	@echo "  make submit MSG=\"...\"            — Submit with a custom message"
	@echo "  make clean                       — Remove __pycache__ and .pyc files"
	@echo ""

MODELS ?=
_MODEL_ARGS := $(if $(MODELS),$(foreach m,$(MODELS),--models $(m)),)

pipeline:
	uv run python -m src.pipeline $(_MODEL_ARGS)

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

submit:
	@test -f $(SUBMISSION) || { echo "Error: $(SUBMISSION) not found. Run the pipeline first."; exit 1; }
	kaggle competitions submit -c $(COMPETITION) -f $(SUBMISSION) -m "$(MSG)"
	@echo "✓ Submitted $(SUBMISSION) — message: $(MSG)"

clean:
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "✓ Clean done"