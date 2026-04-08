# ============================================================
# Makefile — kaggle-advanced-ml-course-bbva
# ============================================================

SRC_DIR  := src
DATA_DIRS := data/raw data/processed
NB_DIR   := notebooks

.PHONY: init clean help

help:
	@echo ""
	@echo "Available commands:"
	@echo "  make init    — Create project folder structure"
	@echo "  make clean   — Remove __pycache__ and .pyc files"
	@echo ""

init:
	@echo "→ Creating project structure..."
	@mkdir -p $(DATA_DIRS) $(NB_DIR) $(SRC_DIR)
	@touch data/raw/.gitkeep data/processed/.gitkeep
	@touch $(NB_DIR)/.gitkeep
	@touch $(SRC_DIR)/__init__.py
	@echo "✓ Done"

clean:
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "✓ Clean done"