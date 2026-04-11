"""
Pipeline configuration — the single file to edit when adapting this pipeline to a new project.

All project-specific settings (experiment name, target column, submission schema,
feature-selection thresholds) live here. The code itself is project-agnostic.

To adapt to a new project:
    1. Copy this file to the new project.
    2. Change `DEFAULT_CONFIG` values (experiment, id_col, target_col).
    3. Optionally tweak the feature-selection thresholds.
    4. Everything else (pipeline.py, train.py, models, evaluations) runs unchanged.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class PipelineConfig:  # pylint: disable=too-many-instance-attributes
    """All project-specific settings in one place.

    Frozen so it can't be mutated accidentally mid-pipeline — if you need a
    different config, create a new instance.
    """

    # ── Required: business identifiers ──────────────────────────────────
    experiment: str        # MLflow experiment name (also used by the custom tracker)
    id_col: str            # name of the ID column in raw CSVs and submission files
    target_col: str        # name of the target column in the train set

    # ── Feature selection thresholds ────────────────────────────────────
    top_n_features: int        = 15    # final PFI top-N after all filters
    missing_threshold: float    = 0.5   # drop features with more than X fraction missing
    correlation_threshold: float = 0.9  # drop features with abs correlation > X
    mi_top_k: int               = 20    # keep top-K by mutual information with target

    # ── Threshold selection ─────────────────────────────────────────────
    # Within the Youden-tolerance band, pick the threshold that maximises this metric.
    # Valid values: "accuracy", "precision", "recall", "f1", "specificity".
    # Default "accuracy" matches the competition metric; use "f1" or "recall"
    # for imbalanced problems where false negatives matter more.
    secondary_metric: str      = "accuracy"

    # ── Cross-validation (only used when --cv flag is passed to the pipeline) ──
    # 5 is a standard choice. Use 3 for faster iteration on slow models (MLP/GP/SVM),
    # or 10 for more robust estimates on small datasets. Runtime scales ~linearly.
    cv_splits: int             = 5

    # ── Reporting thresholds ────────────────────────────────────────────
    overfit_warning_gap: float = 0.02   # flag models where train_auc − val_auc > this


# ── Default config for THIS project ─────────────────────────────────────
# Edit the three required fields when adapting to a new project.
DEFAULT_CONFIG = PipelineConfig(
    experiment="banking-marketing-classification",
    id_col="Id",
    target_col="subscribed",
)
