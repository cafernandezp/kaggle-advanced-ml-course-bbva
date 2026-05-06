"""
Inference CLI — load a trained run and score a new CSV.

Usage:
    uv run python -m src.predict \\
        --run reports/runs/<ts>_<model>_optuna \\
        --input data/raw/new_data.csv \\
        --output data/processed/my_predictions.csv

The run folder must contain `model.pkl`, `run.json`, and `features.json`.
The global `reports/runs/preprocessing/` directory must contain the fitted
`imputer.pkl`, `scaler.pkl`, and column-order JSONs.
"""
import json
import logging
import pickle
import sys
from pathlib import Path

import click
import pandas as pd

from src.config import DEFAULT_CONFIG
from src.preprocessing import (  # pylint: disable=no-name-in-module
    PREPROCESSING_DIR_NAME,
    _scale,
    _to_numeric,
    build_features,
    load_preprocessing_artifacts,
)

ROOT = Path(__file__).parent.parent
RUNS_DIR = ROOT / "reports/runs"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def _load_run(run_dir: Path) -> dict:
    """Load model.pkl, run.json, and features.json from a run folder."""
    if not run_dir.exists():
        raise FileNotFoundError(f"Run folder not found: {run_dir}")

    model_path    = run_dir / "model.pkl"
    run_json      = run_dir / "run.json"
    features_json = run_dir / "features.json"
    for required in (model_path, run_json, features_json):
        if not required.exists():
            raise FileNotFoundError(f"Missing required file: {required}")

    with open(model_path, "rb") as fh:
        model = pickle.load(fh)
    run_meta = json.loads(run_json.read_text())
    features = json.loads(features_json.read_text())
    return {
        "model":    model,
        "run_meta": run_meta,
        "features": features,
    }


def _prepare_features(
    raw_df: pd.DataFrame,
    features: dict,
    preprocessing_state: dict,
) -> pd.DataFrame:
    """Apply the same feature engineering the training run saw.

    The `data_variant` key in features.json tells us whether the model expects
    the tree variant (category dtypes) or the scaled variant (one-hot + scaled).
    """
    tree = build_features(raw_df)
    variant = features["data_variant"]

    if variant == "tree":
        return tree[features["tree_features"]]

    if variant == "scaled":
        num, _ = _to_numeric(tree, median_imputer=preprocessing_state["imputer"])
        # Re-align columns with the train-time column order (new categorical levels → 0)
        num = num.reindex(columns=preprocessing_state["numeric_columns"], fill_value=0)
        scaled = _scale(num, preprocessing_state["scaler"])
        return scaled[features["scaled_features"]]

    raise ValueError(f"Unknown data_variant: {variant!r} (expected 'tree' or 'scaled')")


@click.command()
@click.option(
    "--run", "run_dir", required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Path to a trained run folder (e.g. reports/runs/<ts>_xgb_optuna/).",
)
@click.option(
    "--input", "input_path", required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="CSV file to score (same schema as data/raw/train_set.csv minus the target).",
)
@click.option(
    "--output", "output_path", required=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Destination CSV for predictions.",
)
@click.option(
    "--threshold", type=float, default=None,
    help="Override the threshold saved in run.json. Otherwise uses run.json value.",
)
def main(run_dir: Path, input_path: Path, output_path: Path, threshold: float | None):
    """Score a new CSV using a previously trained model run."""
    logger.info("Loading run from %s", run_dir)
    run = _load_run(run_dir)
    saved_threshold = run["run_meta"]["metrics"]["threshold"]
    threshold = threshold if threshold is not None else saved_threshold
    logger.info(
        "Model: %s  |  threshold: %.4f  (%s)",
        run["run_meta"]["params"].get("model", "?"),
        threshold,
        "override" if threshold != saved_threshold else "from run.json",
    )

    preprocessing_dir = RUNS_DIR / PREPROCESSING_DIR_NAME
    logger.info("Loading preprocessing artifacts from %s", preprocessing_dir)
    preprocessing_state = load_preprocessing_artifacts(preprocessing_dir)

    logger.info("Reading input CSV: %s", input_path)
    raw = pd.read_csv(input_path, index_col=DEFAULT_CONFIG.id_col)
    logger.info("Input shape: %s", raw.shape)

    X = _prepare_features(raw, run["features"], preprocessing_state)
    logger.info("Prepared shape (after feature selection): %s", X.shape)

    proba = run["model"].predict_proba(X)[:, 1]
    preds = (proba >= threshold).astype(int)

    output = pd.DataFrame({
        DEFAULT_CONFIG.id_col: X.index,
        DEFAULT_CONFIG.target_col: preds,
        "y_proba": proba.round(6),
    })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)

    logger.info("Predictions saved → %s  (%d rows)", output_path, len(output))
    logger.info("Predicted positive rate: %.2f%%", output[DEFAULT_CONFIG.target_col].mean() * 100)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
