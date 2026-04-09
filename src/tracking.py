"""
Lightweight experiment tracker — stores runs as JSON, generates comparison reports.

Usage (mirrors MLflow's API so migration is easy later):

    tracker = ExperimentTracker("banking-marketing")
    with tracker.start_run("lgbm_baseline"):
        tracker.log_params({"learning_rate": 0.05, ...})
        tracker.log_metrics({"val_accuracy": 0.917, ...})

Reports:
    tracker.generate_report()  →  reports/runs/comparison.csv + comparison.png
"""
import json
import logging
import pickle
import shutil
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
RUNS_DIR = ROOT / "reports/runs"


class ExperimentTracker:
    """Lightweight JSON-based tracker for ML experiments."""

    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        self._run_dir: Path | None = None
        self._data: dict = {}

    @contextmanager
    def start_run(self, run_name: str | None = None):
        """Context manager for a single training run."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = run_name or "run"
        self._run_dir = RUNS_DIR / f"{ts}_{name}"
        self._run_dir.mkdir(parents=True, exist_ok=True)

        self._data = {
            "experiment": self.experiment_name,
            "run_name": name,
            "timestamp": ts,
            "params": {},
            "metrics": {},
        }

        logger.info("run started → %s", self._run_dir.name)
        try:
            yield self
        finally:
            self._save()
            logger.info("run saved   → %s", self._run_dir.name)

    def log_params(self, params: dict) -> None:
        """Bulk-update the run's parameter dict."""
        self._data["params"].update(params)

    def log_param(self, key: str, value) -> None:
        """Log a single hyperparameter."""
        self._data["params"][key] = value

    def log_metrics(self, metrics: dict) -> None:
        """Bulk-update the run's metrics dict."""
        self._data["metrics"].update(metrics)

    def log_metric(self, key: str, value: float) -> None:
        """Log a single metric value."""
        self._data["metrics"][key] = value

    def log_artifact(self, src_path: str | Path) -> None:
        """Copy a file into the run directory."""
        if self._run_dir is None:
            raise RuntimeError("No active run — use start_run() context manager.")
        dst = self._run_dir / Path(src_path).name
        shutil.copy2(src_path, dst)

    def log_dataframe(self, df: pd.DataFrame, filename: str) -> None:
        """Save a DataFrame as CSV directly into the run directory."""
        if self._run_dir is None:
            raise RuntimeError("No active run — use start_run() context manager.")
        df.to_csv(self._run_dir / filename, index=False)

    def log_study(self, study) -> None:
        """Save Optuna study: trials as CSV and full study object as pkl."""
        if self._run_dir is None:
            raise RuntimeError("No active run — use start_run() context manager.")
        trials_path = self._run_dir / "optuna_trials.csv"
        study.trials_dataframe().to_csv(trials_path, index=False)
        pkl_path = self._run_dir / "optuna_study.pkl"
        with open(pkl_path, "wb") as fh:
            pickle.dump(study, fh)
        logger.info("Optuna trials  → %s", trials_path.name)
        logger.info("Optuna study   → %s", pkl_path.name)

    def log_model(self, model, filename: str = "model.pkl") -> None:
        """Pickle a trained model into the run directory."""
        if self._run_dir is None:
            raise RuntimeError("No active run — use start_run() context manager.")
        path = self._run_dir / filename
        with open(path, "wb") as fh:
            pickle.dump(model, fh)
        logger.info("Model saved → %s", path.name)

    def _save(self) -> None:
        out = self._run_dir / "run.json"
        out.write_text(json.dumps(self._data, indent=2, default=str))

    # ── Reporting ─────────────────────────────────────────────────────────────

    def load_all_runs(self) -> pd.DataFrame:
        """Load every saved run into a flat DataFrame."""
        records = []
        for f in sorted(RUNS_DIR.glob("*/run.json")):
            data = json.loads(f.read_text())
            row = {
                "run_name": data["run_name"],
                "timestamp": data["timestamp"],
                **{f"param_{k}": v for k, v in data.get("params", {}).items()},
                **data.get("metrics", {}),
            }
            records.append(row)
        return pd.DataFrame(records)

    def generate_report(self) -> None:
        """Save a comparison CSV and accuracy bar chart across all runs."""
        df = self.load_all_runs()
        if df.empty:
            logger.info("No runs found.")
            return

        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        csv_path = RUNS_DIR / "comparison.csv"
        df.to_csv(csv_path, index=False)
        logger.info("Comparison table → %s", csv_path)

        # Accuracy & overfit chart
        plot_cols = [c for c in ["val_accuracy", "train_accuracy", "overfit_gap"] if c in df.columns]
        if not plot_cols:
            return

        labels = df["run_name"] + "\n" + df["timestamp"]
        x = range(len(df))
        _, axes = plt.subplots(1, len(plot_cols), figsize=(5 * len(plot_cols), 4))
        if len(plot_cols) == 1:
            axes = [axes]

        for ax, col in zip(axes, plot_cols):
            ax.bar(x, df[col], color="#4878d0", edgecolor="white")
            ax.set_xticks(list(x))
            ax.set_xticklabels(labels, fontsize=7, rotation=30, ha="right")
            ax.set_title(col)
            ax.set_ylim(0, max(df[col].max() * 1.1, 0.01))
            for i, v in enumerate(df[col]):
                ax.text(i, v + 0.001, f"{v:.4f}", ha="center", fontsize=7)

        plt.suptitle(f"Experiment: {self.experiment_name}", fontsize=11)
        plt.tight_layout()
        png_path = RUNS_DIR / "comparison.png"
        plt.savefig(png_path, dpi=130)
        plt.close()
        logger.info("Comparison chart → %s", png_path)

        # Feature importance chart across runs
        imp_files = list(RUNS_DIR.glob("*/feature_importance.csv"))
        if len(imp_files) > 1:
            frames = []
            for f in imp_files:
                run_name = f.parent.name
                tmp = pd.read_csv(f)
                tmp.columns = ["feature", run_name]
                frames.append(tmp.set_index("feature"))
            imp_df = pd.concat(frames, axis=1).fillna(0)
            _, ax = plt.subplots(figsize=(10, max(4, len(imp_df) * 0.4)))
            imp_df.plot(kind="barh", ax=ax)
            ax.invert_yaxis()
            ax.set_title("Feature importance across runs")
            plt.tight_layout()
            plt.savefig(RUNS_DIR / "feature_importance_comparison.png", dpi=130)
            plt.close()
            cmp_path = RUNS_DIR / "feature_importance_comparison.png"
            logger.info("Feature importance comparison → %s", cmp_path)

        logger.info("\n%s", df[["run_name", "timestamp"] + plot_cols].to_string(index=False))
