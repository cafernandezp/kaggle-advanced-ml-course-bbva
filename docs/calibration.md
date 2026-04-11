# Probability Calibration

> Last updated: 10-04-2026

## What we measure (automatic)

Every trained model is scored with two calibration metrics and a diagnostic plot:

| Metric / artifact | Where it's saved | Meaning |
|---|---|---|
| `brier_score` | `run.json` metrics, `evaluation_results.csv` | Mean squared error between `y_proba` and `y_true`. Lower = better. Range [0, 1]. |
| `ece` (Expected Calibration Error) | `run.json` metrics, `evaluation_results.csv` | Binned gap between predicted confidence and actual accuracy. Lower = better. Range [0, 1]. |
| `reliability_diagram.png` | `reports/runs/<ts>_<model>/` | Visual: points on the diagonal = perfect calibration. Uses 10 bins by default. |

### Interpreting the reliability diagram

- **Diagonal line** = perfect calibration (a predicted probability of 0.7 corresponds to 70% actually positive).
- **Points above the diagonal** = underconfident (model predicts 0.3 but reality is 0.5).
- **Points below the diagonal** = overconfident (model predicts 0.9 but only 0.7 are actually positive).
- **Bottom histogram** shows how the predicted probabilities are distributed — many bins with zero samples means you can't trust their calibration scores.

Tree ensembles (LightGBM, XGBoost) are often overconfident at the tails, MLPs vary wildly, and isotonic regression on top tends to fix most of it.

## Why we don't auto-calibrate

Our competition metric is **accuracy**, and the decision threshold is already optimised
via Youden Index. In that setup, calibration has essentially zero impact on the
final score: we only care about whether `y_proba >= threshold`, not about the
actual magnitude of `y_proba`.

Also, applying calibration during training (e.g., wrapping every model in
`CalibratedClassifierCV`) would:
1. Change the distribution of `val_proba`, which invalidates the previously
   selected threshold.
2. Add ~3× training time because calibration needs its own held-out fold.
3. Make the threshold sweep CSV harder to interpret.

So we **measure** calibration (Brier + ECE + plot) so you can see if a model's
probabilities are trustworthy — but we don't **act** on it unless you need to.

## When to actually re-calibrate

Enable calibration if any of these apply:

- **Risk scoring / credit modelling**: decisions depend on the probability itself
  (e.g., "deny loans above 80% default probability"), not just a binary outcome.
- **Cost-sensitive decisions**: false positives and false negatives have
  different costs, and you use `y_proba` in expected-value calculations.
- **Probabilistic ensembling**: you're averaging probabilities across models,
  and mis-calibrated ones will dominate.
- **Regulatory requirements**: some domains require models to be statistically
  calibrated (e.g., insurance pricing).

## How to re-calibrate (if needed)

The cleanest path is wrapping the trained model with sklearn's
`CalibratedClassifierCV` using the `prefit` strategy, so the base model
doesn't retrain:

```python
from sklearn.calibration import CalibratedClassifierCV
import pickle

# Load an existing trained model
with open("reports/runs/20260410_220413_xgb_optuna/model.pkl", "rb") as fh:
    base = pickle.load(fh)

# Wrap with isotonic regression (non-parametric, handles arbitrary miscalibration)
calibrated = CalibratedClassifierCV(base, method="isotonic", cv="prefit")
calibrated.fit(X_val, y_val)   # fits the isotonic map on val data

# For Platt scaling (parametric, assumes sigmoidal miscalibration):
# calibrated = CalibratedClassifierCV(base, method="sigmoid", cv="prefit")

# The calibrated model has the same interface
proba_calibrated = calibrated.predict_proba(X_test)[:, 1]
```

**Important**: after re-calibrating, re-run the threshold optimisation — the
previously selected threshold is no longer optimal for the new probability
distribution.

## Method choice

| Method | When to use | Notes |
|---|---|---|
| **Platt scaling** (`method="sigmoid"`) | Small val set (< 1000), moderate miscalibration | Fits a 2-parameter sigmoid; assumes the distortion is monotonic and sigmoidal |
| **Isotonic regression** (`method="isotonic"`) | Larger val set (> 1000), arbitrary miscalibration shape | Non-parametric, more flexible, but can overfit small val sets |

For our dataset (5k val samples, ~11% positive rate) **isotonic** is the safer
choice if you ever need it.
