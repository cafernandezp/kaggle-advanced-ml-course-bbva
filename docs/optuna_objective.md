# Optuna Objective Design: Single vs Multi-Objective

> Last updated: 09-04-2026

## Current approach: single combined objective

Each Optuna trial maximises a single score that balances validation performance against overfitting:

```
score = val_auc − 0.5 × max(0, train_auc − val_auc)
```

Where:
- `val_auc` rewards discriminative power on unseen data
- `max(0, train_auc - val_auc)` is the overfitting gap (only penalises when train > val)
- `0.5` is the penalty weight (OVERFIT_PENALTY constant in each model file)

This is mathematically equivalent to a **scalarised multi-objective optimisation** where the two objectives (maximise AUC, minimise overfitting) are collapsed into one scalar via a fixed weight.

---

## Why not two separate objectives?

Optuna supports multi-objective optimisation via `directions=["maximize", "minimize"]`. With two objectives (`val_auc` and `overfit_gap`), the study would produce a **Pareto front** — a set of non-dominated trials:

```
        ▲ val_auc
   0.95 │         ● ●
        │       ●     ●          ← Pareto front
   0.94 │     ●         ●
        │   ●
   0.93 │ ●
        └──────────────────► overfit_gap
        0.00  0.01  0.02  0.03
```

No single trial on the Pareto front is "the best" — each represents a different trade-off between AUC and overfitting. You'd need to manually select one, which breaks automation.

### Comparison

| | Single objective (current) | Multi-objective (Pareto) |
|---|---|---|
| **Output** | One best trial with `study.best_params` | A set of non-dominated trials (`study.best_trials`) |
| **Automation** | Fully automated — pipeline extracts best params directly | Requires manual selection or a secondary heuristic |
| **Trade-off control** | Fixed via `OVERFIT_PENALTY = 0.5` | Deferred to after optimisation — full trade-off surface visible |
| **Best for** | Production pipelines, automated runs | Exploratory analysis, understanding model behaviour |
| **Optuna direction** | `direction="maximize"` | `directions=["maximize", "minimize"]` |

### When single objective is sufficient

- The penalty weight `0.5` is reasonable for most cases: it means 1% overfitting costs 0.5% in the score. A model with `val_auc=0.94` and `gap=0.02` scores `0.94 - 0.5×0.02 = 0.93`, while a model with `val_auc=0.93` and `gap=0.00` scores `0.93`. The second wins — correct intuition.
- For this competition (metric = accuracy, single submission), we need one model, not a set of candidates.

### When to switch to multi-objective

- If you want to **visualise** the AUC-vs-overfitting trade-off across all trials
- If the penalty weight `0.5` produces models that overfit too much or are too conservative
- If you're comparing model families (e.g., "does XGBoost Pareto-dominate LightGBM?")

---

## The `max(0, ...)` guard

The overfitting gap is clipped at zero:

```python
overfit_gap = max(0.0, train_auc - val_auc)
```

This means **underfitting is not penalised**. If `val_auc > train_auc` (which can happen with heavy regularisation or small training sets), the penalty is zero. This is intentional:
- Underfitting means the model hasn't memorised the training data — that's not a problem we need to penalise
- Penalising underfitting would discourage regularisation, which is counterproductive

---

## Adjusting the penalty weight

The weight `0.5` is defined as `OVERFIT_PENALTY` in each model file (`src/models/*_model.py`). To change sensitivity:

| Weight | Behaviour |
|---|---|
| `0.0` | Pure AUC optimisation — no overfitting penalty (risky) |
| `0.5` | Current default — moderate penalty |
| `1.0` | Strong penalty — effectively maximises `2×val_auc - train_auc` |
| `> 1.0` | Aggressive — prefers underfitting over any overfitting |

---

## Stratified CV interaction

When `--cv` is enabled, the combined score is computed from fold means:

```python
score = mean(val_auc_per_fold) − 0.5 × max(0, mean(train_auc − val_auc per fold))
```

This makes the score more robust (less variance from a single split) while keeping the same single-objective structure. CV does NOT change the objective function — it changes how each trial is evaluated.

---

## Implementation files

- `src/models/lgbm_model.py:OVERFIT_PENALTY` — penalty weight (same in all model files)
- `src/models/*_model.py:objective()` — single-split objective
- `src/models/*_model.py:cv_objective()` — CV objective (same combined score, K-fold evaluation)
- `src/train.py:run_study()` — runs the Optuna study with `direction="maximize"`
