# STACK.md

Stack tecnologico utilizado en el proyecto.

---

## Lenguaje y entorno

| Herramienta | Version | Proposito |
|---|---|---|
| Python | >= 3.12 | Lenguaje principal |
| uv | — | Gestor de dependencias y entornos virtuales (reemplaza pip + venv) |
| Make | — | Automatizacion de comandos (pipeline, lint, submit, eval-summary) |

---

## Machine Learning y modelado

| Libreria | Version | Proposito |
|---|---|---|
| scikit-learn | >= 1.8.0 | Preprocessing, metricas, Gaussian Process Classification, permutation importance |
| LightGBM | >= 4.6.0 | Gradient boosting con soporte nativo de categoricas |
| XGBoost | >= 3.2.0 | Gradient boosting alternativo |
| PyTorch | >= 2.11.0 | MLP con dropout (red neuronal feed-forward) |
| Optuna | >= 4.8.0 | Optimizacion de hiperparametros con TPE sampler |

---

## Tracking y registro de experimentos

| Herramienta | Version | Proposito |
|---|---|---|
| MLflow | >= 3.9.0 | Registro de modelos, comparacion visual de runs (UI), model registry |
| Custom tracker (`src/tracking.py`) | — | Tracking ligero basado en JSON + CSV, independiente de MLflow |

Ambos sistemas corren en paralelo: MLflow para la UI y el registro de modelos, el tracker custom para los artefactos estructurados (CSVs de evaluacion, threshold sweep, submission por modelo).

---

## Analisis y visualizacion

| Libreria | Version | Proposito |
|---|---|---|
| pandas | >= 2.0.0, < 3 | Manipulacion de datos tabulares (limitado a 2.x por compatibilidad con MLflow) |
| NumPy | >= 2.4.4 | Operaciones numericas |
| Matplotlib | >= 3.10.8 | Graficos: ROC, loss curves, feature importance, distribuciones |
| seaborn | >= 0.13.2 | Graficos estadisticos (disponible, usado opcionalmente) |

---

## CLI y orquestacion

| Libreria | Version | Proposito |
|---|---|---|
| Click | >= 8.3.2 | CLI del pipeline (`--models`, `--n-trials`, `--report`) |

---

## Calidad de codigo

| Herramienta | Version | Proposito |
|---|---|---|
| pylint | >= 4.0.5 | Linting estatico (score actual: 10.00/10) |

---

## Competencia y datos

| Herramienta | Version | Proposito |
|---|---|---|
| Kaggle CLI | >= 2.0.0 | Descarga de datos y envio de submissions |

---

## Infraestructura del proyecto

```
src/
  pipeline.py        -- Orquestador CLI (Click)
  train.py           -- Funciones de entrenamiento (HPO + fit)
  evaluations.py     -- Metricas por split (train / val / test)
  metrics.py         -- Threshold sweep, Youden, KS, Gini
  plots.py           -- Graficos de comparacion
  tracking.py        -- Tracker custom (JSON + CSV + pkl)
  preprocessing.py   -- Feature engineering + splits
  eda.py             -- Analisis exploratorio reutilizable
  models/
    lgbm_model.py    -- LightGBM: objetivo Optuna + train_final
    xgb_model.py     -- XGBoost: objetivo Optuna + train_final
    mlp_model.py     -- MLP PyTorch: objetivo Optuna + train_final + early stopping
    gp_model.py      -- Gaussian Process: seleccion de kernels via Optuna
```

---

## Notas

- **pandas < 3**: MLflow no soporta pandas 3.x (todas las versiones requieren pandas < 3). Se mantiene pandas 2.x por compatibilidad.
- **GPy**: no compila en Python 3.12+ (header `longintrepr.h` eliminado). Se usa `sklearn.gaussian_process.GaussianProcessClassifier` como alternativa.
- **Dual tracking**: si MLflow falla, los artefactos estructurados del tracker custom siguen disponibles en `reports/runs/`.
