"""
PyTorch MLP with dropout: Optuna objective + final training with loss history.
Input must be fully numeric (use build_features_numeric from preprocessing.py).
"""

import logging

import numpy as np
import optuna
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)

RANDOM_STATE = 42
OVERFIT_PENALTY = 0.5
OPTUNA_EPOCHS = 50
FINAL_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class MLP(nn.Module):
    """Feed-forward network: [Linear → ReLU → Dropout] × n_layers → Linear(1)."""

    def __init__(
        self, input_size: int, hidden_size: int, n_layers: int, dropout: float
    ):
        super().__init__()
        layers: list[nn.Module] = []
        for i in range(n_layers):
            in_size = input_size if i == 0 else hidden_size
            layers += [nn.Linear(in_size, hidden_size), nn.ReLU(), nn.Dropout(dropout)]
        layers.append(nn.Linear(hidden_size, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass — returns logits (shape: batch,)."""
        return self.net(x).squeeze(1)


def _to_tensors(X, y=None):
    X_t = torch.tensor(X.values if hasattr(X, "values") else X, dtype=torch.float32)
    if y is not None:
        y_t = torch.tensor(y.values if hasattr(y, "values") else y, dtype=torch.float32)
        return X_t, y_t
    return X_t


def _train_loop(model, optimizer, criterion, loader):
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(X_batch), y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def _eval_loss(model, criterion, X_t, y_t):
    model.eval()
    X_t, y_t = X_t.to(DEVICE), y_t.to(DEVICE)
    return criterion(model(X_t), y_t).item()


@torch.no_grad()
def _predict_proba(model, X_t):
    model.eval()
    X_t = X_t.to(DEVICE)
    return torch.sigmoid(model(X_t)).cpu().numpy()


def get_mlp_params(trial: optuna.Trial) -> dict:
    """Return Optuna-suggested MLP hyperparameters."""
    return {
        "n_layers": trial.suggest_int("n_layers", 1, 3, step=1),
        "hidden_size": trial.suggest_categorical("hidden_size", [64, 128, 256]),
        "dropout": trial.suggest_float("dropout", 0.1, 0.5, log=False),
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [128, 256]),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
    }


def objective(trial, X_train, y_train, X_val, y_val) -> float:
    """Optuna objective: combined val_auc − penalty * overfit_gap."""
    torch.manual_seed(RANDOM_STATE)
    params = get_mlp_params(trial)

    X_tr_t, y_tr_t = _to_tensors(X_train, y_train)
    X_val_t, _ = _to_tensors(X_val, y_val)

    loader = DataLoader(
        TensorDataset(X_tr_t, y_tr_t),
        batch_size=params["batch_size"],
        shuffle=True,
    )

    model = MLP(
        X_tr_t.shape[1], params["hidden_size"], params["n_layers"], params["dropout"]
    )
    model.to(DEVICE)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"]
    )
    criterion = nn.BCEWithLogitsLoss()

    for _ in range(OPTUNA_EPOCHS):
        _train_loop(model, optimizer, criterion, loader)

    train_proba = _predict_proba(model, X_tr_t)
    val_proba = _predict_proba(model, X_val_t)

    train_auc = roc_auc_score(y_train, train_proba)
    val_auc = roc_auc_score(y_val, val_proba)
    overfit_gap = max(0.0, train_auc - val_auc)
    return val_auc - OVERFIT_PENALTY * overfit_gap


def cv_objective(trial, X, y, n_splits=5) -> float:
    """Optuna objective with Stratified K-Fold CV for MLP."""
    torch.manual_seed(RANDOM_STATE)
    params = get_mlp_params(trial)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    y_arr = np.asarray(y)
    val_aucs, train_aucs = [], []
    for train_idx, val_idx in skf.split(X, y):
        X_tr_t, y_tr_t = _to_tensors(X.iloc[train_idx], y_arr[train_idx])
        X_vl_t, _ = _to_tensors(X.iloc[val_idx], y_arr[val_idx])

        loader = DataLoader(
            TensorDataset(X_tr_t, y_tr_t), batch_size=params["batch_size"], shuffle=True,
        )
        model = MLP(X_tr_t.shape[1], params["hidden_size"], params["n_layers"], params["dropout"])
        model.to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
        criterion = nn.BCEWithLogitsLoss()

        for _ in range(OPTUNA_EPOCHS):
            _train_loop(model, optimizer, criterion, loader)

        train_aucs.append(roc_auc_score(y_arr[train_idx], _predict_proba(model, X_tr_t)))
        val_aucs.append(roc_auc_score(y_arr[val_idx], _predict_proba(model, X_vl_t)))

    mean_val = np.mean(val_aucs)
    mean_gap = np.mean([t - v for t, v in zip(train_aucs, val_aucs)])
    return mean_val - OVERFIT_PENALTY * max(0.0, mean_gap)


class MLPSklearnWrapper:
    """Sklearn-compatible wrapper around a trained PyTorch MLP.
    Enables permutation_importance and other sklearn inspection utilities.
    """

    def __init__(self, model: MLP):
        self._model = model

    def predict_proba(self, X) -> np.ndarray:
        """Return [P(0), P(1)] for each sample."""
        arr = X.values if hasattr(X, "values") else X
        X_t = torch.tensor(arr, dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            proba = torch.sigmoid(self._model(X_t)).cpu().numpy()
        return np.column_stack([1 - proba, proba])

    def score(self, X, y) -> float:
        """Return ROC-AUC score (used by sklearn's permutation_importance)."""
        proba = self.predict_proba(X)[:, 1]
        y_arr = y.values if hasattr(y, "values") else np.asarray(y)
        return roc_auc_score(y_arr, proba)


def train_final(
    params: dict, X_train, y_train, X_val, y_val, epochs: int = FINAL_EPOCHS
):
    """Train with best params and early stopping; return (model, wrapper, loss_history).

    Training stops if val_loss doesn't improve for EARLY_STOPPING_PATIENCE epochs.
    The model with the lowest val_loss is restored before returning.
    """
    torch.manual_seed(RANDOM_STATE)

    X_tr_t, y_tr_t = _to_tensors(X_train, y_train)
    X_val_t, y_val_t = _to_tensors(X_val, y_val)

    loader = DataLoader(
        TensorDataset(X_tr_t, y_tr_t),
        batch_size=params["batch_size"],
        shuffle=True,
    )

    model = MLP(
        X_tr_t.shape[1], params["hidden_size"], params["n_layers"], params["dropout"]
    )
    model.to(DEVICE)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"]
    )
    criterion = nn.BCEWithLogitsLoss()

    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        tr_loss = _train_loop(model, optimizer, criterion, loader)
        val_loss = _eval_loss(model, criterion, X_val_t, y_val_t)
        train_losses.append(tr_loss)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            logger.info(
                "  MLP epoch %3d/%d — train_loss: %.4f  val_loss: %.4f  patience: %d/%d",
                epoch + 1,
                epochs,
                tr_loss,
                val_loss,
                patience_counter,
                EARLY_STOPPING_PATIENCE,
            )

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            logger.info(
                "  MLP early stopping at epoch %d (best val_loss: %.4f)",
                epoch + 1,
                best_val_loss,
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    history = {"train_loss": train_losses, "val_loss": val_losses}
    return model, MLPSklearnWrapper(model), history
