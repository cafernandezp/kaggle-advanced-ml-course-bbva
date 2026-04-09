"""
PyTorch MLP with dropout: Optuna objective + final training with loss history.
Input must be fully numeric (use build_features_numeric from preprocessing.py).
"""
import numpy as np
import optuna
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

RANDOM_STATE = 42
OVERFIT_PENALTY = 0.5
OPTUNA_EPOCHS = 50
FINAL_EPOCHS = 150
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class MLP(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, n_layers: int, dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        for i in range(n_layers):
            in_size = input_size if i == 0 else hidden_size
            layers += [nn.Linear(in_size, hidden_size), nn.ReLU(), nn.Dropout(dropout)]
        layers.append(nn.Linear(hidden_size, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
    return {
        "n_layers":    trial.suggest_int(        "n_layers",     1,    4,               step=1),
        "hidden_size": trial.suggest_categorical( "hidden_size",  [64, 128, 256, 512]),
        "dropout":     trial.suggest_float(       "dropout",      0.1,  0.5,             log=False),
        "lr":          trial.suggest_float(       "lr",           1e-4, 1e-2,            log=True),
        "batch_size":  trial.suggest_categorical( "batch_size",   [64, 128, 256]),
        "weight_decay":trial.suggest_float(       "weight_decay", 1e-6, 1e-2,            log=True),
    }


def objective(trial, X_train, y_train, X_val, y_val) -> float:
    torch.manual_seed(RANDOM_STATE)
    params = get_mlp_params(trial)

    X_tr_t, y_tr_t = _to_tensors(X_train, y_train)
    X_val_t, y_val_t = _to_tensors(X_val, y_val)

    loader = DataLoader(
        TensorDataset(X_tr_t, y_tr_t),
        batch_size=params["batch_size"],
        shuffle=True,
    )

    model = MLP(X_tr_t.shape[1], params["hidden_size"], params["n_layers"], params["dropout"])
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
    criterion = nn.BCEWithLogitsLoss()

    for _ in range(OPTUNA_EPOCHS):
        _train_loop(model, optimizer, criterion, loader)

    train_proba = _predict_proba(model, X_tr_t)
    val_proba   = _predict_proba(model, X_val_t)

    train_auc = roc_auc_score(y_train, train_proba)
    val_auc   = roc_auc_score(y_val,   val_proba)
    overfit_gap = max(0.0, train_auc - val_auc)
    return val_auc - OVERFIT_PENALTY * overfit_gap


def train_final(params: dict, X_train, y_train, X_val, y_val, epochs: int = FINAL_EPOCHS):
    """Train with best params, return (model, loss_history).

    loss_history = {"train_loss": [...], "val_loss": [...]} per epoch
    """
    torch.manual_seed(RANDOM_STATE)

    X_tr_t, y_tr_t   = _to_tensors(X_train, y_train)
    X_val_t, y_val_t = _to_tensors(X_val, y_val)

    loader = DataLoader(
        TensorDataset(X_tr_t, y_tr_t),
        batch_size=params["batch_size"],
        shuffle=True,
    )

    model = MLP(X_tr_t.shape[1], params["hidden_size"], params["n_layers"], params["dropout"])
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
    criterion = nn.BCEWithLogitsLoss()

    train_losses, val_losses = [], []
    for epoch in range(epochs):
        tr_loss = _train_loop(model, optimizer, criterion, loader)
        val_loss = _eval_loss(model, criterion, X_val_t, y_val_t)
        train_losses.append(tr_loss)
        val_losses.append(val_loss)
        if (epoch + 1) % 25 == 0:
            print(f"  [MLP] epoch {epoch+1:3d}/{epochs} — train_loss: {tr_loss:.4f}  val_loss: {val_loss:.4f}")

    history = {"train_loss": train_losses, "val_loss": val_losses}
    return model, history
