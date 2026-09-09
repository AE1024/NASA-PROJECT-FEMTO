# arxiv.org/abs/2201.01769 — Weibull-Loss NN with Optuna + MLflow
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
import mlflow
import mlflow.pytorch
import optuna
import polars as pl
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import r2_score, mean_squared_error

from model.nn_model import RULnet

# ── Paths ─────────────────────────────────────────────────────────────────────
TRAIN_PATH = "C:\\Users\\elmaz\\AIProject\\NASA_Project\\data\\processed_train_data\\train.parquet"
VAL_PATH = "C:\\Users\\elmaz\\AIProject\\NASA_Project\\data\\processed_val_test\\val\\val.parquet"
TEST_PATH  = "C\\Users\\elmaz\\AIProject\\NASA_Project\\data\\processed_val_test\\test\\test.parquet"

META_COLS  = {"bearing_id", "step", "RUL_pct"}

# ── MLflow ────────────────────────────────────────────────────────────────────
MLFLOW_URI= "http://127.0.0.1:5000" 
EXPERIMENT_NAME = "RUL_Weibull_NN"
_LAMBDA_CANDIDATES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]

# ── Weibull Loss ──────────────────────────────────────────────────────────────
# TODO: implement WeibullLoss(beta, eta, lambda_w)
class WeibullLoss(nn.Module):
    def __init__(self, lambda_w: float,beta=1.7801, eta=1425.6653):
        super().__init__()
        if lambda_w not in _LAMBDA_CANDIDATES:
            raise ValueError(f"Wrong lambda : {_LAMBDA_CANDIDATES}")
        self.beta = beta
        self.eta = (eta / 2803) * 100
        self.lambda_w = lambda_w

    def forward(self ,predictions,targets):
        #β=1.7801  η=1425.6653 : notebooks/val_test.ipynb
        #calculating weibull cdf in pytorch format
        mse_loss = F.mse_loss(predictions, targets)

        F_pred = 1 - torch.exp(-((predictions / 100) / self.eta) ** self.beta)
        F_true = 1 - torch.exp(-((targets / 100) / self.eta) ** self.beta)
        weibull_term = torch.mean(torch.abs(F_pred - F_true))

        hybrid_loss  = mse_loss + self.lambda_w * weibull_term

        return hybrid_loss

# ── Data loading ──────────────────────────────────────────────────────────────
def load_tensors(path: str, device: torch.device) -> TensorDataset:
    df = pd.read_parquet(path)
    drop = [c for c in META_COLS if c in df.columns]
    X = df.drop(columns=drop + ["RUL_pct"]).values.astype("float32")
    y = df["RUL_pct"].values.astype("float32")
    return TensorDataset(
        torch.tensor(X, device=device),
        torch.tensor(y, device=device),
    )

# ── Train loop ────────────────────────────────────────────────────────────────
# TODO: train_one_epoch(model, loader, optimizer, loss_fn, device) → float

# ── Evaluation ────────────────────────────────────────────────────────────────
# TODO: evaluate(model, loader, device) → dict(R2, RMSE, MAE)

# ── Optuna objective ──────────────────────────────────────────────────────────
# TODO: objective(trial) → val_rmse
#   trial.suggest_int("n_layers", 2, 7)
#   trial.suggest_categorical("n_units", [16, 32, 64, 128, 256])
#   trial.suggest_categorical("dropout_rate", [0.1, 0.2, 0.25, 0.4, 0.5, 0.6])
#   trial.suggest_categorical("lr", [0.1, 0.01, 0.001, 0.0001])
#   trial.suggest_categorical("batch_size", [32, 64, 128, 256, 512])
#   trial.suggest_float("lambda_weibull", 0.0, 3.0)
#   trial.suggest_categorical("activation", ["relu", "tanh", "leaky_relu"])

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    study = optuna.create_study(direction="minimize")
    # study.optimize(objective, n_trials=50)
