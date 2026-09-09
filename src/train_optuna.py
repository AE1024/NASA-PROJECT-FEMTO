"""
Optuna + MLflow hyperparameter search for RUL prediction NN.
Reference: von Hahn & Mechefske (2022) arXiv:2201.01769
"""

import mlflow
import optuna
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.nn_model import RULnet, WeibullLoss

# ── Paths ──────────────────────────────────────────────────────────────────────
TRAIN_PATH = r"C:\Users\elmaz\AIProject\NASA_Project\data\processed_train_data\train.parquet"
VAL_PATH   = r"C:\Users\elmaz\AIProject\NASA_Project\data\processed_val_test\val\val.parquet"

# ── MLflow ─────────────────────────────────────────────────────────────────────
MLFLOW_URI      = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "RUL_Weibull_NN"

# ── Constants ──────────────────────────────────────────────────────────────────
META_COLS   = {"bearing_id", "step", "RUL_pct"}
INPUT_DIM   = 21
N_TRIALS    = 50
N_EPOCHS    = 100
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Data helpers ───────────────────────────────────────────────────────────────

def load_tensors(path: str, device: torch.device) -> TensorDataset:
    df = pd.read_parquet(path)
    y  = torch.tensor(df["RUL_pct"].values, dtype=torch.float32, device=device)
    X  = torch.tensor(
        df.drop(columns=list(META_COLS), errors="ignore").values,
        dtype=torch.float32,
        device=device,
    )
    return TensorDataset(X, y)


# ── Training helpers ───────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        preds = model(X_batch)
        loss  = loss_fn(preds, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
    return total_loss / len(loader.dataset)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    preds_all, targets_all = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            preds_all.append(model(X_batch.to(device)).cpu())
            targets_all.append(y_batch.cpu())
    preds   = torch.cat(preds_all)
    targets = torch.cat(targets_all)

    mse  = torch.mean((preds - targets) ** 2).item()
    rmse = mse ** 0.5
    ss_res = torch.sum((targets - preds) ** 2).item()
    ss_tot = torch.sum((targets - targets.mean()) ** 2).item()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"rmse": rmse, "mse": mse, "r2": r2}


# ── Optuna objective ───────────────────────────────────────────────────────────

def objective(trial: optuna.Trial, train_ds: TensorDataset, val_ds: TensorDataset) -> float:
    # Hyperparameter search space (Table 2 of paper + λ)
    n_layers     = trial.suggest_int("n_layers",     2,    7)
    n_units      = trial.suggest_categorical("n_units", [16, 32, 64, 128, 256])
    dropout_rate = trial.suggest_categorical("dropout_rate", [0.1, 0.2, 0.25, 0.4, 0.5])
    lr           = trial.suggest_categorical("lr", [0.1, 0.01, 0.001, 0.0001])
    batch_size   = trial.suggest_categorical("batch_size", [32, 64, 128, 256, 512])
    activation   = trial.suggest_categorical("activation", ["relu", "tanh", "leaky_relu", "silu"])
    # λ continuous in [0, 3] as per paper (0 = pure MSE baseline)
    lambda_w     = trial.suggest_float("lambda_w", 0.0, 3.0)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=512,        shuffle=False)

    model    = RULnet(
        input_dim=INPUT_DIM,
        n_layers=n_layers,
        n_units=n_units,
        dropout_rate=dropout_rate,
        activation=activation,
    ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn   = WeibullLoss(lambda_w=lambda_w)

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=f"trial_{trial.number}"):
        mlflow.log_params({
            "n_layers": n_layers, "n_units": n_units,
            "dropout_rate": dropout_rate, "lr": lr,
            "batch_size": batch_size, "activation": activation,
            "lambda_w": lambda_w,
        })

        best_val_rmse = float("inf")
        for epoch in range(N_EPOCHS):
            train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, DEVICE)
            val_metrics = evaluate(model, val_loader, DEVICE)

            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_rmse":   val_metrics["rmse"],
                "val_r2":     val_metrics["r2"],
            }, step=epoch)

            if val_metrics["rmse"] < best_val_rmse:
                best_val_rmse = val_metrics["rmse"]

            # Optuna pruning
            trial.report(val_metrics["rmse"], epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        mlflow.log_metric("best_val_rmse", best_val_rmse)

    return best_val_rmse


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Load once — shared across all trials
    print("Loading datasets...")
    train_ds = load_tensors(TRAIN_PATH, DEVICE)
    val_ds   = load_tensors(VAL_PATH,   DEVICE)
    print(f"  Train: {len(train_ds)} samples | Val: {len(val_ds)} samples")

    sampler = optuna.samplers.TPESampler(seed=42)
    pruner  = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=20)

    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        study_name=EXPERIMENT_NAME,
    )
    study.optimize(
        lambda trial: objective(trial, train_ds, val_ds),
        n_trials=N_TRIALS,
        show_progress_bar=True,
    )

    print("\n=== Best Trial ===")
    best = study.best_trial
    print(f"  Val RMSE : {best.value:.4f}")
    print(f"  Params   : {best.params}")
