# arxiv.org/abs/2201.01769 — Weibull-Loss NN with Optuna + MLflow

import copy
import json
import sys
from pathlib import Path

import mlflow
import mlflow.pytorch
import optuna
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import DataLoader, TensorDataset

from model.nn_model import RULnet

# MLflow prints run URLs containing emoji; a non-UTF-8 console (e.g. cp1254 on
# Turkish Windows) raises UnicodeEncodeError mid-trial and kills the study.
sys.stdout.reconfigure(errors="replace")
sys.stderr.reconfigure(errors="replace")

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR    = PROJECT_ROOT / "model"

TRAIN_PATH = PROJECT_ROOT / "data" / "processed_train_data" / "train.parquet"
VAL_PATH   = PROJECT_ROOT / "data" / "processed_val_test" / "val"  / "val.parquet"
TEST_PATH  = PROJECT_ROOT / "data" / "processed_val_test" / "test" / "test.parquet"

BEST_PARAMS_PATH = MODEL_DIR / "best_params.json"
BEST_MODEL_PATH  = MODEL_DIR / "best_model.pt"
STUDY_DB_PATH    = MODEL_DIR / "optuna_study.db"

META_COLS = {"bearing_id", "step", "RUL_pct"}

# ── MLflow ─────────────────────────────────────────────────────────────────────
# v2 separates this corrected run from the earlier runs, whose reported metric did
# not correspond to the weights left in memory (no best-epoch checkpointing).
MLFLOW_URI      = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "RUL_Weibull_NN_v2"

# ── Constants ──────────────────────────────────────────────────────────────────
N_EPOCHS  = 100
N_TRIALS  = 50
SEED      = 42
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Tracks the best trial across the whole study so only the winning weights are saved.
_GLOBAL_BEST: dict = {"rmse": float("inf"), "trial": None, "params": None}


# ── Weibull Loss ───────────────────────────────────────────────────────────────
class WeibullLoss(nn.Module):
    """
    Hybrid loss: L = MSE + lambda_w * mean|F(pred) - F(true)|

    Weibull CDF: F(t) = 1 - exp(-(t / eta)^beta)

    beta and eta are fit from train bearing lifetimes expressed as LifePercentage:
        lifetimes_pct = [18.41, 28.47, 31.11, 32.54, 58.44, 100.0]
        weibull_min.fit(lifetimes_pct, floc=0) -> beta=1.7801, eta=50.8620

    Both predictions and targets are LifePercentage in [0, 100], matching eta's units.

    Reference: eq. (4-5) in arXiv:2201.01769
    """

    def __init__(self, lambda_w: float, beta: float = 1.7801, eta: float = 50.8620):
        super().__init__()
        self.lambda_w = lambda_w
        self.beta     = beta
        self.eta      = eta

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        mse_loss = F.mse_loss(predictions, targets)
        if self.lambda_w == 0.0:
            return mse_loss
        t_pred = torch.clamp(predictions, min=1e-6)
        t_true = torch.clamp(targets,     min=1e-6)
        F_pred = 1 - torch.exp(-((t_pred / self.eta) ** self.beta))
        F_true = 1 - torch.exp(-((t_true / self.eta) ** self.beta))
        weibull_term = torch.mean(torch.abs(F_pred - F_true))
        return mse_loss + self.lambda_w * weibull_term


# ── Data loading ───────────────────────────────────────────────────────────────
def load_tensors(path: str | Path) -> TensorDataset:
    """Read a parquet file and return a TensorDataset of (X, y) on CPU.

    Tensors stay on CPU so DataLoader workers and pinned memory remain usable;
    batches are moved to the compute device inside the training/eval loops.

    Drops META_COLS (bearing_id, step, RUL_pct) from features;
    y is RUL_pct (LifePercentage in [0, 100]).
    """
    df = pd.read_parquet(path)
    feature_cols = [c for c in df.columns if c not in META_COLS]
    X = df[feature_cols].values.astype("float32")
    y = df["RUL_pct"].values.astype("float32")
    return TensorDataset(torch.from_numpy(X), torch.from_numpy(y))


# ── Train loop ─────────────────────────────────────────────────────────────────
def train_one_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn:   nn.Module,
    device:    torch.device,
) -> float:
    """Run one full pass over the training DataLoader. Returns mean loss."""
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        pred = model(X_batch)
        loss = loss_fn(pred, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
    return total_loss / len(loader.dataset)


# ── Evaluation ─────────────────────────────────────────────────────────────────
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    """Evaluate model on a DataLoader. Returns {"rmse": float, "r2": float}."""
    model.eval()
    preds_all, targets_all = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            preds_all.append(model(X_batch).cpu())
            targets_all.append(y_batch)
    preds   = torch.cat(preds_all).numpy()
    targets = torch.cat(targets_all).numpy()
    return {
        "rmse": float(mean_squared_error(targets, preds) ** 0.5),
        "r2":   float(r2_score(targets, preds)),
    }


def build_optimizer(name: str, model: nn.Module, lr: float) -> torch.optim.Optimizer:
    if name == "adam":
        return optim.Adam(model.parameters(), lr=lr)
    if name == "rmsprop":
        return optim.RMSprop(model.parameters(), lr=lr)
    return optim.SGD(model.parameters(), lr=lr)


# ── Optuna objective ───────────────────────────────────────────────────────────
def objective(
    trial:     optuna.Trial,
    train_ds:  TensorDataset,
    val_ds:    TensorDataset,
    input_dim: int,
) -> float:
    """Single Optuna trial: sample hyperparams, train N_EPOCHS, return best val RMSE.

    Search space follows Table 2 of arXiv:2201.01769 with continuous lambda_w.
    Weights from the best-scoring epoch are restored before the trial returns, so
    the reported metric and the persisted model always correspond to the same epoch.
    Each trial is logged as a separate MLflow run.
    """
    # Same init seed for every trial so hyperparameters are compared fairly.
    torch.manual_seed(SEED)

    n_layers  = trial.suggest_int("n_layers", 2, 7)
    n_units   = trial.suggest_categorical("n_units", [16, 32, 64, 128, 256])
    dropout_rate   = trial.suggest_categorical("dropout_rate", [0.1, 0.2, 0.25, 0.4, 0.5, 0.6])
    lr   = trial.suggest_categorical("lr", [0.1, 0.01, 0.001, 0.0001])
    batch_size     = trial.suggest_categorical("batch_size", [32, 64, 128, 256, 512])
    lambda_w    = trial.suggest_float("lambda_w", 0.0, 3.0)
    activation     = trial.suggest_categorical("activation", ["relu", "tanh", "leaky_relu"])
    optimizer_name = trial.suggest_categorical("optimizer", ["adam", "sgd", "rmsprop"])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=512,        shuffle=False)

    model = RULnet(
        input_dim=input_dim,
        n_layers=n_layers,
        n_units=n_units,
        dropout_rate=dropout_rate,
        activation=activation,
    ).to(DEVICE)

    opt = build_optimizer(optimizer_name, model, lr)
    loss_fn = WeibullLoss(lambda_w=lambda_w)

    best_val_rmse = float("inf")
    best_val_r2   = float("-inf")
    best_epoch  = -1
    best_state  = None
    pruned  = False

    with mlflow.start_run(run_name=f"trial_{trial.number}"):
        mlflow.set_tag("optuna_trial", trial.number)
        mlflow.log_params({
            "n_layers": n_layers, "n_units": n_units,
            "dropout_rate": dropout_rate, "lr": lr,
            "batch_size": batch_size, "lambda_w": lambda_w,
            "activation": activation, "optimizer": optimizer_name,
        })

        for epoch in range(N_EPOCHS):
            train_loss  = train_one_epoch(model, train_loader, opt, loss_fn, DEVICE)
            val_metrics = evaluate(model, val_loader, DEVICE)

            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_rmse":   val_metrics["rmse"],
                "val_r2":     val_metrics["r2"],
            }, step=epoch)

            if val_metrics["rmse"] < best_val_rmse:
                best_val_rmse = val_metrics["rmse"]
                best_val_r2   = val_metrics["r2"]
                best_epoch    = epoch
                best_state    = copy.deepcopy(model.state_dict())

            # Report to Optuna pruner; stop the loop if clearly underperforming.
            # The exception is raised after the MLflow run closes cleanly.
            trial.report(val_metrics["rmse"], epoch)
            if trial.should_prune():
                pruned = True
                break

        mlflow.log_metrics({
            "best_val_rmse": best_val_rmse,
            "best_val_r2":   best_val_r2,
            "best_epoch":    best_epoch,
        })
        mlflow.set_tag("pruned", str(pruned))

        # Restore the weights that produced best_val_rmse.
        if best_state is not None:
            model.load_state_dict(best_state)

        # Persist only the study-wide best model to keep artifact size down.
        if not pruned and best_val_rmse < _GLOBAL_BEST["rmse"]:
            _GLOBAL_BEST.update({
                "rmse":   best_val_rmse,
                "r2":     best_val_r2,
                "trial":  trial.number,
                "epoch":  best_epoch,
                "params": dict(trial.params),
            })
            torch.save(
                {"state_dict": model.state_dict(), "params": dict(trial.params)},
                BEST_MODEL_PATH,
            )
            # Pickle keeps the artifact a real nn.Module. The default pt2 format
            # traces the graph against input_example and specializes the batch
            # dimension to its size, which makes batched inference fail later.
            mlflow.pytorch.log_model(
                model,
                name="model",
                serialization_format=mlflow.pytorch.SERIALIZATION_FORMAT_PICKLE,
                input_example=val_ds.tensors[0][:4].numpy(),
            )
            mlflow.set_tag("global_best", "true")

    if pruned:
        raise optuna.TrialPruned()

    return best_val_rmse


# ── Entry point ────────────────────────────────────────────────────────────────
def main() -> None:
    torch.manual_seed(SEED)

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    print("Loading datasets...")
    train_ds = load_tensors(TRAIN_PATH)
    val_ds   = load_tensors(VAL_PATH)
    test_ds  = load_tensors(TEST_PATH)
    input_dim = train_ds.tensors[0].shape[1]
    print(
        f"  Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)} "
        f"| Features: {input_dim} | Device: {DEVICE}"
    )

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=20),
        study_name=EXPERIMENT_NAME,
        storage=f"sqlite:///{STUDY_DB_PATH.as_posix()}",
        load_if_exists=True,
    )
    study.optimize(
        lambda trial: objective(trial, train_ds, val_ds, input_dim),
        n_trials=N_TRIALS,
        show_progress_bar=True,
    )

    best = study.best_trial
    print("\n=== Best Trial ===")
    print(f"  Trial    : {best.number}")
    print(f"  Val RMSE : {best.value:.4f}")
    print(f"  Params   : {best.params}")

    # Final held-out evaluation using the persisted best weights.
    checkpoint = torch.load(BEST_MODEL_PATH, weights_only=False)
    final_model = RULnet(
        input_dim=input_dim,
        n_layers=best.params["n_layers"],
        n_units=best.params["n_units"],
        dropout_rate=best.params["dropout_rate"],
        activation=best.params["activation"],
    ).to(DEVICE)
    final_model.load_state_dict(checkpoint["state_dict"])

    val_metrics  = evaluate(final_model, DataLoader(val_ds,  batch_size=512), DEVICE)
    test_metrics = evaluate(final_model, DataLoader(test_ds, batch_size=512), DEVICE)
    print(f"  Val      : RMSE={val_metrics['rmse']:.4f}  R2={val_metrics['r2']:.4f}")
    print(f"  Test     : RMSE={test_metrics['rmse']:.4f}  R2={test_metrics['r2']:.4f}")

    results = {
        "best_trial":   best.number,
        "best_epoch":   _GLOBAL_BEST.get("epoch"),
        "params":    best.params,
        "val_metrics":  val_metrics,
        "test_metrics": test_metrics,
        "n_trials":  len(study.trials),
        "n_pruned":   len(study.get_trials(states=(optuna.trial.TrialState.PRUNED,))),
        "n_epochs": N_EPOCHS,
        "seed": SEED,
        "input_dim": input_dim,
    }
    with open(BEST_PARAMS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n  Weights  -> {BEST_MODEL_PATH}")
    print(f"  Results  -> {BEST_PARAMS_PATH}")
    print(f"  Study DB -> {STUDY_DB_PATH}")


if __name__ == "__main__":
    main()
