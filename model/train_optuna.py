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

# ── Dataset selection ──────────────────────────────────────────────────────────
# Two disjoint bearing splits live under data/experiments/ (see data/bearing_identity_map.json):
#   paper_replica -> arXiv:2201.01769 Table 6 exactly, 9 bearings, 1 per condition per split
#   full_data     -> all 17 PRONOSTIA bearings, disjoint val/test, maximizes sample size
_VALID_DATASETS = {"paper_replica", "full_data"}
DATASET = sys.argv[1] if len(sys.argv) > 1 else "full_data"
if DATASET not in _VALID_DATASETS:
    raise ValueError(f"Unknown dataset '{DATASET}'. Choose from {_VALID_DATASETS}")

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR    = PROJECT_ROOT / "model"
DATASET_DIR  = PROJECT_ROOT / "data" / "experiments" / DATASET
RESULTS_DIR  = MODEL_DIR / "results" / DATASET
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = DATASET_DIR / "train.parquet"
VAL_PATH   = DATASET_DIR / "val.parquet"
TEST_PATH  = DATASET_DIR / "test.parquet"

BEST_PARAMS_PATH = RESULTS_DIR / "best_params.json"
BEST_MODEL_PATH  = RESULTS_DIR / "best_model.pt"
STUDY_DB_PATH    = RESULTS_DIR / "optuna_study.db"

META_COLS = {"bearing_id", "step", "RUL_pct"}

# ── MLflow ─────────────────────────────────────────────────────────────────────
# v4 separates this run from v3, whose WeibullLoss computed MSE on the raw [0, 100]
# scale; the Weibull CDF term (always in [0, 1]) was ~1000x smaller and had no real
# effect on training regardless of lambda_w. v3 also predates the val/test bearing
# leakage fix (data/bearing_identity_map.json), so its results aren't reusable either.
MLFLOW_URI      = "http://127.0.0.1:5000"
EXPERIMENT_NAME = f"RUL_Weibull_NN_v4_{DATASET}"

# ── Constants ──────────────────────────────────────────────────────────────────
# arXiv:2201.01769 lets early stopping run PRONOSTIA models for up to ~1950 epochs
# (median 98, 75th pct 273 for Weibull-loss models, their Table 10). N_EPOCHS is a
# hard ceiling; EARLY_STOP_PATIENCE lets a converged trial stop well before it.
N_EPOCHS             = 300
EARLY_STOP_PATIENCE  = 30
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

    beta is NOT fit from data. With only 6 train-bearing failures, a joint 2-parameter
    MLE fit (the previous scipy.stats.weibull_min.fit approach) is exactly the unstable
    small-sample case arXiv:2201.01769 Section 1.2.2 warns against. Following the paper
    (Table 1), beta is fixed from reliability-engineering literature for ball bearings
    (Abernethy's Weibull Handbook): beta=2.0. Only eta is then estimated from data via
    the Weibayes equation (paper Eq. 2):
        eta = (sum(t_i ** beta) / r) ** (1 / beta)
    using train lifetimes t_i = [515, 797, 871, 911, 1637, 2803] steps, r=6 complete
    failures (no censoring) -> eta = 1473.41 steps.

    eta is rescaled from steps to LifePercentage units (divide by the longest train
    lifetime, 2803 steps, times 100) to match the [0, 100] scale of predictions/targets:
    eta = 52.5654.

    MSE is computed on predictions/targets normalized to [0, 1], not the raw [0, 100]
    scale. F(t) is always a probability in [0, 1] (~0.001-0.5 in practice), so an MSE
    computed on [0, 100] values (~400-2500) outweighs it by ~1000x -- lambda_w could
    not meaningfully move the loss anywhere in its [0, 3] search range. Table 8/9 of
    arXiv:2201.01769 show the paper's own RMSE (~0.13-0.26) is on this same [0, 1]
    scale, so this normalization also matches their convention. eta/beta still operate
    on the original [0, 100] scale, since eta was derived in those units.

    The Weibull term is squared (F(pred)-F(true))^2, not |F(pred)-F(true)|, matching
    Eq. 5 exactly. Squaring (vs. absolute value) also keeps it the same order of
    magnitude as the now-normalized MSE term (both ~1e-3 to 1e-2 near convergence),
    instead of the absolute-value form which stayed ~10-25x larger than MSE.

    Reference: eq. (1-2, 4-5) and Table 1 in arXiv:2201.01769
    """

    def __init__(self, lambda_w: float, beta: float = 2.0, eta: float = 52.5654):
        super().__init__()
        self.lambda_w = lambda_w
        self.beta     = beta
        self.eta      = eta

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        mse_loss = F.mse_loss(predictions / 100.0, targets / 100.0)
        if self.lambda_w == 0.0:
            return mse_loss
        t_pred = torch.clamp(predictions, min=1e-6)
        t_true = torch.clamp(targets,     min=1e-6)
        F_pred = 1 - torch.exp(-((t_pred / self.eta) ** self.beta))
        F_true = 1 - torch.exp(-((t_true / self.eta) ** self.beta))
        weibull_term = torch.mean((F_pred - F_true) ** 2)
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

    n_layers      = trial.suggest_int("n_layers", 2, 7)
    n_units       = trial.suggest_categorical("n_units", [16, 32, 64, 128, 256])
    dropout_rate  = trial.suggest_categorical("dropout_rate", [0.1, 0.2, 0.25, 0.4, 0.5, 0.6])
    lr            = trial.suggest_categorical("lr", [0.1, 0.01, 0.001, 0.0001])
    batch_size    = trial.suggest_categorical("batch_size", [32, 64, 128, 256, 512])
    lambda_w      = trial.suggest_float("lambda_w", 0.0, 3.0)
    activation    = trial.suggest_categorical("activation", ["relu", "tanh", "leaky_relu"])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=512,        shuffle=False)

    model = RULnet(
        input_dim=input_dim,
        n_layers=n_layers,
        n_units=n_units,
        dropout_rate=dropout_rate,
        activation=activation,
    ).to(DEVICE)

    # arXiv:2201.01769 uses Adam exclusively (Section 3); optimizer choice was not
    # part of their random search, so it is fixed here too rather than tuned.
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = WeibullLoss(lambda_w=lambda_w)

    best_val_rmse = float("inf")
    best_val_r2   = float("-inf")
    best_epoch    = -1
    best_state    = None
    pruned        = False
    epochs_since_improvement = 0

    with mlflow.start_run(run_name=f"trial_{trial.number}"):
        mlflow.set_tag("optuna_trial", trial.number)
        mlflow.log_params({
            "n_layers": n_layers, "n_units": n_units,
            "dropout_rate": dropout_rate, "lr": lr,
            "batch_size": batch_size, "lambda_w": lambda_w,
            "activation": activation, "optimizer": "adam",
            "early_stop_patience": EARLY_STOP_PATIENCE,
        })

        stopped_epoch = N_EPOCHS - 1
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
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1

            # Report to Optuna pruner; kills trials that are clearly underperforming
            # relative to other trials, even before they've had time to converge.
            trial.report(val_metrics["rmse"], epoch)
            if trial.should_prune():
                pruned = True
                stopped_epoch = epoch
                break

            # Early stopping (arXiv:2201.01769, Section 3: "Early stopping, based on
            # the validation loss, was used to prevent overfitting"). This is a
            # per-trial convergence check, distinct from the pruner above, which
            # compares a trial against the other trials in the study.
            if epochs_since_improvement >= EARLY_STOP_PATIENCE:
                stopped_epoch = epoch
                break

        mlflow.log_metrics({
            "best_val_rmse": best_val_rmse,
            "best_val_r2":   best_val_r2,
            "best_epoch":    best_epoch,
            "stopped_epoch": stopped_epoch,
        })
        mlflow.set_tag("pruned", str(pruned))
        mlflow.set_tag(
            "early_stopped",
            str(not pruned and stopped_epoch < N_EPOCHS - 1),
        )

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

    print(f"Dataset: {DATASET}  ({DATASET_DIR})")
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
