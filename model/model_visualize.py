# Visualize Optuna study results and best-model predictions

import json

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import optuna
import torch
from torch.utils.data import DataLoader

from model.nn_model import RULnet
from model.train_optuna import (
    BEST_MODEL_PATH,
    BEST_PARAMS_PATH,
    DEVICE,
    EXPERIMENT_NAME,
    MLFLOW_URI,
    MODEL_DIR,
    STUDY_DB_PATH,
    TEST_PATH,
    VAL_PATH,
    load_tensors,
)


def load_best_model(input_dim: int, params: dict) -> RULnet:
    """Rebuild the winning architecture and load its trained weights."""
    checkpoint = torch.load(BEST_MODEL_PATH, weights_only=False)
    model = RULnet(
        input_dim=input_dim,
        n_layers=params["n_layers"],
        n_units=params["n_units"],
        dropout_rate=params["dropout_rate"],
        activation=params["activation"],
    ).to(DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def predict(model: RULnet, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    preds_all, targets_all = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            preds_all.append(model(X_batch.to(DEVICE)).cpu().numpy())
            targets_all.append(y_batch.numpy())
    return np.concatenate(preds_all), np.concatenate(targets_all)


def plot_study(study: optuna.Study, out_path) -> None:
    """Trial-by-trial val RMSE with running best, plus hyperparameter importances."""
    # Pruned trials also carry a value (their last intermediate report), so filter
    # on state rather than on value to keep the two groups distinct.
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    numbers   = [t.number for t in completed]
    values    = [t.value for t in completed]
    best_so_far = np.minimum.accumulate(values)

    pruned = [t.number for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].scatter(numbers, values, alpha=0.6, s=30, label="Completed trial")
    axes[0].plot(numbers, best_so_far, color="crimson", linewidth=2, label="Best so far")
    for p in pruned:
        axes[0].axvline(p, color="grey", alpha=0.15, linewidth=1)
    axes[0].set_xlabel("Trial")
    axes[0].set_ylabel("Val RMSE")
    axes[0].set_title(f"Optuna progression ({len(completed)} completed, {len(pruned)} pruned)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    try:
        importances = optuna.importance.get_param_importances(study)
        names  = list(importances.keys())[::-1]
        scores = [importances[n] for n in names]
        axes[1].barh(names, scores, color="steelblue")
        axes[1].set_xlabel("Relative importance")
        axes[1].set_title("Hyperparameter importance")
        axes[1].grid(True, alpha=0.3, axis="x")
    except (ValueError, RuntimeError) as exc:
        axes[1].text(0.5, 0.5, f"Importance unavailable:\n{exc}",
                     ha="center", va="center", transform=axes[1].transAxes)
        axes[1].set_axis_off()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"  saved {out_path}")


def plot_predictions(model: RULnet, val_loader, test_loader, out_path) -> None:
    """Predicted vs true RUL scatter for validation and test splits."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    for ax, (name, loader) in zip(axes, [("Validation", val_loader), ("Test", test_loader)]):
        preds, targets = predict(model, loader)
        rmse = float(np.sqrt(np.mean((preds - targets) ** 2)))
        ss_res = float(np.sum((targets - preds) ** 2))
        ss_tot = float(np.sum((targets - targets.mean()) ** 2))
        r2 = 1 - ss_res / ss_tot

        ax.scatter(targets, preds, alpha=0.25, s=8, edgecolors="none")
        ax.plot([0, 100], [0, 100], "r--", linewidth=1.5, label="Perfect prediction")
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_xlabel("True RUL_pct")
        ax.set_ylabel("Predicted RUL_pct")
        ax.set_title(f"{name} — RMSE={rmse:.2f}, R2={r2:.3f}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"  saved {out_path}")


def plot_training_curves(best_trial_number: int, out_path) -> None:
    """Per-epoch train loss and val RMSE for the winning trial, read from MLflow."""
    mlflow.set_tracking_uri(MLFLOW_URI)
    client = mlflow.tracking.MlflowClient()

    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        print("  MLflow experiment not found — skipping training curves")
        return

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.optuna_trial = '{best_trial_number}'",
        max_results=1,
    )
    if not runs:
        print(f"  No MLflow run tagged optuna_trial={best_trial_number} — skipping curves")
        return

    run_id = runs[0].info.run_id
    train_hist = client.get_metric_history(run_id, "train_loss")
    val_hist   = client.get_metric_history(run_id, "val_rmse")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    axes[0].plot([m.step for m in train_hist], [m.value for m in train_hist], color="darkorange")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Hybrid loss")
    axes[0].set_title(f"Trial {best_trial_number} — train loss")
    axes[0].grid(True, alpha=0.3)

    steps  = [m.step for m in val_hist]
    values = [m.value for m in val_hist]
    axes[1].plot(steps, values, color="steelblue")
    if values:
        best_idx = int(np.argmin(values))
        axes[1].scatter([steps[best_idx]], [values[best_idx]], color="crimson", zorder=5,
                        label=f"Best: epoch {steps[best_idx]} ({values[best_idx]:.2f})")
        axes[1].legend()
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Val RMSE")
    axes[1].set_title(f"Trial {best_trial_number} — validation RMSE")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"  saved {out_path}")


def main() -> None:
    with open(BEST_PARAMS_PATH, "r", encoding="utf-8") as f:
        results = json.load(f)

    print("Best configuration:")
    for key, value in results["params"].items():
        print(f"  {key:14s}: {value}")
    print(f"  val  RMSE={results['val_metrics']['rmse']:.4f}  R2={results['val_metrics']['r2']:.4f}")
    print(f"  test RMSE={results['test_metrics']['rmse']:.4f}  R2={results['test_metrics']['r2']:.4f}")

    val_ds  = load_tensors(VAL_PATH)
    test_ds = load_tensors(TEST_PATH)
    model   = load_best_model(results["input_dim"], results["params"])

    print("\nGenerating plots...")
    study = optuna.load_study(
        study_name=EXPERIMENT_NAME,
        storage=f"sqlite:///{STUDY_DB_PATH.as_posix()}",
    )
    plot_study(study, MODEL_DIR / "optuna_progression.png")
    plot_predictions(
        model,
        DataLoader(val_ds,  batch_size=512),
        DataLoader(test_ds, batch_size=512),
        MODEL_DIR / "predictions.png",
    )
    plot_training_curves(results["best_trial"], MODEL_DIR / "training_curves.png")


if __name__ == "__main__":
    main()
