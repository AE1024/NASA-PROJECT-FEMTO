# FEMTO/PRONOSTIA Bearing RUL Prediction — Weibull-Loss Neural Network

Remaining Useful Life (RUL) prediction on the FEMTO-ST / PRONOSTIA bearing
vibration dataset, using a knowledge-informed loss function that blends MSE with
a Weibull-distribution reliability term. Implementation follows
[von Hahn & Mechefske (2022), "Knowledge Informed Machine Learning using a
Weibull-based Loss Function," arXiv:2201.01769](https://arxiv.org/abs/2201.01769).

## What this project does

Raw accelerometer signals (`ax`, `ay`, `mag`) from bearings run to failure under
three operating conditions (1800/1650/1500 RPM) are converted into 21 statistical
features (RMS, kurtosis, skew, rolling-slope trends, a PCA/Mahalanobis health
index, PELT-detected regime labels) and used to predict `RUL_pct` — a bearing's
remaining life as a percentage (100 = healthy, 0 = failure). A feedforward neural
network is trained with a hybrid loss:

```
L = MSE(pred, true) + lambda_w * (F(pred) - F(true))^2
```

where `F` is the Weibull CDF, parametrized by `beta` (fixed from reliability-
engineering literature for ball bearings) and `eta` (solved via the Weibayes
equation from train-set bearing lifetimes). Hyperparameters, including `lambda_w`,
are tuned with Optuna; every trial is tracked in MLflow.

## Pipeline

| Stage | Where |
|---|---|
| Raw data ingestion (HuggingFace `Amgharr/FEMTO-ST_DATASET`) | `notebooks/01_data_ingestion.ipynb` |
| Exploratory analysis | `notebooks/02_eda.ipynb` |
| Time-domain feature extraction | `notebooks/03_feature_extraction.ipynb`, `src/features.py` |
| Statistical validation (Weibull fit, stationarity, etc.) | `notebooks/04_statistical_validation.ipynb` |
| Feature selection (21 final features) | `notebooks/05_feature_selection.ipynb`, `notebooks/features.json` |
| Val/test dataset construction | `notebooks/06_val_test.ipynb`, `src/preprocessing.py` |
| Baseline models (Ridge, Random Forest) | `src/baseline_model_train.py` |
| NN architecture (`RULnet`) | `model/nn_model.py` |
| Training + Optuna + MLflow + Weibull loss | `model/train_optuna.py` |
| Result plots | `model/model_visualize.py` |

## Two experiments

`model/train_optuna.py` takes a dataset argument and trains a 50-trial Optuna
study for it:

```bash
uv run mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db
uv run python -m model.train_optuna paper_replica   # 9 bearings, mirrors the paper's Table 6
uv run python -m model.train_optuna full_data        # all 17 PRONOSTIA bearings, disjoint val/test
uv run python -m model.model_visualize                # regenerates all result plots
```

| | `paper_replica` | `full_data` |
|---|---|---|
| Bearings used | 9 (1 per condition per split, matches arXiv:2201.01769 Table 6) | 17 (all of them, disjoint train/val/test) |
| Val R² | **0.609** (paper: 0.266) | 0.456 |
| Test R² | 0.455 | **0.572** (paper: 0.654 — closest of the two) |

Full results, per-experiment details, and the paper comparison live in
[`COMPARISON.md`](COMPARISON.md), with per-experiment detail in
[`model/results/article/FINDINGS.md`](model/results/article/FINDINGS.md)
(paper replica) and
[`model/results/unified_data/FINDINGS.md`](model/results/unified_data/FINDINGS.md)
(full data).

## Notable finding: val/test bearing identity and leakage

The dataset's `val`/`test` split, as received, had no recorded link between its
generic `bearing_id` values and the underlying PRONOSTIA bearings — and turned out
to cover the *same* 11 physical bearings at two different truncation depths (one
side was the official PHM2012 `Test_set`, the other was `Full_Test_Set`), meaning
val and test were not independent, and the truncated side's `RUL_pct` labels were
computed against the wrong (truncated) lifetime. Both `val`/`test` were rebuilt
with bearing identities recovered by matching lifetimes against the official
[IEEE PHM 2012 Data Challenge repo](https://github.com/wkzs111/phm-ieee-2012-data-challenge-dataset),
producing disjoint, correctly-labeled splits. See
`data/bearing_identity_map.json` and [`COMPARISON.md`](COMPARISON.md) for the full
chronological list of bugs found and fixed.

## Setup

```bash
uv sync
```

Requires Python >=3.13 (see `pyproject.toml` for the full dependency list —
PyTorch, Optuna, MLflow, scikit-learn, ruptures, polars/pandas).

## Reference

Tim von Hahn and Chris K. Mechefske, "Knowledge Informed Machine Learning using a
Weibull-based Loss Function," arXiv:2201.01769 (2022).
[Paper's own code](https://github.com/tvhahn/weibull-knowledge-informed-ml).
