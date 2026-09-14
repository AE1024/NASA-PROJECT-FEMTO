# Paper Replica

A 9-bearing train/val/test split that mirrors
[arXiv:2201.01769](https://arxiv.org/abs/2201.01769)'s Table 6 exactly — one
bearing per operating condition, per split — so the result below is directly
comparable to the paper's own reported numbers, not just an independent run on
different data.

```
Train: Bearing1_1, Bearing2_1, Bearing3_1
Val:   Bearing1_2, Bearing2_2, Bearing3_2
Test:  Bearing1_3, Bearing2_3, Bearing3_3
```

## Best model (trial 26, epoch 92/300)

```
n_layers=7  n_units=32  dropout=0.5  lr=0.001  batch_size=32
activation=leaky_relu  lambda_w=0.938
```

## Results

| split | RMSE (0-100) | RMSE ([0,1]) | MSE ([0,1]) | R2    |
|-------|--------------|---------------|-------------|-------|
| val   | 18.04        | 0.180         | 0.0326      | 0.609 |
| test  | 21.31        | 0.213         | 0.0454      | 0.455 |

Hybrid train loss at best epoch: 0.0342

## vs. paper (Table 9, averaged over the 3 bearings per split, same [0,1] scale)

| split | paper RMSE | ours RMSE | paper R2 | ours R2 |
|-------|-----------|-----------|----------|---------|
| val   | 0.247     | **0.180** | 0.266    | **0.609** |
| test  | 0.167     | 0.213     | 0.654    | 0.455     |

Val beats the paper on both metrics. Test is behind the paper's.

## Hyperparameter importance

| lr | dropout_rate | n_layers | batch_size | lambda_w | activation | n_units |
|---|---|---|---|---|---|---|
| 0.50 | 0.31 | 0.06 | 0.06 | 0.03 | 0.02 | 0.02 |

`lambda_w` ranks 5th — small, but no longer negligible the way it was before the
loss-scale fix (see [COMPARISON.md](../../../COMPARISON.md) for that history),
where it measured ~0.003 regardless of its actual value. On this 9-bearing split,
learning rate and dropout dominate; see `unified_data/FINDINGS.md` for a split
where `lambda_w` matters considerably more (3rd of 7).

## Caveat

The paper reports RMSE/R2 per bearing and averages 3 values; ours is computed by
pooling all rows in a split. Not perfectly apples-to-apples if bearings differ a lot
in variance — "roughly comparable," not "identical methodology."

## Files

- `best_params.json` — full metrics + hyperparameters
- `best_model.pt` — trained weights (state_dict + params, load via `RULnet`)
- `optuna_study.db` — full Optuna study (50 trials, SQLite)
- `optuna_progression.png` — trial-by-trial val RMSE + hyperparameter importance
- `predictions.png` — predicted vs. true RUL scatter (val + test)
- `training_curves.png` — train loss / val RMSE per epoch for the best trial
