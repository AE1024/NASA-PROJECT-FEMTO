# Full Data

All 17 available PRONOSTIA bearings, split so no physical bearing is shared
between train/val/test — see
[`data/bearing_identity_map.json`](../../../data/bearing_identity_map.json) for
how the bearing identities were recovered and the splits rebuilt.

```
Train (6): Learning_set in full — Bearing1_1, 1_2, 2_1, 2_2, 3_1, 3_2
Val   (5): Bearing1_3, Bearing1_5, Bearing1_7, Bearing2_3, Bearing2_5
Test  (6): Bearing1_4, Bearing1_6, Bearing2_4, Bearing2_6, Bearing2_7, Bearing3_3
```

Val/test allocation was a manual choice (not condition-balanced — Condition 3 only
has 1 remaining bearing after train takes 2 of its 3), prioritizing disjointness
over per-split condition balance.

## Best model (trial 31, epoch 120/300)

```
n_layers=4  n_units=32  dropout=0.25  lr=0.001  batch_size=64
activation=relu  lambda_w=2.963
```

## Results

| split | RMSE (0-100) | RMSE ([0,1]) | MSE ([0,1]) | R2    |
|-------|--------------|---------------|-------------|-------|
| val   | 21.29        | 0.213         | 0.0453      | 0.456 |
| test  | 18.38        | 0.184         | 0.0338      | **0.572** |

Hybrid train loss at best epoch: 0.0688

Test outperforms val here — the reverse of the usual pattern. Likely explanation:
the manually-chosen val bearings (1_3, 1_5, 1_7, 2_3, 2_5) happen to be harder to
predict than the test bearings (which include the shorter-lived, more varied set
plus the only remaining Condition-3 bearing) — an artifact of the hand-picked split,
not a modeling issue.

## vs. paper (Table 9, averaged over 3 bearings per split, same [0,1] scale)

| split | paper RMSE | ours RMSE | paper R2 | ours R2 |
|-------|-----------|-----------|----------|---------|
| val   | 0.247     | 0.213     | 0.266    | **0.456** |
| test  | 0.167     | 0.184     | 0.654    | **0.572** |

Val beats the paper; test comes closer to it than the `article` run does, but
doesn't reach it.

## Hyperparameter importance

| dropout_rate | lr | lambda_w | batch_size | n_units | activation | n_layers |
|---|---|---|---|---|---|---|
| 0.53 | 0.21 | 0.13 | 0.08 | 0.05 | 0.00 | 0.00 |

`lambda_w` ranks 3rd of 7 here — a real, measurable effect, and considerably
stronger than on the `article` split (5th of 7 there, see its FINDINGS.md). Before
the loss-scale fix (see [COMPARISON.md](../../../COMPARISON.md)), `lambda_w`
ranked last everywhere at ~0.003 — the model trained as if `lambda_w=0` regardless
of what value Optuna picked. Two bugs caused that: (1) beta/eta from a
2-parameter MLE fit on 6 samples instead of the paper's Weibayes method, and
(2) MSE computed on the raw [0, 100] target scale, ~1000x the Weibull CDF term's
natural [0, 1] scale.

## Files

- `best_params.json` — full metrics + hyperparameters
- `best_model.pt` — trained weights (state_dict + params, load via `RULnet`)
- `optuna_study.db` — full Optuna study (50 trials, SQLite)
- `optuna_progression.png` — trial-by-trial val RMSE + hyperparameter importance
- `predictions.png` — predicted vs. true RUL scatter (val + test)
- `training_curves.png` — train loss / val RMSE per epoch for the best trial
