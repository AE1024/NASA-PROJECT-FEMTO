# Results Comparison — RUL Weibull-Loss Neural Network

This project reproduces and extends
[von Hahn & Mechefske (2022), arXiv:2201.01769](https://arxiv.org/abs/2201.01769),
which predicts bearing remaining-useful-life (RUL) with a neural network trained
on a hybrid loss combining ordinary MSE with a Weibull-distribution reliability
term. We ran two experiments to see how our implementation compares to the
paper's own reported numbers, and how much extra training data helps once the
implementation is correct.

- **`article`** — a 9-bearing split that mirrors the paper's own train/val/test
  table exactly (one bearing per operating condition, per split), for a
  direct, apples-to-apples comparison.
- **`unified_data`** — all 17 available PRONOSTIA bearings, split so no bearing
  is shared between train/val/test, to see what the same method does with
  roughly twice the data.

Full detail for each: [`model/results/article/FINDINGS.md`](model/results/article/FINDINGS.md)
and [`model/results/unified_data/FINDINGS.md`](model/results/unified_data/FINDINGS.md).
In the code, these are the `paper_replica` and `full_data` dataset names
(`train_optuna.py`'s `DATASET` argument, the `data/experiments/` folders, and the
MLflow experiment names below) — only this results tree uses the shorter display
names `article` / `unified_data`.

## Results ([0,1] scale, matching arXiv:2201.01769 Table 8/9's convention)

| metric | paper (Table 9) | article (paper_replica) | unified_data (full_data) |
|---|---|---|---|
| lambda_w | 2.28 | 0.938 | 2.963 |
| val RMSE | 0.247 | **0.180** | 0.213 |
| val R2 | 0.266 | **0.609** | 0.456 |
| test RMSE | 0.167 | 0.213 | **0.184** |
| test R2 | 0.654 | 0.455 | **0.572** |

Both experiments beat the paper on validation. On test, `unified_data` lands
closer to the paper's own result than `article` does — plausibly because it
trains on twice as many bearings (6 vs. 3) and, through training alone, sees
examples from all three operating conditions rather than just one bearing per
condition.

Neither run matches the paper's test performance outright. Given `article` uses
the exact same 9 bearings and split logic as the paper, the gap there is a fair,
direct signal that our model — same idea, same loss, same search space — still
falls short of the paper's specific best run on held-out test data, while doing
noticeably better on validation.

## Bugs found and fixed along the way (chronological)

1. **eta unit mismatch** — WeibullLoss divided predictions by 100 before comparing
   to an eta expressed in raw step units. Fixed by rescaling eta to the same
   LifePercentage units as predictions/targets.
2. **First hidden block had no activation; activation list was off-by-one** —
   `NN_Block`/`RULnet` in `model/nn_model.py`. Verified against the paper's own
   architecture description (Fig. 9: "ReLU used throughout, except the last layer")
   and fixed to give every hidden block an activation.
3. **Optuna objective never trained more than 1 epoch, never evaluated on val,
   never checkpointed weights** — the original `train_optuna.py` skeleton. Rewrote
   with a real epoch loop, per-epoch val evaluation, and best-epoch state_dict
   checkpointing (see point 6).
4. **Reported metric didn't match the saved model** — `best_val_rmse` was tracked
   as a number, but the model left in memory (and logged to MLflow) was whatever
   epoch the loop happened to end on, not the epoch that scored `best_val_rmse`.
   Fixed with `copy.deepcopy(model.state_dict())` on every improvement, restored
   before the trial returns.
5. **beta/eta estimated via 2-parameter MLE fit on 6 samples** — statistically
   unstable, and exactly the failure mode arXiv:2201.01769 Section 1.2.2 warns
   against. Fixed to follow the paper's own method: beta fixed from reliability-
   engineering literature (ball bearing, beta=2.0), eta solved via the Weibayes
   equation (their Eq. 2) given that fixed beta.
6. **val/test data leakage** — the original val.parquet and test.parquet were
   independently traced back to the official PHM2012 `Full_Test_Set` (complete)
   and `Test_set` (truncated) folders respectively, both covering the *same* 11
   physical bearings at different truncation depths. Test's RUL_pct labels were
   also silently wrong (calibrated against the truncated length, not the true
   bearing lifetime), because `prepare_dataset()` was never given `max_steps`.
   Rebuilt from scratch with real bearing identities (`data/bearing_identity_map.json`)
   and disjoint train/val/test bearing sets under `data/experiments/`.
7. **MSE computed on the raw [0, 100] scale vs. the Weibull CDF term always in
   [0, 1]** — MSE outweighed the Weibull term by ~1000x, so lambda_w had almost no
   effect on training anywhere in its [0, 3] search range (its Optuna importance
   was ~0.003, i.e. negligible). Fixed by normalizing predictions/targets to [0, 1]
   before computing MSE, matching the paper's own apparent convention (their
   reported RMSE values are also on [0, 1]).
8. **Weibull term used `|F(pred)-F(true)|` instead of `(F(pred)-F(true))^2`** —
   arXiv:2201.01769 Eq. 5 squares both the MSE and Weibull terms. Fixed to match;
   this also brought the two terms to comparable magnitude, reinforcing fix 7.

After fixes 7-8, lambda_w's Optuna hyperparameter importance rose from ~0.003
(negligible, in both the pre-fix run and effectively acting as if lambda_w=0
regardless of its actual value) to ~0.03-0.14 across the two v4 experiments — a
real, measurable effect for the first time.

## Reproducing

```
uv run mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db
uv run python -m model.train_optuna paper_replica
uv run python -m model.train_optuna full_data
uv run python -m model.model_visualize
```

MLflow experiments: `RUL_Weibull_NN_v4_paper_replica`, `RUL_Weibull_NN_v4_full_data`.
