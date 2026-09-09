import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

_ROOT = Path(__file__).resolve().parent.parent

TRAIN_PATH = _ROOT / "data" / "processed_train_data" / "train.parquet"
VAL_PATH   = _ROOT / "data" / "processed_val_test" / "val" / "val.parquet"
TEST_PATH  = _ROOT / "data" / "processed_val_test" / "test" / "test.parquet"


META_COLS = {"bearing_id", "step", "RUL_pct"}


def load_dataset(
    parquet_path: str | Path,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Loads a processed parquet (bearing_id + step + 21 features + RUL_pct).
    Returns:
        X    : 21 scaled features
        y    : RUL_pct Series
        meta : bearing_id + step DataFrame (for per-bearing evaluation)
    """
    df = pd.read_parquet(parquet_path)
    meta = df[["bearing_id", "step"]].copy()
    y    = df["RUL_pct"]
    X    = df.drop(columns=list(META_COLS), errors="ignore")
    return X, y, meta


def evaluate(model, X: pd.DataFrame, y: pd.Series, label: str = "") -> dict:
    """Returns R², RMSE, MAE for a fitted model."""
    y_pred = model.predict(X)
    r2   = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae  = mean_absolute_error(y, y_pred)
    if label:
        print(f"[{label}]  R²={r2:.4f}  RMSE={rmse:.2f}  MAE={mae:.2f}")
    return {"R2": r2, "RMSE": rmse, "MAE": mae, "y_pred": y_pred}


def train_baseline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> dict:
    """Trains Ridge and RandomForest baselines. Returns fitted model dict."""
    models = {
        "Ridge": Ridge(alpha=10.0),
        "RandomForest": RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        ),
    }
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
    return models

if __name__ == "__main__":
    X_train, y_train, meta_train = load_dataset(TRAIN_PATH)
    X_val,   y_val,   meta_val   = load_dataset(VAL_PATH)
    print(f"Train: {X_train.shape}  Val: {X_val.shape}")
    models = train_baseline(X_train, y_train)
    for name, model in models.items():
        evaluate(model, X_train, y_train, label=f"{name} - Train")
        evaluate(model, X_val,   y_val,   label=f"{name} - Val")

