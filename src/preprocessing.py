import json
import joblib
import pandas as pd
import polars as pl

from src.features import build_feature_pipeline, raw_to_features


def prepare_dataset(
    parquet_path: str,
    scaler_path: str = r"C:\Users\elmaz\AIProject\NASA_Project\src\scaler\robust_scaler.pkl",
    features_path: str = r"C:\Users\elmaz\AIProject\NASA_Project\notebooks\features.json",
    bearing_lifetimes: dict | None = None,
    max_steps: dict | None = None,
    include_target: bool = True,
) -> tuple[pd.DataFrame, pd.Series | None]:
    """
    Converts a raw signal parquet (val/test) into model-ready format (21 features).

    Parameters
    ----------
    parquet_path      : path to features_val.parquet or features_test.parquet
    scaler_path       : path to robust_scaler.pkl (fit on train — transform only)
    features_path     : path to features.json (21 selected feature names)
    bearing_lifetimes : {bearing_id: last_alive_step}. Auto-detected if None.
    max_steps         : {bearing_id: int} effective total lifetime for RUL_pct.
                        None -> inferred from data (run-to-failure).
    include_target    : True -> also return y (RUL_pct). False for inference only.

    Returns
    -------
    X : DataFrame with 21 scaled features
    y : Series of RUL_pct values, or None if include_target=False
    """
    # 1. Load raw signal parquet
    df_raw = pl.read_parquet(parquet_path).to_pandas()

    # 2. Raw signal -> 30 time-domain features with bearing_id and step
    features_df = raw_to_features(df_raw, bearing_lifetimes=bearing_lifetimes)

    # 3. Full feature engineering: HI, PELT, regime, RUL, slopes
    df_model = build_feature_pipeline(features_df, max_steps=max_steps)

    # 4. One-hot encode regime_label
    df_model = pd.get_dummies(df_model, columns=["regime_label"], prefix="regime")

    # 5. Load scaler (fit on train — transform only)
    scaler = joblib.load(scaler_path)

    # 6. Load selected feature names
    with open(features_path, "r") as f:
        selected_features = json.load(f)

    # 7. Scale all numeric features (excluding identifiers and RUL_pct target)
    # RUL is kept here because the scaler was fit on it; it's excluded via selected_features
    drop_cols = {"bearing_id", "step", "RUL_pct"}
    feature_cols = [c for c in df_model.columns if c not in drop_cols]
    X_scaled = pd.DataFrame(
        scaler.transform(df_model[feature_cols]),
        columns=feature_cols,
        index=df_model.index,
    )

    # 8. Select only the 21 model features
    X = X_scaled[selected_features]

    y = df_model["RUL_pct"] if include_target and "RUL_pct" in df_model.columns else None

    return X, y
