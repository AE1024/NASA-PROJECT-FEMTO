# src/features.py
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import ruptures as rpt

RAW_FEATURE_COLS: list[str] = [
    "mean_ax", "rms_ax", "std_ax", "skew_ax", "kurtosis_ax",
    "peak_ax", "p2p_ax", "crest_ax", "shape_ax", "impulse_ax",
    "mean_ay", "rms_ay", "std_ay", "skew_ay", "kurtosis_ay",
    "peak_ay", "p2p_ay", "crest_ay", "shape_ay", "impulse_ay",
    "mean_mag", "rms_mag", "std_mag", "skew_mag", "kurtosis_mag",
    "peak_mag", "p2p_mag", "crest_mag", "shape_mag", "impulse_mag",
]

# Features selected for rolling-slope engineering
# (Mann-Whitney significant across all 6 train bearings, Bonferroni-corrected)
TREND_FEATURES: list[str] = [
    "rms_ax", "kurtosis_ax", "rms_ay", "kurtosis_ay",
    "rms_mag", "kurtosis_mag", "peak_ax", "peak_ay", "peak_mag",
]

TREND_WINDOW: int = 20

# Final model input: 30 raw + 9 slopes = 39 features
MODEL_FEATURES: list[str] = RAW_FEATURE_COLS + [f"{f}_slope" for f in TREND_FEATURES]

# ── 03: Time-domain feature extraction ────────────────────────────────────────

def compute_axis_features(signal: pd.Series, prefix: str) -> dict:
    """Time-domain statistics for one vibration axis (ax / ay / mag)."""
    rms = np.sqrt(np.mean(signal ** 2))
    peak = signal.max()
    mean_abs = signal.abs().mean()
    return {
        f"mean_{prefix}":    signal.mean(),
        f"rms_{prefix}":     rms,
        f"std_{prefix}":     signal.std(),
        f"skew_{prefix}":    signal.skew(),
        f"kurtosis_{prefix}": signal.kurtosis(),
        f"peak_{prefix}":    peak,
        f"p2p_{prefix}":     peak - signal.min(),
        f"crest_{prefix}":   peak / rms,
        f"shape_{prefix}":   rms / mean_abs,
        f"impulse_{prefix}": peak / mean_abs,
    }


def extract_features(group: pd.DataFrame) -> pd.Series:
    """Combines time-domain features for ax, ay, and mag axes (one bearing step block)."""
    return pd.Series({
        **compute_axis_features(group["ax"], "ax"),
        **compute_axis_features(group["ay"], "ay"),
        **compute_axis_features(group["mag"], "mag"),
    })

# ── 04: Health Index via PCA + Mahalanobis ────────────────────────────────────

def _mahalanobis_distance(data: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    mean = np.mean(baseline, axis=0)
    cov_inv = np.linalg.pinv(np.cov(baseline, rowvar=False))
    diff = data - mean
    return np.sqrt(np.einsum("ij,jk,ik->i", diff, cov_inv, diff))


def compute_hi_for_bearing(
    df: pd.DataFrame,
    bearing_id: str,
    feature_cols: list[str],
    baseline_frac: float = 0.10,
) -> pd.DataFrame:
    """
    Computes Health Index (HI) for one bearing via PCA + Mahalanobis distance.
    Scaler and PCA are fit on the first `baseline_frac` of each bearing's life
    so there is no cross-bearing contamination.
    """
    b = df[df["bearing_id"] == bearing_id].sort_values("step").reset_index(drop=True)
    X = b[feature_cols].values.astype(float)
    n_baseline = max(int(len(X) * baseline_frac), 10)

    scaler = StandardScaler().fit(X[:n_baseline])
    X_scaled = scaler.transform(X)
    X_baseline_scaled = X_scaled[:n_baseline]

    pca = PCA(n_components=2).fit(X_baseline_scaled)
    X_pca = pca.transform(X_scaled)

    b["health_index"] = _mahalanobis_distance(X_pca, pca.transform(X_baseline_scaled))
    return b


# ── 04: PELT changepoint detection ────────────────────────────────────────────

def find_hi_changepoint(df: pd.DataFrame, bearing_id: str, pen: float = 50) -> int | None:
    """
    Finds the single most significant changepoint in the HI curve using PELT (rbf kernel).
    Returns the step index of the changepoint, or None if none is found.
    """
    b = df[df["bearing_id"] == bearing_id].sort_values("step")
    breakpoints = rpt.Pelt(model="rbf").fit(b["health_index"].values).predict(pen=pen)
    if len(breakpoints) > 1:
        return int(b["step"].iloc[breakpoints[0] - 1])
    return None


# ── 04: Regime labeling ───────────────────────────────────────────────────────

def assign_final_regime_label(
    df: pd.DataFrame,
    bearing_id: str,
    cutoffs: dict,
    tail_frac: float = 0.10,
) -> pd.Series:
    """
    Labels each step as 'healthy' / 'transition' / 'critical'.
    Uses PELT cutoff when available; falls back to fixed 30% boundary
    for bearings where PELT finds no reliable changepoint (bearing_1, bearing_5).
    """
    b = df[df["bearing_id"] == bearing_id].sort_values("step")
    n = len(b)
    cutoff = cutoffs.get(bearing_id)
    tail_start = b["step"].iloc[int(n * (1 - tail_frac))]

    healthy_end = (
        b["step"].iloc[int(n * 0.3)]
        if (cutoff is None or cutoff >= tail_start)
        else cutoff
    )

    def _label(s):
        if s < healthy_end:   return "healthy"
        if s < tail_start:    return "transition"
        return "critical"

    return b["step"].apply(_label)


# ── 04: RUL target computation ────────────────────────────────────────────────

def add_rul(
    df: pd.DataFrame,
    bearing_id: str,
    max_step: int | None = None,
) -> pd.DataFrame:
    """
    Adds RUL (steps remaining) and RUL_pct (0-100) columns.

    - Train bearings: leave max_step=None → inferred from data (run-to-failure).
    - Val/test bearings: pass max_step = last_observed_step + true_remaining_life
      so RUL_pct is calibrated against the full bearing lifetime.
    """
    b = df[df["bearing_id"] == bearing_id].sort_values("step").copy()
    effective_max = max_step if max_step is not None else int(b["step"].max())
    b["RUL"]     = effective_max - b["step"]
    b["RUL_pct"] = b["RUL"] / effective_max * 100
    return b


# ── 04: Trend feature engineering ─────────────────────────────────────────────

def compute_rolling_slopes(
    df: pd.DataFrame,
    feature_list: list[str] = TREND_FEATURES,
    window: int = TREND_WINDOW,
) -> pd.DataFrame:
    """
    Adds a `{feat}_slope` column for each feature in feature_list.
    Slope = OLS gradient over the last `window` steps, computed per bearing.
    Scale-invariant across bearings and naturally monotonic with degradation.
    """
    x_base = np.arange(window)
    bearing_frames = []
    for bid in df["bearing_id"].unique():
        b = df[df["bearing_id"] == bid].sort_values("step").copy().reset_index(drop=True)
        for feat in feature_list:
            vals = b[feat].values.astype(float)
            slopes = np.empty(len(vals))
            for i in range(len(vals)):
                if i == 0:
                    slopes[i] = 0.0
                elif i < window - 1:
                    x_w = np.arange(i + 1)
                    slopes[i] = np.polyfit(x_w, vals[: i + 1], 1)[0]
                else:
                    slopes[i] = np.polyfit(x_base, vals[i - window + 1 : i + 1], 1)[0]
            b[f"{feat}_slope"] = slopes
        bearing_frames.append(b)
    return pd.concat(bearing_frames, ignore_index=True)


# ── Raw signal → extracted features ──────────────────────────────────────────

def detect_bearing_lifetimes(raw_df: pd.DataFrame) -> dict:
    """
    Auto-detects bearing lifetimes from row-count transitions in raw signal data.
    Each drop in rows-per-file = one bearing dying.
    Returns {bearing_1: last_alive_step, bearing_2: ..., ...} in death order.
    """
    counts = raw_df.groupby("source_file").size()
    counts.index = counts.index.str.extract(r"(\d+)")[0].astype(int)
    counts = counts.sort_index()

    # Find steps where count drops (a bearing died just before this step)
    changes = counts[counts != counts.shift()].iloc[1:]  # skip first (no previous)
    death_steps = (changes.index - 1).tolist()           # last step bearing was alive
    death_steps.append(int(counts.index.max()))           # last surviving bearing

    return {f"bearing_{i+1}": step for i, step in enumerate(sorted(death_steps))}


def assign_bearing_ids_raw(raw_df: pd.DataFrame, bearing_lifetimes: dict) -> pd.DataFrame:
    """
    Assigns bearing_id and step columns to raw signal data.
    Uses (step, block_idx) merge — order-independent.
    """
    df = raw_df.copy()
    df["step"] = df["source_file"].str.extract(r"(\d+)").astype(int)
    df["row_in_file"] = df.groupby("source_file").cumcount()
    df["block_idx"] = df["row_in_file"] // 2560

    lookup_rows = []
    for step in range(1, max(bearing_lifetimes.values()) + 1):
        alive = sorted([b for b, life in bearing_lifetimes.items() if step <= life])
        for idx, bid in enumerate(alive):
            lookup_rows.append({"step": step, "block_idx": idx, "bearing_id": bid})

    lookup = pd.DataFrame(lookup_rows)
    df = df.merge(lookup, on=["step", "block_idx"], how="left")
    return df


def extract_step_features(raw_df_with_ids: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts 30 time-domain features per (bearing_id, step) from raw signal blocks.
    Input must have columns: bearing_id, step, ax, ay, mag.
    Returns one row per (bearing_id, step).
    """
    records = []
    for (bid, step), group in raw_df_with_ids.groupby(["bearing_id", "step"]):
        feats = extract_features(group)
        feats["bearing_id"] = bid
        feats["step"] = step
        records.append(feats)
    return pd.DataFrame(records).reset_index(drop=True)


def raw_to_features(
    raw_df: pd.DataFrame,
    bearing_lifetimes: dict | None = None,
) -> pd.DataFrame:
    """
    Full conversion: raw signal DataFrame → 30-feature DataFrame with bearing_id and step.

    Parameters
    ----------
    raw_df           : DataFrame with columns [ax, ay, mag, source_file]
    bearing_lifetimes: {bearing_id: last_alive_step}. Auto-detected if None.
    """
    if bearing_lifetimes is None:
        bearing_lifetimes = detect_bearing_lifetimes(raw_df)

    df_with_ids = assign_bearing_ids_raw(raw_df, bearing_lifetimes)
    features_df = extract_step_features(df_with_ids)
    return features_df


# ── Pipeline orchestrator ──────────────────────────────────────────────────────

def build_feature_pipeline(
    features_df: pd.DataFrame,
    max_steps: dict | None = None,
) -> pd.DataFrame:
    """
    Full pipeline: raw features → HI → PELT → regime_label → RUL → slopes.
    Returns df_model ready for model training/inference (MODEL_FEATURES columns intact).

    Parameters
    ----------
    features_df : DataFrame with columns [bearing_id, step, <30 raw features>]
    max_steps   : {bearing_id: int} — effective total lifetime per bearing.
                  Pass None for train (inferred from data).
                  Pass a dict for val/test (true remaining life from competition labels).
    """
    max_steps = max_steps or {}

    # Step 1: Health Index
    hi = pd.concat([
        compute_hi_for_bearing(features_df, bid, RAW_FEATURE_COLS)
        for bid in features_df["bearing_id"].unique()
    ])

    # Step 2: PELT changepoints
    cutoffs = {bid: find_hi_changepoint(hi, bid) for bid in hi["bearing_id"].unique()}

    # Step 3: Regime labels
    hi["regime_label"] = pd.concat([
        assign_final_regime_label(hi, bid, cutoffs)
        for bid in hi["bearing_id"].unique()
    ])

    # Step 4: RUL target
    hi = pd.concat([
        add_rul(hi, bid, max_steps.get(bid))
        for bid in hi["bearing_id"].unique()
    ])

    # Step 5: Rolling slope features
    df_model = compute_rolling_slopes(hi, TREND_FEATURES, TREND_WINDOW)

    return df_model
