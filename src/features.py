# src/features.py
import numpy as np
import pandas as pd

#feature extractions

def compute_axis_features(signal: pd.Series, prefix: str) -> dict:
    """Calculates time-domain statistics for an axis (ax/ay/mag)."""
    abs_signal = signal.abs()
    mean_abs = abs_signal.mean()
    rms = np.sqrt(np.mean(signal**2))
    peak = signal.max()
    return {
        f'mean_{prefix}': signal.mean(), f'rms_{prefix}': rms,
        f'std_{prefix}': signal.std(), f'skew_{prefix}': signal.skew(),
        f'kurtosis_{prefix}': signal.kurtosis(), f'peak_{prefix}': peak,
        f'p2p_{prefix}': peak - signal.min(), f'crest_{prefix}': peak / rms,
        f'shape_{prefix}': rms / mean_abs, f'impulse_{prefix}': peak / mean_abs,
    }

def extract_features(group: pd.DataFrame) -> pd.Series:
    """Combines all features on the ax and ay axes for a bearing/step block."""

    return pd.Series({
        **compute_axis_features(group['ax'], 'ax'),
        **compute_axis_features(group['ay'], 'ay'),
    })  