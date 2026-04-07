# src/features.py
from __future__ import annotations

from typing import Dict, Tuple
import numpy as np
import pandas as pd

# We do NOT re-transform here. data_transformation.py already produced:
# NPL (log of decimal), GDP (level), DEGU (log-diff), CBLR (diff), GLA (diff).

HI_FREQ_VARS = ["DEGU", "CBLR", "GLA"]


def _as_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype(float)


def _beta_midas_weights(k: int, a: float, b: float) -> np.ndarray:
    """
    Standard Beta polynomial MIDAS weights (normalized to sum=1) for k months.
    Most recent observation has index 0.
    """
    idx = np.arange(k, dtype=float)  # 0..k-1 (0=most recent)
    # map to (0,1]; avoid exactly 0 for stability
    x = (idx + 1) / k
    w = np.power(x, a - 1.0) * np.power(1.0 - x, b - 1.0)
    w = np.maximum(w, 1e-12)
    w = w / w.sum()
    # reverse so that lag 0 gets the last weight in convolution sense
    return w[::-1]  # so that w[0] multiplies GDP_t, w[1] * GDP_{t-1}, ...


def _make_gdp_midas_feature(gdp: pd.Series, months: int, a: float, b: float) -> pd.Series:
    w = _beta_midas_weights(months, a, b)
    # Convolution-like rolling weighted sum:
    # For date t we want sum_{j=0..months-1} w[j] * gdp_{t-j}
    g = _as_num(gdp).copy()
    mat = np.vstack([g.shift(j).to_numpy() for j in range(months)])
    out = pd.Series(np.dot(w, mat), index=g.index)
    out.name = f"GDP_midas_{months}_{a:.2f}_{b:.2f}"
    return out


def _add_lag_block(df: pd.DataFrame, col: str, max_lag: int, include_lag0: bool = True) -> pd.DataFrame:
    out = {}
    start = 0 if include_lag0 else 1
    for L in range(start, max_lag + 1):
        out[f"{col}_lag{L}"] = _as_num(df[col]).shift(L)
    return pd.DataFrame(out, index=df.index)


def _add_rolling_means(df: pd.DataFrame, cols: list[str], windows: list[int]) -> pd.DataFrame:
    feats = {}
    for c in cols:
        s = _as_num(df[c])
        for w in windows:
            # past-only mean up to and including t (no leakage)
            feats[f"{c}_mean{w}"] = s.rolling(window=w, min_periods=w).mean()
    return pd.DataFrame(feats, index=df.index)


def build_feature_matrix(panel: pd.DataFrame, cfg: Dict, horizon: int) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Construct ML features per horizon h using ONLY the already-transformed series.
    Target y is NPL shifted -h (i.e., predicting y(t+h) from info at t).
    """
    df = panel.copy().sort_index()
    # Ensure the core columns exist (others will be optional)
    for c in ["NPL", "GDP", "DEGU", "CBLR", "GLA"]:
        if c not in df.columns:
            df[c] = np.nan

    fcfg = (cfg.get("features") or {})
    npl_lags = int(fcfg.get("npl_auto_lags", 12))
    hi_lags = int(fcfg.get("hi_freq_lags", 6))
    gdp_m = int(fcfg.get("gdp_months", 12))
    a, b = (fcfg.get("beta_shape") or [3.0, 4.5])
    reverse = bool(fcfg.get("reverse_midas", False))  # keeps compatibility; not used here

    # Target: future NPL (still in log space)
    y = _as_num(df["NPL"]).shift(-horizon)

    blocks = []

    # NPL autoregressive lags (exclude contemporaneous to avoid leaking y)
    if npl_lags > 0:
        blocks.append(_add_lag_block(df, "NPL", npl_lags, include_lag0=False))

    # GDP MIDAS single feature
    if gdp_m > 0:
        blocks.append(_make_gdp_midas_feature(df["GDP"], gdp_m, float(a), float(b)).to_frame())

    # Hi-frequency lags for DEGU/CBLR/GLA (include lag0 since we forecast y(t+h))
    if hi_lags > 0:
        for var in HI_FREQ_VARS:
            if var in df.columns:
                blocks.append(_add_lag_block(df, var, hi_lags, include_lag0=True))

    # Optional: Rolling-window level summaries (no additional transforms)
    add_roll = bool(fcfg.get("add_rolling_windows", False))
    if add_roll:
        windows = list(map(int, fcfg.get("rolling_windows", [3, 6])))
        # Apply to the available core columns
        roll_cols = [c for c in ["NPL", "GDP", "DEGU", "CBLR", "GLA"] if c in df.columns]
        blocks.append(_add_rolling_means(df, roll_cols, windows))

    # Concatenate feature blocks
    X = pd.concat(blocks, axis=1) if blocks else pd.DataFrame(index=df.index)

    # Cut to rows where y and X are fully available
    data = pd.concat([X, y.rename("y")], axis=1)
    data = data.dropna(how="any")

    # Final split
    Xf = data.drop(columns=["y"])
    yf = data["y"]
    return Xf, yf


def build_all_horizons(panel: pd.DataFrame, cfg: Dict) -> Dict[int, Tuple[pd.DataFrame, pd.Series]]:
    horizons = cfg.get("horizons", [0, 1, 2, 3, 4, 5, 6])
    out = {}
    for h in horizons:
        X, y = build_feature_matrix(panel, cfg, int(h))
        out[int(h)] = (X, y)
    return out
