"""
drift.py
Lightweight drift checks for production monitoring (no scipy dependency).
Implements PSI (Population Stability Index) and simple status mapping.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Tuple


def psi_score(ref: pd.Series, cur: pd.Series, bins: int = 10) -> float:
    """Compute PSI between reference and current samples."""
    ref = pd.Series(ref).dropna().astype(float)
    cur = pd.Series(cur).dropna().astype(float)
    if ref.empty or cur.empty:
        return np.nan
    edges = np.quantile(ref, np.linspace(0, 1, bins + 1))
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    r_hist, _ = np.histogram(ref, bins=edges)
    c_hist, _ = np.histogram(cur, bins=edges)
    r_prop = r_hist / r_hist.sum() if r_hist.sum() > 0 else np.ones_like(r_hist) / len(r_hist)
    c_prop = c_hist / c_hist.sum() if c_hist.sum() > 0 else np.ones_like(c_hist) / len(c_hist)
    eps = 1e-12
    return float(np.sum((r_prop - c_prop) * np.log((r_prop + eps) / (c_prop + eps))))


def psi_status(psi: float) -> str:
    """Map PSI to status per common thresholds."""
    if np.isnan(psi):
        return "N/A"
    if psi < 0.1:
        return "OK"
    if psi < 0.25:
        return "WARN"
    return "ALERT"


def drift_report(
    ref_df: pd.DataFrame,
    cur_df: pd.DataFrame,
    features: Tuple[str, ...],
    ref_window: str,
    cur_window: str,
) -> pd.DataFrame:
    """
    Return dataframe with columns: metric, value, status, window_start, window_end.
    """
    rows = []
    for f in features:
        if f not in ref_df.columns or f not in cur_df.columns:
            continue
        v = psi_score(ref_df[f], cur_df[f])
        rows.append({
            "metric": f"PSI::{f}",
            "value": v,
            "status": psi_status(v),
            "window_start": ref_window,
            "window_end": cur_window,
        })
    return pd.DataFrame(rows)
