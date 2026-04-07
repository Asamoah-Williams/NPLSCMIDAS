# -*- coding: utf-8 -*-
"""
NPL Data Quality Framework — DQ-only with Changelog

- Percent -> decimal for NPL/GDP/GLA/CBLR; DEGU is a positive level
- Checks: schema/anchor drift, freshness, completeness, duplicates, invalid negatives
- Outliers: rolling median/MAD with per-series transforms & parameters
    * GLA: level (growth), window=36, z=4.5
    * CBLR: Δlevel (pp), hard 500 bps rule, deadband 5 bps, window=48, z=5.0
    * DEGU: Δlog(level), window=48, z=5.5, percentile hard rule (99th of |Δlog|)
    * Auto-relax z if >3% flags after warm-up (hard rules never relaxed)
- Overall DQ Score in Summary (plus OVERALL rows), Scorecard, Overall sheet
- NEW: DQ Changelog — compares current report to previous report at same path
    * Changelog_Summary: per-dataset deltas (DQ score, freshness band, outlier counts)
    * Changelog_Outliers_Added / Removed: outlier timestamps that appeared/disappeared
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pathlib import Path

import yaml
import pyodbc
from npl_src.db import DatabaseConnection

THIS_FILE = Path(__file__).resolve()  #new_14/10/25        # .../src/npl_dq/cli.py
SRC_DIR = THIS_FILE.parents[0]   #new_14/10/25             # .../src
PROJECT_ROOT = THIS_FILE.parents[0] #new_14/10/25          # .../project root
#PKG_DIR = SRC_DIR / "npl_dq" #new_14/10/25
ROOT = Path(__file__).resolve().parents[1]
CFG  = yaml.safe_load((ROOT / "config.yml").read_text())

db_config = CFG["database"]

# =============================================================================
# Configuration
# =============================================================================

PERCENT_DATASETS = {"NPL", "GDP", "GLA", "CBLR"}  # DEGU is not a percent

# Series kind (for invalid-negative rule only)
SERIES_KIND = {
    "NPL":  "level_pct",   # decimal >= 0
    "GDP":  "change_pct",  # pct change can be negative
    "GLA":  "change_pct",  # pct change can be negative (growth rate)
    "CBLR": "level_pct",   # decimal >= 0
    "DEGU": "level_level"  # positive level (>0)
}

FRESHNESS_THRESHOLDS = {
    "M": (45, 60),     # Green ≤45, Amber ≤60, else Red
    "Q": (120, 180),   # Green ≤120, Amber ≤180, else Red
}

EXPECTED_SCHEMAS = {
    "NPL":  {"columns": ["DATE", "NPL"],  "freq": "M", "anchor": "ME"},
    "GDP":  {"columns": ["DATE", "GDP"],  "freq": "Q", "anchor": "QE-DEC"},
    "GLA":  {"columns": ["DATE", "GLA"],  "freq": "M", "anchor": "ME"},
    "CBLR": {"columns": ["DATE", "CBLR"], "freq": "M", "anchor": "ME"},
    "DEGU": {"columns": ["DATE", "DEGU"], "freq": "M", "anchor": "ME"},
}

DQ_SCORE_WEIGHTS = {
    "default": {
        "Completeness":     0.25,
        "Timeliness":       0.20,
        "Uniqueness":       0.20,
        "Outliers":         0.25,
        "InvalidNegatives": 0.10,
    }
}

# -------- Outlier tuning (per-series) --------
# mode:
#   "level"       -> detect on level
#   "change"      -> detect on Δlevel (period-over-period)
#   "log_level"   -> detect on log(level)
#   "pct_change"  -> detect on relative change
#   "log_return"  -> detect on Δlog(level)  (≈ percentage change for small moves)
SERIES_OUTLIER_PARAMS = {
    "NPL":  {"window": 24, "z": 3.75, "mode": "level"},
    "GDP":  {"window": 12, "z": 4.00, "mode": "level"},      # quarterly
    # GLA = Average Growth in Loan Portfolio → detect on the growth level itself
    "GLA":  {"window": 36, "z": 4.50, "mode": "level"},
    # CBLR = lending rate: Δlevel (decimals), hard 500 bps + 5 bps deadband
    "CBLR": {"window": 48, "z": 5.00, "mode": "change",
             "deadband": 0.0005,          # ignore < 5 bps
             "hard_pp_threshold": 0.05},  # 500 bps = 5pp in decimals
    # DEGU = FX level: Δlog(level)
    "DEGU": {"window": 48, "z": 5.50, "mode": "log_return",
             "percentile_hard_rule": 0.99}  # flag if |Δlog| above 99th pct
}

# Safety valve: cap post-warmup outlier share (per series)
MAX_OUTLIER_SHARE = 0.03
Z_MAX = 7.0

# =============================================================================
# Utilities
# =============================================================================

def _to_datetime_index(df: pd.DataFrame, date_col: str = "DATE") -> pd.DataFrame:
    """Parse dates robustly (avoid deprecated infer_datetime_format)."""
    out = df.copy()
    ser = out[date_col].astype(str).str.strip()
    iso_mask = ser.str.match(r"^\d{4}-\d{2}-\d{2}$")
    if iso_mask.all():
        out[date_col] = pd.to_datetime(ser, format="%Y-%m-%d", errors="coerce")
    else:
        out[date_col] = pd.to_datetime(ser, errors="coerce", dayfirst=True)
    out = out.dropna(subset=[date_col]).set_index(date_col).sort_index()
    return out

def _infer_month_anchor(idx: pd.DatetimeIndex) -> str:
    if len(idx) == 0:
        return "M"
    s = pd.Series(idx)
    if s.dt.is_month_end.all():
        return "ME"
    if s.dt.is_month_start.all():
        return "MS"
    return "M"

def _freq_code(idx: pd.DatetimeIndex) -> str:
    if len(idx) == 0:
        return "M"
    if len(idx.to_period("Q-DEC").unique()) == len(idx):
        return "Q"
    if len(idx.to_period("M").unique()) == len(idx):
        return "M"
    return "M"

def _freshness_days(idx: pd.DatetimeIndex) -> Optional[int]:
    if len(idx) == 0:
        return None
    last = idx.max()
    return int((pd.Timestamp.today().normalize() - last.normalize()).days)

def _freshness_band(freq_code: str, days: Optional[int]) -> str:
    if days is None:
        return "unknown"
    green_max, amber_max = FRESHNESS_THRESHOLDS.get(freq_code, (45, 60))
    if days <= green_max:
        return "green"
    if days <= amber_max:
        return "amber"
    return "red"

def mad(arr: np.ndarray) -> float:
    m = np.nanmedian(arr)
    return np.nanmedian(np.abs(arr - m))

def robust_outliers_rolling(x: pd.Series, z_thresh: float, window: int) -> pd.Series:
    """Median/MAD rolling z-score outliers."""
    x = x.astype(float)
    med = x.rolling(window, min_periods=window).median()
    rolling_mad = x.rolling(window, min_periods=window).apply(lambda s: mad(s.values), raw=False)
    z = 0.6745 * (x - med) / rolling_mad.replace(0, np.nan)
    return (z.abs() > z_thresh).fillna(False)

def missing_periods(idx_like, freq_hint: str) -> List[pd.Timestamp]:
    """Return missing expected period-end timestamps between min(idx) and max(idx)."""
    if not isinstance(idx_like, pd.DatetimeIndex):
        idx = pd.DatetimeIndex(pd.to_datetime(idx_like, errors="coerce").dropna())
    else:
        idx = idx_like
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    idxn = idx.normalize()
    if len(idxn) == 0:
        return []
    if freq_hint == "Q":
        rng = pd.period_range(
            idxn.min().to_period("Q-DEC"), idxn.max().to_period("Q-DEC"), freq="Q-DEC"
        ).to_timestamp(how="end").normalize()
    else:
        rng = pd.period_range(
            idxn.min().to_period("M"), idxn.max().to_period("M"), freq="M"
        ).to_timestamp(how="end").normalize()
    miss = rng.difference(idxn)
    return list(miss)

def check_schema(df: pd.DataFrame, expected_cols: List[str]) -> Dict[str, object]:
    got = set([c.strip().lower() for c in df.columns])
    want = set([c.strip().lower() for c in expected_cols])
    return {"schema_drift": got != want, "missing": sorted(list(want - got)), "extra": sorted(list(got - want))}

def check_anchor(idx: pd.DatetimeIndex, expect_anchor: str) -> Tuple[bool, str]:
    if expect_anchor.startswith("QE"):
        qe = idx.to_period("Q-DEC").end_time.normalize()
        inferred = "QE-DEC" if (qe == idx.normalize()).all() else "Q"
        return (inferred != expect_anchor), inferred
    else:
        inferred = _infer_month_anchor(idx)
        return (inferred != expect_anchor), inferred

def as_quarter_end(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return idx.to_period("Q-DEC").end_time.normalize()

# Prune any legacy policy/shock/winsor columns from Validity sheets
PRUNE_COLS = [
    "plaus_low", "plaus_high", "breach_policy", "implausible_npl",
    "CBLR_hard500", "FX_shock", "det_series_win"  # legacy columns if present
]
def _prune_validity_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols_to_drop = [c for c in PRUNE_COLS if c in df.columns]
    if cols_to_drop:
        return df.drop(columns=cols_to_drop)
    return df

# How to transform the series before outlier detection
def _prep_for_detection(series: pd.Series, mode: str) -> pd.Series:
    s = series.astype(float)
    if mode == "level":
        return s
    if mode == "log_level":
        return np.log(s.where(s > 0))
    if mode == "change":
        return s.diff()
    if mode == "pct_change":
        return s.pct_change()
    if mode == "log_return":
        return np.log(s.where(s > 0)).diff()
    return s

# =============================================================================
# DQ Changelog helpers
# =============================================================================

def _read_prev_report(xlsx_path: str):
    """Return (prev_summary_df, prev_outliers_df) from an existing report, else (None, None)."""
    if not os.path.isfile(xlsx_path):
        return None, None
    try:
        with pd.ExcelFile(xlsx_path) as xf:
            prev_summary = pd.read_excel(xf, "Summary")
            prev_outliers = pd.read_excel(xf, "Outliers")
        return prev_summary, prev_outliers
    except Exception:
        return None, None

def _build_changelog(curr_summary: pd.DataFrame,
                     curr_outliers: pd.DataFrame,
                     prev_summary: pd.DataFrame,
                     prev_outliers: pd.DataFrame):
    """
    Build:
      - summary_change: per-dataset deltas
      - outliers_added: rows present now but not before
      - outliers_removed: rows present before but not now
    """
    # --- Summary deltas
    keep_cols = ["dataset", "dq_score", "freshness_band", "stat_outliers", "n_obs"]
    cs = curr_summary.copy()
    ps = prev_summary.copy()
    for df in (cs, ps):
        for c in keep_cols:
            if c not in df.columns:
                df[c] = np.nan
    cs = cs[keep_cols]
    ps = ps[keep_cols]

    # exclude OVERALL rows from per-dataset deltas
    overall_tags = {"OVERALL_MACRO", "OVERALL_WEIGHTED"}
    cs_ = cs[~cs["dataset"].isin(overall_tags)].copy()
    ps_ = ps[~ps["dataset"].isin(overall_tags)].copy()

    merged = cs_.merge(ps_, on="dataset", how="outer", suffixes=("_curr", "_prev"))
    merged["dq_score_delta"] = merged["dq_score_curr"] - merged["dq_score_prev"]
    merged["stat_outliers_delta"] = merged["stat_outliers_curr"] - merged["stat_outliers_prev"]
    merged["freshness_change"] = merged["freshness_band_prev"].fillna("NA") + " → " + merged["freshness_band_curr"].fillna("NA")
    summary_change = merged[[
        "dataset", "n_obs_prev", "n_obs_curr",
        "dq_score_prev", "dq_score_curr", "dq_score_delta",
        "stat_outliers_prev", "stat_outliers_curr", "stat_outliers_delta",
        "freshness_band_prev", "freshness_band_curr", "freshness_change"
    ]].sort_values("dataset")

    # --- Outliers added/removed
    def _norm_outliers(df: pd.DataFrame):
        if df is None or df.empty:
            return pd.DataFrame(columns=["dataset", "export_date"])
        cols = df.columns.str.lower()
        df = df.copy()
        df.columns = cols
        # minimal key: dataset + export_date (date for M, quarter-end for Q)
        if "export_date" not in df.columns:
            if "date" in df.columns:
                df["export_date"] = pd.to_datetime(df["date"], errors="coerce")
            else:
                df["export_date"] = pd.NaT
        else:
            df["export_date"] = pd.to_datetime(df["export_date"], errors="coerce")
        if "dataset" not in df.columns:
            df["dataset"] = "UNKNOWN"
        return df[["dataset", "export_date"]].dropna()

    curr_o = _norm_outliers(curr_outliers)
    prev_o = _norm_outliers(prev_outliers)

    key = ["dataset", "export_date"]

    # Added = in current but not in previous
    outliers_added = curr_o.merge(prev_o, on=key, how="left", indicator=True)
    outliers_added = outliers_added[outliers_added["_merge"] == "left_only"].drop(columns=["_merge"]).sort_values(key)

    # Removed = in previous but not in current
    outliers_removed = prev_o.merge(curr_o, on=key, how="left", indicator=True)
    outliers_removed = outliers_removed[outliers_removed["_merge"] == "left_only"].drop(columns=["_merge"]).sort_values(key)

    # Add run metadata
    summary_change.insert(1, "run_time", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    return summary_change, outliers_added, outliers_removed

# =============================================================================
# DQ Result Type
# =============================================================================

@dataclass
class DQResult:
    name: str
    validity: pd.DataFrame
    summary: Dict[str, object]
    outliers: pd.DataFrame
    issues: List[Dict[str, object]]
    modelling_notes: pd.DataFrame
    duplicates: pd.DataFrame
    scorecard: Dict[str, float]

# =============================================================================
# Core evaluation
# =============================================================================

def evaluate_series(name: str, df_raw: pd.DataFrame, source_path: str) -> DQResult:
    name_u = name.upper()
    spec = EXPECTED_SCHEMAS.get(name_u, None)
    issues: List[Dict[str, object]] = []

    # Schema
    if spec:
        sch = check_schema(df_raw, spec["columns"])
        if sch["schema_drift"]:
            issues.append({
                "dataset": name_u, "date": None, "rule_id": "SCHEMA_DRIFT",
                "severity": "critical",
                "message": f"Schema drift detected. Missing={sch['missing']}, Extra={sch['extra']}",
                "owner": "Data Steward", "status": "open"
            })

    # Index
    date_col = [c for c in df_raw.columns if c.lower() == "date"][0]
    df = _to_datetime_index(df_raw, date_col=date_col)

    # Duplicates
    dup_mask = df.index.duplicated(keep=False)
    duplicates_df = pd.DataFrame()
    if dup_mask.any():
        duplicates_df = df[dup_mask].reset_index().rename(columns={date_col: "DATE"})
        issues.append({
            "dataset": name_u, "date": duplicates_df["DATE"].min(), "rule_id": "DUPLICATE_DATES",
            "severity": "critical", "message": f"Found {dup_mask.sum()} rows with duplicate dates.",
            "owner": "Data Steward", "status": "open"
        })
        df = df[~df.index.duplicated(keep="last")]

    # Value series
    value_col = [c for c in df.columns if c.lower() != "date"]
    if len(value_col) != 1:
        value_col = [df.columns[-1]]
    series_raw = pd.to_numeric(df[value_col[0]], errors="coerce").rename("value_raw")

    # Frequency & freshness
    freq_code = _freq_code(df.index)
    fresh_days = _freshness_days(df.index)
    fresh_band = _freshness_band(freq_code, fresh_days)
    fresh_band_safe = fresh_band if fresh_band in {"green", "amber", "red"} else "unknown"

    # Anchor drift
    if spec:
        anchor_drift, inferred_anchor = check_anchor(df.index, spec["anchor"])
        if anchor_drift:
            issues.append({
                "dataset": name_u, "date": df.index.min(), "rule_id": "ANCHOR_DRIFT",
                "severity": "critical",
                "message": f"Anchor drift: expected {spec['anchor']}, inferred {inferred_anchor}",
                "owner": "Data Steward", "status": "open"
            })

    # Completeness
    misses = missing_periods(df.index, freq_code)

    # Validity frame
    valid = pd.DataFrame({"date": df.index, "value_raw": series_raw.values}).set_index("date")

    # Normalize units
    if name_u in PERCENT_DATASETS:
        valid[f"{name_u.lower()}_was_percent"] = True
        valid["value"] = (pd.to_numeric(valid["value_raw"], errors="coerce")) / 100.0
    else:
        valid["value"] = pd.to_numeric(valid["value_raw"], errors="coerce")

    # GDP: quarter-end convenience index
    if name_u == "GDP":
        valid = valid.copy()
        valid["quarter_end_index"] = as_quarter_end(valid.index)

    # Invalid negatives (by kind)
    kind = SERIES_KIND.get(name_u, "change_pct")
    if kind == "level_pct":
        valid["invalid_negative"] = valid["value"] < 0
    elif kind == "level_level":
        valid["invalid_negative"] = valid["value"] <= 0
    else:
        valid["invalid_negative"] = False
    if isinstance(valid["invalid_negative"], pd.Series) and valid["invalid_negative"].any():
        issues.append({
            "dataset": name_u, "date": valid.index[valid["invalid_negative"]].min(),
            "rule_id": "NEGATIVE_VALUE", "severity": "warn",
            "message": "Negative values observed where not permitted by series kind.",
            "owner": "Data Steward", "status": "open"
        })

    # ---------------- Outlier detection (per-series params + adaptive relaxation + hard rules)
    p = SERIES_OUTLIER_PARAMS.get(name_u)
    if p is None:
        det = valid["value"]
        window = 12 if freq_code == "Q" else 24
        z0 = 3.5
    else:
        det = _prep_for_detection(valid["value"], p["mode"])
        # Deadband for small changes if configured (e.g., CBLR 5 bps)
        if p.get("deadband") and p["mode"] in ("change", "pct_change", "log_return"):
            det = det.mask(det.abs() < float(p["deadband"]), 0.0)
        window = int(p["window"])
        z0 = float(p["z"])

    # Baseline robust-MAD flags
    flags = robust_outliers_rolling(det, z_thresh=z0, window=window)

    # Hard rules
    if name_u == "CBLR":
        hard_thr = float(p.get("hard_pp_threshold", 0.05))  # 0.05 = 5pp = 500 bps
        flags = flags | (det.abs() > hard_thr)
    if name_u == "DEGU" and p.get("percentile_hard_rule"):
        pr = float(p["percentile_hard_rule"])
        cutoff = det.abs().quantile(pr)
        flags = flags | (det.abs() > cutoff)

    # Auto-relax z if too many flags post-warmup
    eff = det.rolling(window, min_periods=window).apply(lambda s: 1, raw=True).astype(bool)
    eff_total = int(eff.sum())
    if eff_total:
        share = float((flags & eff).sum()) / eff_total
        if share > MAX_OUTLIER_SHARE:
            scale = np.sqrt(share / MAX_OUTLIER_SHARE)
            z_relaxed = float(min(Z_MAX, z0 * max(1.0, scale)))
            flags = robust_outliers_rolling(det, z_thresh=z_relaxed, window=window)
            # re-apply hard rules (never relaxed)
            if name_u == "CBLR":
                flags = flags | (det.abs() > hard_thr)
            if name_u == "DEGU" and p.get("percentile_hard_rule"):
                cutoff = det.abs().quantile(pr)
                flags = flags | (det.abs() > cutoff)

    valid["is_stat_outlier"] = flags

    # DQ diagnostics (DQ-only)
    det_mode = p["mode"] if p else ("level" if freq_code != "Q" else "level")
    valid["det_series"] = det
    valid["detector_mode"] = det_mode

    # Outliers export
    out = valid[valid["is_stat_outlier"]].copy().reset_index()
    if name_u == "GDP":
        out["export_date"] = (
            out["date"].dt.to_period("Q-DEC")
                         .dt.to_timestamp(how="end")
                         .dt.normalize()
        )
    else:
        out["export_date"] = out["date"]

    # Modelling notes (as DQ guidance)
    notes = []
    for _, row in out.iterrows():
        notes.append({
            "dataset": name_u,
            "date": row["export_date"],
            "reason": "stat_outlier",
            "recommended_treatment": "winsorize_1_99"
        })
    modelling_notes = pd.DataFrame(notes) if notes else pd.DataFrame(columns=["dataset","date","reason","recommended_treatment"])

    # ---------------- DQ Score (0–100)
    w = DQ_SCORE_WEIGHTS["default"]
    sum_w = float(sum(w.values())) if sum(w.values()) != 0 else 1.0

    idxn = pd.DatetimeIndex(pd.to_datetime(valid.reset_index()["date"], errors="coerce").dropna())
    if len(idxn) > 1:
        if freq_code == "Q":
            expected = len(pd.period_range(idxn.min().to_period("Q-DEC"), idxn.max().to_period("Q-DEC"), freq="Q-DEC"))
        else:
            expected = len(pd.period_range(idxn.min().to_period("M"), idxn.max().to_period("M"), freq="M"))
    else:
        expected = len(idxn)
    missing_count = len(misses)
    completeness = float(max(0.0, 1.0 - (missing_count / expected))) if expected else 1.0

    timeliness_map = {"green": 1.0, "amber": 0.6, "red": 0.0, "unknown": 0.3}
    timeliness = timeliness_map.get(fresh_band_safe, 0.3)

    dup_rows = int(duplicates_df.shape[0])
    n_obs_final = int(valid["value"].notna().sum())
    denom = n_obs_final + dup_rows if (n_obs_final + dup_rows) > 0 else 1
    uniqueness = float(max(0.0, 1.0 - (dup_rows / denom)))

    total_pts = int(valid["value"].notna().sum())
    out_share = float(valid["is_stat_outlier"].sum() / total_pts) if total_pts else 0.0
    outliers_score = float(max(0.0, 1.0 - out_share))

    invneg_share = float(valid["invalid_negative"].sum() / total_pts) if (total_pts and isinstance(valid["invalid_negative"], pd.Series)) else 0.0
    invalid_negatives = float(max(0.0, 1.0 - invneg_share))

    score_0_1_raw = (w["Completeness"]     * completeness +
                     w["Timeliness"]       * timeliness +
                     w["Uniqueness"]       * uniqueness +
                     w["Outliers"]         * outliers_score +
                     w["InvalidNegatives"] * invalid_negatives)
    score_0_1 = score_0_1_raw / sum_w
    dq_score = round(100.0 * score_0_1, 1)

    scorecard = {
        "dataset": name_u,
        "Completeness": round(completeness, 3),
        "Timeliness": round(timeliness, 3),
        "Uniqueness": round(uniqueness, 3),
        "Outliers": round(outliers_score, 3),
        "InvalidNegatives": round(invalid_negatives, 3),
        "DQ_Score": dq_score
    }

    # ---------------- Summary (per dataset)
    summ = {
        "dataset": name_u,
        "start": valid.reset_index()["date"].min(),
        "end": valid.reset_index()["date"].max(),
        "freq_inferred": freq_code,
        "freshness_days": fresh_days,
        "freshness_band": fresh_band_safe,
        "missing_periods_count": len(misses),
        "duplicate_rows": dup_rows,
        "stat_outliers": int(valid["is_stat_outlier"].sum()),
        "invalid_negatives": int(valid["invalid_negative"].sum()) if isinstance(valid["invalid_negative"], pd.Series) else 0,
        "dq_score": dq_score,
        "source_file": source_path,
        "n_obs": int(valid["value"].notna().sum()),
    }

    # Prune any legacy policy/shock/winsor columns and return
    valid = _prune_validity_columns(valid.reset_index())

    return DQResult(
        name=name_u,
        validity=valid,
        summary=summ,
        outliers=out,
        issues=issues,
        modelling_notes=modelling_notes,
        duplicates=duplicates_df,
        scorecard=scorecard
    )

# =============================================================================
# Overall helpers & Report writer
# =============================================================================

def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df

def _overall_rows(scorecard_df: pd.DataFrame, summary_df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """
    Build macro-average and observation-weighted average rows for Scorecard metrics.
    Returns (macro_row, weighted_row) with dataset labels 'OVERALL_MACRO' and 'OVERALL_WEIGHTED'.
    """
    wanted = ["Completeness", "Timeliness", "Uniqueness", "Outliers", "InvalidNegatives", "DQ_Score"]
    if scorecard_df is None or scorecard_df.empty:
        base = {k: np.nan for k in wanted}
        return pd.Series({"dataset": "OVERALL_MACRO", **base}), pd.Series({"dataset": "OVERALL_WEIGHTED", **base})

    sc = scorecard_df.copy()
    for c in wanted:
        if c in sc.columns:
            sc[c] = pd.to_numeric(sc[c], errors="coerce")

    macro_vals = sc[wanted].mean(numeric_only=True)
    macro_row = pd.Series({"dataset": "OVERALL_MACRO", **macro_vals.to_dict()})

    if summary_df is not None and not summary_df.empty and "n_obs" in summary_df.columns:
        w_map = dict(zip(summary_df["dataset"], pd.to_numeric(summary_df["n_obs"], errors="coerce").fillna(0)))
        weights = sc["dataset"].map(w_map).fillna(0).astype(float)
        if weights.sum() > 0:
            weighted_vals = (sc[wanted].multiply(weights, axis=0).sum(numeric_only=True) / weights.sum())
        else:
            weighted_vals = macro_vals
    else:
        weighted_vals = macro_vals
    weighted_row = pd.Series({"dataset": "OVERALL_WEIGHTED", **weighted_vals.to_dict()})
    return macro_row, weighted_row

def write_report(results: List[DQResult], xlsx_path: str):
    # Read previous report for changelog (if any)
    prev_summary_df, prev_outliers_df = _read_prev_report(xlsx_path)

    Path(xlsx_path).parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as xw:
        # Build Summary and Scorecard; compute OVERALL rows
        summary_df = pd.DataFrame([r.summary for r in results])

        sc = pd.DataFrame([r.scorecard for r in results]) if results else pd.DataFrame()
        if sc.empty:
            sc = pd.DataFrame(columns=["dataset", "Completeness", "Timeliness", "Uniqueness", "Outliers", "InvalidNegatives", "DQ_Score"])

        macro_row, weighted_row = _overall_rows(sc, summary_df)
        sc_overall = pd.concat([sc, pd.DataFrame([macro_row, weighted_row])], ignore_index=True)

        # Ensure dq_score exists; append OVERALL rows to Summary (dq_score only)
        if "dq_score" not in summary_df.columns:
            summary_df["dq_score"] = np.nan
        overall_summary_rows = []
        for row in (macro_row, weighted_row):
            overall_summary_rows.append({
                "dataset": row["dataset"],
                "dq_score": round(float(row["DQ_Score"]), 1) if pd.notna(row.get("DQ_Score", np.nan)) else np.nan
            })
        summary_df = pd.concat([summary_df, pd.DataFrame(overall_summary_rows)], ignore_index=True)

        # Write Summary
        summary_df.to_excel(xw, sheet_name="Summary", index=False)

        # Validity sheets (pruned)
        for r in results:
            sh = f"{r.name}_Validity"
            v = _prune_validity_columns(r.validity.copy())
            v.to_excel(xw, sheet_name=sh, index=False)

        # Missing Periods
        miss_rows = []
        for r in results:
            idx = pd.to_datetime(r.validity["date"], errors="coerce")
            freq_code = r.summary["freq_inferred"]
            miss = missing_periods(idx, freq_code)
            for ts in miss:
                miss_rows.append({"dataset": r.name, "date": ts})
        mp = pd.DataFrame(miss_rows) if miss_rows else pd.DataFrame(columns=["dataset", "date"])
        mp.to_excel(xw, sheet_name="Missing_Periods", index=False)

        # Duplicates
        dups = []
        for r in results:
            if not r.duplicates.empty:
                df = r.duplicates.copy()
                df.insert(0, "dataset", r.name)
                dups.append(df)
        dups_df = pd.concat(dups, ignore_index=True) if dups else pd.DataFrame(columns=["dataset"])
        dups_df.to_excel(xw, sheet_name="Duplicates", index=False)

        # Outliers (collect before changelog)
        outs = []
        for r in results:
            df = r.outliers.copy()
            if not df.empty:
                df.insert(0, "dataset", r.name)
                outs.append(df)
        out_df = pd.concat(outs, ignore_index=True) if outs else pd.DataFrame(columns=["dataset"])
        out_df.to_excel(xw, sheet_name="Outliers", index=False)

        # Issues
        issues_rows = []
        for r in results:
            issues_rows.extend(r.issues)
        issues_df = pd.DataFrame(issues_rows) if issues_rows else pd.DataFrame(
            columns=["dataset", "date", "rule_id", "severity", "message", "owner", "status"]
        )
        issues_df.to_excel(xw, sheet_name="Issues", index=False)

        # Modelling Notes (as DQ guidance)
        mn = pd.concat([r.modelling_notes for r in results], ignore_index=True) if results else pd.DataFrame()
        if mn.empty:
            mn = pd.DataFrame(columns=["dataset", "date", "reason", "recommended_treatment"])
        mn.to_excel(xw, sheet_name="Modelling_Notes", index=False)

        # Scorecard (with OVERALL rows)
        sc_overall.to_excel(xw, sheet_name="Scorecard", index=False)

        # Overall sheet (macro & weighted)
        overall_tbl = pd.DataFrame([macro_row, weighted_row])
        overall_tbl.to_excel(xw, sheet_name="Overall", index=False)

        # Metadata
        meta_rows = []
        for r in results:
            meta_rows.append({"dataset": r.name, **r.summary})
        pd.DataFrame(meta_rows).to_excel(xw, sheet_name="Metadata", index=False)

        # Contents
        contents = pd.DataFrame({
            "Sheet": ["Summary"]
                    + [f"{r.name}_Validity" for r in results]
                    + ["Missing_Periods", "Duplicates", "Outliers", "Issues", "Modelling_Notes",
                       "Scorecard", "Overall", "Metadata", "Changelog_Summary",
                       "Changelog_Outliers_Added", "Changelog_Outliers_Removed"],
            "Description": (
                ["High-level metrics incl. individual + OVERALL DQ score"]
                + [f"Row-level checks for {r.name}" for r in results]
                + [
                    "List of missing timestamps by dataset",
                    "Duplicate date rows (if any)",
                    "Statistical outliers (robust MAD with tuned params)",
                    "Audit-style issues register",
                    "Suggested modelling treatments per flagged point",
                    "Per-dataset sub-scores + OVERALL rows",
                    "Overall DQ: macro & observation-weighted averages",
                    "Technical summary incl. DQ fields",
                    "Per-dataset changes vs previous report",
                    "Newly flagged outlier timestamps",
                    "Outlier timestamps that disappeared"
                ]
            )
        })
        contents.to_excel(xw, sheet_name="Contents", index=False)

        # ---------- CHANGELOG (if previous report available) ----------
        if prev_summary_df is not None and prev_outliers_df is not None:
            chg_summary, chg_added, chg_removed = _build_changelog(
                summary_df.copy(), out_df.copy(),
                prev_summary_df, prev_outliers_df
            )
            chg_summary.to_excel(xw, sheet_name="Changelog_Summary", index=False)
            chg_added.to_excel(xw, sheet_name="Changelog_Outliers_Added", index=False)
            chg_removed.to_excel(xw, sheet_name="Changelog_Outliers_Removed", index=False)
        else:
            pd.DataFrame({
                "note": ["No previous report found at this path; changelog will start next run."]
            }).to_excel(xw, sheet_name="Changelog_Summary", index=False)

# =============================================================================
# Main (paths)
# =============================================================================
def load_sql(con:pyodbc.Connection, table_name: str) -> pd.DataFrame:
    """Query a database table and return a DataFrame with stripped column names."""
    query = f"SELECT * FROM {table_name}"
    df = pd.read_sql(query, con)
    df.columns = [c.strip() for c in df.columns]
    return df


def main():

    db_conn = DatabaseConnection()#new_14/10/25
    con = db_conn.get_db_connection()#new_14/10/25
    files = {
        "NPL":   "NPL_raw",
        "GDP":   "GDP_raw",
        "GLA":  "GLA_raw",
        "CBLR": "CBLR_raw",
        "DEGU": "DEGU_raw",
    }

    results: List[DQResult] = []
    for name in ["NPL", "GDP", "GLA", "CBLR", "DEGU"]:
        table_name = files[name]
        try:
            df = load_sql(con, table_name)
    
        except Exception as e:
            results.append(DQResult(
                name=name,
                validity=pd.DataFrame(columns=["date", "value_raw", "value"]),
                summary={"dataset": name, "n_obs": 0, "start": None, "end": None,
                         "freq_inferred": "M", "freshness_days": None, "freshness_band": "unknown",
                         "missing_periods_count": 0, "duplicate_rows": 0,
                         "stat_outliers": 0, "invalid_negatives": 0,
                         "dq_score": 0.0, "source_file": table_name},
                outliers=pd.DataFrame(),
                issues=[{
                    "dataset": name,
                    "date": None,
                    "rule_id": "FILE_MISSING",
                    "severity": "critical",
                    "message": f"Unable to read file: {e}",
                    "owner": "Data Steward",
                    "status": "open"
                }],
                modelling_notes=pd.DataFrame(columns=["dataset", "date", "reason", "recommended_treatment"]),
                duplicates=pd.DataFrame(),
                scorecard={"dataset": name, "Completeness": 0, "Timeliness": 0, "Uniqueness": 0, "Outliers": 0, "InvalidNegatives": 0, "DQ_Score": 0}
            ))
            continue
        results.append(evaluate_series(name, df, table_name))

    # Save to your reports folder (same path enables changelog diffs across runs)
    # report_path = '/Users/sheilaamoafo/Desktop/NPLscmidas - FINAL_13_10_2025_V.1/NPLscmidas/NPLscmidas/reports/NPL_DQ_Report_updated.xlsx'
    # write_report(results, report_path)

    #Saving to SQL database

    for r in results:
        r.validity["train_date"] = datetime.now()
        r.validity.to_sql(f'{r.name}_validity', if_exists='append', index=False,schema='dbo', con=con)

    for r in results:
        r.outliers['train_date'] = datetime.now()
        r.outliers.to_sql(f'{r.name}_outliers', if_exists='append', index=False,schema='dbo', con=con)

    for r in results:
        if len(r.duplicates) > 0:
            r.duplicates['train_date'] = datetime.now()
            r.duplicates.to_sql('duplicates', if_exists='append', index=False,schema='dbo', con=con)

    for r in results:
        r.modelling_notes['train_date'] = datetime.now()
        r.modelling_notes.to_sql(f'{r.name}_modelling_notes', if_exists='append', index=False,schema='dbo', con=con)

    for r in results:
        sm = pd.DataFrame([r.summary])
        sm['train_date'] = datetime.now()
        sm.to_sql(f'{r.name}_summary', if_exists='append', index=False,schema='dbo', con=con)


    for r in results:
        sc = pd.DataFrame([r.scorecard])
        sc['train_date'] = datetime.now()    
        sc.to_sql(f'{r.name}_scorecard', if_exists='append', index=False,schema='dbo', con=con)

    for r in results:
        if r.issues:
            iss = pd.DataFrame(r.issues)
            iss['train_date'] = datetime.now()
            iss.to_sql(f'{r.name}_issues', if_exists='append', index=False,schema='dbo', con=con)

if __name__ == "__main__":
    main()