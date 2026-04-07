"""
monitoring.py
Production monitoring for industry NPL forecasts.

Aligned to your codebase:
- Model types & names (from pipeline logic):
    * midas_ml     -> data/interim/models/lgbm_midas_h{h}.pkl (joblib)
    * sc_midas_ols -> data/interim/models/sc_midas_ols_h{h}.json (SCMIDASOLS.load)
  Rule: cfg.model_type_per_horizon overrides; else 'midas_ml' if h<=2, 'sc_midas_ols' if h>2.
- Features use features.build_feature_matrix(panel, cfg, h); autodetects (panel,h,cfg) too.
- SCMIDASOLS.predict(panel, horizon=h) returns log NPL at time t; we convert to % and shift index to t+h.
- Actuals detect your 'NPL' (log of decimal) -> exp * 100.

New artifacts this file writes each run (CSV + PNG):
- Forecast path (t+1..t+6)
- Backtest heatmap(s) across full history (abs error; optionally SMAPE)
- Actual vs Pred (full history; overlay + per-horizon)
- Feature importance per horizon
- Model comparison across horizons
"""

from __future__ import annotations

import os
os.environ.setdefault("MPLBACKEND","Agg")

from pathlib import Path
from typing import Dict, Tuple, Optional, Iterable
from datetime import datetime, UTC
import importlib
import json
import re

import numpy as np
import pandas as pd
import yaml

from report_paths import ReportRoot, repro_manifest
from kpi_reporter import (
    MetricThresholds, RunStamp,
    rmse_pp, mae_pp, smape, mase, r2_log,
    log_monitoring, log_gates
)
from drift import drift_report
from notifier import send_alert
from pathlib import Path
import pandas as pd
import numpy as np
from report_paths import ReportRoot  # use your existing path helper

from npl_dq.db import DatabaseConnection # use your existing DB helper

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "config.yml").read_text()) or {}
MODELS_BASE = Path(CFG.get("models_folder") or "data/interim/models")
CAND_SUB = str(CFG.get("candidate_subdir", "candidate"))
MODELS_CAND = ROOT / MODELS_BASE / CAND_SUB

db_con = DatabaseConnection()

def _build_forward_path_block(df_all_actual_pred: pd.DataFrame,
                              panel: pd.DataFrame,
                              as_of_date: pd.Timestamp) -> pd.DataFrame:
    """
    Build the authoritative 6-month forward block using the last month that has a real actual.
    If no actual is present in df_all_actual_pred, fall back to the panel's last month.
    Always produce horizons 1..6 starting from base_month + 1M.
    """
    as_of = pd.to_datetime(as_of_date).normalize()

    df = df_all_actual_pred.copy()
    df["target_month"] = pd.to_datetime(df["target_month"], errors="coerce")
    df["as_of_date"]   = pd.to_datetime(df.get("as_of_date"), errors="coerce")
    df["horizon"]      = pd.to_numeric(df.get("horizon"), errors="coerce").astype("Int64")

    # 1) Determine the base (last actual) month dynamically from the reconciled data
    has_actual = df.dropna(subset=["y_true_pct"])
    if not has_actual.empty:
        base_month = pd.to_datetime(has_actual["target_month"].max()).normalize()
    else:
        # Fallback to the panel's last timestamp
        base_month = (pd.to_datetime(panel.index.max()) + pd.offsets.MonthEnd(0)).normalize()

    # 2) For each horizon 1..6, pick the latest prediction for the exact forward month
    rows = []
    for h in [1, 2, 3, 4, 5, 6]:
        tm = (base_month + pd.offsets.MonthEnd(h)).normalize()
        pool = df[df["horizon"] == h]
        exact = pool[pool["target_month"] == tm].sort_values("as_of_date")

        if not exact.empty:
            y = float(pd.to_numeric(exact["y_pred_pct"], errors="coerce").iloc[-1])
            rows.append((as_of, base_month, h, tm, y))
        else:
            # No exact row for that month: take the most recent available pred for this horizon (if any)
            if not pool.empty:
                pool = pool.sort_values(["target_month", "as_of_date"])
                last = pool.iloc[-1]
                y = float(pd.to_numeric(last["y_pred_pct"], errors="coerce"))
                rows.append((as_of, base_month, h, tm, y))  # keep tm as the forward month for clarity
            else:
                rows.append((as_of, base_month, h, tm, np.nan))

    out = pd.DataFrame(rows, columns=["as_of_date", "info_month", "horizon", "target_month", "y_pred_pct"])
    return out.sort_values("horizon")


def _upsert_csv(csv_path: Path, df: pd.DataFrame, key_cols, date_cols=(), sort_by=(), round_cols=None) -> Path:
    """
    Idempotent CSV upsert (append-with-history then de-duplicate on key_cols).
    This is *file-based* (not SQL). Keeps your monitoring artifacts deterministic and scheduler-safe.
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df = df.copy()

    # normalize datetimes
    for c in date_cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    if round_cols:
        for c, d in round_cols.items():
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").round(d)

    if csv_path.exists():
        prev = pd.read_csv(csv_path)
        for c in date_cols:
            if c in prev.columns:
                prev[c] = pd.to_datetime(prev[c], errors="coerce")
        merged = pd.concat([prev, df], ignore_index=True)
        merged = merged.drop_duplicates(subset=list(key_cols), keep="last")
    else:
        merged = df

    if sort_by:
        cols = [c for c in sort_by if c in merged.columns]
        if cols:
            merged = merged.sort_values(cols)

    merged.to_csv(csv_path, index=False)
    return csv_path


# --- ReportRoot directory resolvers# --- ReportRoot directory resolvers (compatible with different implementations) ---

def _rr_base_dir(reports_root) -> Path:
    """
    Find the base 'reports' directory from your ReportRoot object.
    Tries common attribute names; falls back to treating the object as a path,
    and finally to ROOT/'reports'.
    """
    for attr in ("root", "base", "reports_root", "root_dir", "base_dir", "reports_dir"):
        val = getattr(reports_root, attr, None)
        if val:
            try:
                return Path(val)
            except Exception:
                pass
    try:
        return Path(reports_root)
    except Exception:
        pass
    return ROOT / "reports"

def _rr_dirs(reports_root) -> Tuple[Path, Path]:
    """
    Return (data_dir, charts_dir) under the base reports directory.
    If your ReportRoot exposes data_dir/charts_dir properties or callables,
    we use them; otherwise construct base/_data and base/charts.
    """
    # explicit methods/properties
    if hasattr(reports_root, "data_dir"):
        val = getattr(reports_root, "data_dir")
        try:
            if callable(val):  # method
                data_dir = Path(val())
            else:              # property/attr
                data_dir = Path(val)
            charts_attr = getattr(reports_root, "charts_dir", None)
            if charts_attr is not None:
                charts_dir = Path(charts_attr() if callable(charts_attr) else charts_attr)
            else:
                charts_dir = _rr_base_dir(reports_root) / "charts"
            return data_dir, charts_dir
        except Exception:
            pass
    # fallback to base
    base = _rr_base_dir(reports_root)
    return base / "_data", base / "charts"

def save_forecast_path_authoritative(df_path: pd.DataFrame, reports_root: ReportRoot) -> dict:
    data_dir, charts_dir = _rr_dirs(reports_root)
    data_dir.mkdir(parents=True, exist_ok=True); charts_dir.mkdir(parents=True, exist_ok=True)

    csv_path = data_dir / "forecast_path.csv"
    _upsert_csv(
        csv_path, df_path,
        key_cols=["as_of_date","info_month","horizon"],
        date_cols=["as_of_date","info_month","target_month"],
        sort_by=["as_of_date","horizon"],
        round_cols={"y_pred_pct": 6},
    )

    # latest block PNG
    try:
        import matplotlib.pyplot as plt
        df_plot = pd.read_csv(csv_path, parse_dates=["as_of_date","info_month","target_month"])
        latest_asof = df_plot["as_of_date"].max()
        d = df_plot[df_plot["as_of_date"] == latest_asof].sort_values("horizon")
        fig_path = charts_dir / "forecast_path_latest.png"
        plt.figure(figsize=(8,4.5))
        if not d.dropna(subset=["y_pred_pct"]).empty:
            plt.plot(d["target_month"], d["y_pred_pct"], marker="o")
        else:
            plt.text(0.5,0.5,"No forward values available for the latest run", ha="center", va="center", transform=plt.gca().transAxes)
        plt.title("6-Month Forward Forecast Path")
        plt.xlabel("Target month"); plt.ylabel("NPL forecast (%)")
        plt.grid(True, alpha=0.3); plt.tight_layout(); plt.savefig(fig_path, dpi=160); plt.close()
        print(f"[monitoring] Forecast path CSV  => {csv_path.resolve()}")
        print(f"[monitoring] Forecast path PNG  => {fig_path.resolve()}")
    except Exception as e:
        print(f"[monitoring] NOTE: forecast path chart failed: {e}")

    return {"csv": str(csv_path)}

def write_forecast_path_table_latest(df_path: pd.DataFrame, reports_root: ReportRoot) -> str:
    data_dir, _ = _rr_dirs(reports_root)
    data_dir.mkdir(parents=True, exist_ok=True)
    d = df_path.copy()
    for c in ("as_of_date","target_month"):
        if c in d.columns:
            d[c] = pd.to_datetime(d[c], errors="coerce")
    if "as_of_date" in d.columns:
        d = d[d["as_of_date"] == d["as_of_date"].max()].copy()
    d = d.sort_values("horizon")
    d["month"] = d["target_month"].dt.strftime("%b-%Y")
    out = d[["as_of_date","horizon","month","y_pred_pct"]].rename(columns={"y_pred_pct":"npl_forecast_pct"})
    csv_path = data_dir / "forecast_path_table_latest.csv"
    _upsert_csv(
        csv_path, out,
        key_cols=["as_of_date","horizon"],
        date_cols=["as_of_date"],
        sort_by=["as_of_date","horizon"],
        round_cols={"npl_forecast_pct": 6},
    )
    print(f"[monitoring] Forecast path TABLE => {csv_path.resolve()}")
    return str(csv_path)

def write_forecast_path_metrics_latest(df_actual_pred: pd.DataFrame, reports_root: ReportRoot, window_months: int = 12) -> str:
    data_dir, _ = _rr_dirs(reports_root)
    data_dir.mkdir(parents=True, exist_ok=True)
    df = df_actual_pred.copy()
    df["target_month"] = pd.to_datetime(df["target_month"], errors="coerce")
    df = df.dropna(subset=["target_month","horizon","y_true_pct","y_pred_pct"])
    if df.empty:
        out = data_dir / "forecast_path_metrics_latest.csv"
        _upsert_csv(out, pd.DataFrame(columns=["as_of_date","horizon","metric","value","window"]),
                    key_cols=["as_of_date","horizon","metric","window"])
        return str(out)
    recent = sorted(df["target_month"].unique())[-window_months:]
    d = df[df["target_month"].isin(recent)].sort_values(["horizon","target_month"])
    d["y_naive"] = d.groupby("horizon")["y_true_pct"].shift(1)
    as_of = pd.to_datetime(df.get("as_of_date", pd.Timestamp("today"))).max().normalize()
    rows = []
    for h in [1,2,3,4,5,6]:
        z = d[d["horizon"] == h].dropna(subset=["y_true_pct","y_pred_pct"])
        if z.empty: continue
        yt, yp = pd.to_numeric(z["y_true_pct"]), pd.to_numeric(z["y_pred_pct"])
        rmse = float(np.sqrt(np.mean((yt-yp)**2)))
        mae  = float(np.mean(np.abs(yt-yp)))
        denom = (yt - pd.to_numeric(z["y_naive"])).abs()
        mase_val = float(np.mean(np.abs(yt-yp)) / (np.mean(denom) if np.isfinite(denom).any() else np.nan))
        smape_val = float(np.mean(200.0*np.abs(yp-yt)/(np.abs(yt)+np.abs(yp))))
        lt = np.log(np.clip(yt/100.0, 1e-6, None)); lp = np.log(np.clip(yp/100.0, 1e-6, None))
        r2 = float(1.0 - np.sum((lt-lp)**2)/np.sum((lt-lt.mean())**2)) if np.isfinite(lt).any() else np.nan
        rows += [
            {"as_of_date":as_of, "horizon":h, "metric":"RMSE_pp", "value":rmse},
            {"as_of_date":as_of, "horizon":h, "metric":"MAE_pp",  "value":mae},
            {"as_of_date":as_of, "horizon":h, "metric":"SMAPE",   "value":smape_val},
            {"as_of_date":as_of, "horizon":h, "metric":"MASE",    "value":mase_val},
            {"as_of_date":as_of, "horizon":h, "metric":"R2_log",  "value":r2},
        ]
    out = pd.DataFrame(rows); out["window"] = f"last_{window_months}m_with_actuals"
    csv_path = data_dir / "forecast_path_metrics_latest.csv"
    _upsert_csv(
        csv_path, out,
        key_cols=["as_of_date","horizon","metric","window"],
        date_cols=["as_of_date"],
        sort_by=["as_of_date","horizon","metric"],
        round_cols={"value": 6},
    )
    print(f"[monitoring] Forecast performance METRICS => {csv_path.resolve()}")
    return str(csv_path)

# ---------- Repo root ----------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]

# ---------- Config (multi-document YAML) --------------------------------------

def load_config(cfg_path: Path) -> dict:
    docs = list(yaml.safe_load_all(cfg_path.read_text()))
    cfg: Dict = {}
    for d in docs:
        if isinstance(d, dict):
            cfg.update(d)

    cfg.setdefault("data_folder", "data")
    cfg.setdefault("models_folder", "data/interim/models")
    cfg.setdefault("reports_folder", "reports")

    cfg.setdefault("horizons", [0, 1, 2, 3, 4, 5, 6])
    cfg.setdefault("model_version", "v0.0.0")
    cfg.setdefault("cfg_hash", "NA")
    cfg.setdefault("code_commit", "")

    mon = cfg.setdefault("monitoring", {})
    mon.setdefault("thresholds", {
        "rmse_pp_max": 3.0, "mae_pp_max": 2.5, "smape_max": 30.0, "mase_max": 1.0, "r2_log_min": 0.20
    })
    mon.setdefault("gate_window", 12)

    cfg.setdefault("features", {}).setdefault("scoring_entrypoint", "auto")
    npl_cfg = cfg.setdefault("npl", {})
    npl_cfg.setdefault("ln_column_override", "")
    npl_cfg.setdefault("value_column_override", "")
    npl_cfg.setdefault("value_unit_override", "")   # "decimal" or "percent"

    return cfg

# ---------- Panel loader (expects 'date' in CSV, any case) --------------------

def _load_panel_df(cfg: Dict) -> pd.DataFrame:
    data_dir = ROOT / cfg["data_folder"]
    pq = data_dir / "t_panel.parquet"
    csv = data_dir / "t_panel.csv"

    df = db_con._read_sql(pq)
    if "date" in df.columns:
        idx = pd.to_datetime(df["date"], errors="raise")
        df = df.drop(columns=["date"]).set_index(idx)
    elif isinstance(df.index, (pd.DatetimeIndex, pd.PeriodIndex)):
        if isinstance(df.index, pd.PeriodIndex):
            df.index = df.index.to_timestamp("M")
    

    if not isinstance(df.index, pd.DatetimeIndex):
        head = db_con._read_sql(csv)
        lower = {c.lower(): c for c in head.columns}
        if "date" not in lower:
            raise ValueError(f"'date' column not found in {csv}.")
        date_col = lower["date"]
        df = pd.read_csv(csv, parse_dates=[date_col], low_memory=False)
        df = df.rename(columns={date_col: "date"}).set_index("date")


    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="raise")

    df.index = (df.index + pd.offsets.MonthEnd(0))
    df.index.name = "DATE"
    df = df.sort_index()

    print(f"Panel loaded from {pq if pq.exists() else csv}: {df.shape[0]} rows, {df.shape[1]} cols; "
          f"first={df.index.min().date()} last={df.index.max().date()}")
    return df


# ---------- Panel health checks -------------------------------------------------

def validate_panel_health(panel: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    """
    Lightweight but production-grade panel validation:
    - index is monthly and increasing
    - required target column exists
    - staleness check vs as_of_date (if provided)
    - missingness summary for last N months
    Returns a dataframe of checks (check, status, detail).
    """
    checks = []
    if panel is None or panel.empty:
        checks.append({"check":"panel_nonempty","status":"FAIL","detail":"panel is empty"})
        return pd.DataFrame(checks)

    # index checks
    if not isinstance(panel.index, pd.DatetimeIndex):
        checks.append({"check":"index_datetime","status":"FAIL","detail":f"type={type(panel.index)}"})
    else:
        if not panel.index.is_monotonic_increasing:
            checks.append({"check":"index_monotonic","status":"FAIL","detail":"index not increasing"})
        # monthly cadence (MonthEnd)
        diffs = panel.index.to_series().diff().dropna()
        if not diffs.empty:
            ok = diffs.dt.days.between(27, 32).mean() >= 0.9  # tolerant
            checks.append({"check":"monthly_cadence","status":"PASS" if ok else "WARN",
                           "detail":f"pct_monthlike={diffs.dt.days.between(27,32).mean():.2f}"})

    # target presence
    try:
        _ = actuals_from_panel(panel, cfg)
        checks.append({"check":"npl_detectable","status":"PASS","detail":"target detected"})
    except Exception as e:
        checks.append({"check":"npl_detectable","status":"FAIL","detail":str(e)})

    # staleness
    as_of = pd.Timestamp(datetime.now(UTC).date())
    last = pd.to_datetime(panel.index.max()).normalize()
    age_days = int((as_of - last).days)
    checks.append({"check":"panel_staleness_days","status":"PASS" if age_days <= 45 else "WARN",
                   "detail":str(age_days)})

    # missingness for recent window
    win = int((cfg.get("monitoring", {}) or {}).get("gate_window", 12))
    recent = panel.loc[panel.index >= (panel.index.max() - pd.offsets.MonthEnd(win))]
    miss = (recent.isna().mean().sort_values(ascending=False) if not recent.empty else pd.Series(dtype=float))
    if not miss.empty:
        top = miss.head(8).to_dict()
        checks.append({"check":"missingness_recent_top8","status":"PASS","detail":json.dumps(top)})
    return pd.DataFrame(checks)


# ---------- Features glue (robust calling order) ------------------------------

def build_features_for_scoring(panel: pd.DataFrame, horizon: int, cfg: Dict | None = None) -> pd.DataFrame:
    import importlib, inspect
    F = importlib.import_module("features")

    entry = (cfg or {}).get("features", {}).get("scoring_entrypoint", "auto")
    if entry != "auto":
        if not hasattr(F, entry):
            raise AttributeError(f"features.py has no function named '{entry}'.")
        fn = getattr(F, entry)
    else:
        for name in ("build_feature_matrix", "make_X_for_h", "build_X_for_h",
                     "make_xy", "build_xy", "build_features", "make_design_matrix"):
            if hasattr(F, name):
                fn = getattr(F, name)
                break
        else:
            raise AttributeError("features.py lacks a recognized builder; set features.scoring_entrypoint.")

    attempts = [
        lambda: fn(panel, cfg, horizon),
        lambda: fn(panel, horizon, cfg),
        lambda: fn(panel, h=horizon, cfg=cfg),
        lambda: fn(panel, cfg=cfg, h=horizon),
        lambda: fn(panel, horizon),
        lambda: fn(panel, cfg),
        lambda: fn(panel, horizon=horizon),
        lambda: fn(panel, cfg=cfg),
    ]

    out = None; errors = []
    for call in attempts:
        try:
            out = call(); break
        except TypeError as e:
            errors.append(str(e)); continue
    if out is None:
        sig = inspect.signature(fn)
        kw = {}
        if "h" in sig.parameters: kw["h"] = horizon
        if "horizon" in sig.parameters: kw["horizon"] = horizon
        if "cfg" in sig.parameters and cfg is not None: kw["cfg"] = cfg
        out = fn(panel, **kw)

    X = out[0] if isinstance(out, tuple) else out

    if X.index.name is None or str(X.index.name).upper() in {"DATE", "TIME"}:
        X = X.copy()
        X.index = (pd.to_datetime(X.index) + pd.offsets.MonthEnd(horizon))
        X.index.name = "target_month"
    else:
        if X.index.name != "target_month":
            X = X.copy()
            X.index.name = "target_month"
    return X


# ---------- Model registry resolution (aligned to pipeline promotion) ------------

def _models_base(cfg: Dict) -> Path:
    base = Path(cfg.get("models_folder") or "data/interim/models")
    if not base.is_absolute():
        base = (ROOT / base).resolve()
    return base

def _resolve_registry(cfg: Dict) -> Dict:
    return (cfg.get("models_registry") or {}) if isinstance(cfg.get("models_registry"), dict) else {}

def resolve_models_dir(cfg: Dict, run_mode: str | None = None) -> Path:
    """
    Resolve which directory to read model artifacts from.

    run_mode:
      - 'approved'  : use pointer file (LATEST_APPROVED.json) to find approved_dir; fallback to latest approved run_*
      - 'candidate' : use candidate folder
      - 'run_<id>' or '<id>' : use approved/run_<id> if present (else candidate/run_<id>)
    """
    reg = _resolve_registry(cfg)
    cand_sub = str(reg.get("candidate_subdir", "candidate"))
    appr_sub = str(reg.get("approved_subdir", "approved"))
    pointer_name = str(reg.get("pointer_file", "LATEST_APPROVED.json"))

    base = _models_base(cfg)
    cand_dir = base / cand_sub
    appr_dir = base / appr_sub
    cand_dir.mkdir(parents=True, exist_ok=True)
    appr_dir.mkdir(parents=True, exist_ok=True)

    mode = (run_mode or (cfg.get("monitoring", {}) or {}).get("run_mode") or "approved").strip().lower()

    def _latest_run_dir(parent: Path) -> Optional[Path]:
        runs = [p for p in parent.glob("run_*") if p.is_dir()]
        if not runs:
            return None
        runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return runs[0]

    if mode in ("candidate", "candidates"):
        return cand_dir

    if mode.startswith("run_"):
        run_id = mode.replace("run_", "").strip()
    elif mode not in ("approved", "appr"):
        run_id = mode
    else:
        run_id = ""

    if run_id:
        p1 = appr_dir / f"run_{run_id}"
        if p1.exists():
            return p1
        p2 = cand_dir / f"run_{run_id}"
        if p2.exists():
            return p2
        # fallback
        return appr_dir if appr_dir.exists() else cand_dir

    # approved mode: use pointer file
    pointer = appr_dir / pointer_name
    if pointer.exists():
        try:
            j = json.loads(pointer.read_text(encoding="utf-8"))
            appr_path = j.get("approved_dir") or j.get("approved_path") or j.get("models_dir")
            if appr_path:
                p = Path(appr_path)
                if not p.is_absolute():
                    p = (ROOT / p).resolve()
                if p.exists():
                    return p
            rid = j.get("approved_run_id")
            if rid:
                p = appr_dir / f"run_{rid}"
                if p.exists():
                    return p
        except Exception:
            pass

    # fallback: latest approved run_*, else approved root, else candidate
    latest = _latest_run_dir(appr_dir)
    if latest is not None:
        return latest
    return appr_dir if appr_dir.exists() else cand_dir

# ---------- Model type & loader (matching your pipeline) ----------------------

def _model_type_for_h(h: int, cfg: Dict) -> str:
    per = cfg.get("model_type_per_horizon") or {}
    if h in per: return str(per[h]).lower()
    if str(h) in per: return str(per[str(h)]).lower()
    return "midas_ml" if int(h) <= 2 else "sc_midas_ols"

def load_model_for_h(h: int, cfg: Dict, run_mode: str | None = None):
    model_dir = MODELS_CAND
    mtype = _model_type_for_h(h, cfg)

    if mtype == "midas_ml":
        p = model_dir / f"lgbm_midas_h{h}.pkl"
        if not p.exists():
            raise FileNotFoundError(f"[midas_ml] expected model file missing: {p}")
        try:
            import joblib
        except Exception as e:
            raise RuntimeError(f"joblib is required to load {p} ({e})")
        m = joblib.load(p)
        if not hasattr(m, "predict"):
            raise TypeError(f"Loaded object from {p} has no .predict(...) method.")
        try: setattr(m, "version_", cfg.get("model_version", "v0.0.0"))
        except Exception: pass
        return m

    if mtype == "sc_midas_ols":
        from sc_midas import SCMIDASOLS
        p = model_dir / f"sc_midas_ols_h{h}.json"
        if not p.exists():
            raise FileNotFoundError(f"[sc_midas_ols] expected model file missing: {p}")
        m = SCMIDASOLS.load(p)
        try: setattr(m, "version_", cfg.get("model_version", "v0.0.0"))
        except Exception: pass
        return m

    raise ValueError(f"Unsupported model type '{mtype}' for horizon {h}.")


# ---------- Actuals derivation# ---------- Actuals derivation (robust; includes your 'NPL' log column) -------

def _infer_unit(series: pd.Series) -> str:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty: return "unknown"
    p95 = float(s.quantile(0.95))
    if p95 <= 1.0: return "decimal"
    if p95 <= 100.0: return "percent"
    return "percent"

def _find_first(panel: pd.DataFrame, names: Iterable[str]) -> Optional[str]:
    lower = {c.lower(): c for c in panel.columns}
    for n in names:
        if n.lower() in lower: return lower[n.lower()]
    return None

def actuals_from_panel(panel: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    ycol = _find_first(panel, ["y_true_pct"])
    if ycol:
        return (panel[[ycol]].rename(columns={ycol: "y_true_pct"})
                .rename_axis("target_month").reset_index()[["target_month", "y_true_pct"]])

    npl_cfg = cfg.get("npl", {})
    ln_override = (npl_cfg.get("ln_column_override") or "").strip()
    if ln_override:
        if ln_override not in panel.columns:
            raise KeyError(f"npl.ln_column_override '{ln_override}' not found.")
        s = pd.to_numeric(panel[ln_override], errors="coerce")
        y = 100.0 * np.exp(s)
        return (pd.DataFrame({"y_true_pct": y}, index=panel.index)
                .rename_axis("target_month").reset_index()[["target_month", "y_true_pct"]])

    val_override = (npl_cfg.get("value_column_override") or "").strip()
    if val_override:
        if val_override not in panel.columns:
            raise KeyError(f"npl.value_column_override '{val_override}' not found.")
        unit = (npl_cfg.get("value_unit_override") or "").lower().strip()
        s = pd.to_numeric(panel[val_override], errors="coerce")
        if unit not in ("decimal", "percent"):
            unit = _infer_unit(s)
        y = 100.0 * s if unit == "decimal" else s
        return (pd.DataFrame({"y_true_pct": y}, index=panel.index)
                .rename_axis("target_month").reset_index()[["target_month", "y_true_pct"]])

    ln_candidates = ["NPL", "ln_npl", "npl_ln", "log_npl", "npl_log", "ln(npl)", "ln_npl_decimal", "npl_ln_decimal"]
    ln_col = _find_first(panel, ln_candidates)
    if ln_col:
        s = pd.to_numeric(panel[ln_col], errors="coerce")
        y = 100.0 * np.exp(s)
        return (pd.DataFrame({"y_true_pct": y}, index=panel.index)
                .rename_axis("target_month").reset_index()[["target_month", "y_true_pct"]])

    val_candidates = ["npl", "npl_pct", "npl_percent", "npl_percentage", "npl_rate"]
    vcol = _find_first(panel, val_candidates)
    if vcol:
        s = pd.to_numeric(panel[vcol], errors="coerce")
        unit = _infer_unit(s)
        y = 100.0 * s if unit == "decimal" else s
        return (pd.DataFrame({"y_true_pct": y}, index=panel.index)
                .rename_axis("target_month").reset_index()[["target_month", "y_true_pct"]])

    raise KeyError("Could not detect the NPL target in panel; set npl.*_override in config.yml.")

# ---------- Helpers: dirs & conversions ---------------------------------------

def _ensure_dirs(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def _to_pp_from_log(x: np.ndarray | pd.Series) -> np.ndarray:
    return np.exp(np.asarray(x, dtype=float)) * 100.0

# ---------- Forecast path (t+1..t+6) ------------------------------------------

def build_forecast_path(df_preds: pd.DataFrame, panel: pd.DataFrame, cfg: Dict, as_of: pd.Timestamp) -> pd.DataFrame:
    t = pd.Timestamp(panel.index.max())
    horizons = [h for h in range(1, 7) if h in set(cfg.get("horizons", []))]
    if not horizons:
        return pd.DataFrame(columns=["as_of_date","info_month","horizon","target_month","y_pred_pct"])
    wanted = []
    for h in horizons:
        tm = (t + pd.offsets.MonthEnd(h))
        m = df_preds[(df_preds["horizon"] == h) & (pd.to_datetime(df_preds["target_month"]) == tm)]
        if not m.empty:
            m = m.sort_values("as_of_date").tail(1)
            y = float(m["y_pred_pct"].iloc[0])
            wanted.append((as_of.normalize(), t.normalize(), h, tm.normalize(), y))
        else:
            wanted.append((as_of.normalize(), t.normalize(), h, tm.normalize(), np.nan))
    out = pd.DataFrame(wanted, columns=["as_of_date","info_month","horizon","target_month","y_pred_pct"])
    return out.sort_values("horizon")

def write_forecast_path_reports(df_path: pd.DataFrame, cfg: Dict) -> Dict[str, str]:
    reports_dir = ROOT / cfg.get("reports_folder", "reports")
    data_dir = reports_dir / "_data"; charts_dir = reports_dir / "charts"
    _ensure_dirs(data_dir); _ensure_dirs(charts_dir)
    csv_path = data_dir / "forecast_path.csv"
    if csv_path.exists():
        prev = pd.read_csv(csv_path, parse_dates=["as_of_date","info_month","target_month"])
        keep = ["as_of_date","info_month","horizon"]
        merged = (pd.concat([prev, df_path], ignore_index=True)
                    .sort_values(keep).drop_duplicates(subset=keep, keep="last"))
        merged.to_csv(csv_path, index=False)
    else:
        df_path.to_csv(csv_path, index=False)
    png_path = charts_dir / "forecast_path_latest.png"
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 4.5))
        plt.plot(df_path["target_month"], df_path["y_pred_pct"], marker="o")
        plt.title("6-Month Forward Forecast Path"); plt.xlabel("Target month"); plt.ylabel("NPL forecast (%)")
        plt.grid(True, alpha=0.3); plt.tight_layout(); plt.savefig(png_path, dpi=160); plt.close()
    except Exception as e:
        print(f"[monitoring] NOTE: could not render forecast path chart: {e}")
        png_path = None
    return {"csv": str(csv_path), "png": str(png_path) if png_path else ""}

# ---------- Backtest heatmap ---------------------------------------------------


def build_backtest_heatmap_matrix(df_rec: pd.DataFrame, metric: str = "abs_error_pp") -> pd.DataFrame:
    """
    Returns a numeric matrix indexed by target_month with horizon columns.
    (No reset_index; this avoids accidental mixing of datetime and float data.)
    """
    df = df_rec.dropna(subset=["y_true_pct", "y_pred_pct"]).copy()
    if df.empty:
        return pd.DataFrame()

    df["target_month"] = pd.to_datetime(df["target_month"], errors="coerce")
    df = df.dropna(subset=["target_month"])

    if metric == "smape":
        denom = (df["y_pred_pct"].abs() + df["y_true_pct"].abs())
        df["metric_val"] = np.where(
            denom == 0, 0.0, 200.0 * (df["y_pred_pct"] - df["y_true_pct"]).abs() / denom
        )
    else:
        df["metric_val"] = (df["y_pred_pct"] - df["y_true_pct"]).abs()

    mat = df.pivot_table(index="target_month", columns="horizon", values="metric_val", aggfunc="mean")
    mat = mat.sort_index()
    mat.columns = sorted(mat.columns)
    return mat

def write_backtest_heatmap_reports(mat: pd.DataFrame, cfg: Dict, metric: str = "abs_error_pp") -> Dict[str, str]:
    reports_dir = ROOT / cfg.get("reports_folder", "reports")
    data_dir = reports_dir / "_data"; charts_dir = reports_dir / "charts"
    _ensure_dirs(data_dir); _ensure_dirs(charts_dir)

    csv_path = data_dir / f"backtest_heatmap_{metric}.csv"
    png_path = charts_dir / f"backtest_heatmap_{metric}_latest.png"

    if mat is None or mat.empty:
        pd.DataFrame().to_csv(csv_path, index=False)
        return {"csv": str(csv_path), "png": ""}

    # persist matrix (with index)
    mat.to_csv(csv_path, index=True)

    # render
    try:
        import matplotlib.pyplot as plt
        A = mat.to_numpy(dtype=float)
        A_masked = np.ma.masked_invalid(A)

        plt.figure(figsize=(10, max(4, 0.25 * len(mat.index))))
        im = plt.imshow(A_masked, aspect="auto", interpolation="nearest", cmap="Reds", origin="upper")

        plt.yticks(np.arange(len(mat.index)), [pd.to_datetime(ix).strftime("%Y-%m") for ix in mat.index])
        plt.xticks(np.arange(len(mat.columns)), [str(c) for c in mat.columns])

        plt.xlabel("Horizon (months ahead)")
        title = "Backtest Heatmap — Abs Error (pp)" if metric != "smape" else "Backtest Heatmap — SMAPE (%)"
        plt.title(title)
        cbar = plt.colorbar(im)
        cbar.set_label("Abs error (pp)" if metric != "smape" else "SMAPE (%)")

        plt.tight_layout()
        plt.savefig(png_path, dpi=160)
        plt.close()
    except Exception as e:
        print(f"[monitoring] NOTE: could not render backtest heatmap ({metric}): {e}")
        png_path = None

    return {"csv": str(csv_path), "png": str(png_path) if png_path else ""}


# ---------- Feature importance# ---------- Feature importance -------------------------------------------------

def _safe_feature_names(X: pd.DataFrame):
    try: return list(X.columns)
    except Exception: return [f"f{i}" for i in range(X.shape[1])]

def _fi_from_model_and_X(model, X: pd.DataFrame, mtype: str) -> pd.DataFrame:
    fnames = _safe_feature_names(X)
    if mtype == "midas_ml":
        imp = None
        for attr in ("feature_importances_", "feature_importance_"):
            if hasattr(model, attr): imp = getattr(model, attr); break
        if imp is None:
            booster = getattr(model, "booster_", None)
            if booster is not None and hasattr(booster, "feature_importance"):
                imp = booster.feature_importance(importance_type="gain")
        if imp is None: raise ValueError("Could not read feature_importances_ from midas_ml model.")
        imp = np.maximum(np.asarray(imp, dtype=float), 0.0)
        s = imp.sum(); imp = imp/s if s>0 else imp
        return pd.DataFrame({"feature": fnames[:len(imp)], "importance": imp, "method": "gain_norm"})
    if mtype == "sc_midas_ols":
        coef = None
        if hasattr(model, "coef_"):
            coef = np.asarray(getattr(model, "coef_")).ravel()
            if len(coef) != X.shape[1]: coef = None
        if coef is None and hasattr(model, "params_"):
            params = getattr(model, "params_")
            if hasattr(params, "reindex"):
                coef = params.reindex(fnames).fillna(0.0).to_numpy(dtype=float)
        if coef is None:
            params = getattr(model, "params", None)
            if isinstance(params, dict):
                coef = np.array([params.get(f, 0.0) for f in fnames], dtype=float)
        if coef is None: raise ValueError("Could not read coefficients from sc_midas_ols model.")
        std = np.asarray(X.std(ddof=0), dtype=float); std[~np.isfinite(std)] = 0.0
        scaled = np.abs(coef) * np.where(std>0, std, 1.0); s = scaled.sum()
        imp = scaled/s if s>0 else scaled
        return pd.DataFrame({"feature": fnames, "importance": imp, "method": "abs_beta_x_std_norm"})
    raise ValueError(f"Unknown model type '{mtype}' for FI.")

def write_feature_importance_reports(h: int, df_fi: pd.DataFrame, cfg: Dict) -> Dict[str, str]:
    reports_dir = ROOT / cfg.get("reports_folder", "reports")
    data_dir = reports_dir / "_data"; charts_dir = reports_dir / "charts"
    _ensure_dirs(data_dir); _ensure_dirs(charts_dir)
    csv_path = data_dir / f"feature_importance_h{h}.csv"
    df_sorted = df_fi.sort_values("importance", ascending=False).reset_index(drop=True)
    df_sorted.to_csv(csv_path, index=False)
    df_sorted.to_csv(csv_path, index=False)

    try:
        import matplotlib.pyplot as plt
        top = df_sorted.head(25)
        plt.figure(figsize=(8, max(4, 0.25*len(top))))
        plt.barh(top["feature"], top["importance"]); plt.gca().invert_yaxis()
        plt.title(f"Feature Importance (h={h}) — {top['method'].iloc[0] if not top.empty else ''}")
        plt.xlabel("Normalized importance"); plt.tight_layout()
        png_path = charts_dir / f"feature_importance_h{h}_latest.png"
        plt.savefig(png_path, dpi=160); plt.close()
    except Exception as e:
        print(f"[monitoring] NOTE: could not render FI chart for h={h}: {e}"); png_path = None
    return {"csv": str(csv_path), "png": str(png_path) if png_path else ""}

# ---------- Actual vs Pred + Model comparison ---------------------------------

def write_actual_vs_pred_reports(df_rec: pd.DataFrame, cfg: Dict) -> Dict[str, str]:
    reports_dir = ROOT / cfg.get("reports_folder", "reports")
    data_dir = reports_dir / "_data"; charts_dir = reports_dir / "charts"
    _ensure_dirs(data_dir); _ensure_dirs(charts_dir)
    csv_path = data_dir / "actual_vs_pred.csv"
    long_cols = ["target_month","horizon","y_true_pct","y_pred_pct","residual_pp","as_of_date"]
    cols = [c for c in long_cols if c in df_rec.columns]
    df_rec[cols].to_csv(csv_path, index=False)
    df_rec[cols].to_csv(csv_path, index=False)
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 5))
        for h, g in df_rec.dropna(subset=["y_true_pct","y_pred_pct"]).groupby("horizon"):
            e = (g.sort_values("target_month")
                   .assign(err=lambda x: (x["y_pred_pct"]-x["y_true_pct"])**2)
                   .set_index("target_month")["err"].rolling(6, min_periods=1).mean()**0.5)
            plt.plot(e.index, e.values, label=f"h={h}")
        plt.title("Rolling RMSE (6m) by Horizon"); plt.xlabel("Month"); plt.ylabel("RMSE (pp)")
        plt.legend(ncol=3, fontsize=8); plt.grid(alpha=0.3); plt.tight_layout()
        plt.savefig(charts_dir / "actual_vs_pred_latest.png", dpi=160); plt.close()
        # per-horizon last 50 months
        for h, g in df_rec.sort_values("target_month").groupby("horizon"):
            g2 = g.dropna(subset=["y_true_pct","y_pred_pct"]).tail(50)
            if g2.empty: continue
            plt.figure(figsize=(10, 4))
            plt.plot(g2["target_month"], g2["y_true_pct"], label="Actual")
            plt.plot(g2["target_month"], g2["y_pred_pct"], label="Predicted")
            plt.title(f"Actual vs Predicted — h={h} (last 50 months)")
            plt.xlabel("Month"); plt.ylabel("NPL (%)"); plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
            plt.savefig(charts_dir / f"actual_vs_pred_h{h}_latest.png", dpi=160); plt.close()
    except Exception as e:
        print(f"[monitoring] NOTE: could not render actual-vs-pred charts: {e}")
    return {"csv": str(csv_path)}

def build_model_comparison(df_rec: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    """
    Summarize performance by (horizon, model_type, model_version), then
    add an overall_rank per horizon using (MASE, SMAPE, RMSE_pp) as tie-breakers.
    """
    rows = []
    # If df_rec already carries model_type/version, we'll keep them; else derive.
    has_type = "model_type" in df_rec.columns
    has_ver  = "model_version" in df_rec.columns

    for h, g in df_rec.dropna(subset=["y_true_pct","y_pred_pct"]).groupby("horizon"):
        # Derive type/version when not present in df_rec
        model_type = (g["model_type"].iloc[0] if has_type else _model_type_for_h(int(h), cfg))
        model_version = (g["model_version"].iloc[0] if has_ver else cfg.get("model_version", "v0.0.0"))

        y_t = g["y_true_pct"]
        y_p = g["y_pred_pct"]
        # naive baseline: last actual shifted by horizon
        y_n = y_t.sort_index().shift(int(h))

        rows.append({
            "horizon":      int(h),
            "model_type":   str(model_type),
            "model_version":str(model_version),
            "rmse_pp":      rmse_pp(y_t, y_p),
            "mae_pp":       mae_pp(y_t, y_p),
            "smape":        smape(y_t, y_p),
            "mase":         mase(y_t, y_p, y_n),
            "r2_log":       r2_log(y_t, y_p),
            "n_obs":        float(len(g)),
        })

    if not rows:
        return pd.DataFrame(columns=[
            "horizon","model_type","model_version","rmse_pp","mae_pp","smape","mase","r2_log","n_obs","overall_rank"
        ])

    df = pd.DataFrame(rows)

    # Rank within each horizon (lower MASE wins; then SMAPE; then RMSE_pp)
    df = df.sort_values(["horizon","mase","smape","rmse_pp"]).reset_index(drop=True)
    df["overall_rank"] = df.groupby("horizon").cumcount() + 1

    # Nice ordering
    cols = ["horizon","model_type","model_version","overall_rank","mase","smape","rmse_pp","mae_pp","r2_log","n_obs"]
    return df[cols]

def write_model_comparison_reports(df_comp: pd.DataFrame, cfg: Dict) -> Dict[str, str]:
    reports_dir = ROOT / cfg.get("reports_folder", "reports")
    data_dir = reports_dir / "_data"; charts_dir = reports_dir / "charts"
    _ensure_dirs(data_dir); _ensure_dirs(charts_dir)
    csv_path = data_dir / "model_comparison.csv"
    df_comp.to_csv(csv_path, index=False)
    df_comp.to_csv(csv_path, index=False)
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 4))
        plt.bar(df_comp["horizon"].astype(str), df_comp["mase"])
        plt.title("Model Comparison across Horizons — MASE (lower is better)")
        plt.xlabel("Horizon"); plt.ylabel("MASE"); plt.grid(axis="y", alpha=0.3); plt.tight_layout()
        plt.savefig(charts_dir / "model_comparison_MASE_latest.png", dpi=160); plt.close()
    except Exception as e:
        print(f"[monitoring] NOTE: could not render model comparison chart: {e}")
    # Optional strict H0..H6 subset
    h_subset = df_comp[df_comp["horizon"].between(0, 6)]
    if not h_subset.empty and len(h_subset) != len(df_comp):
        h_subset.to_csv(data_dir / "model_comparison_h0_h6.csv", index=False)
        h_subset.to_csv(data_dir / "model_comparison_h0_h6.csv", index=False)
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(8, 4))
            plt.bar(h_subset["horizon"].astype(int).astype(str), h_subset["mase"])
            plt.title("Model Comparison (H0..H6) — MASE"); plt.xlabel("Horizon"); plt.ylabel("MASE")
            plt.grid(axis="y", alpha=0.3); plt.tight_layout()
            plt.savefig(charts_dir / "model_comparison_h0_h6_MASE_latest.png", dpi=160); plt.close()
        except Exception as e:
            print(f"[monitoring] NOTE: could not render H0..H6 comparison: {e}")
    return {"csv": str(csv_path)}

# ---------- Scoring & KPIs ----------------------------------------------------


def _get_model_feature_names(model) -> Optional[list]:
    try:
        if hasattr(model, "feature_name_"):
            fn = list(getattr(model, "feature_name_"))
            return fn if fn else None
        booster = getattr(model, "booster_", None)
        if booster is not None and hasattr(booster, "feature_name"):
            fn = list(booster.feature_name())
            return fn if fn else None
        booster2 = getattr(model, "booster_", None)
        if booster2 is not None and hasattr(booster2, "feature_name"):
            return list(booster2.feature_name())
    except Exception:
        return None
    return None

def _infer_gdp_midas_from_feature_names(feature_names: list) -> Optional[tuple]:
    pat = re.compile(r"^GDP_midas_(\d+)_([0-9.]+)_([0-9.]+)$")
    for c in feature_names or []:
        m = pat.match(str(c))
        if m:
            return int(m.group(1)), float(m.group(2)), float(m.group(3))
    return None

def _enforce_feature_parity(X: pd.DataFrame, feature_names: list, h: int) -> pd.DataFrame:
    missing = [c for c in feature_names if c not in X.columns]
    if missing:
        extra = [c for c in X.columns if c not in feature_names]
        raise ValueError(
            f"H{h} feature mismatch: trained={len(feature_names)} inferred={X.shape[1]} "
            f"missing={missing[:10]}{'...' if len(missing)>10 else ''} "
            f"extra={extra[:10]}{'...' if len(extra)>10 else ''}"
        )
    return X.reindex(columns=feature_names, copy=False)

def score_horizon(h: int, panel: pd.DataFrame, model, as_of: pd.Timestamp, cfg: Dict) -> pd.DataFrame:
    """
    Score a horizon using the *deployed* feature contract.
    For midas_ml we enforce exact training feature set + order (feature parity), aligned to forecast.py.
    """
    mtype = _model_type_for_h(h, cfg)

    if mtype == "midas_ml":
        feature_names = _get_model_feature_names(model)

        # If GDP MIDAS params differ between config and trained model, override for scoring
        if feature_names:
            gdp_meta = _infer_gdp_midas_from_feature_names(feature_names)
            if gdp_meta is not None:
                gdp_months, a, b = gdp_meta
                cfg.setdefault("features", {})
                cfg["features"]["gdp_months"] = int(gdp_months)
                cfg["features"]["beta_shape"] = [float(a), float(b)]

        X = build_features_for_scoring(panel, horizon=h, cfg=cfg)
        if X is None or X.empty:
            return pd.DataFrame(columns=["target_month","y_pred_pct","horizon","as_of_date"])

        if feature_names:
            X = _enforce_feature_parity(X, feature_names, h)

        yhat_log = model.predict(X, num_iteration=getattr(model, "best_iteration_", None))

        bias = float(getattr(model, "bias_log_add_", 0.0)) if hasattr(model, "bias_log_add_") else 0.0
        yhat_pp = _to_pp_from_log(np.asarray(yhat_log, dtype=float) + bias)

        return pd.DataFrame({"target_month": X.index, "y_pred_pct": yhat_pp, "horizon": h, "as_of_date": as_of})

    if mtype == "sc_midas_ols":
        yhat_log = model.predict(panel, horizon=h)
        target_month = (pd.to_datetime(yhat_log.index) + pd.offsets.MonthEnd(h))
        yhat_pp = _to_pp_from_log(yhat_log.values)
        out = pd.DataFrame({"target_month": target_month, "y_pred_pct": yhat_pp, "horizon": h, "as_of_date": as_of})
        return out.dropna(subset=["target_month"]).sort_values("target_month")

    raise ValueError(f"Unsupported model type for h={h}")


def reconcile_actuals(df_preds: pd.DataFrame, actuals: pd.DataFrame) -> pd.DataFrame:
    m = df_preds.merge(actuals, on="target_month", how="left")
    m["residual_pp"] = m["y_pred_pct"] - m["y_true_pct"]
    return m


def aggregate_kpis_monthly(df_h: pd.DataFrame) -> pd.DataFrame:
    def _one(g: pd.DataFrame):
        y_t = g["y_true_pct"].dropna(); y_p = g["y_pred_pct"].dropna()
        if len(y_t)==0 or len(y_p)==0:
            return pd.Series({"rmse_pp": np.nan, "mae_pp": np.nan, "smape": np.nan, "mase": np.nan, "r2_log": np.nan, "n_obs": 0})
        y_n = y_t.sort_index().shift(int(g["horizon"].iloc[0]))
        return pd.Series({
            "rmse_pp": rmse_pp(y_t, y_p),
            "mae_pp":  mae_pp(y_t, y_p),
            "smape":   smape(y_t, y_p),
            "mase":    mase(y_t, y_p, y_n),
            "r2_log":  r2_log(y_t, y_p),
            "n_obs":   float(len(g))
        })
    agg = df_h.groupby("target_month", as_index=False).apply(_one)
    return agg

def build_drift_report(panel: pd.DataFrame, train_window: Tuple[str, str], cur_window: Tuple[str, str]) -> pd.DataFrame:
    drivers = tuple([c for c in panel.columns
                     if c.lower() in {"degu","cblr","gla","gdp","npl","ln_npl"}
                     or c.lower().startswith(("degu_","cblr_","gla_"))])
    ref = panel.loc[train_window[0]:train_window[1]].copy()
    cur = panel.loc[cur_window[0]:cur_window[1]].copy()
    if ref.empty or cur.empty:
        return pd.DataFrame(columns=["metric","value","status","window_start","window_end"])
    return drift_report(ref, cur, drivers, f"{train_window[0]}..{train_window[1]}", f"{cur_window[0]}..{cur_window[1]}")

# --- Recent window KPIs --------------------------------------------------------
def compute_recent_kpis(df_rec: pd.DataFrame, window_months: int = 12) -> pd.DataFrame:
    """
    Compute KPIs on the most recent `window_months` with actuals for each horizon.
    Returns columns:
      horizon, window_start, window_end, n_obs_recent,
      rmse_pp_recent, mae_pp_recent, smape_recent, mase_recent, r2_log_recent
    """
    d = df_rec.dropna(subset=["target_month", "y_true_pct", "y_pred_pct"]).copy()
    d["target_month"] = pd.to_datetime(d["target_month"])
    if d.empty:
        return pd.DataFrame(columns=[
            "horizon","window_start","window_end","n_obs_recent",
            "rmse_pp_recent","mae_pp_recent","smape_recent","mase_recent","r2_log_recent"
        ])

    # Most recent target months with actuals
    months = sorted(d["target_month"].unique())
    if not months:
        return pd.DataFrame(columns=[
            "horizon","window_start","window_end","n_obs_recent",
            "rmse_pp_recent","mae_pp_recent","smape_recent","mase_recent","r2_log_recent"
        ])
    use_months = months[-min(window_months, len(months)):]
    d = d[d["target_month"].isin(use_months)].copy()
    win_start = pd.to_datetime(use_months[0]).date()
    win_end   = pd.to_datetime(use_months[-1]).date()

    rows = []
    for h, g in d.groupby("horizon"):
        yt = g["y_true_pct"]
        yp = g["y_pred_pct"]
        yn = yt.sort_index().shift(int(h))  # naive baseline
        rows.append({
            "horizon": int(h),
            "window_start": str(win_start),
            "window_end": str(win_end),
            "n_obs_recent": int(len(g)),
            "rmse_pp_recent": rmse_pp(yt, yp),
            "mae_pp_recent":  mae_pp(yt, yp),
            "smape_recent":   smape(yt, yp),
            "mase_recent":    mase(yt, yp, yn),
            "r2_log_recent":  r2_log(yt, yp),
        })
    out = pd.DataFrame(rows).sort_values("horizon").reset_index(drop=True)
    return out

# --- Champion selection / fallback -------------------------------------------
def _recent_thresholds(cfg: Dict):
    th = cfg.get("monitoring", {}).get("thresholds", {}) or {}
    # Defaults if not provided in your config
    return {
        "mase_max_recent": float(th.get("mase_max_recent", 1.0)),
        "smape_max_recent": float(th.get("smape_max_recent", 30.0)),
        "r2_log_min_recent": float(th.get("r2_log_min_recent", 0.20)),
    }

def _breached_recent(row: pd.Series, th: Dict) -> bool:
    if pd.isna(row.get("mase_recent")) or pd.isna(row.get("smape_recent")) or pd.isna(row.get("r2_log_recent")):
        return True
    return (
        (row["mase_recent"]  > th["mase_max_recent"]) or
        (row["smape_recent"] > th["smape_max_recent"]) or
        (row["r2_log_recent"] < th["r2_log_min_recent"])
    )

def select_champions(df_comp: pd.DataFrame, df_recent: pd.DataFrame, cfg: Dict, as_of: pd.Timestamp) -> pd.DataFrame:
    """
    Returns a table keyed by horizon with your champion decision:
      horizon, selected_model, model_type, model_version, decision, rationale,
      window_start, window_end, mase_recent, smape_recent, r2_log_recent, as_of_date
    Logic:
      - If recent KPIs breach thresholds => fallback to naive_shift
      - Else keep the current model (the one that produced df_comp)
    NOTE: If/when you log multiple candidates per horizon, you can extend this
          to compare champion vs challenger using df_comp['overall_rank'].
    """
    th = _recent_thresholds(cfg)
    as_of_norm = pd.to_datetime(as_of).normalize()

    # If multiple rows per horizon, we keep rank==1 as current champion.
    if "overall_rank" in df_comp.columns:
        base = df_comp.sort_values(["horizon","overall_rank"]).groupby("horizon").head(1).copy()
    else:
        base = df_comp.sort_values("horizon").copy()

    recent = df_recent.set_index("horizon")
    rows = []
    for _, r in base.iterrows():
        h = int(r["horizon"])
        model_type = r.get("model_type", _model_type_for_h(h, cfg))
        model_version = r.get("model_version", cfg.get("model_version", "v0.0.0"))
        rr = recent.loc[h] if h in recent.index else pd.Series(dtype="float64")

        breach = _breached_recent(rr, th) if not rr.empty else True
        if breach:
            selected_model = "naive_shift"
            decision = "fallback"
            rationale = "Recent KPIs breached thresholds; using last-actual shifted baseline."
        else:
            selected_model = "current"
            decision = "champion"
            rationale = "Recent KPIs within thresholds; keep current model."

        rows.append({
            "horizon": h,
            "selected_model": selected_model,
            "model_type": model_type,
            "model_version": model_version,
            "decision": decision,
            "rationale": rationale,
            "window_start": rr.get("window_start", ""),
            "window_end": rr.get("window_end", ""),
            "mase_recent": float(rr.get("mase_recent", np.nan)),
            "smape_recent": float(rr.get("smape_recent", np.nan)),
            "r2_log_recent": float(rr.get("r2_log_recent", np.nan)),
            "as_of_date": as_of_norm,
        })
    return pd.DataFrame(rows).sort_values("horizon").reset_index(drop=True)

def write_champions(reports_root, df_champion: pd.DataFrame) -> str:
    """Append/update the champion-by-horizon table for audit and downstream routing."""
    data_dir, _ = _rr_dirs(reports_root)
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_path = data_dir / "champion_by_horizon.csv"
    _upsert_csv(
        csv_path,
        df_champion,
        key_cols=["as_of_date","horizon"],
        date_cols=["as_of_date"],
        sort_by=["as_of_date","horizon"]
    )
    print(f"[monitoring] Champions table => {csv_path.resolve()}")
    return str(csv_path)

# --- Optional: linear post-model calibration (off by default) -----------------
def calibrate_linear_recent(df_rec: pd.DataFrame, h: int, window_months: int = 12):
    """
    Fit y_true_pct ~ a + b * y_pred_pct on the last `window_months` (for horizon h).
    Returns (a, b) or (0, 1) if not enough data.
    """
    g = df_rec[df_rec["horizon"] == h].dropna(subset=["y_true_pct","y_pred_pct"]).copy()
    g = g.sort_values("target_month").tail(window_months)
    if len(g) < max(6, window_months // 3):
        return 0.0, 1.0
    # Solve least squares: [1, y_pred] -> y_true
    X = np.column_stack([np.ones(len(g)), g["y_pred_pct"].to_numpy(float)])
    y = g["y_true_pct"].to_numpy(float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    a, b = float(beta[0]), float(beta[1])
    return a, b

def write_calibration_coeffs(reports_root, h: int, a: float, b: float, as_of: pd.Timestamp) -> str:
    data_dir, _ = _rr_dirs(reports_root)
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_path = data_dir / f"calibration_h{h}.csv"
    df = pd.DataFrame([{
        "as_of_date": pd.to_datetime(as_of).normalize(),
        "horizon": h, "intercept_a": a, "slope_b": b
    }])
    _upsert_csv(csv_path, df, key_cols=["as_of_date","horizon"], date_cols=["as_of_date"], sort_by=["as_of_date","horizon"])
    print(f"[monitoring] Calibration h={h} => {csv_path.resolve()}")
    return str(csv_path)

def write_recent_snapshot_block(reports_root, cfg: Dict, title_prefix: str = "Snapshot by horizon") -> Tuple[str, str]:
    """
    Build a human-readable summary like:
      Snapshot by horizon (last 12 months with actuals)
      h=1: MAE 1.20pp, RMSE 1.45pp, SMAPE 5.30%, MASE 2.48, R²_log -1.90
      ...
    Saves to reports/_data/recent_snapshot.txt and returns (path, text).
    """
    # resolve _data dir
    def _rr_base_dir(rr) -> Path:
        for attr in ("root","base","reports_root","root_dir","base_dir","reports_dir"):
            v = getattr(rr, attr, None)
            if v:
                try: return Path(v)
                except Exception: pass
        try: return Path(rr)
        except Exception: return ROOT / "reports"

    def _rr_dirs(rr) -> Tuple[Path, Path]:
        # prefer explicit methods/properties if exposed by ReportRoot
        if hasattr(rr, "data_dir"):
            val = getattr(rr, "data_dir")
            try:
                data_dir = Path(val()) if callable(val) else Path(val)
                charts_attr = getattr(rr, "charts_dir", None)
                charts_dir = Path(charts_attr() if callable(charts_attr) else charts_attr) if charts_attr else _rr_base_dir(rr) / "charts"
                return data_dir, charts_dir
            except Exception:
                pass
        base = _rr_base_dir(rr)
        return base / "_data", base / "charts"

    data_dir, _ = _rr_dirs(reports_root)
    data_dir.mkdir(parents=True, exist_ok=True)
    fp = data_dir / "forecast_path_metrics_latest.csv"

    if not fp.exists():
        text = f"{title_prefix} (no recent metrics found)"
        out = data_dir / "recent_snapshot.txt"
        out.write_text(text, encoding="utf-8")
        return str(out), text

    df = pd.read_csv(fp, parse_dates=["as_of_date"])
    if df.empty:
        text = f"{title_prefix} (no rows)"
        out = data_dir / "recent_snapshot.txt"
        out.write_text(text, encoding="utf-8")
        return str(out), text

    # pick latest as_of_date
    latest_asof = df["as_of_date"].max()
    df = df[df["as_of_date"] == latest_asof].copy()

    # choose the window based on config.gate_window (defaults to 12 if missing)
    win = int(cfg.get("monitoring", {}).get("gate_window", 12))
    window_label = f"last_{win}m_with_actuals"
    df = df[df["window"] == window_label].copy()

    if df.empty:
        text = f"{title_prefix} ({window_label}) — no rows"
        out = data_dir / "recent_snapshot.txt"
        out.write_text(text, encoding="utf-8")
        return str(out), text

    # Pivot to get metrics as columns per horizon
    piv = (df.pivot_table(index=["horizon"], columns="metric", values="value", aggfunc="last")
             .reset_index()
             .sort_values("horizon"))

    # formatting helpers
    def fmt_pp(x):  # percentage points
        try:
            return f"{float(x):.2f}pp"
        except Exception:
            return "NA"
    def fmt_pct(x):  # percent
        try:
            return f"{float(x):.2f}%"
        except Exception:
            return "NA"
    def fmt_num(x):  # plain number
        try:
            return f"{float(x):.2f}"
        except Exception:
            return "NA"

    # compose lines
    header = f"{title_prefix} ({window_label.replace('_',' ')})"
    lines = [header]
    # metrics we expect to exist (case matches how we wrote them)
    # RMSE_pp, MAE_pp, SMAPE, MASE, R2_log
    for _, r in piv.iterrows():
        h = int(r["horizon"])
        mae = fmt_pp(r.get("MAE_pp"))
        rmse = fmt_pp(r.get("RMSE_pp"))
        smape_v = fmt_pct(r.get("SMAPE"))
        mase_v = fmt_num(r.get("MASE"))
        r2 = fmt_num(r.get("R2_log"))
        # U+00B2 for the superscript 2 in R²
        lines.append(f"h={h}: MAE {mae}, RMSE {rmse}, SMAPE {smape_v}, MASE {mase_v}, R\u00b2_log {r2}")

    text = "\n".join(lines)

    out = data_dir / "recent_snapshot.txt"
    out.write_text(text, encoding="utf-8")
    #also print to console so it shows up in your run logs
    print("\n" + text + "\n")

    return str(out), text

# def _append_sql(df: pd.DataFrame, path: Path):
#     conn = db_con.get_db_connection()
#     table_name = path.stem  # Use the file name (without extension) as the table name
#     df.to_sql(table_name, conn, if_exists="append", index=False)

# def _read_sql(path: Path):
#     conn = db_con.get_db_connection()
#     table_name = path.stem  # Use the file name (without extension) as the table name
#     return pd.read_sql(f"SELECT * FROM {table_name}", conn)
# ---------- Main ---------------------------------------------------------------

def main(argv=None):
    cfg_path = ROOT / "config.yml"
    if not cfg_path.exists(): raise FileNotFoundError(f"config.yml not found at {cfg_path}")
    cfg = load_config(cfg_path)


    import argparse
    p = argparse.ArgumentParser(description="NPL forecast monitoring job")
    p.add_argument("--as-of", default="", help="As-of date YYYY-MM-DD (default: today UTC)")
    p.add_argument("--run-mode", default="", help="approved|candidate|<run_id>|run_<run_id>")
    p.add_argument("--horizons", nargs="*", default=None, help="Optional list of horizons to score")
    args = p.parse_args()

    run_mode = (args.run_mode or (cfg.get("monitoring", {}) or {}).get("run_mode") or "approved")
    if args.as_of:
        as_of = pd.to_datetime(args.as_of).normalize()
    else:
        as_of = pd.Timestamp(datetime.now(UTC).date())

    if  cfg.get("horizons", None) is not None:
        cfg["horizons"] = [int(x) for x in cfg.get("horizons")]

        thresholds = MetricThresholds(**cfg["monitoring"]["thresholds"])
        stamp = RunStamp(as_of_date=as_of, model_version=cfg.get("model_version", "v0.0.0"),
                         cfg_hash=cfg.get("cfg_hash","NA"), code_commit=cfg.get("code_commit",""))

        reports_root = ReportRoot.from_reports_root(ROOT / cfg["reports_folder"])

        # 1) Panel
        panel = _load_panel_df(cfg)

        # 1b) Panel health checks
        df_health = validate_panel_health(panel, cfg)
        try:
            data_dir, _ = _rr_dirs(reports_root)
            data_dir.mkdir(parents=True, exist_ok=True)
            (data_dir / "panel_health.csv").write_text(df_health.to_csv(index=False), encoding="utf-8")
        except Exception as e:
            print(f"[monitoring] NOTE: could not write panel health report: {e}")

        # 2) Actuals (NPL %)
        actuals = actuals_from_panel(panel, cfg)
        actuals["target_month"] = pd.to_datetime(actuals["target_month"])

        # 3) Score horizons (skip missing models gracefully)
        preds_all = []; skipped = []
        for h in cfg["horizons"]:
            try:
                model = load_model_for_h(h, cfg, run_mode=run_mode)
            except FileNotFoundError as e:
                print(f"[monitoring] SKIP h={h}: {e}"); skipped.append(h); continue
            df_h = score_horizon(h, panel, model, as_of, cfg)

            if df_h is not None and not df_h.empty: preds_all.append(df_h)
        if not preds_all:
            raise RuntimeError("No predictions generated; all requested horizons missing models.")
        if skipped:
            print(f"[monitoring] Missing horizons skipped: {sorted(skipped)}")
        df_preds = pd.concat(preds_all, ignore_index=True)

        # 4) Reconcile with actuals
        df_rec = reconcile_actuals(df_preds, actuals)
        print("888888888888888888888888888888888888888888888888888888888888")
        print(df_rec)
        print("888888888888888888888888888888888888888888888888888888888888")
        exit()


        # 5) Forecast path (t+1..t+6) — UPDATED to use the authoritative forward-path writers
        df_forward_block = _build_forward_path_block(df_rec, panel, as_of)
        save_forecast_path_authoritative(df_forward_block, reports_root)
        write_forecast_path_table_latest(
            df_forward_block[["as_of_date","horizon","target_month","y_pred_pct"]],
            reports_root
        )
        win_eval = int(cfg.get("monitoring", {}).get("gate_window", 36))
        write_forecast_path_metrics_latest(df_rec, reports_root, window_months=win_eval)
        print(f"[monitoring] Forecast metrics window = {win_eval} months")
        #print("[monitoring] Forward block emitted:")
        print(
            df_forward_block
              .assign(target_month=lambda d: d["target_month"].dt.strftime("%Y-%m"))
              .to_string(index=False)
        )

        # Build and save the human-readable snapshot for the configured window
        _ = write_recent_snapshot_block(reports_root, cfg)

        # 6) Backtest heatmaps
        heat_abs = build_backtest_heatmap_matrix(df_rec, metric="abs_error_pp")
        paths_abs = write_backtest_heatmap_reports(heat_abs, cfg, metric="abs_error_pp")
        print(f"[monitoring] Backtest heatmap (abs_error_pp): CSV={paths_abs['csv']}, PNG={paths_abs['png']}")
        # (optional SMAPE) — uncomment if desired:
        # heat_smape = build_backtest_heatmap_matrix(df_rec, metric="smape")
        # write_backtest_heatmap_reports(heat_smape, cfg, metric="smape")

        # 7) Feature importance per horizon
        for h in sorted(set(df_preds["horizon"])):
            try:
                m = load_model_for_h(h, cfg, run_mode=run_mode); mtype = _model_type_for_h(h, cfg)
                X = build_features_for_scoring(panel, horizon=h, cfg=cfg)
                df_fi = _fi_from_model_and_X(m, X, mtype); df_fi["horizon"] = h
                _ = write_feature_importance_reports(h, df_fi, cfg)
            except Exception as e:
                print(f"[monitoring] NOTE: could not compute feature importance for h={h}: {e}")

        # 8) Actual vs Predicted (full history)
        avp_paths = write_actual_vs_pred_reports(df_rec, cfg)
        print(f"[monitoring] Actual vs Predicted saved: CSV={avp_paths['csv']}")

        # 9) Model comparison across horizons (full backtest)
        df_comp = build_model_comparison(df_rec, cfg)
        mc_paths = write_model_comparison_reports(df_comp, cfg)
        print(f"[monitoring] Model comparison saved: CSV={mc_paths['csv']}")

        # 9b) NEW — recent-window KPIs per horizon (explicit table)
        recent_win = int(cfg.get("monitoring", {}).get("gate_window", 12))
        df_recent = compute_recent_kpis(df_rec, window_months=recent_win)

        # 9c) NEW — choose champion or fallback (naïve) per horizon, log for routing/audit
        df_champ = select_champions(df_comp, df_recent, cfg, as_of)
        champions_csv = write_champions(reports_root, df_champ)

        # 9d) NEW — optional: write per-horizon calibration (disabled by default)
        if bool(cfg.get("monitoring", {}).get("calibration", {}).get("enable_recent_linear", False)):
            for h in sorted(df_rec["horizon"].unique()):
                a, b = calibrate_linear_recent(df_rec, h, window_months=recent_win)
                write_calibration_coeffs(reports_root, h, a, b, as_of)

        # 10) Per-horizon monthly KPIs, drift, gates, alerts
        for h in sorted(df_rec["horizon"].unique()):
            df_h = df_rec[df_rec["horizon"] == h].dropna(subset=["y_true_pct"])
            if df_h.empty: continue
            kpi_ts = aggregate_kpis_monthly(df_h)
            resid = df_h.dropna(subset=["residual_pp"])[["target_month","residual_pp"]].sort_values("target_month")

            if len(panel) >= 36:
                train_w = (str(panel.index.min().date()), str((panel.index.min() + pd.offsets.MonthEnd(24)).date()))
                cur_w = (str((panel.index.max() - pd.offsets.MonthEnd(11)).date()), str(panel.index.max().date()))
                drift_df = build_drift_report(panel, train_w, cur_w)
            else:
                drift_df = None

            _ = log_monitoring(reports_root, h=h, df_prod_kpis=kpi_ts, df_resid=resid, df_drift=drift_df)

            win = int(cfg["monitoring"].get("gate_window", 24))
            recent = kpi_ts.tail(win).select_dtypes(include=[float, int]).mean(numeric_only=True).to_dict()
            gates_paths = log_gates(reports_root, h=h, kpis=recent, thresholds=thresholds, stamp=stamp)

            breach = (
                (recent.get("mase", 0) > thresholds.mase_max) or
                (recent.get("smape", 0) > thresholds.smape_max) or
                (recent.get("rmse_pp", 0) > thresholds.rmse_pp_max) or
                (recent.get("mae_pp", 0) > thresholds.mae_pp_max) or
                (recent.get("r2_log", 1) < thresholds.r2_log_min)
            )
            if breach:
                send_alert(
                    subject=f"[Monitoring] KPI gate breach on h={h}",
                    message=f"Recent {win}-month KPIs failed thresholds for h={h}. See {gates_paths.get('gates_html','')}.",
                    extra=recent
                )

        # 11) Repro manifest
        repro = {
            "as_of_date": str(as_of.date()),
            "model_version": cfg.get("model_version"),
            "cfg_hash": cfg.get("cfg_hash"),
            "code_commit": cfg.get("code_commit"),
            "tz": "Africa/Accra"
        }
        repro_manifest(ReportRoot.from_reports_root(ROOT / cfg["reports_folder"])).write_text(json.dumps(repro, indent=2))
        print("Monitoring run complete.")

# # Optional entry points
# def run(cfg: Optional[Dict] = None): main()
# def monitor(cfg: Optional[Dict] = None): main()

if __name__ == "__main__":
    main()

