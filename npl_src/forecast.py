#!/usr/bin/env python3
# forecast_pipeline.py
"""
Production inference / forecasting pipeline (no training).

Fixes the common production error:
  "Model trained on X features but inference has Y features"

Root cause (in your case):
- Training features include optional rolling means (add_rolling_windows=true),
  which adds 5 * len(rolling_windows) features. For windows [3,6] => +10 features.
- Earlier inference code omitted this block, so inference had fewer features (e.g., 34 vs 44).

This script:
- Builds inference features to exactly mirror src/features.py (minus target shift),
  including rolling means when enabled.
- Aligns inference columns to the model's stored feature list (order + membership).
- Writes forecasts to SQL (append-with-history) + optional CSV mirror.
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional
import re

import numpy as np
import pandas as pd
import yaml
import joblib
import lightgbm as lgb

from npl_src.db import DatabaseConnection
from npl_src.sc_midas import SCMIDASOLS, _make_lags  # reuse SCMIDAS helper for lags

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "config.yml").read_text()) or {}

MODELS_BASE = Path(CFG.get("models_folder") or "data/interim/models")
CAND_SUB = str(CFG.get("candidate_subdir", "candidate"))
MODELS_CAND = ROOT / MODELS_BASE / CAND_SUB


class forecast:

    def __init__(self):
        self.db_con = DatabaseConnection()

    def _utc_now_iso(self) -> str:
        return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    def _to_pp(self, log_vals: np.ndarray) -> np.ndarray:
        return np.exp(np.asarray(log_vals, dtype=float)) * 100.0

    def _append_sql(self, df: pd.DataFrame, table: str) -> None:
        try:
            self.db_con._append_sql(df, table)  # your infra if available
            return
        except Exception:
            conn = self.db_con.get_db_connection()
            df.to_sql(table, conn, if_exists="append", index=False)

    def _load_panel(self) -> pd.DataFrame:
        conn = self.db_con.get_db_connection()
        df = pd.read_sql("SELECT * FROM t_panel", conn, parse_dates=["date"]).set_index("date").sort_index()
        if "NPL" in df.columns and "ln_npl" not in df.columns:
            df["ln_npl"] = df["NPL"]  # training expects ln(NPL_decimal) in NPL
        if "GDP" in df.columns and "gdp" not in df.columns:
            df["gdp"] = df["GDP"]
        return df

    # ─────────────────────────────────────────────────────────────────────────────
    # Inference-safe feature building (mirrors features.build_feature_matrix blocks)
    # ─────────────────────────────────────────────────────────────────────────────
    HI_FREQ_VARS = ["DEGU", "CBLR", "GLA"]

    def _as_num(self, s: pd.Series) -> pd.Series:
        return pd.to_numeric(s, errors="coerce").astype(float)

    def _beta_midas_weights(self, k: int, a: float, b: float) -> np.ndarray:
        idx = np.arange(k, dtype=float)
        x = (idx + 1) / k
        w = np.power(x, a - 1.0) * np.power(1.0 - x, b - 1.0)
        w = np.maximum(w, 1e-12)
        w = w / w.sum()
        return w[::-1]

    def _make_gdp_midas_feature(self, gdp: pd.Series, months: int, a: float, b: float) -> pd.Series:
        w = self._beta_midas_weights(months, a, b)
        g = self._as_num(gdp).copy()
        mat = np.vstack([g.shift(j).to_numpy() for j in range(months)])
        out = pd.Series(np.dot(w, mat), index=g.index)
        out.name = f"GDP_midas_{months}_{a:.2f}_{b:.2f}"
        return out

    def _add_lag_block(self, df: pd.DataFrame, col: str, max_lag: int, include_lag0: bool) -> pd.DataFrame:
        out = {}
        start = 0 if include_lag0 else 1
        for L in range(start, max_lag + 1):
            out[f"{col}_lag{L}"] = self._as_num(df[col]).shift(L)
        return pd.DataFrame(out, index=df.index)

    def _add_rolling_means(self, df: pd.DataFrame, cols: List[str], windows: List[int]) -> pd.DataFrame:
        feats = {}
        for c in cols:
            s = self._as_num(df[c])
            for w in windows:
                feats[f"{c}_mean{w}"] = s.rolling(window=w, min_periods=w).mean()
        return pd.DataFrame(feats, index=df.index)

    def build_feature_matrix_inference(self, panel: pd.DataFrame, cfg: Dict, horizon: int) -> pd.DataFrame:
        df = panel.copy().sort_index()
        for c in ["NPL", "GDP", "DEGU", "CBLR", "GLA"]:
            if c not in df.columns:
                df[c] = np.nan

        fcfg = (cfg.get("features") or {})
        npl_lags = int(fcfg.get("npl_auto_lags", 12))
        hi_lags = int(fcfg.get("hi_freq_lags", 6))
        gdp_m = int(fcfg.get("gdp_months", 12))
        a, b = (fcfg.get("beta_shape") or [3.0, 4.5])

        blocks = []

        # NPL AR lags (exclude lag0 as in training)
        if npl_lags > 0:
            blocks.append(self._add_lag_block(df, "NPL", npl_lags, include_lag0=False))

        # GDP MIDAS
        if gdp_m > 0:
            blocks.append(self._make_gdp_midas_feature(df["GDP"], gdp_m, float(a), float(b)).to_frame())

        # Hi-freq lags include lag0
        if hi_lags > 0:
            for var in self.HI_FREQ_VARS:
                if var in df.columns:
                    blocks.append(self._add_lag_block(df, var, hi_lags, include_lag0=True))

        # Rolling means (THIS IS THE CAUSE OF YOUR 44 vs 34 MISMATCH WHEN ENABLED)
        add_roll = bool(fcfg.get("add_rolling_windows", False))
        if add_roll:
            windows = list(map(int, fcfg.get("rolling_windows", [3, 6])))
            roll_cols = [c for c in ["NPL", "GDP", "DEGU", "CBLR", "GLA"] if c in df.columns]
            blocks.append(self._add_rolling_means(df, roll_cols, windows))

        X = pd.concat(blocks, axis=1) if blocks else pd.DataFrame(index=df.index)
        return X.dropna(how="any")

    def _infer_gdp_midas_from_feature_names(self, feature_names: List[str]) -> Optional[tuple]:
        """Infer (months, a, b) from a feature named like GDP_midas_<months>_<a>_<b>."""
        pat = re.compile(r"^GDP_midas_(\d+)_([0-9.]+)_([0-9.]+)$")
        for c in feature_names:
            m = pat.match(c)
            if m:
                return int(m.group(1)), float(m.group(2)), float(m.group(3))
        return None

    # ─────────────────────────────────────────────────────────────────────────────
    # SCMIDAS inference design
    # ─────────────────────────────────────────────────────────────────────────────
    def _scmidas_design_inference(self, m: SCMIDASOLS, panel: pd.DataFrame) -> pd.DataFrame:
        cols = m._resolve_columns(panel)
        ln_npl = self._as_num(panel[cols["ln_npl"]])

        lag_cfg = m._lag_config()
        X_parts = []
        if lag_cfg.get("include_const", True):
            X_parts.append(pd.DataFrame({"const": 1.0}, index=panel.index))

        ar = int(lag_cfg.get("ar_lags", 0))
        if ar > 0:
            X_parts.append(_make_lags(ln_npl, ar, "ln_npl"))

        def add_exog(key: str, label: str, n_lags: int):
            cname = cols.get(key)
            if cname is None or int(n_lags) <= 0:
                return
            s = self._as_num(panel[cname])
            X_parts.append(_make_lags(s, int(n_lags), label))

        add_exog("gdp", "gdp", lag_cfg.get("gdp_lags", 0))
        add_exog("degu", "degu", lag_cfg.get("degu_lags", 0))
        add_exog("cblr", "cblr", lag_cfg.get("cblr_lags", 0))
        add_exog("gla", "gla", lag_cfg.get("gla_lags", 0))

        X = pd.concat(X_parts, axis=1).astype("float64").dropna(how="any")
        if m.columns_ is not None:
            X = X.reindex(columns=m.columns_, copy=False)
        return X

    # ─────────────────────────────────────────────────────────────────────────────
    # Model artifact resolution (simple: load from models_folder directly)
    # If you use approved/candidate registry, adapt this to read your pointer file.
    # ─────────────────────────────────────────────────────────────────────────────
    MODELS_DIR = Path(CFG.get("models_folder", "data/interim/models"))
    if not MODELS_DIR.is_absolute():
        MODELS_DIR = (ROOT / MODELS_DIR).resolve()

    def _model_type_for_h(self, h: int) -> str:
        per = CFG.get("model_type_per_horizon") or {}
        if str(h) in per:
            return str(per[str(h)]).lower()
        if h in per:
            return str(per[h]).lower()
        return str(CFG.get("model_type", "midas_ml")).lower()

    def run_main(self) -> pd.DataFrame:
        run_ts_utc = self._utc_now_iso()
        panel = self._load_panel()
        if panel.empty:
            raise RuntimeError("Panel is empty")

        anchor_date = panel.index.max()
        horizons = [int(h) for h in (CFG.get("horizons") or [0, 1, 2, 3, 4, 5, 6])]

        rows = []
        for h in horizons:
            mtype = self._model_type_for_h(h)

            if mtype in ("midas_ml", "lgbm", "lightgbm"):
                mp = MODELS_CAND / f"lgbm_midas_h{h}.pkl"
                if not mp.exists():
                    raise FileNotFoundError(f"Missing model artifact: {mp}")
                model: lgb.LGBMRegressor = joblib.load(mp)

                # Determine trained feature list first (for exact alignment)
                feature_names = None
                if hasattr(model, "feature_name_"):
                    feature_names = list(getattr(model, "feature_name_"))
                elif hasattr(model, "booster_") and model.booster_ is not None:
                    feature_names = list(model.booster_.feature_name())

                # If GDP MIDAS params were grid-selected during training, config.yml may differ.
                # Infer required (months, a, b) from the model feature list and override feature config for inference.
                gdp_meta = self._infer_gdp_midas_from_feature_names(feature_names) if feature_names is not None else None
                if gdp_meta is not None:
                    gdp_months, a, b = gdp_meta
                    CFG.setdefault("features", {})
                    CFG["features"]["gdp_months"] = int(gdp_months)
                    CFG["features"]["beta_shape"] = [float(a), float(b)]

                # Build inference features (no target shift) using the (possibly overridden) config
                Xinf = self.build_feature_matrix_inference(panel, CFG, horizon=h)
                if Xinf.empty:
                    raise RuntimeError(
                        f"Inference features empty for horizon {h}. Check lag lengths/data completeness.")
                x_last = Xinf.iloc[[-1]]
                feature_date = x_last.index[0]

                # Enforce exact training feature set + order
                if feature_names is not None:
                    missing = [c for c in feature_names if c not in x_last.columns]
                    extra = [c for c in x_last.columns if c not in feature_names]
                    if missing:
                        raise ValueError(
                            f"H{h} feature mismatch: trained={len(feature_names)} inferred={x_last.shape[1]}"
                            f"Missing: {missing[:10]}{'...' if len(missing) > 10 else ''} "
                            f"(also found extra: {extra[:10]}{'...' if len(extra) > 10 else ''})"
                        )
                    x_last = x_last.reindex(columns=feature_names, copy=False)
                yhat_log = float(model.predict(x_last, num_iteration=getattr(model, "best_iteration_", None))[0])
                bias = float(getattr(model, "bias_log_add_", 0.0)) if hasattr(model, "bias_log_add_") else 0.0
                yhat_log_adj = yhat_log + bias
                yhat_pp = float(self._to_pp(np.array([yhat_log_adj]))[0])

                rows.append({
                    "run_ts_utc": run_ts_utc,
                    "anchor_date": str(anchor_date),
                    "feature_date": str(feature_date),
                    "forecast_date": str(feature_date + pd.DateOffset(months=int(h))),
                    "horizon": int(h),
                    "model_type": "midas_ml",
                    "model_artifact": mp.name,
                    "bias_log": bias,
                    "y_pred_log": yhat_log_adj,
                    "y_pred_pp": yhat_pp,
                })

            elif mtype in ("sc_midas_ols", "sc_midas", "ols"):
                mp = MODELS_CAND / f"sc_midas_ols_h{h}.json"
                if not mp.exists():
                    raise FileNotFoundError(f"Missing model artifact: {mp}")
                m = SCMIDASOLS.load(mp)
                Xinf = self._scmidas_design_inference(m, panel)
                if Xinf.empty:
                    raise RuntimeError(f"SCMIDAS inference design empty for horizon {h}.")
                x_last = Xinf.iloc[[-1]]
                feature_date = x_last.index[0]
                # Robust linear prediction: ensure scalar output regardless of coef_ shape
                coef = np.asarray(m.coef_, dtype=float).reshape(-1)
                pred = x_last.to_numpy(dtype=float) @ coef
                yhat_log = float(np.asarray(pred).reshape(-1)[0])
                yhat_pp = float(self._to_pp(np.array([yhat_log]))[0])

                rows.append({
                    "run_ts_utc": run_ts_utc,
                    "anchor_date": str(anchor_date),
                    "feature_date": str(feature_date),
                    "forecast_date": str(feature_date + pd.DateOffset(months=int(h))),
                    "horizon": int(h),
                    "model_type": "sc_midas_ols",
                    "model_artifact": mp.name,
                    "bias_log": 0.0,
                    "y_pred_log": yhat_log,
                    "y_pred_pp": yhat_pp,
                })
            else:
                raise ValueError(f"Unknown model_type for horizon {h}: {mtype}")

        df = pd.DataFrame(rows).sort_values(["forecast_date", "horizon"]).reset_index(drop=True)

        # SQL output (append-with-history)
        forecast_table = str(CFG.get("forecast_table", "t_npl_forecasts"))
        self._append_sql(df, forecast_table)

        # Optional CSV mirror
        # if bool(CFG.get("forecast_to_csv", False)):
        reports = Path(CFG.get("reports_folder", "reports"))
        if not reports.is_absolute():
            reports = (ROOT / reports).resolve()
        tables = reports / "tables"
        tables.mkdir(parents=True, exist_ok=True)
        out_csv = tables / "npl_forecasts.csv"
        header = not out_csv.exists()
        # df.to_csv(out_csv, mode="a", header=header, index=False)

        #print("✔ Forecast complete.")
        return df

    # def call_main(self):
    #     print(self.run_main())

    # if __name__ == "__main__":
    #     main()

# obj = forecast()
# print(obj.run_main())
