# src/sc_midas.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, List

import numpy as np
import pandas as pd
import yaml


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _root() -> Path:
    fp = globals().get("__file__")
    return Path(fp).resolve().parents[1] if fp else Path.cwd().resolve()


ROOT = _root()
CFG_PATH = ROOT / "config.yml"
CFG: Dict[str, Any] = {}
if CFG_PATH.exists():
    try:
        CFG = yaml.safe_load(CFG_PATH.read_text()) or {}
    except Exception:
        CFG = {}


def _get_cfg_block() -> Dict[str, Any]:
    return (CFG.get("sc_midas_ols") or {}) if isinstance(CFG, dict) else {}


def _find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    lower = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name in df.columns:
            return name
        if name.lower() in lower:
            return lower[name.lower()]
    return None


def _make_lags(s: pd.Series, n_lags: int, prefix: str) -> pd.DataFrame:
    cols = {}
    for k in range(1, int(max(0, n_lags)) + 1):
        cols[f"{prefix}_lag{k}"] = s.shift(k)
    return pd.DataFrame(cols, index=s.index)


def _as_float_array(x) -> np.ndarray:
    return np.asarray(x, dtype=float)


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class SCMIDASOLS:
    """
    Simple SCMIDAS-style OLS with ridge regularization (alpha).
    - Target: level `ln_npl_{t+h}` (no extra transforms here).
    - Predictors: constant (optional), AR lags of ln_npl, and stacked lags of exogenous
      variables (GDP, DEGU, CBLR, GLA). No re-transforms are applied.
    - Lags are read from config.yml -> sc_midas_ols (optional; sensible defaults otherwise).
    - Ridge uses (X'X + alpha * I) beta = X'y, excluding constant columns from penalty.
    """

    alpha: float = 0.0  # ridge penalty; 0.0 => plain OLS

    # learned parameters
    coef_: Optional[np.ndarray] = None
    columns_: Optional[List[str]] = None
    meta_: Dict[str, Any] = None

    def __post_init__(self):
        if self.meta_ is None:
            self.meta_ = {}

    # ─────────────────────────────────────────────────────────────────────────
    # Design construction
    # ─────────────────────────────────────────────────────────────────────────
    def _lag_config(self) -> Dict[str, int]:
        cfg = _get_cfg_block()
        return {
            "ar_lags": int(cfg.get("ar_lags", 3)),
            "gdp_lags": int(cfg.get("gdp_lags", 4)),
            "degu_lags": int(cfg.get("degu_lags", 4)),
            "cblr_lags": int(cfg.get("cblr_lags", 4)),
            "gla_lags": int(cfg.get("gla_lags", 4)),
            "include_const": bool(cfg.get("include_const", True)),
        }

    def _resolve_columns(self, panel: pd.DataFrame) -> Dict[str, Optional[str]]:
        """
        Resolve column names in a case/alias tolerant manner.
        We DO NOT transform; we only select the already-transformed series.
        """
        col_ln_npl = _find_col(panel, ["ln_npl", "NPL"])  # transformation writes ln to NPL; pipeline aliases ln_npl
        col_gdp = _find_col(panel, ["gdp", "GDP"])
        col_degu = _find_col(panel, ["DEGU"])
        col_cblr = _find_col(panel, ["CBLR"])
        col_gla = _find_col(panel, ["GLA"])

        return {
            "ln_npl": col_ln_npl,
            "gdp": col_gdp,
            "degu": col_degu,
            "cblr": col_cblr,
            "gla": col_gla,
        }

    def _build_design(self, panel: pd.DataFrame, horizon: int) -> (pd.DataFrame, pd.Series):
        """
        Build X (design) and y for OLS:
          y_t = ln_npl_{t+h}
          X_t = [const?, AR lags of ln_npl, lags of GDP/DEGU/CBLR/GLA]
        We align on common non-null rows and drop any missing.
        """
        if not isinstance(panel.index, pd.DatetimeIndex):
            # enforce DatetimeIndex for stable alignment
            panel = panel.copy()
            panel.index = pd.to_datetime(panel.index)

        cols = self._resolve_columns(panel)
        if cols["ln_npl"] is None:
            raise KeyError("SCMIDASOLS: required column 'ln_npl' or 'NPL' not found in panel.")
        ln_npl = pd.to_numeric(panel[cols["ln_npl"]], errors="coerce")

        # Target (level): ln_npl shifted by -horizon
        y = ln_npl.shift(-int(horizon)).rename("y")

        # Predictors
        lag_cfg = self._lag_config()
        X_parts = []

        # Constant
        if lag_cfg["include_const"]:
            X_parts.append(pd.DataFrame({"const": 1.0}, index=panel.index))

        # AR lags of ln_npl
        if lag_cfg["ar_lags"] > 0:
            X_parts.append(_make_lags(ln_npl, lag_cfg["ar_lags"], "ln_npl"))

        # Exogenous lags (already transformed by data_transformation.py)
        def add_exog_lags(key: str, label: str, n_lags: int):
            cname = cols[key]
            if cname is None or n_lags <= 0:
                return
            s = pd.to_numeric(panel[cname], errors="coerce")
            X_parts.append(_make_lags(s, n_lags, label))

        add_exog_lags("gdp", "gdp", lag_cfg["gdp_lags"])
        add_exog_lags("degu", "degu", lag_cfg["degu_lags"])
        add_exog_lags("cblr", "cblr", lag_cfg["cblr_lags"])
        add_exog_lags("gla", "gla", lag_cfg["gla_lags"])

        if not X_parts:
            raise ValueError("SCMIDASOLS: no predictors constructed (check lag settings).")

        X = pd.concat(X_parts, axis=1).astype("float64")

        # Align X and y; drop rows with any NA
        df = pd.concat([y, X], axis=1)
        df = df.dropna(how="any")

        y_final = df["y"]
        X_final = df.drop(columns=["y"])

        # Store columns order
        self.columns_ = list(X_final.columns)
        return X_final, y_final

    # ─────────────────────────────────────────────────────────────────────────
    # Fitting and solving
    # ─────────────────────────────────────────────────────────────────────────
    def fit(self, panel: pd.DataFrame, horizon: int = 0) -> "SCMIDASOLS":
        """
        Fit ridge-regularized OLS on design built from the provided panel.
        """
        X, y = self._build_design(panel, horizon)
        Xv = _as_float_array(X.values)
        yv = _as_float_array(y.values)

        # (X'X + αI)β = X'y, with no penalty on constant columns
        XtX = Xv.T @ Xv
        Xty = Xv.T @ yv

        if self.alpha and self.alpha > 0.0:
            I = np.eye(XtX.shape[0], dtype=float)

            # Do not penalize constant columns (near-zero std)
            stds = np.nanstd(Xv, axis=0)
            const_mask = np.isfinite(stds) & (stds < 1e-12)
            if const_mask.any():
                I[const_mask, const_mask] = 0.0

            XtX = XtX + self.alpha * I

        beta = np.linalg.solve(XtX, Xty)

        self.coef_ = beta
        self.meta_.update({
            "horizon": int(horizon),
            "n_obs": int(Xv.shape[0]),
            "n_features": int(Xv.shape[1]),
            "alpha": float(self.alpha),
            "columns": self.columns_,
            "lags": self._lag_config(),
        })
        return self

    # ─────────────────────────────────────────────────────────────────────────
    # Predict
    # ─────────────────────────────────────────────────────────────────────────
    def predict(self, panel: pd.DataFrame, horizon: Optional[int] = None) -> pd.Series:
        """
        Predict ln_npl_{t+h} given a panel. If horizon is None, uses the training horizon.
        Returns a Series indexed like the input panel (trimmed to valid rows).
        """
        if self.coef_ is None or self.columns_ is None:
            raise RuntimeError("SCMIDASOLS not fitted. Call fit(...) first.")

        if horizon is None:
            horizon = int(self.meta_.get("horizon", 0))

        X, _y_dummy = self._build_design(panel, horizon)
        # ensure columns order matches training
        X = X.reindex(columns=self.columns_, copy=False)
        Xv = _as_float_array(X.values)
        preds = Xv @ _as_float_array(self.coef_)
        return pd.Series(preds, index=X.index, name="ln_npl_hat")

    # ─────────────────────────────────────────────────────────────────────────
    # Persistence
    # ─────────────────────────────────────────────────────────────────────────
    def save(self, path: Path | str):
        """
        Save model as JSON (coef, columns, meta).
        """
        path = Path(path)
        obj = {
            "alpha": float(self.alpha),
            "coef": self.coef_.tolist() if self.coef_ is not None else None,
            "columns": self.columns_,
            "meta": self.meta_,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(obj, f, indent=2)

    @classmethod
    def load(cls, path: Path | str) -> "SCMIDASOLS":
        path = Path(path)
        with open(path, "r") as f:
            obj = json.load(f)
        model = cls(alpha=float(obj.get("alpha", 0.0)))
        model.coef_ = np.asarray(obj.get("coef", []), dtype=float) if obj.get("coef") is not None else None
        model.columns_ = obj.get("columns", None)
        model.meta_ = obj.get("meta", {}) or {}
        return model
