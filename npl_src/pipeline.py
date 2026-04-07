#!/usr/bin/env python3
# pipeline.py
from __future__ import annotations

import os, json, copy, shutil, warnings
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yaml

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score

import lightgbm as lgb
import joblib
import optuna

from npl_src.features import build_all_horizons
from npl_src.sc_midas import SCMIDASOLS
from npl_src.db import DatabaseConnection

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parents[1]
CFG  = yaml.safe_load((ROOT / "config.yml").read_text()) or {}

REPORTS = Path(CFG.get("reports_folder", "reports"))
if not REPORTS.is_absolute():
    REPORTS = (ROOT / REPORTS).resolve()
REPORTS.mkdir(parents=True, exist_ok=True)
TABLES = REPORTS / "tables"
TABLES.mkdir(parents=True, exist_ok=True)

MODELS_BASE = Path(CFG.get("models_folder") or "data/interim/models")
if not MODELS_BASE.is_absolute():
    MODELS_BASE = (ROOT / MODELS_BASE).resolve()
MODELS_BASE.mkdir(parents=True, exist_ok=True)

REG = CFG.get("models_registry", {}) or {}
CAND_SUB = str(REG.get("candidate_subdir", "candidate"))
APPR_SUB = str(REG.get("approved_subdir", "approved"))
POINTER  = str(REG.get("pointer_file", "LATEST_APPROVED.json"))

MODELS_CAND = MODELS_BASE / CAND_SUB
MODELS_APPR = MODELS_BASE / APPR_SUB
MODELS_CAND.mkdir(parents=True, exist_ok=True)
MODELS_APPR.mkdir(parents=True, exist_ok=True)

db_con = DatabaseConnection()

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

def _to_pp(log_vals: np.ndarray) -> np.ndarray:
    return np.exp(np.asarray(log_vals, dtype=float)) * 100.0

def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    return float(np.sqrt(np.mean((a-b)**2)))

@dataclass
class KPIs:
    rmse_pp: float
    mae_pp: float
    smape: float
    mase: float
    r2_log: float

def _kpis(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> KPIs:
    y_true_log = np.asarray(y_true_log, dtype=float)
    y_pred_log = np.asarray(y_pred_log, dtype=float)

    y_true_pp = _to_pp(y_true_log)
    y_pred_pp = _to_pp(y_pred_log)

    rmse_pp = _rmse(y_true_pp, y_pred_pp)
    mae_pp  = float(np.mean(np.abs(y_true_pp - y_pred_pp)))

    denom = (np.abs(y_true_pp) + np.abs(y_pred_pp))
    smape = float(np.mean(np.where(denom==0, 0.0, 2.0*np.abs(y_pred_pp - y_true_pp)/denom)))

    naive = np.abs(np.diff(y_true_pp))
    mase_denom = float(np.mean(naive)) if len(naive) else np.nan
    mase = float(mae_pp / mase_denom) if np.isfinite(mase_denom) and mase_denom > 0 else np.nan

    r2 = float(r2_score(y_true_log, y_pred_log)) if len(y_true_log) else np.nan
    return KPIs(rmse_pp, mae_pp, smape, mase, r2)

def _append_sql(df: pd.DataFrame, table: str):
    """Try to append to SQL via db_con; fall back to pandas.to_sql if needed."""
    try:
        # preferred: user's helper (handles audit + csv mirror in their codebase)
        db_con._append_sql(df, table)
        return
    except Exception:
        conn = db_con.get_db_connection()
        df.to_sql(table, conn, if_exists="append", index=False)

def _beta_midas_weights(k: int, a: float, b: float) -> np.ndarray:
    idx = np.arange(k, dtype=float)
    x = (idx + 1) / k
    w = np.power(x, a - 1.0) * np.power(1.0 - x, b - 1.0)
    w = np.maximum(w, 1e-12)
    w = w / w.sum()
    return w[::-1]

def _dynamic_leaves_cap(n_rows: int, n_features: int) -> int:
    base = max(31, min(255, int(np.sqrt(max(1, n_rows)) + 2*n_features)))
    return int(max(31, min(255, base)))

# ─────────────────────────────────────────────────────────────────────────────
# CV engines: fast (tscv) vs faithful (rolling-origin)
# ─────────────────────────────────────────────────────────────────────────────
def _rolling_origin_splits(n: int, min_train: int, valid: int, step: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    splits = []
    end_train = min_train
    while end_train + valid <= n:
        tr = np.arange(0, end_train)
        va = np.arange(end_train, end_train + valid)
        splits.append((tr, va))
        end_train += step
    return splits

def _cv_splits(X: pd.DataFrame, mode: str, n_splits: int, min_train: int, valid: int, step: int):
    mode = (mode or "rolling_origin").lower().strip()
    n = len(X)
    if mode == "rolling_origin":
        splits = _rolling_origin_splits(n, min_train=min_train, valid=valid, step=step)
        if len(splits) >= 2:
            return splits
        # fallback
    tscv = TimeSeriesSplit(n_splits=n_splits)
    return list(tscv.split(X))

# ─────────────────────────────────────────────────────────────────────────────
# Bias correction (optional)
# ─────────────────────────────────────────────────────────────────────────────
def _compute_bias_log_from_folds(
    folds: List[pd.DataFrame],
    method: str = "median",
    window_points: int = 12,
    min_points: int = 6
) -> float:
    if not folds:
        return 0.0
    df = pd.concat(folds, axis=0, ignore_index=True)
    if "date" not in df.columns:
        return 0.0
    df = df.dropna(subset=["y_true_log", "y_pred_log"]).copy()
    if df.empty:
        return 0.0
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    tail = df.tail(int(window_points)) if window_points and window_points > 0 else df
    if len(tail) < int(min_points):
        return 0.0
    resid = (tail["y_true_log"].values - tail["y_pred_log"].values)
    method = (method or "median").lower().strip()
    if method == "mean":
        return float(np.mean(resid))
    return float(np.median(resid))

def _apply_bias_log(y_pred_log: np.ndarray, bias_log: float) -> np.ndarray:
    return np.asarray(y_pred_log, dtype=float) + float(bias_log)

# ─────────────────────────────────────────────────────────────────────────────
# LightGBM training per horizon
# ─────────────────────────────────────────────────────────────────────────────
def _cv_fit_predict_lgbm(
    X: pd.DataFrame,
    y: pd.Series,
    seed: int,
    lgb_params: dict,
    cv_mode: str,
    n_splits: int,
    early_stopping_rounds: int,
) -> Tuple[np.ndarray, np.ndarray, List[pd.DataFrame]]:

    cv_cfg = CFG.get("cv", {}) or {}
    min_train = int(cv_cfg.get("min_train_points", 84))
    valid = int(cv_cfg.get("min_valid_points", 12))
    step = int(cv_cfg.get("backtest_step", 3))

    splits = _cv_splits(X, cv_mode, n_splits=n_splits, min_train=min_train, valid=valid, step=step)

    y_true_cv, y_pred_cv = [], []
    fold_preds: List[pd.DataFrame] = []

    for fold, (tr, va) in enumerate(splits):
        Xtr, Xva = X.iloc[tr], X.iloc[va]
        ytr, yva = y.iloc[tr], y.iloc[va]

        model = lgb.LGBMRegressor(random_state=seed, **lgb_params)
        model.fit(
            Xtr, ytr,
            eval_set=[(Xva, yva)],
            eval_metric="l1",
            callbacks=[lgb.early_stopping(int(early_stopping_rounds), verbose=False)]
        )

        pred = model.predict(Xva, num_iteration=getattr(model, "best_iteration_", None))
        y_true_cv.append(yva.to_numpy())
        y_pred_cv.append(pred)
        fold_preds.append(pd.DataFrame({"date": Xva.index, "y_true_log": yva.values, "y_pred_log": pred}))

    return np.concatenate(y_true_cv), np.concatenate(y_pred_cv), fold_preds

def _tune_lgbm_params(X: pd.DataFrame, y: pd.Series, seed: int) -> dict:
    opt = CFG.get("optuna", {}) or {}
    if not bool(opt.get("enabled", True)):
        cap = _dynamic_leaves_cap(len(X), X.shape[1])
        lgbm_cfg = CFG.get("lgbm", {}) or {}
        return dict(
            n_estimators=int(lgbm_cfg.get("n_estimators", 2000)),
            learning_rate=float(lgbm_cfg.get("learning_rate", 0.03)),
            num_leaves=int(lgbm_cfg.get("num_leaves", min(63, cap))),
            min_child_samples=int(lgbm_cfg.get("min_child_samples", 15)),
            subsample=float(lgbm_cfg.get("subsample", 0.8)),
            colsample_bytree=float(lgbm_cfg.get("colsample_bytree", 0.85)),
            reg_alpha=float(lgbm_cfg.get("reg_alpha", 0.05)),
            reg_lambda=float(lgbm_cfg.get("reg_lambda", 0.5)),
            min_split_gain=float(lgbm_cfg.get("min_split_gain", 0.0)),
            objective=str(lgbm_cfg.get("objective", "huber")),
            alpha=float(lgbm_cfg.get("alpha", 0.9)),
        )

    # FAST CV for tuning
    cv_mode = str(opt.get("cv_mode", "tscv"))
    cv_n = int(opt.get("cv_n_splits", 3))
    esr = int(opt.get("early_stopping_rounds", 100))
    trials = int(opt.get("trials", opt.get("n_trials", 60)))
    timeout = opt.get("timeout_sec", opt.get("timeout", None))
    timeout = int(timeout) if timeout is not None else None

    cap = _dynamic_leaves_cap(len(X), X.shape[1])

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 800, 2200),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 31, min(255, cap)),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 60),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
            "objective": "huber",
            "alpha": trial.suggest_float("alpha", 0.85, 0.95),
        }
        yt, yp, _ = _cv_fit_predict_lgbm(X, y, seed, params, cv_mode=cv_mode, n_splits=cv_n, early_stopping_rounds=esr)
        return _kpis(yt, yp).mae_pp

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=trials, timeout=timeout, show_progress_bar=False)
    best = study.best_params
    best["objective"] = "huber"
    return best

# ─────────────────────────────────────────────────────────────────────────────
# GDP MIDAS grid selection (fast CV)
# ─────────────────────────────────────────────────────────────────────────────
def _baseline_lgb_params_for_grid(X: pd.DataFrame) -> dict:
    cap = _dynamic_leaves_cap(len(X), X.shape[1])
    lgbm_cfg = CFG.get("lgbm", {}) or {}
    return dict(
        n_estimators=int(lgbm_cfg.get("n_estimators", 2000)),
        learning_rate=float(lgbm_cfg.get("learning_rate", 0.03)),
        num_leaves=int(lgbm_cfg.get("num_leaves", min(63, cap))),
        min_child_samples=int(lgbm_cfg.get("min_child_samples", 15)),
        subsample=float(lgbm_cfg.get("subsample", 0.8)),
        colsample_bytree=float(lgbm_cfg.get("colsample_bytree", 0.85)),
        reg_alpha=float(lgbm_cfg.get("reg_alpha", 0.05)),
        reg_lambda=float(lgbm_cfg.get("reg_lambda", 0.5)),
        min_split_gain=float(lgbm_cfg.get("min_split_gain", 0.0)),
        objective=str(lgbm_cfg.get("objective", "huber")),
        alpha=float(lgbm_cfg.get("alpha", 0.9)),
    )

def _grid_select_gdp_midas_params(panel: pd.DataFrame, seed: int) -> Optional[dict]:
    gs = (CFG.get("grid_search", {}) or {}).get("gdp_midas", {}) or {}
    if not bool(gs.get("enabled", False)):
        return None

    months_grid = [int(x) for x in (gs.get("months_grid") or [6,9,12,18])]
    beta_grid = [[float(a), float(b)] for a,b in (gs.get("beta_grid") or [[3.0,4.5]])]
    horizons = [int(h) for h in (gs.get("horizons") or [0,6])]
    max_candidates = gs.get("max_candidates", None)
    candidates = [(m,(ab[0],ab[1])) for m in months_grid for ab in beta_grid]
    if max_candidates is not None:
        candidates = candidates[:int(max_candidates)]

    cv_mode = str(gs.get("cv_mode", "tscv"))
    cv_n = int(gs.get("cv_n_splits", 3))
    esr = int(gs.get("early_stopping_rounds", 80))

    rows = []
    best = None

    for gdp_months,(a,b) in candidates:
        cfg_tmp = copy.deepcopy(CFG)
        cfg_tmp.setdefault("features", {})
        cfg_tmp["features"]["gdp_months"] = int(gdp_months)
        cfg_tmp["features"]["beta_shape"] = [float(a), float(b)]
        try:
            datasets = build_all_horizons(panel, cfg_tmp)
        except Exception as e:
            rows.append({"gdp_months": gdp_months,"beta_a":a,"beta_b":b,"mae_pp_mean":np.nan,"status":f"feature_fail:{type(e).__name__}"})
            continue

        maes = []
        status = "ok"
        for h in horizons:
            if h not in datasets:
                continue
            X, y = datasets[h]
            if len(X) < int(CFG.get("min_rows_after_lag", 100)):
                continue
            params = _baseline_lgb_params_for_grid(X)
            try:
                yt, yp, _ = _cv_fit_predict_lgbm(X, y, seed, params, cv_mode=cv_mode, n_splits=cv_n, early_stopping_rounds=esr)
                maes.append(float(_kpis(yt, yp).mae_pp))
            except Exception as e:
                status = f"cv_fail:{type(e).__name__}"
                maes = []
                break

        mae_mean = float(np.mean(maes)) if len(maes) else np.nan
        rows.append({"gdp_months": gdp_months,"beta_a":a,"beta_b":b,"mae_pp_mean":mae_mean,"status":status})
        if np.isfinite(mae_mean) and (best is None or mae_mean < best["mae_pp_mean"]):
            best = {"gdp_months": gdp_months, "beta_shape": [a,b], "mae_pp_mean": mae_mean}

    if best is None:
        return None
    return {"best": best, "grid_scores": rows}

# ─────────────────────────────────────────────────────────────────────────────
# Data load
# ─────────────────────────────────────────────────────────────────────────────
def _load_panel() -> pd.DataFrame:
    conn = db_con.get_db_connection()
    df = pd.read_sql("SELECT * FROM t_panel", conn, parse_dates=["date"]).set_index("date").sort_index()
    if "NPL" in df.columns and "ln_npl" not in df.columns:
        df["ln_npl"] = df["NPL"]
    if "GDP" in df.columns and "gdp" not in df.columns:
        df["gdp"] = df["GDP"]
    return df

# ─────────────────────────────────────────────────────────────────────────────
# Training orchestration with model promotion
# ─────────────────────────────────────────────────────────────────────────────
def main():
    seed = int(CFG.get("seed", CFG.get("random_state", 42)))
    run_ts_utc = _utc_now_iso()
    run_id = run_ts_utc.replace(":", "").replace("-", "")
    run_models_dir = MODELS_CAND / f"run_{run_id}"
    run_models_dir.mkdir(parents=True, exist_ok=True)

    panel = _load_panel()

    # optional grid select
    gs_out = _grid_select_gdp_midas_params(panel, seed)
    if gs_out is not None:
        CFG.setdefault("features", {})
        CFG["features"]["gdp_months"] = int(gs_out["best"]["gdp_months"])
        CFG["features"]["beta_shape"] = [float(gs_out["best"]["beta_shape"][0]), float(gs_out["best"]["beta_shape"][1])]

    datasets = build_all_horizons(panel, CFG)

    horizons = [int(h) for h in (CFG.get("horizons") or sorted(datasets.keys()))]
    model_type_default = str(CFG.get("model_type", "midas_ml")).lower()
    per_h = CFG.get("model_type_per_horizon") or {}

    run_rows = []
    cv_rows = []
    saved = []
    gates_ok = True

    for h in horizons:
        model_type = str(per_h.get(str(h), per_h.get(h, model_type_default))).lower()
        if h not in datasets:
            continue
        X, y = datasets[h]
        if len(X) < int(CFG.get("min_rows_after_lag", 100)):
            continue

        if model_type in ("sc_midas_ols", "sc_midas", "ols"):
            m = SCMIDASOLS()
            m.fit(panel, horizon=h)
            #out = run_models_dir / f"sc_midas_ols_h{h}.json"
            out = MODELS_CAND / f"sc_midas_ols_h{h}.json"
            m.save(out)
            saved.append(out.name)

            # No CV here (you can add if needed)
            run_rows.append({
                "run_ts_utc": run_ts_utc, "h": h, "model_type": "sc_midas_ols",
                "rows": int(len(X)), "n_features": int(X.shape[1]),
                "rmse_pp": np.nan, "mae_pp": np.nan, "smape": np.nan, "mase": np.nan, "r2_log": np.nan,
                "bias_log": 0.0, "params": json.dumps({"sc_midas": CFG.get("sc_midas", {})}),
                "model_artifact": out.name, "status": "trained"
            })
            continue

        # LGBM MIDAS-ML
        best_params = _tune_lgbm_params(X, y, seed)
        # FINAL evaluation uses rolling-origin CV for gating
        yt, yp, folds = _cv_fit_predict_lgbm(
            X, y, seed, best_params,
            cv_mode="rolling_origin",
            n_splits=int(CFG.get("cv", {}).get("n_splits", 5)),
            early_stopping_rounds=int(CFG.get("early_stopping_rounds", 200))
        )

        # bias correction
        bc = CFG.get("bias_correction", {}) or {}
        bc_enabled = bool(bc.get("enabled", False))
        bias_log = 0.0
        if bc_enabled:
            bias_log = _compute_bias_log_from_folds(
                folds,
                method=str(bc.get("method", "median")),
                window_points=int(bc.get("window_points", 12)),
                min_points=int(bc.get("min_points", 6))
            )
        yp_adj = _apply_bias_log(yp, bias_log) if bc_enabled else yp

        k_raw = _kpis(yt, yp)
        k_adj = _kpis(yt, yp_adj)

        # Export CV predictions (raw + adj)
        for i, dfp in enumerate(folds):
            dfp = dfp.copy()
            dfp["run_ts_utc"] = run_ts_utc
            dfp["h"] = h
            dfp["fold"] = i
            dfp["y_true_pp"] = _to_pp(dfp["y_true_log"].values)
            dfp["y_pred_pp"] = _to_pp(dfp["y_pred_log"].values)
            dfp["bias_log"] = float(bias_log) if bc_enabled else 0.0
            dfp["y_pred_log_adj"] = _apply_bias_log(dfp["y_pred_log"].values, bias_log) if bc_enabled else dfp["y_pred_log"].values
            dfp["y_pred_pp_adj"] = _to_pp(dfp["y_pred_log_adj"].values)
            cv_rows.append(dfp)

        # Gate check (production: use adjusted metrics)
        gates = (CFG.get("gates") or {}) if isinstance(CFG, dict) else {}
        max_mae = float(gates.get("max_mae_pp", np.inf))
        if np.isfinite(max_mae) and k_adj.mae_pp > max_mae:
            gates_ok = False

        # Fit final model on full data (simple holdout for early stopping)
        holdout = int(max(6, round(len(X) * 0.1)))
        Xtr, Xva = X.iloc[:-holdout], X.iloc[-holdout:]
        ytr, yva = y.iloc[:-holdout], y.iloc[-holdout:]

        model = lgb.LGBMRegressor(random_state=seed, **best_params)
        model.fit(
            Xtr, ytr,
            eval_set=[(Xva, yva)],
            eval_metric="l1",
            callbacks=[lgb.early_stopping(int(CFG.get("early_stopping_rounds", 200)), verbose=False)]
        )

        # Persist bias inside model artifact for inference
        model.bias_log_add_ = float(bias_log) if bc_enabled else 0.0
        model.bias_method_ = str(bc.get("method", "median")) if bc_enabled else "none"
        model.bias_window_points_ = int(bc.get("window_points", 12)) if bc_enabled else 0

        # out = run_models_dir / f"lgbm_midas_h{h}.pkl"
        out = MODELS_CAND / f"lgbm_midas_h{h}.pkl"
        joblib.dump(model, out)
        saved.append(out.name)

        run_rows.append({
            "run_ts_utc": run_ts_utc, "h": h, "model_type": "midas_ml",
            "rows": int(len(X)), "n_features": int(X.shape[1]),
            "rmse_pp": k_raw.rmse_pp, "mae_pp": k_raw.mae_pp, "smape": k_raw.smape, "mase": k_raw.mase, "r2_log": k_raw.r2_log,
            "rmse_pp_adj": k_adj.rmse_pp, "mae_pp_adj": k_adj.mae_pp, "smape_adj": k_adj.smape, "mase_adj": k_adj.mase, "r2_log_adj": k_adj.r2_log,
            "bias_log": float(bias_log) if bc_enabled else 0.0,
            "params": json.dumps(best_params),
            "model_artifact": out.name,
            "status": "trained"
        })

    # Persist training logs to SQL
    if run_rows:
        _append_sql(pd.DataFrame(run_rows), str(CFG.get("train_run_table", "t_npl_train_runs")))
    if cv_rows:
        _append_sql(pd.concat(cv_rows, axis=0, ignore_index=True), str(CFG.get("cv_pred_table", "t_npl_cv_predictions")))

    # Promotion: only if gates OK
    promotion = {"run_ts_utc": run_ts_utc, "run_id": run_id, "gates_ok": bool(gates_ok), "saved": saved}
    if gates_ok:
        # copy artifacts to approved/run_<id>
        appr_run = MODELS_APPR / f"run_{run_id}"
        if appr_run.exists():
            shutil.rmtree(appr_run)
        shutil.copytree(run_models_dir, appr_run)

        pointer = {
            "approved_run_id": run_id,
            "approved_run_ts_utc": run_ts_utc,
            "approved_dir": str(appr_run),
            "artifacts": saved
        }
        (MODELS_APPR / POINTER).write_text(json.dumps(pointer, indent=2))
        promotion["promoted_to"] = str(appr_run)
    else:
        promotion["promoted_to"] = None

    (TABLES / "train_promotion_manifest.json").write_text(json.dumps(promotion, indent=2))
    print("✔ Training complete.")
    print(json.dumps(promotion, indent=2))

if __name__ == "__main__":
    main()
