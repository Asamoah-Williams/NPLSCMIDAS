# kpi_reporter.py
# Lightweight reporting utilities used by monitoring.py and processor.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from npl_src.db import DatabaseConnection

# Keep import for type compatibility; we won't assume specific methods exist on ReportRoot
try:
    from report_paths import ReportRoot
except Exception:  # pragma: no cover
    ReportRoot = object  # type: ignore

db_con = DatabaseConnection()


# --------------------------------------------------------------------------------------
# Public dataclasses (API used by monitoring.py / processor.py)
# --------------------------------------------------------------------------------------

@dataclass
class MetricThresholds:
    rmse_pp_max: float = 3.0
    mae_pp_max: float = 2.5
    smape_max: float = 30.0
    mase_max: float = 1.0
    r2_log_min: float = 0.20


@dataclass
class RunStamp:
    as_of_date: pd.Timestamp
    model_version: str = "v0.0.0"
    cfg_hash: str = "NA"
    code_commit: str = ""


# --------------------------------------------------------------------------------------
# Metrics (percent-based). These are kept simple & deterministic for auditing.
# --------------------------------------------------------------------------------------

def rmse_pp(y_true: pd.Series, y_pred: pd.Series) -> float:
    y_true = pd.to_numeric(y_true, errors="coerce")
    y_pred = pd.to_numeric(y_pred, errors="coerce")
    e2 = (y_true - y_pred) ** 2
    return float(np.sqrt(np.nanmean(e2))) if len(e2) else float("nan")


def mae_pp(y_true: pd.Series, y_pred: pd.Series) -> float:
    y_true = pd.to_numeric(y_true, errors="coerce")
    y_pred = pd.to_numeric(y_pred, errors="coerce")
    e = (y_true - y_pred).abs()
    return float(np.nanmean(e)) if len(e) else float("nan")


def smape(y_true: pd.Series, y_pred: pd.Series) -> float:
    y_true = pd.to_numeric(y_true, errors="coerce").astype(float)
    y_pred = pd.to_numeric(y_pred, errors="coerce").astype(float)
    denom = y_true.abs() + y_pred.abs()
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(denom == 0.0, 0.0, 200.0 * np.abs(y_pred - y_true) / denom)
    return float(np.nanmean(out)) if len(out) else float("nan")


def mase(y_true: pd.Series, y_pred: pd.Series, y_naive: pd.Series) -> float:
    """
    Mean Absolute Scaled Error (lower is better).
    y_true, y_pred in percent; y_naive should be the naive forecast of y_true (shifted series).
    """
    y_true = pd.to_numeric(y_true, errors="coerce")
    y_pred = pd.to_numeric(y_pred, errors="coerce")
    y_naive = pd.to_numeric(y_naive, errors="coerce")
    de = (y_true - y_pred).abs()
    dn = (y_true - y_naive).abs()
    denom = np.nanmean(dn) if np.isfinite(dn).any() else np.nan
    return float(np.nanmean(de) / denom) if denom and np.isfinite(denom) else float("nan")


def r2_log(y_true: pd.Series, y_pred: pd.Series) -> float:
    """
    R^2 computed on log scale of percentages converted back to decimal (robust for NPL).
    We guard against <=0 by flooring at 1e-6.
    """
    yt = pd.to_numeric(y_true, errors="coerce") / 100.0
    yp = pd.to_numeric(y_pred, errors="coerce") / 100.0
    yt = yt.clip(lower=1e-6)
    yp = yp.clip(lower=1e-6)
    lt = np.log(yt)
    lp = np.log(yp)
    ss_res = np.nansum((lt - lp) ** 2)
    ss_tot = np.nansum((lt - np.nanmean(lt)) ** 2)
    if ss_tot == 0 or not np.isfinite(ss_tot):
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


# --------------------------------------------------------------------------------------
# HTML helpers
# --------------------------------------------------------------------------------------

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _html_page(title: str, content: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>{title}</title>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif; margin: 16px; }}
    h1, h2, h3 {{ margin: 0.6em 0 0.4em; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ padding: 6px 8px; border-bottom: 1px solid #eee; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    .pill {{ display:inline-block; padding:2px 8px; border-radius: 12px; font-size: 12px; }}
    .pill.ok {{ background:#e8f5e9; color:#2e7d32; }}
    .pill.warn {{ background:#fff8e1; color:#8d6e63; }}
    .pill.bad {{ background:#ffebee; color:#c62828; }}
    .meta {{ color:#666; font-size: 12px; }}
    .grid {{ display:grid; grid-template-columns: repeat(2, minmax(240px, 1fr)); gap: 16px; }}
    .card {{ border:1px solid #eee; border-radius: 8px; padding: 12px; }}
    img {{ max-width: 100%; height: auto; border: 0; }}
    a {{ color: #1565c0; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
  </style>
</head>
<body>
{content}
</body>
</html>
"""


def _write_html(path: Path, body: str, title: str = "Report") -> None:
    _ensure_dir(path.parent)
    path.write_text(_html_page(title, body), encoding="utf-8")


def _render_table(df: pd.DataFrame, max_rows: int = 12) -> str:
    """
    Simple HTML table renderer used in dashboards. Non-destructive; caps rows for readability.
    """
    if df is None or df.empty:
        return "<em>No data</em>"
    head = df.head(max_rows)
    cols = head.columns.tolist()
    th = "".join(f"<th>{c}</th>" for c in cols)
    trs = []
    for _, r in head.iterrows():
        tds = "".join(f"<td>{r[c]}</td>" for c in cols)
        trs.append(f"<tr>{tds}</tr>")
    return f"<table><thead><tr>{th}</tr></thead><tbody>{''.join(trs)}</tbody></table>"


# --------------------------------------------------------------------------------------
# Robust path resolution against different ReportRoot implementations
# --------------------------------------------------------------------------------------

def _resolve_dirs(reports_root: ReportRoot, h: int) -> Dict[str, Path]:
    """
    Resolve horizon, data, and charts directories even if ReportRoot does NOT have
    horizon_dir()/data_dir()/charts_dir() helpers. Fall back to root/h{h}, root/_data, root/charts.
    """
    # Root path
    root: Optional[Path] = None
    for attr in ("root", "base", "reports_root", "path"):
        if hasattr(reports_root, attr):
            val = getattr(reports_root, attr)
            if isinstance(val, (str, Path)):
                root = Path(val)
                break
    if root is None:
        if isinstance(reports_root, (str, Path)):
            root = Path(reports_root)
        else:
            root = Path.cwd() / "reports"

    # Try helper methods if present
    h_dir = None
    if hasattr(reports_root, "horizon_dir"):
        try:
            h_dir = Path(getattr(reports_root, "horizon_dir")(h))
        except Exception:
            h_dir = None
    data_dir = None
    if hasattr(reports_root, "data_dir"):
        try:
            data_dir = Path(getattr(reports_root, "data_dir")())
        except Exception:
            data_dir = None
    charts_dir = None
    if hasattr(reports_root, "charts_dir"):
        try:
            charts_dir = Path(getattr(reports_root, "charts_dir")())
        except Exception:
            charts_dir = None

    # Fallbacks
    if h_dir is None:
        h_dir = root / f"h{h}"
    if data_dir is None:
        data_dir = root / "_data"
    if charts_dir is None:
        charts_dir = root / "charts"

    return {"h_dir": h_dir, "data_dir": data_dir, "charts_dir": charts_dir}


# --- EXTRA WIDGETS: forecast path, heatmap, AVP, comparison, optional FI(h=6) ----
def _summary_widgets_block(dashboard_html_path: Path) -> str:
    """
    Returns an HTML block that embeds global monitoring visuals and CSV links.
    The block assumes the per-horizon dashboard is written under reports/h{h}/,
    so ../charts and ../_data point to shared artifacts.
    We also conditionally show Feature Importance (h=6) if files exist.
    """
    out_dir = dashboard_html_path.parent
    charts_dir = out_dir.parent / "charts"
    data_dir = out_dir.parent / "_data"

    bits = []
    bits.append("""
<hr/>
<h2>Forecast Path (t+1..t+6)</h2>
<p><img src="../charts/forecast_path_latest.png" style="max-width:100%;" alt="Forecast Path">
   [<a href="../_data/forecast_path.csv">CSV</a>]</p>

<h2>Backtest Heatmap (Abs Error)</h2>
<p><img src="../charts/backtest_heatmap_abs_error_pp_latest.png" style="max-width:100%;" alt="Backtest Heatmap Abs Error">
   [<a href="../_data/backtest_heatmap_abs_error_pp.csv">CSV</a>]</p>

<h2>Actual vs Predicted (All Horizons)</h2>
<p><img src="../charts/actual_vs_pred_latest.png" style="max-width:100%;" alt="Actual vs Predicted">
   [<a href="../_data/actual_vs_pred.csv">CSV</a>]</p>

<h2>Model Comparison (MASE)</h2>
<p><img src="../charts/model_comparison_MASE_latest.png" style="max-width:100%;" alt="Model Comparison (MASE)">
   [<a href="../_data/model_comparison.csv">CSV</a>]</p>
""")

    # Optional: compact Feature Importance (h=6) widget, only if files exist to avoid broken links
    fi6_png = charts_dir / "feature_importance_h6_latest.png"
    fi6_csv = data_dir / "feature_importance_h6.csv"
    if fi6_png.exists() or fi6_csv.exists():
        fi_bits = '<h2>Feature Importance — h=6</h2><p>'
        if fi6_png.exists():
            fi_bits += '<img src="../charts/feature_importance_h6_latest.png" style="max-width:100%;" alt="Feature Importance h=6"> '
        if fi6_csv.exists():
            fi_bits += '[<a href="../_data/feature_importance_h6.csv">CSV</a>]'
        fi_bits += '</p>'
        bits.append(fi_bits)

    return "\n".join(bits)


# --------------------------------------------------------------------------------------
# MAIN REPORTERS (used by monitoring.py)
# --------------------------------------------------------------------------------------


# def _append_sql(df: pd.DataFrame, path: Path):
#     db_con = db.DatabaseConnection()
#     conn = db_con.get_db_connection()
#     table_name = path.stem  # Use the file name (without extension) as the table name
#     df.to_sql(table_name, conn, if_exists="append", index=False)

def log_monitoring(
    reports_root: ReportRoot,
    h: int,
    df_prod_kpis: pd.DataFrame,
    df_resid: pd.DataFrame,
    df_drift: Optional[pd.DataFrame] = None,
) -> Dict[str, str]:
    """
    Write per-horizon CSVs + HTML dashboard.
    Robust to ReportRoot implementations that may not expose horizon_dir/data_dir/charts_dir.
    """
    # Resolve directories (robust)
    dirs = _resolve_dirs(reports_root, h)
    h_dir, data_dir, charts_dir = dirs["h_dir"], dirs["data_dir"], dirs["charts_dir"]
    _ensure_dir(h_dir); _ensure_dir(data_dir); _ensure_dir(charts_dir)

    # ---- Save CSVs
    kpi_csv = h_dir / "kpis_monthly.csv"
    df_prod_kpis = df_prod_kpis.copy()
    if "target_month" in df_prod_kpis.columns:
        df_prod_kpis["target_month"] = pd.to_datetime(df_prod_kpis["target_month"]).dt.strftime("%Y-%m-%d")
    df_prod_kpis.to_csv(kpi_csv, index=False)
    # _append_csv(df_prod_kpis, kpi_csv)
    db_con._append_sql(df_prod_kpis, kpi_csv)

    resid_csv = h_dir / "residuals.csv"
    df_resid = df_resid.copy()
    if "target_month" in df_resid.columns:
        df_resid["target_month"] = pd.to_datetime(df_resid["target_month"]).dt.strftime("%Y-%m-%d")
    df_resid.to_csv(resid_csv, index=False)
    # _append_csv(df_resid, resid_csv)
    db_con._append_sql(df_resid, resid_csv)

    drift_csv = None
    if df_drift is not None and not df_drift.empty:
        drift_csv = h_dir / "drift.csv"
        df_drift.to_csv(drift_csv, index=False)
        # _append_csv(df_drift, drift_csv)
        db_con._append_sql(df_drift, drift_csv)

    # ---- Build HTML content
    title = f"Monitoring — Horizon h={h}"
    meta = f"""<p class="meta">
      KPI rows: {len(df_prod_kpis)} &middot; Residual rows: {len(df_resid)}
      {"&middot; Drift rows: " + str(len(df_drift)) if df_drift is not None else ""}
    </p>"""

    content = []
    content.append(f"<h1>{title}</h1>{meta}")

    content.append("<h2>KPIs (Monthly)</h2>")
    content.append(_render_table(df_prod_kpis))

    content.append("<h2>Residuals (pp)</h2>")
    if "residual_pp" in df_resid.columns:
        content.append(_render_table(df_resid[["target_month","residual_pp"]]))
    else:
        content.append(_render_table(df_resid))

    if drift_csv:
        content.append("<h2>Drift (PSI / Summary)</h2>")
        content.append(_render_table(df_drift))

    # ---- EXTRA: Embed summary visuals + CSV links (global artifacts written by monitoring.py)
    content.append(_summary_widgets_block(h_dir / "index.html"))

    # ---- Footer
    content.append("""
<hr/>
<p class="meta">This report aggregates operational monitoring outputs for the selected horizon.
See the <a href="../">reports root</a> for other horizons.</p>
""")

    index_html = h_dir / "index.html"
    _write_html(index_html, "\n".join(content), title=title)

    return {
        "kpi_csv": str(kpi_csv),
        "residual_csv": str(resid_csv),
        "drift_csv": str(drift_csv) if drift_csv else "",
        "html": str(index_html),
    }


def log_gates(
    reports_root: ReportRoot,
    h: int,
    kpis: Dict[str, float],
    thresholds: MetricThresholds,
    stamp: RunStamp,
) -> Dict[str, str]:
    """
    Write a simple gating summary (CSV + HTML card) comparing recent KPIs vs thresholds.
    Robust to ReportRoot implementations that may not expose helper methods.
    """
    dirs = _resolve_dirs(reports_root, h)
    gates_dir = dirs["h_dir"]
    _ensure_dir(gates_dir)

    # Status evaluation
    def _status(metric: str, val: float) -> str:
        if val is None or not np.isfinite(val):  # unknown
            return "warn"
        if metric == "r2_log":
            return "ok" if val >= thresholds.r2_log_min else "bad"
        bounds = {
            "rmse_pp": thresholds.rmse_pp_max,
            "mae_pp": thresholds.mae_pp_max,
            "smape": thresholds.smape_max,
            "mase": thresholds.mase_max,
        }
        lim = bounds.get(metric, None)
        if lim is None:
            return "warn"
        return "ok" if val <= lim else "bad"

    rows = []
    for m in ("rmse_pp", "mae_pp", "smape", "mase", "r2_log"):
        v = kpis.get(m, np.nan)
        rows.append({"metric": m, "value": float(v) if v is not None else np.nan, "status": _status(m, v)})

    gates_df = pd.DataFrame(rows)
    gates_csv = gates_dir / "gates_summary.csv"
    gates_df.to_csv(gates_csv, index=False)
    # _append_csv(gates_df, gates_csv)
    db_con._append_sql(gates_df, gates_csv)

    # HTML card
    def _pill(s: str) -> str:
        cls = "ok" if s == "ok" else ("bad" if s == "bad" else "warn")
        return f'<span class="pill {cls}">{s.upper()}</span>'

    items = "".join(
        f"<tr><td>{r['metric']}</td><td>{r['value']:.4g}</td><td>{_pill(r['status'])}</td></tr>"
        for _, r in gates_df.iterrows()
    )
    meta = f"""<p class="meta">
      As-of: {stamp.as_of_date.date()} &middot; Model: {stamp.model_version}
      &middot; cfg: {stamp.cfg_hash} &middot; commit: {stamp.code_commit}
    </p>"""
    body = f"""
<h2>Gating Summary (Recent Window)</h2>
{meta}
<table>
  <thead><tr><th>Metric</th><th>Value</th><th>Status</th></tr></thead>
  <tbody>{items}</tbody>
</table>
"""
    gates_html = gates_dir / "gates.html"
    _write_html(gates_html, body, title=f"Gates — h={h}")

    return {"gates_csv": str(gates_csv), "gates_html": str(gates_html)}


# --------------------------------------------------------------------------------------
# NEW: Backtest logger used by processor.py (adds missing log_backtest)
# --------------------------------------------------------------------------------------

def log_backtest(
    reports_root: ReportRoot,
    df_bt: pd.DataFrame,
    stamp: RunStamp,
) -> Dict[str, str]:
    """
    Publish all-horizons backtest artifacts:
      - Long CSV of backtest
      - Matrix CSV of abs error by (target_month x horizon)
      - Heatmap PNG (Reds: darker=larger error)
    Expected columns in df_bt: as_of_date, target_month, horizon, y_true_pct, y_pred_pct
    """
    # Validate required columns
    required = {"as_of_date", "target_month", "horizon", "y_true_pct", "y_pred_pct"}
    missing = required - set(df_bt.columns)
    if missing:
        raise ValueError(f"log_backtest: missing required columns: {missing}")

    # Resolve directories (re-use robust path logic; h is irrelevant for global artifacts, use h=0)
    dirs = _resolve_dirs(reports_root, h=0)
    root = dirs["h_dir"].parent  # reports root path
    data_dir = root / "_data" / "backtest"
    charts_dir = root / "charts"
    _ensure_dir(data_dir); _ensure_dir(charts_dir)

    # Clean frame
    df = df_bt.copy()
    df["as_of_date"] = pd.to_datetime(df["as_of_date"])
    df["target_month"] = pd.to_datetime(df["target_month"])
    df["horizon"] = pd.to_numeric(df["horizon"], errors="coerce").astype("Int64")
    df["y_true_pct"] = pd.to_numeric(df["y_true_pct"], errors="coerce")
    df["y_pred_pct"] = pd.to_numeric(df["y_pred_pct"], errors="coerce")
    df["abs_error_pp"] = (df["y_pred_pct"] - df["y_true_pct"]).abs()

    # Save long CSV (idempotent overwrite; processor controls versioning)
    long_csv = data_dir / "backtest_long.csv"
    df.sort_values(["target_month", "horizon"]).to_csv(long_csv, index=False)
    # _append_csv(df, long_csv)
    db_con._append_sql(df, long_csv)

    # Build matrix: abs error by month x horizon
    mat = (df
           .dropna(subset=["target_month", "horizon", "abs_error_pp"])
           .pivot_table(index="target_month", columns="horizon", values="abs_error_pp", aggfunc="mean")
           .sort_index())
    mat.columns = sorted(mat.columns.astype(int))
    matrix_csv = data_dir / "errors_by_h_month.csv"
    mat.to_csv(matrix_csv, index=True)
    # _append_csv(mat.reset_index(), matrix_csv)
    db_con._append_sql(mat.reset_index(), matrix_csv)

    # Heatmap PNG
    png_path = charts_dir / "backtest_heatmap_abs_error_pp_latest.png"
    try:
        import matplotlib.pyplot as plt
        A = mat.to_numpy(dtype=float)
        A_masked = np.ma.masked_invalid(A)
        plt.figure(figsize=(10, max(4, 0.25 * len(mat.index))))
        im = plt.imshow(A_masked, aspect="auto", interpolation="nearest", cmap="Reds", origin="upper")
        plt.yticks(np.arange(len(mat.index)), [pd.to_datetime(ix).strftime("%Y-%m") for ix in mat.index])
        plt.xticks(np.arange(len(mat.columns)), [str(c) for c in mat.columns])
        plt.xlabel("Horizon (months ahead)")
        plt.title("Backtest Heatmap — Abs Error (pp)")
        cbar = plt.colorbar(im)
        cbar.set_label("Abs error (pp)")
        plt.tight_layout()
        plt.savefig(png_path, dpi=160)
        plt.close()
    except Exception as e:
        print(f"[kpi_reporter] NOTE: could not render backtest heatmap: {e}")
        png_path = None

    # Optionally, add a tiny HTML to browse results (not required by processor)
    html = data_dir / "index.html"
    try:
        body = f"""
<h1>Backtest Artifacts</h1>
<p class="meta">As-of: {stamp.as_of_date.date()} &middot; Model: {stamp.model_version}
 &middot; cfg: {stamp.cfg_hash} &middot; commit: {stamp.code_commit}</p>
<h2>Files</h2>
<ul>
  <li><a href="backtest_long.csv">backtest_long.csv</a></li>
  <li><a href="errors_by_h_month.csv">errors_by_h_month.csv</a></li>
</ul>
<p><img src="../charts/backtest_heatmap_abs_error_pp_latest.png" alt="Backtest Heatmap" style="max-width:100%;"/></p>
"""
        _write_html(html, body, title="Backtest")
    except Exception:
        pass

    return {
        "backtest_long_csv": str(long_csv),
        "errors_matrix_csv": str(matrix_csv),
        "heatmap_png": str(png_path) if png_path else "",
        "html": str(html),
    }

