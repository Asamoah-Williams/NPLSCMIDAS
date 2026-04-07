"""
report_paths.py
Canonical locations for all reports & data artifacts (BCBS-239 traceability).
Dependency-free (std lib only) to avoid import headaches early in pipeline runs.
"""

from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict


@dataclass(frozen=True)
class ReportRoot:
    """Roots for human-readable and machine-readable outputs."""
    human: Path       # e.g., ROOT / "reports"
    data: Path        # e.g., ROOT / "reports" / "_data"
    meta: Path        # e.g., ROOT / "reports" / "_meta"
    controls: Path    # e.g., ROOT / "reports" / "_controls"
    validation: Path  # e.g., ROOT / "reports" / "_validation"

    @staticmethod
    def from_reports_root(reports_root: Path) -> "ReportRoot":
        return ReportRoot(
            human=reports_root,
            data=reports_root / "_data",
            meta=reports_root / "_meta",
            controls=reports_root / "_controls",
            validation=reports_root / "_validation",
        )


def ensure_dirs(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


# ---------------- Forecast (horizon-specific) ----------------

def forecast_human(root: ReportRoot, h: int) -> Path:
    p = root.human / "forecast" / f"h{h}"
    ensure_dirs(p)
    return p

def forecast_data(root: ReportRoot, h: int) -> Path:
    p = root.data / "forecast" / f"h{h}"
    ensure_dirs(p)
    return p

def forecast_files(root: ReportRoot, h: int) -> Dict[str, Path]:
    hdir = forecast_human(root, h)
    ddir = forecast_data(root, h)
    return {
        "html": hdir / f"npl_forecast_path_h{h}.html",
        "png":  hdir / f"npl_forecast_path_h{h}.png",
        "csv":  ddir / f"npl_forecast_path_h{h}.csv",
    }


# ---------------- Backtest (all horizons + per horizon) ----------------

def backtest_heatmap_files(root: ReportRoot, *, per_h: Optional[int]=None) -> Dict[str, Path]:
    if per_h is None:
        hdir = root.human / "backtest" / "heatmap"
        ddir = root.data / "backtest"
        ensure_dirs(hdir, ddir)
        return {
            "html": hdir / "backtest_heatmap_all_horizons.html",
            "png":  hdir / "backtest_heatmap_all_horizons.png",
            "errors_csv": ddir / "errors_by_h_month.csv",
            "kpis_csv":   ddir / "kpis_by_h_month.csv",
        }
    else:
        hdir = root.human / "backtest" / "heatmap" / f"h{per_h}"
        ensure_dirs(hdir)
        return {
            "png": hdir / f"backtest_heatmap_h{per_h}.png",
        }


# ---------------- Metrics (cross-validation, exec, monitoring) ----------------

def cv_metrics_files(root: ReportRoot, h: int) -> Dict[str, Path]:
    hdir = root.human / "metrics" / f"h{h}"
    ddir = root.data / "metrics" / f"h{h}"
    ensure_dirs(hdir, ddir)
    return {
        "cv_html":     hdir / f"cv_kpis_h{h}.html",
        "cv_vs_html":  hdir / f"cv_vs_naive_h{h}.html",
        "cv_folds":    ddir / f"cv_folds_h{h}.csv",
        "cv_summary":  ddir / f"cv_summary_h{h}.csv",
    }

def exec_metrics_files(root: ReportRoot) -> Dict[str, Path]:
    hdir = root.human / "metrics" / "exec"
    ddir = root.data / "metrics" / "exec"
    ensure_dirs(hdir, ddir)
    return {
        "dashboard_html": hdir / "exec_kpi_dashboard.html",
        "model_card_html": hdir / "model_card.html",
        "snapshot_json":   ddir / "exec_kpi_snapshot.json",
        # history CSV for snapshots (append-only)
        "snapshot_hist_csv": ddir / "exec_kpi_snapshot_history.csv",
    }

def monitoring_files(root: ReportRoot, h: int) -> Dict[str, Path]:
    hdir = root.human / "monitoring" / f"h{h}"
    ddir = root.data / "monitoring" / f"h{h}"
    ensure_dirs(hdir, ddir)
    return {
        "kpi_ts_html": hdir / f"kpi_timeseries_h{h}.html",
        "resid_html":  hdir / f"residuals_h{h}.html",
        "drift_html":  hdir / f"drift_h{h}.html",
        "kpi_ts_csv":  ddir / f"kpi_timeseries_h{h}.csv",
        "resid_csv":   ddir / f"residuals_h{h}.csv",
        "drift_csv":   ddir / f"drift_h{h}.csv",
    }


# ---------------- Importance & Scenarios ----------------

def importance_files(root: ReportRoot, h: int, *, latest_window: bool=False) -> Dict[str, Path]:
    base = root.human / "importance" / f"h{h}"
    data = root.data / "importance" / f"h{h}"
    ensure_dirs(base, data)
    stem = f"feature_importance_h{h}" if not latest_window else f"feature_importance_h{h}_latest_window"
    return {
        "html": base / f"{stem}.html",
        "png":  base / f"{stem}.png",
        "csv":  data / f"{stem}.csv",
    }

def scenario_files(root: ReportRoot, h: int) -> Dict[str, Path]:
    base = root.human / "scenario" / f"h{h}"
    data = root.data / "scenario" / f"h{h}"
    ensure_dirs(base, data)
    return {
        "html": base / f"top3_scenarios_h{h}.html",
        "png":  base / f"top3_scenarios_h{h}.png",
        "csv":  data / f"top3_scenarios_h{h}.csv",
        "manifest": data / f"scenario_manifest_h{h}.json",
    }


# ---------------- Controls / Meta ----------------

def gates_files(root: ReportRoot, h: int) -> Dict[str, Path]:
    hdir = root.controls / f"h{h}"
    ensure_dirs(hdir, root.meta)
    return {
        "gates_html": hdir / f"gates_report_h{h}.html",
        "gates_csv":  hdir / f"gates_h{h}.csv",
    }

def repro_manifest(root: ReportRoot) -> Path:
    ensure_dirs(root.meta)
    return root.meta / "repro_manifest.json"

def signoff_file(root: ReportRoot) -> Path:
    ensure_dirs(root.meta)
    return root.meta / "signoff.md"

def index_md(root: ReportRoot) -> Path:
    ensure_dirs(root.meta)
    return root.meta / "index.md"
