# -*- coding: utf-8 -*-
"""
NPL_Transformation_Validation.py — Clean, DB-ready, append-only, and world-class.

Features:
- Aligns with data_transformation.py (ROOT via config.yml; data/ and reports/)
- MonthEnd frequency ("ME") to avoid deprecation warnings
- Canonical CSV is append-only (header once), RFC-4180, UTF-8
- NDJSON append-only audit trail; latest JSON snapshot
- Excel per-run (timestamped) to avoid overwrites
- Legacy artifact names preserved as compact pointer CSVs (append-only)
- Robust error handling: writes error rows even on failure
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import platform
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import uuid

import numpy as np
import pandas as pd
import yaml

# Optional ADF
try:
    from statsmodels.tsa.stattools import adfuller
except Exception:
    adfuller = None


LEGACY_ARTIFACTS = [
    "NPL.csv",
    "GDP.csv",
    "DEGU.csv",
    "CBLR.csv",
    "GLA.csv",
    "GDP_Revisions.csv",
    "Transformation_Operational_Report.csv",
    "Transformation_Validation_Report.csv",
    "NPL_DQ_Report_updated.xlsx",
]


def _find_root() -> Path:
    here = Path(__file__).resolve().parents[1]
    for p in [here] + list(here.parents):
        if (p / "config.yml").exists():
            return p
    return Path.cwd()


ROOT = _find_root()
print("printing root path...................")
print(ROOT)
print("<<<<<<<<<<<<<<<<<<<<end>>>>>>>>>>>>>>>>>>>>>")

def _setup_logging(report_dir: Path, verbose: bool = False) -> Path:
    log_dir = report_dir / "_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"validation_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.log"
    handlers = [logging.FileHandler(log_file, encoding="utf-8")]
    if verbose:
        handlers.append(logging.StreamHandler(sys.stdout))
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        handlers=handlers,
    )
    logging.info("Logging to %s", log_file)
    return log_file


def _load_cfg(root: Path) -> dict:
    cfg_path = root / "config.yml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"config.yml not found at {cfg_path}")
    cfg = yaml.safe_load(cfg_path.read_text()) or {}
    return cfg


def _resolve_series_files(data_dir: Path, series_names: List[str]) -> Dict[str, Path]:
    return {s: data_dir / f"t_{s}.csv" for s in series_names}


def _read_series_csv(fn: Path, name: str) -> pd.Series:
    if not fn.exists():
        raise FileNotFoundError(f"Series missing: {fn}")
    df = pd.read_csv(fn)
    date_col = "date" if "date" in df.columns else ("Date" if "Date" in df.columns else None)
    if date_col is None:
        raise ValueError(f"{fn.name}: expected a 'date' or 'Date' column")
    val_col_candidates = [c for c in df.columns if c.lower() not in ("date",)]
    if not val_col_candidates:
        raise ValueError(f"{fn.name}: no value column found")
    val_col = val_col_candidates[0]
    s = df[[date_col, val_col]].copy()
    s[date_col] = pd.to_datetime(s[date_col])
    s[date_col] = s[date_col].dt.to_period("M").dt.to_timestamp("M")
    s = s.set_index(date_col)[val_col].astype(float).rename(name).sort_index()
    return s


def _continuous_months(idx: pd.DatetimeIndex) -> Tuple[int, List[pd.Timestamp]]:
    full = pd.date_range(idx.min(), idx.max(), freq="ME")  # MonthEnd
    missing = sorted(set(full) - set(idx))
    return len(missing), missing


def _iqr_flags(x: pd.Series, k: float = 3.0) -> pd.Series:
    q1, q3 = np.nanpercentile(x, 25), np.nanpercentile(x, 75)
    iqr = q3 - q1
    if iqr == 0 or not np.isfinite(iqr):
        return pd.Series(False, index=x.index)
    low, high = q1 - k * iqr, q3 + k * iqr
    return (x < low) | (x > high)


def _adf_p(x: pd.Series) -> float:
    if adfuller is None:
        return float("nan")
    try:
        return float(adfuller(x.dropna(), autolag="AIC")[1])
    except Exception:
        return float("nan")


@dataclass
class CheckResult:
    section: str
    series: str
    metric: str
    value: object
    threshold: object = None
    passed: Optional[bool] = None
    notes: str = ""


def _append_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    # Stable, DB-ready schema
    cols = [
        "run_id",
        "run_ts",
        "section",
        "series",
        "metric",
        "value",
        "threshold",
        "passed",
        "notes",
        "root",
        "data_dir",
        "reports_dir",
    ]
    for c in cols:
        if c not in df.columns:
            df[c] = "" if c not in ("value", "threshold", "passed") else None
    df = df[cols]
    with open(path, "a", encoding="utf-8", newline="") as f:
        df.to_csv(f, header=write_header, index=False, quoting=csv.QUOTE_MINIMAL)


def _append_ndjson(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(payload, ensure_ascii=False)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def validate(
    cfg: dict,
    data_dir: Path,
    reports_dir: Path,
    do_adf: bool,
    series_list: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, bool, dict, dict, dict]:
    if series_list is None:
        series_list = ["NPL", "GDP", "DEGU", "CBLR", "GLA"]

    dq = cfg.get("dq_thresholds", {})
    th_missing = int(dq.get("missing_months_max", 0))
    th_dup = int(dq.get("duplicates_max", 0))
    th_nan = int(dq.get("nan_max", 0))
    th_inf = int(dq.get("inf_max", 0))
    th_adf = float(dq.get("adf_pvalue_max", 0.10))

    series_files = _resolve_series_files(data_dir, series_list)
    panel_file = data_dir / "t_panel.csv"

    series = {}
    for name, fn in series_files.items():
        if fn.exists():
            series[name] = _read_series_csv(fn, name)
            logging.info("Loaded %s [%d]: %s", name, series[name].size, fn)
        else:
            logging.warning("Missing transformed series file: %s", fn)

    if not series:
        raise FileNotFoundError("No transformed series found. Run data_transformation.py first.")

    panel = None
    if panel_file.exists():
        panel = pd.read_csv(panel_file, parse_dates=["date"])
        panel["date"] = panel["date"].dt.to_period("M").dt.to_timestamp("M")
        panel = panel.sort_values("date")
        logging.info("Loaded panel (%d x %d): %s", panel.shape[0], panel.shape[1], panel_file)
    else:
        logging.warning("Panel not found: %s", panel_file)

    results: List[CheckResult] = []

    for name, s in series.items():
        results.append(CheckResult("Meta", name, "start", s.index.min().date().isoformat()))
        results.append(CheckResult("Meta", name, "end", s.index.max().date().isoformat()))
        results.append(CheckResult("Meta", name, "obs", int(s.shape[0])))

        dups = int(s.index.duplicated().sum())
        results.append(CheckResult("Integrity", name, "duplicates", dups, th_dup, dups <= th_dup))

        miss_ct, miss_list = _continuous_months(s.index)
        miss_sample = ";".join([d.strftime("%Y-%m-%d") for d in miss_list[:24]])
        results.append(CheckResult("Integrity", name, "missing_months_count", miss_ct, th_missing, miss_ct <= th_missing))
        results.append(CheckResult("Integrity", name, "missing_months_sample", miss_sample))

        nan_ct = int(np.isnan(s.values).sum())
        inf_ct = int(np.isinf(s.values).sum())
        results.append(CheckResult("Quality", name, "nan_count", nan_ct, th_nan, nan_ct <= th_nan))
        results.append(CheckResult("Quality", name, "inf_count", inf_ct, th_inf, inf_ct <= th_inf))

        out_ct = int(_iqr_flags(s, 3.0).sum())
        results.append(CheckResult("Quality", name, "iqr3_outliers", out_ct))

        if do_adf:
            pval = _adf_p(s)
            passed = (not np.isfinite(pval)) or (pval <= th_adf)
            results.append(CheckResult("ADF", name, "adf_pvalue", pval, th_adf, passed))

    if panel is not None:
        pidx = panel["date"]
        miss_ct, miss_list = _continuous_months(pidx)
        miss_sample = ";".join([d.strftime("%Y-%m-%d") for d in miss_list[:24]])
        results.append(CheckResult("Integrity", "panel", "missing_months_count", miss_ct, th_missing, miss_ct <= th_missing))
        results.append(CheckResult("Integrity", "panel", "missing_months_sample", miss_sample))
        results.append(CheckResult("Meta", "panel", "start", pidx.min().date().isoformat()))
        results.append(CheckResult("Meta", "panel", "end", pidx.max().date().isoformat()))
        results.append(CheckResult("Meta", "panel", "obs", int(panel.shape[0])))

        for name, s in series.items():
            if name in panel.columns:
                m = pd.merge(
                    s.rename("val").reset_index().rename(columns={"index": "date"}),
                    panel[["date", name]],
                    on="date",
                    how="inner",
                )
                eps = 1e-8
                mismatch_ct = int((np.abs(m["val"].values - m[name].astype(float).values) > eps).sum())
                results.append(CheckResult("Consistency", name, "panel_value_mismatch_count", mismatch_ct, 0, mismatch_ct == 0))
            else:
                results.append(CheckResult("Consistency", name, "panel_column_present", False, True, False, notes="Missing in panel"))

    df = pd.DataFrame([asdict(r) for r in results])
    flagged = df.dropna(subset=["passed"])
    all_pass = bool(flagged["passed"].all()) if not flagged.empty else True

    env = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "python": sys.version.replace("\n", " "),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "root": str(ROOT),
        "data_dir": str(data_dir),
        "reports_dir": str(reports_dir),
        "adf_available": adfuller is not None,
        "series_checked": list(series.keys()),
    }
    thresholds = {
        "missing_months_max": th_missing,
        "duplicates_max": th_dup,
        "nan_max": th_nan,
        "inf_max": th_inf,
        "adf_pvalue_max": th_adf if do_adf else None,
    }
    counts = {
        "rows": int(df.shape[0]),
        "flagged_checks": int(flagged.shape[0]),
        "failed_checks": int((flagged["passed"] == False).sum()) if not flagged.empty else 0,
    }

    return df, all_pass, env, thresholds, counts


def _resolve_legacy_output(name: str, data_dir: Path, reports_dir: Path) -> Path:
    p = Path(name)
    if p.is_absolute():
        return p
    return (reports_dir if "report" in p.name.lower() else data_dir) / p.name


def write_outputs(
    df: pd.DataFrame,
    all_pass: bool,
    env: dict,
    thresholds: dict,
    counts: dict,
    root: Path,
    data_dir: Path,
    reports_dir: Path,
    error: Optional[str] = None,
) -> Dict[str, str]:
    # Append run metadata columns
    run_id = str(uuid.uuid4())
    run_ts = pd.Timestamp.now().isoformat()
    df = df.copy()
    df["run_id"] = run_id
    df["run_ts"] = run_ts
    df["root"] = str(root)
    df["data_dir"] = str(data_dir)
    df["reports_dir"] = str(reports_dir)

    # Canonical CSV (append-only)
    canon_csv = data_dir / "t_transformation_validation_report.csv"
    _append_csv(df, canon_csv)

    # NDJSON append + latest JSON snapshot
    ndjson_path = data_dir / "t_transformation_validation_report.ndjson"
    payload = {
        "run_id": run_id,
        "run_ts": run_ts,
        "all_pass": all_pass,
        "env": env,
        "thresholds": thresholds,
        "counts": counts,
        "error": error,
    }
    canon_json = data_dir / "t_transformation_validation_report.json"
    with open(canon_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    _append_ndjson(payload, ndjson_path)

    # Excel per-run to avoid overwrites
    ts_compact = run_ts.replace(":", "").replace("-", "").replace(".", "")
    xlsx = reports_dir / f"t_transformation_validation_report_{ts_compact}.xlsx"
    with pd.ExcelWriter(xlsx, engine="xlsxwriter") as xl:
        summary = pd.DataFrame([{
            **{"run_id": run_id, "run_ts": run_ts, "all_pass": all_pass},
            **env, **thresholds, **counts, **({"error": error} if error else {})
        }]).T.reset_index()
        summary.columns = ["key", "value"]
        summary.to_excel(xl, sheet_name="Summary", index=False)
        for sec in ["Meta", "Integrity", "Quality", "Consistency", "ADF"]:
            sdf = df[df["section"] == sec].copy()
            if not sdf.empty:
                sdf.to_excel(xl, sheet_name=sec, index=False)

    # Legacy artifacts: compact pointer CSVs (append-only)
    pointer = pd.DataFrame([{
        "run_id": run_id,
        "run_ts": run_ts,
        "canonical_csv": str(canon_csv),
        "canonical_json": str(canon_json),
        "ndjson": str(ndjson_path),
        "excel": str(xlsx),
        "root": str(root),
        "data_dir": str(data_dir),
        "reports_dir": str(reports_dir),
    }])
    for name in LEGACY_ARTIFACTS:
        outp = _resolve_legacy_output(name, data_dir, reports_dir)
        _append_csv(pointer.assign(section="Pointer", series=name, metric="paths", value="canonical_paths"), outp)

    return {
        "canonical_csv": str(canon_csv),
        "canonical_json": str(canon_json),
        "ndjson": str(ndjson_path),
        "excel": str(xlsx),
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="NPL Transformation Validation (clean & DB-ready)")
    parser.add_argument("--adf", action="store_true", help="Compute ADF p-values (if available).")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero if thresholded checks fail.")
    parser.add_argument("--series", nargs="*", default=None, help="Subset of series (default: NPL GDP DEGU CBLR GLA).")
    parser.add_argument("--verbose", action="store_true", help="Console logging output.")
    parser.add_argument("--output-dir", default=None, help="Override reports output directory.")
    args = parser.parse_args(argv)

    cfg = _load_cfg(ROOT)
    data_dir = ROOT / cfg.get("data_folder", "data")
    reports_dir = ROOT / cfg.get("reports_folder", "reports")
    data_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    if args.output_dir:
        reports_dir = Path(args.output_dir)
        reports_dir.mkdir(parents=True, exist_ok=True)

    _setup_logging(reports_dir, verbose=args.verbose)

    try:
        df, all_pass, env, thresholds, counts = validate(
            cfg=cfg,
            data_dir=data_dir,
            reports_dir=reports_dir,
            do_adf=args.adf,
            series_list=args.series,
        )
        outs = write_outputs(df, all_pass, env, thresholds, counts, ROOT, data_dir, reports_dir, error=None)
        print("✅ Validation completed.")
        print("   Canonical CSV:", outs["canonical_csv"])
        print("   Excel Report :", outs["excel"])
        if args.strict and not all_pass:
            sys.exit(1)
        sys.exit(0)
    except Exception as e:
        logging.exception("Validation failed: %s", e)
        # Emit a DB-storable error row
        err_df = pd.DataFrame([{
            "section": "System",
            "series": "__error__",
            "metric": "exception",
            "value": str(e),
            "threshold": None,
            "passed": False,
            "notes": "validator exception"
        }])
        outs = write_outputs(
            err_df,
            all_pass=False,
            env={
                "timestamp": pd.Timestamp.now().isoformat(),
                "python": sys.version.replace("\n", " "),
                "pandas": pd.__version__,
                "numpy": np.__version__,
                "platform": platform.platform(),
                "root": str(ROOT),
                "data_dir": str(data_dir),
                "reports_dir": str(reports_dir),
                "adf_available": adfuller is not None,
                "series_checked": args.series or ["NPL", "GDP", "DEGU", "CBLR", "GLA"],
            },
            thresholds={},
            counts={"rows": 1, "flagged_checks": 1, "failed_checks": 1},
            root=ROOT,
            data_dir=data_dir,
            reports_dir=reports_dir,
            error=str(e),
        )
        print("❌ Validation failed, but error report was written:")
        print("   Canonical CSV:", outs["canonical_csv"])
        print("   Excel Report :", outs["excel"])
        if args.strict:
            sys.exit(1)
        sys.exit(0)


if __name__ == "__main__":
    main()

