# src/data_transformation.py
from __future__ import annotations
import os
from pathlib import Path
from typing import Optional, Tuple, Iterable
from datetime import datetime, UTC

import numpy as np
import pandas as pd
import yaml
from npl_src.db import DatabaseConnection

db_con = DatabaseConnection()


# ────────────────────────────────────────────────────────────────────────────
# Project paths (works even when __file__ is missing)
# ────────────────────────────────────────────────────────────────────────────
def _root() -> Path:
    fp = globals().get("__file__")
    if fp:
        return Path(fp).resolve().parents[1]
    cur = Path.cwd().resolve()
    for base in [cur] + list(cur.parents):
        if (base / "config.yml").exists():
            return base
    return cur


ROOT = _root()
CFG = yaml.safe_load((ROOT / "config.yml").read_text())
DATA = ROOT / CFG.get("data_folder", "data")

# Source files live directly in DATA/ (no subfolder)
SRC_DIR = Path(os.environ.get("DATA_SRC_DIR", DATA)).resolve()

# Outputs (CSV with t_ prefix in the same DATA folder)
PANEL_CSV = DATA / "t_panel.csv"
GDP_REV_CSV = DATA / "t_gdp_revisions.csv"
RUN_REP_CSV = DATA / "t_transformation_report.csv"

SERIES_FILES = {
    "NPL": DATA / "t_NPL.csv",
    "GDP": DATA / "t_GDP.csv",
    "DEGU": DATA / "t_DEGU.csv",
    "CBLR": DATA / "t_CBLR.csv",
    "GLA": DATA / "t_GLA.csv",
}

GDP_REV_LOOKBACK_M = int(CFG.get("gdp_revision_lookback_months", 24))

# Optional stems override from config.yml
RAW_STEMS: dict = CFG.get("raw_stems", {}) or {}

# ────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ────────────────────────────────────────────────────────────────────────────
_EXTS = (".csv", ".parquet", ".xlsx", ".xls")


def _list_src_files() -> list[Path]:
    return sorted([p for p in SRC_DIR.iterdir() if p.is_file()]) if SRC_DIR.exists() else []


def _candidate_stems(key: str, default_fallbacks: Iterable[str]) -> list[str]:
    cfg_list = RAW_STEMS.get(key)
    if isinstance(cfg_list, str):
        cfg_list = [cfg_list]
    stems = []
    if cfg_list:
        stems.extend(cfg_list)
    stems.extend(default_fallbacks)
    out, seen = [], set()
    for s in stems:
        s2 = str(s).strip()
        if s2 and s2.lower() not in seen:
            out.append(s2);
            seen.add(s2.lower())
    return out


def _find_src_by_stems(stems: Iterable[str]) -> Optional[Path]:
    # exact match first
    for stem in stems:
        for ext in _EXTS:
            p = SRC_DIR / f"{stem}{ext}"
            if p.exists():
                return p
    # prefix / case-insensitive
    for stem in stems:
        for ext in _EXTS:
            for p in SRC_DIR.glob(f"{stem}*{ext}"):
                if p.exists():
                    return p
            for p in _list_src_files():
                if p.suffix.lower() in _EXTS and stem.lower() in p.stem.lower():
                    return p
    return None


def _read_any(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf == ".csv":
        return pd.read_csv(path)
    if suf == ".parquet":
        return pd.read_parquet(path)
    if suf in (".xlsx", ".xls"):
        return pd.read_excel(path)
    raise ValueError(f"Unsupported file type: {path.suffix}")


def _read_sql(path: str) -> pd.DataFrame:
    db_con = DatabaseConnection()
    con = db_con.get_db_connection()
    query = f"SELECT * FROM {path}"
    df = pd.read_sql(query, con)
    return df


def _detect_cols(df: pd.DataFrame) -> Tuple[str, str]:
    # date column: first with >=80% parseable datetimes
    date_col = None
    for c in df.columns:
        s = pd.to_datetime(df[c], errors="coerce")
        if s.notna().mean() >= 0.80:
            date_col = c;
            break
    if date_col is None:
        date_col = df.columns[0]
    # value column: first numeric-like not equal to date col
    value_col = None
    for c in df.columns:
        if c == date_col: continue
        v = pd.to_numeric(df[c], errors="coerce")
        if v.notna().mean() >= 0.80:
            value_col = c;
            break
    if value_col is None:
        value_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
    return date_col, value_col


def _read_series(stems_or_path, prefer_quarterly: bool = False) -> pd.Series:
    df = _read_sql(stems_or_path[0])
    """
    Read a series from DATA/, detect monthly vs quarterly robustly,
    align to month-end; if quarterly, upsample to month-end via ffill.
    """
    path: Optional[Path]
    if not df.empty:
        path = stems_or_path
    elif isinstance(stems_or_path, str) and Path(stems_or_path).exists():
        path = Path(stems_or_path)
    else:
        stems = stems_or_path if isinstance(stems_or_path, (list, tuple)) else [str(stems_or_path)]
        path = _find_src_by_stems(stems)

    if path is None:
        files = _list_src_files()
        file_list = "\n    ".join(str(p.name) for p in files) if files else "(no files found)"
        raise FileNotFoundError(
            "Missing source file. Tried stems:\n"
            f"  - {stems_or_path}\n"
            f"in directory:\n"
            f"  - {SRC_DIR}\n"
            f"Allowed extensions: {', '.join(_EXTS)}\n"
            f"Files present:\n    {file_list}\n\n"
            "Fix by either:\n"
            "  • Renaming your file to one of the expected stems (see config.yml raw_stems), or\n"
            "  • Adding your actual filename stem under config.yml -> raw_stems, or\n"
            "  • Setting env var DATA_SRC_DIR to the folder with the files."
        )

    df = _read_sql(stems_or_path[0])
    dcol, vcol = _detect_cols(df)

    idx = pd.to_datetime(df[dcol], errors="coerce")
    val = pd.to_numeric(df[vcol], errors="coerce")
    s = pd.Series(val.values, index=idx).dropna().sort_index()
    s = s[s.index.notna()]
    s = s[~s.index.duplicated(keep="last")]

    # Robust monthly vs quarterly detection
    per_m = s.index.to_period("M")
    if len(per_m) >= 6:
        years = per_m.year
        months = per_m.month
        tmp = pd.DataFrame({"year": years, "month": months})
        months_per_year = tmp.groupby("year")["month"].nunique()
        is_quarterly = bool(months_per_year.median() <= 4)
    else:
        is_quarterly = False

    is_quarterly = is_quarterly or prefer_quarterly

    if is_quarterly:
        s.index = s.index.to_period("Q").to_timestamp("Q")
        s = s.asfreq("ME").ffill()  # 'ME' = month-end (fix deprecation)
    else:
        s.index = s.index.to_period("M").to_timestamp("M")
    return s


# ────────────────────────────────────────────────────────────────────────────
# Transform helpers
# ────────────────────────────────────────────────────────────────────────────
def _percent_to_decimal_if_needed(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    if not x.notna().any(): return x
    q95 = x.quantile(0.95)
    return x / 100.0 if q95 > 1.5 else x


def _safe_log(level: pd.Series) -> pd.Series:
    x = pd.to_numeric(level, errors="coerce").clip(lower=1e-6).astype(float)
    return np.log(x)


def _diff(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype(float).diff()


def _logdiff(level: pd.Series) -> pd.Series:
    x = pd.to_numeric(level, errors="coerce").clip(lower=1e-6).astype(float)
    return np.log(x).diff()


def _atomic_write_csv(df: pd.DataFrame, path: Path):
    tmp = path.with_name(f"{path.stem}.__tmp__{path.suffix}")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def _append_csv_atomic(rows: pd.DataFrame, path: Path, pk_cols: Optional[list[str]] = None):
    """
    Append rows to CSV 'path'. If exists and pk_cols provided, read, concat, drop dups on pk, then write atomically.
    """
    rows = rows.copy()
    if path.exists() and pk_cols:
        old = pd.read_csv(path)
        all_df = pd.concat([old, rows], axis=0, ignore_index=True)
        all_df = all_df.drop_duplicates(subset=pk_cols, keep="last")
        _atomic_write_csv(all_df, path)
    elif path.exists() and not pk_cols:
        # Fast append (no de-dup)
        rows.to_csv(path, mode="a", header=False, index=False)
    else:
        _atomic_write_csv(rows, path)


# ────────────────────────────────────────────────────────────────────────────
# GDP revision auditing (→ CSV)
# ────────────────────────────────────────────────────────────────────────────
def _audit_gdp_revisions(old: pd.Series, new: pd.Series, as_of: str, lookback_months: int) -> int:
    if old is None or old.empty or new is None or new.empty:
        return 0
    max_date = new.index.max()
    start_lb = (max_date.to_period("M") - lookback_months + 1).to_timestamp("M")
    idx = old.index.intersection(new.index)
    idx = idx[idx >= start_lb]
    if idx.empty:
        return 0

    merged = pd.DataFrame({"old": old.reindex(idx), "new": new.reindex(idx)})
    mask = np.isfinite(merged["old"].values) & np.isfinite(merged["new"].values)
    merged = merged[mask]
    if merged.empty:
        return 0

    eps = 1e-9
    neq_mask = ~np.isclose(merged["old"].to_numpy(), merged["new"].to_numpy(), rtol=0.0, atol=eps)
    changed = merged[neq_mask]
    if changed.empty:
        return 0

    logs = []
    for dt, row in changed.iterrows():
        old_v = float(row["old"])
        new_v = float(row["new"])
        abs_delta = new_v - old_v
        pct_delta = (abs_delta / old_v * 100.0) if old_v != 0 else np.nan
        logs.append({
            "as_of": as_of,
            "date": pd.Timestamp(dt).to_period("M").to_timestamp("M").date(),
            "old_gdp": old_v,
            "new_gdp": new_v,
            "abs_change": abs_delta,
            "pct_change": pct_delta,
            "lookback_months": lookback_months,
        })
    if logs:
        df_logs = pd.DataFrame(logs)
        _append_csv_atomic(df_logs, GDP_REV_CSV, pk_cols=["as_of", "date"])

        conn = db_con.get_db_connection()

        df_logs.to_sql('t_gdp_revisions', con=conn, if_exists='append', index=False)

    return len(logs)


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────
def main():
    # print(f"SOURCE dir: {SRC_DIR}")
    conn = db_con.get_db_connection()

    # Build stem lists (config-overrideable, with defaults)
    stems_NPL = _candidate_stems("NPL", ["NPL_raw", "NPL"])
    stems_DEGU = _candidate_stems("DEGU", ["DEGU_raw", "USDGHS", "DEGU"])
    stems_CBLR = _candidate_stems("CBLR", ["CBLR_raw", "CBLR"])
    stems_GLA = _candidate_stems("GLA", ["GLA_raw", "GLA"])
    stems_GDP = _candidate_stems("GDP", ["GDP_raw", "DGP_raw", "GDP", "DGP"])

    # print(f"Using source stems:{stems_NPL}, {stems_DEGU}, {stems_CBLR}, {stems_GLA}, {stems_GDP}")

    # ['NPL_raw', 'NPL'], ['DEGU_raw', 'USDGHS', 'DEGU'], ['CBLR_raw', 'CBLR'], ['GLA_raw', 'GLA'], ['GDP_raw', 'DGP_raw', 'GDP', 'DGP']
    # Read source series
    NPL_level = _read_series(stems_NPL, prefer_quarterly=False)
    DEGU_lvl = _read_series(stems_DEGU, prefer_quarterly=False)  # USD/GHS FX
    CBLR_lvl = _read_series(stems_CBLR, prefer_quarterly=False)
    GLA_lvl = _read_series(stems_GLA, prefer_quarterly=False)
    GDP_level = _read_series(stems_GDP, prefer_quarterly=True)

    # Harmonize units
    NPL_level = _percent_to_decimal_if_needed(NPL_level)  # must be decimal 0..1
    CBLR_lvl = _percent_to_decimal_if_needed(CBLR_lvl)
    DEGU_lvl = _percent_to_decimal_if_needed(DEGU_lvl)
    GLA_lvl = _percent_to_decimal_if_needed(GLA_lvl)
    # GDP stays as numeric level

    # Final transforms (aligned with modeling stack)
    NPL_log = _safe_log(NPL_level);
    NPL_log.name = "NPL"  # target: ln(NPL_decimal)
    GDP_f = pd.to_numeric(GDP_level, errors="coerce").astype(float);
    GDP_f.name = "GDP"
    DEGU_f = _logdiff(DEGU_lvl);
    DEGU_f.name = "DEGU"  # FX log return (stationary)
    CBLR_f = _diff(CBLR_lvl);
    CBLR_f.name = "CBLR"  # diff(level)
    GLA_f = _diff(GLA_lvl);
    GLA_f.name = "GLA"  # diff(level)

    # New panel
    new_panel = pd.concat([NPL_log, GDP_f, DEGU_f, CBLR_f, GLA_f], axis=1) \
        .sort_index() \
        .dropna(how="all")
    new_panel.index = new_panel.index.to_period("M").to_timestamp("M")

    # Load existing panel (if present)
    if db_con.check_db_table_exists(PANEL_CSV):
        # old_panel = pd.read_csv(PANEL_CSV, parse_dates=["date"])
        old_panel = pd.read_sql('t_panel', con=conn, parse_dates=["date"])
        old_panel = old_panel.set_index("date").sort_index()
    else:
        old_panel = pd.DataFrame(columns=["NPL", "GDP", "DEGU", "CBLR", "GLA"])

    # GDP revision audit (in lookback window)
    as_of = datetime.now(UTC).isoformat(timespec="seconds").replace(":", "-") + "Z"
    n_gdp_revisions = 0
    if not old_panel.empty and "GDP" in old_panel.columns:
        old_gdp = old_panel["GDP"]
        old_gdp.index = pd.to_datetime(old_gdp.index).to_period("M").to_timestamp("M")
        n_gdp_revisions = _audit_gdp_revisions(
            old=old_gdp, new=new_panel["GDP"], as_of=as_of, lookback_months=GDP_REV_LOOKBACK_M
        )

    # Merge/upsert: union of dates; overwrite overlaps with NEW values
    combined = old_panel.combine_first(new_panel)
    combined.update(new_panel)

    # Ensure canonical column order and clean index
    for col in ["NPL", "GDP", "DEGU", "CBLR", "GLA"]:
        if col not in combined.columns:
            combined[col] = np.nan
    combined = combined[["NPL", "GDP", "DEGU", "CBLR", "GLA"]].sort_index()
    combined.index.name = "date"

    # Truncate up to the last date where ALL key variables have non-missing values
    _KEYS = ["NPL", "CBLR", "GLA", "DEGU", "GDP"]
    if all(k in combined.columns for k in _KEYS):
        _mask_full = combined[_KEYS].notna().all(axis=1)
        if _mask_full.any():
            _last_full_date = combined.index[_mask_full][-1]
            combined = combined.loc[:_last_full_date]
    # End truncation guard

    # (4) Monthly gap check (informational only; does not change output)
    if len(combined) > 1:
        full_idx = pd.date_range(
            start=combined.index.min().to_period("M").to_timestamp("M"),
            end=combined.index.max().to_period("M").to_timestamp("M"),
            freq="ME",
            inclusive="both",
        )
        missing = full_idx.difference(combined.index)
        if len(missing) > 0:
            print(
                f"⚠️  Gap check: {len(missing)} missing month(s) in panel:",
                ", ".join(d.strftime("%Y-%m") for d in missing[:12]),
                "..." if len(missing) > 12 else "",
            )

    # (3) Enforce numeric dtypes before writing
    for col in ["NPL", "GDP", "DEGU", "CBLR", "GLA"]:
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col], errors="coerce").astype("float64")

    # ── Write per-series CSVs (t_*.csv) ─────────────────────────────────────
    for name, path in SERIES_FILES.items():
        s = combined[[name]].dropna().reset_index()
        s.columns = ["date", name]
        _atomic_write_csv(s, path)

    # ── Write combined panel CSV (t_panel.csv) ──────────────────────────────
    _atomic_write_csv(combined.reset_index(), PANEL_CSV)

    combined.reset_index().to_sql('t_panel', con=conn, if_exists='replace', index=False)

    # Summary
    print(f"✅ Wrote/updated panel -> {PANEL_CSV}")
    if len(combined):
        print(f"   Rows: {len(combined):,} | Range: {combined.index.min().date()} … {combined.index.max().date()}")
    print("   Columns:", ", ".join(combined.columns))
    if n_gdp_revisions:
        print(f"   GDP revisions logged (last {GDP_REV_LOOKBACK_M} months): {n_gdp_revisions} → {GDP_REV_CSV.name}")

    # Append operational run report (CSV append with de-dup on run_ts)
    run_row = pd.DataFrame([{
        "run_ts": as_of,
        "src_dir": str(SRC_DIR),
        "rows_panel": int(len(combined)),
        "start": combined.index.min().date() if len(combined) else None,
        "end": combined.index.max().date() if len(combined) else None,
        "gdp_revisions_logged": int(n_gdp_revisions),
        "gdp_revision_lookback_m": int(GDP_REV_LOOKBACK_M),
        "sources": f"NPL={stems_NPL} | DEGU={stems_DEGU} | CBLR={stems_CBLR} | GLA={stems_GLA} | GDP={stems_GDP}",
        "policy": "NPL=log(level decimal); GDP=level; DEGU=logdiff; CBLR=diff; GLA=diff",
    }])
    _append_csv_atomic(run_row, RUN_REP_CSV, pk_cols=["run_ts"])

    run_row.reset_index().to_sql('t_transformation_report', con=conn, if_exists='append', index=False)

    # Sanity hint for NPL (force numeric before exp to avoid dtype issues)
    if "NPL" in combined.columns and combined["NPL"].notna().any():
        npl_numeric = pd.to_numeric(combined["NPL"], errors="coerce")
        probe = np.exp(npl_numeric.dropna().astype(float).values)
        if not ((probe > 0).all() and (probe <= 1.5).all()):
            print("⚠️  'NPL' may not look like ln(NPL_decimal). "
                  "Check raw NPL units (percent vs decimal).")


if __name__ == "__main__":
    main()
