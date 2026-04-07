# src/npl_dq/cli.py
# -*- coding: utf-8 -*-
"""
NPL DQ CLI — robust launcher for the NPL data-quality framework.

Works in BOTH modes:
  1) Module:  python -m npl_dq.cli --data-dir "./data" --out "./reports/NPL_DQ_Report.xlsx"
  2) Script:  python src/npl_dq/cli.py --data-dir "./data" --out "./reports/NPL_DQ_Report.xlsx"
"""

from __future__ import annotations
from pathlib import Path
import yaml
from npl_src.db import DatabaseConnection  # Ensure db.py is importable


THIS_FILE = Path(__file__).resolve()          # .../src/npl_dq/cli.py
SRC_DIR = THIS_FILE.parents[1]                # .../src
PROJECT_ROOT = THIS_FILE.parents[2]           # .../project root
PKG_DIR = SRC_DIR / "npl_dq" 
CFG = yaml.safe_load((PROJECT_ROOT / "config.yml").read_text())
db_conn = DatabaseConnection()#new_14/10/25  
con = db_conn.get_db_connection()#new_14/10/25

# # Ensure ./src is importable when run as a script
# if str(SRC_DIR) not in sys.path:
#     sys.path.insert(0, str(SRC_DIR))

# # Ensure package marker exists (non-fatal if it can't be created)
# INIT_FILE = PKG_DIR / "__init__.py"
# try:
#     if not INIT_FILE.exists():
#         INIT_FILE.touch()
# except Exception as _e:
#     print(f"⚠️  Could not create {INIT_FILE}: {_e}", file=sys.stderr)


# # Import framework entrypoint robustly
# try:
#     from .framework import main as run_framework
# except Exception:
#     from npl_dq.framework import main as run_framework  # type: ignore

# def parse_args() -> argparse.Namespace:
#     p = argparse.ArgumentParser(
#         prog="npl_dq.cli",
#         description="Run the NPL Data Quality framework and produce a database table."#new_14/10/25
#     )
#     p.add_argument(
#         "--data-dir",
#         default=str((PROJECT_ROOT / "data").resolve()),
#         help="Folder containing the CSVs (default: ./data under project root)",
#     )
#     p.add_argument(
#         "--out",
#         default=str((PROJECT_ROOT / "reports" / "NPL_DQ_Report.xlsx").resolve()),
#         help="Output Excel path (default: ./reports/NPL_DQ_Report.xlsx under project root)",
#     )
#     p.add_argument(
#         "--csv-sidecars",
#         action="store_true",
#         help="Also write CSV copies of key sheets next to the XLSX alongside database output.",#new_14/10/25
#     )
#     p.add_argument(
#         "--quiet",
#         action="store_true",
#         help="Reduce console output.",
#     )
#     return p.parse_args()


# def ensure_parent_dir(path_str: str) -> None:
#     Path(path_str).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)

# def patch_framework_constants(data_dir: str, out_path: str, csv_sidecars: bool) -> None:
#     """
#     Patch framework globals before running.
#     """
#     import npl_dq.framework as fw
#     fw.DATA_DIR = str(Path(data_dir).expanduser().resolve())
#     fw.OUT_XLSX = str(Path(out_path).expanduser().resolve())
#     fw.CSV_SIDECARS = bool(csv_sidecars)

# def print_startup_info(data_dir: str, out_path: str, csv_sidecars: bool) -> None:
#     print("------------------------------------------------------------")
#     print("NPL DQ — launch details")
#     print(f" Project root : {PROJECT_ROOT}")
#     print(f" Source dir   : {SRC_DIR}")
#     print(f" Package dir  : {PKG_DIR}")
#     print(f" Data dir     : {Path(data_dir).resolve()}")
#     print(f" Output table : NPL_DQ_Report")#new_14/10/25
#     print(f" CSV sidecars : {csv_sidecars}")
#     print(f"Database      : {CFG["database"]["database"]} on {CFG['database']['server']}")#new_14/10/25
#     print(" Required CSVs: NPL_raw, GLA_raw, CBLR_raw, DEGU_raw, GDP_raw")#new_14/10/25
#     print("------------------------------------------------------------")



# def main() -> None:
#     args = parse_args()

#     data_dir_abs = str(Path(args.data_dir).expanduser().resolve())
#     out_abs = str(Path(args.out).expanduser().resolve())

#     if not args.quiet:
#         print_startup_info(data_dir_abs, out_abs, args.csv_sidecars)

#     ensure_parent_dir(out_abs)
#     patch_framework_constants(data_dir_abs, out_abs, args.csv_sidecars)

#     # Soft preflight (warn only)
#     required = ["NPL_raw.csv", "GLA_raw.csv", "CBLR_raw.csv", "DEGU_raw.csv", "GDP_raw.csv"]
#     missing = [f for f in required if not (Path(data_dir_abs) / f).exists()]
#     if missing and not args.quiet:
#         print(f"⚠️  Missing in data dir: {', '.join(missing)}")

#     run_framework()

# if __name__ == "__main__":
#     main()

print(PROJECT_ROOT)

