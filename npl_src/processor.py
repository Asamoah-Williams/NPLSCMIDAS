"""
processor.py
End-to-end orchestration for the Industry NPL workflow.

Stages
  1) Data Quality (framework.py)        [optional but recommended]
  2) Data Transformation (data_transformation.py) -> <data_folder>/t_panel.csv
  3) Training & CV (pipeline_V1.py)        -> models + CV reports
  4) Backtest reporting (optional)      -> heatmap + CSVs
  5) Production Monitoring (monitoring.py, Option A) -> KPIs, drift, gates

This variant:
- uses a multi-document YAML loader (handles '...' in your config.yml),
- respects config paths: data_folder, models_folder, reports_folder,
- prefers direct function calls, with fallback to "python -m <module>".
"""

from __future__ import annotations
import sys
import subprocess
import importlib
import inspect
import traceback
from pathlib import Path
from datetime import datetime, UTC, timezone


import pandas as pd
import yaml

#from build.nsis.pkgs.sympy.physics.units import microsecond
from npl_src.db import DatabaseConnection

# ---------- Reporting hooks (optional backtest summary) -----------------------
from npl_src.kpi_reporter import RunStamp, log_backtest
from npl_src.report_paths import ReportRoot

# === Post-processing & safe entrypoint shim (src-aware) =======================
from pathlib import Path as _P
import sys as _sys
import subprocess as _subprocess
import os as _os

# ---------- Repo roots & import path ------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))  # ensure "src" modules import cleanly


class processor:
    def __init__(self):
        self.db_con = DatabaseConnection()

    # ===== Utilities ==============================================================

    def _log(self, msg: str) -> None:
        ts = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{ts} UTC] {msg}", flush=True)

    def load_config(self, cfg_path: Path) -> dict:
        """
        Multi-document YAML loader (handles '...' document terminator).
        Later documents override earlier keys. Also sets safe defaults.
        """
        docs = list(yaml.safe_load_all(cfg_path.read_text()))
        cfg = {}
        for d in docs:
            if isinstance(d, dict):
                cfg.update(d)

        # Defaults referenced downstream
        cfg.setdefault("data_folder", "data")
        cfg.setdefault("models_folder", "data/interim/models")
        cfg.setdefault("reports_folder", "reports")
        cfg.setdefault("horizons", [0, 1, 2, 3, 4, 5, 6])
        cfg.setdefault("model_version", "v0.0.0")
        cfg.setdefault("cfg_hash", "NA")

        # Monitoring defaults (if a later doc didn't specify)
        mon = cfg.setdefault("monitoring", {})
        mon.setdefault("thresholds", {
            "rmse_pp_max": 3.0, "mae_pp_max": 2.5, "smape_max": 30.0, "mase_max": 1.0, "r2_log_min": 0.20
        })
        mon.setdefault("gate_window", 12)

        return cfg

    def _import_module(self, modname: str):
        try:
            return importlib.import_module(modname)
        except Exception as e:
            self._log(f"ERROR importing {modname}: {e}")
            return None

    def _call_first_available(self, mod, candidates: list[str], *args, **kwargs) -> bool:
        """
        Try calling the first function in `mod` from `candidates`.
        Returns True if a function was successfully called, False otherwise.
        """
        for fn_name in candidates:
            fn = getattr(mod, fn_name, None)
            if fn is None or not callable(fn):
                continue
            self._log(f"Calling {mod.__name__}.{fn_name}()")
            try:
                sig = inspect.signature(fn)
                if len(sig.parameters) == 0:
                    fn()
                    return True
                # If the function takes a 'cfg' argument, pass cfg; else call without kwargs
                if "cfg" in [p.name for p in sig.parameters.values()] and "cfg" in kwargs:
                    fn(kwargs["cfg"])
                else:
                    fn()
                return True
            except Exception:
                traceback.print_exc()
                raise
        return False

    def _run_module(self, modname: str, required: bool, candidates: list[str], *args, **kwargs) -> None:
        """
        Prefer calling a function in the module (from candidates).
        Fall back to running 'python -m modname'.
        """
        self._log(f"--- Stage start: {modname} ---")
       #  modulename = "npl_src."+modname
        mod = self._import_module(modname)
        if mod:
            try:
                called = self._call_first_available(mod, candidates, *args, **kwargs)
                if called:
                    self._log(f"--- Stage complete (function): {modname} ---")
                    return
            except Exception as e:
                self._log(f"ERROR in {modname} via direct function call: {e}")
                if required:
                    raise

        # Fallback to module execution
        self._log(f"Falling back to 'python -m npl_src.{modname}'")
        try:
            modulename = "npl_src."+modname
            subprocess.run([sys.executable, "-m", modulename], cwd=str(ROOT), check=True)
            self._log(f"--- Stage complete (-m): {modname} ---")
        except subprocess.CalledProcessError as e:
            self._log(f"ERROR running {modname} as a module: {e}")
            if required:
                raise

    def _maybe_backtest_report(self, cfg: dict, stamp: RunStamp, reports_root: ReportRoot) -> None:
        """
        If a backtest dataset exists, publish the all-horizons heatmap and CSVs.
        Optional; skipped if nothing is found.
        """
        candidates = [
            ROOT / cfg["data_folder"] / "backtest_long.csv",
            ROOT / cfg["data_folder"] / "backtest_long.parquet",
            ROOT / cfg["reports_folder"] / "_data" / "backtest" / "errors_by_h_month.csv",
        ]
        path = next((p for p in candidates if p.exists()), None)
        if path is None:
            self._log("No backtest dataset found; skipping backtest reporting.")
            return

        self._log(f"Backtest dataset detected: {path}")
        if path.suffix == ".parquet":
            df_bt = pd.read_parquet(path)
        else:
            # Try to parse date columns if present
            preview_cols = self.db_con._read_sql(path).columns.tolist()
            # preview_cols = pd.read_csv(path, nrows=0).columns.tolist()
            date_cols = [c for c in ("as_of_date", "target_month") if c in preview_cols]
            df_bt = self.db_con._read_sql(path) if date_cols else self.db_con._read_sql(path)
            # df_bt = pd.read_csv(path, parse_dates=date_cols) if date_cols else pd.read_csv(path)

        required = {"as_of_date", "target_month", "horizon", "y_true_pct", "y_pred_pct"}
        if not required.issubset(df_bt.columns):
            self._log(f"Backtest file missing columns: {required - set(df_bt.columns)}; skipping.")
            return

        # Build a naive baseline if missing: carry-forward within each horizon
        if "y_naive_pct" not in df_bt.columns:
            self._log("y_naive_pct not present—constructing naive baseline (carry-forward).")
            df_bt = df_bt.sort_values(["horizon", "target_month"]).copy()
            df_bt["y_naive_pct"] = df_bt.groupby("horizon")["y_true_pct"].shift(1)

        log_backtest(reports_root, df_bt, stamp)
        self._log("Backtest reporting complete.")

    # ===== Orchestration ==========================================================

    def run_main(self):
        # save start date and time in a variable
        train_start = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

        self._log("Processor started.")
        cfg_path = ROOT / "config.yml"
        if not cfg_path.exists():
            raise FileNotFoundError(f"config.yml not found at {cfg_path}")
        cfg = self.load_config(cfg_path)

        reports_root = ReportRoot.from_reports_root(ROOT / cfg["reports_folder"])
        stamp = RunStamp(
            as_of_date=pd.Timestamp(datetime.now(UTC).date()),
            model_version=cfg.get("model_version", "v0.0.0"),
            cfg_hash=cfg.get("cfg_hash", "NA"),
            code_commit=cfg.get("code_commit"),
        )


        # 1) Data Quality (framework.py) — optional but recommended
        self._run_module(
            modname="framework",
            required=False,
            candidates=["run_dq", "run_checks", "dq_main", "main"],
            cfg=cfg
        )

        # 2) Transformation (data_transformation.py) — required
        self._run_module(
            modname="data_transformation",
            required=True,
            candidates=["transform", "run", "build_panel", "main"],
            cfg=cfg
        )

        # 3) Training & CV (pipeline_V1.py) — required
        self._run_module(
            modname="pipeline",
            required=True,
            candidates=["train_all_horizons", "train", "run", "main"],
            cfg=cfg
        )

        # 4) Optional: backtest heatmap + CSVs
        self._maybe_backtest_report(cfg, stamp, reports_root)

        # 5) Monitoring (monitoring.py) — required
        self._run_module(
            modname="monitoring",
            required=True,
            candidates=["run", "monitor", "main"],
            cfg=cfg
        )
        self._log("Processor finished successfully.")

        # log training completion - insert into the database
        train_end =  datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        train_log_rows = []
        train_log_rows.append({"train_start":train_start,"train_end":train_end})
        df = pd.DataFrame(train_log_rows)
        self.db_con._append_sql(df,Path(cfg.get("train_run_log_pro", "t_npl_train_run_log")))


        try:
            _here = self._P(__file__).resolve()
            _root = _here.parents[1]
            _cfg = yaml.safe_load((_root / "config.yml").read_text()) or {}
        except Exception:
            _cfg = {}
        self._run_post_processing(_cfg)

    def _safe_run_py(self, script_path: _P, args=None, required=True, env=None):
        if args is None:
            args = []
        if not script_path.exists():
            msg = f"[processor] Skipping: {script_path} not found."
            print(msg)
            if required:
                raise FileNotFoundError(msg)
            return 0
        cmd = [_sys.executable, str(script_path)] + list(args)
        print(f"[processor] ▶ Running: {' '.join(cmd)}")
        rc = _subprocess.call(cmd, env=env or _os.environ.copy())
        if rc != 0:
            msg = f"[processor] ✖ {script_path.name} exited with code {rc}"
            print(msg)
            if required:
                raise SystemExit(rc)
        else:
            print(f"[processor] ✓ {script_path.name} completed.")
        return rc

    def _run_post_processing(self, _cfg: dict):
        here = _P(__file__).resolve()
        scripts_dir = here.parent  # src/
        project_root = scripts_dir.parent

        def _pick(*cands):
            for c in cands:
                if c.exists():
                    return c
            return cands[0]

        drift = _pick(scripts_dir / "drift.py", project_root / "drift.py")
        validator = _pick(scripts_dir / "NPL_Transformation_Validation.py",
                          project_root / "NPL_Transformation_Validation.py")
        notifier = _pick(scripts_dir / "notifier.py", project_root / "notifier.py")

        # Build validator args from config (best-effort)
        val_args = []
        try:
            import yaml as _yaml
            cfg_file = project_root / "config.yml"
            _cfg = _yaml.safe_load(cfg_file.read_text()) or {}
        except Exception:
            pass
        if isinstance(_cfg, dict):
            if _cfg.get("dq_thresholds"):
                val_args.append("--strict")
            if _cfg.get("validator_adf", True):
                val_args.append("--adf")
            if _cfg.get("verbose", True):
                val_args.append("--verbose")

        # Run scripts
        self._safe_run_py(drift, required=False)
        self._safe_run_py(validator, args=val_args, required=True)
        self._safe_run_py(notifier, required=False)

    # Ensure a main() symbol exists even if earlier code renamed it.
    # if "main" not in globals():
    #     def main(self):
    #         for fn in ("_original_main", "run", "run_pipeline", "execute", "start"):
    #             _f = globals().get(fn)
    #             if callable(_f):
    #                 return _f()
    #         raise RuntimeError("No entrypoint found (expected main/_original_main/run).")
    #
    # def __entry_main(self):
    #     rv = self.main()
    #     try:
    #         import yaml as _yaml
    #         _here = self._P(__file__).resolve()
    #         _root = _here.parents[1]
    #         _cfg = _yaml.safe_load((_root / "config.yml").read_text()) or {}
    #     except Exception:
    #         _cfg = {}
    #     self._run_post_processing(_cfg)
    #     return rv

    # if __name__ == "__main__":
    #     __entry_main()


# obj = processor()
# obj.run_main()