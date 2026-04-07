from pathlib import Path
from itertools import product
from npl_src.db import DatabaseConnection


class extract():
    def __init__(self):
        self.db_con = DatabaseConnection()

        self.table_names = ["CBLR", "DEGU", "GDP", "NPL", "GLA", "cv_metrics",
                            "cv_predictions", "feature_importance", "forecast_path",
                            "actual_vs_pred", "backtest_heatmap_abs_error_pp",
                            "champion_by_horizon", "drift", "gates_summary", "kpis_monthly",
                            "model_comparison", "residuals", "t_transformation_report",
                            "train_run_log", "t_gdp_revisions", "NPL_raw", "t_npl"]

        self.table_types = ["issues", "modelling_notes", "outliers", "scorecard", "summary",
                            "validity", "h0", "h1", "h2", "metrics_latest", "table_latest","forecasts","train_run_log"]

        self.table_dict = {}
        self.not_av_set = set()  # use a set to avoid duplicates


    def db_t(self):

        for tbl, tbl_type in product(self.table_names, self.table_types):
            primary = f"{tbl}_{tbl_type}"  # e.g., CBLR_h0
            primary_path = Path(f"{primary}.csv")

            # 1) Try the primary table
            try:
                df_primary = self.db_con._read_sql_latest_date(primary_path)
                # Save primary under its own key; this key is unique per combo
                self.table_dict[primary] = df_primary
                continue  # done with this combo
            except Exception:
                # Primary not found -> record it
                self.not_av_set.add(primary)

                # 2) Try the base table; save it ONLY IF it exists and hasn't been saved yet
                if tbl not in self.table_dict:
                    try:
                        df_base = self.db_con._read_sql_latest_date(Path(f"{tbl}.csv"))
                        self.table_dict[tbl] = df_base  # saved once total
                    except Exception:
                        # Base also not found -> record it
                        self.not_av_set.add(tbl)
                # If base is already saved, do nothing (no duplicate save)

        # Convert set to list if you need a list
        not_av = sorted(self.not_av_set)

        return self.table_dict