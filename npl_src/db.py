# src/npl_dq/db.py
from __future__ import annotations
from pathlib import Path
import yaml
from sqlalchemy import create_engine, inspect
import urllib
import pandas as pd
import datetime

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "config.yml").read_text())


class DatabaseConnection:
    def __init__(self, db_config=None):
        if db_config is None:
            db_config = CFG["database"]
        self.db_config = db_config

    def get_db_connection(self):
        trust_value = self.db_config.get('trust_server_certificate', False)
        trust_str = "yes" if trust_value else "no"  # Convert boolean to "yes" or "no"
        conn_str = (
            f"DRIVER={{{self.db_config['driver']}}};"
            f"SERVER={self.db_config['server']};"
            f"DATABASE={self.db_config['database']};"
            f"UID={self.db_config['username']};"
            f"PWD={self.db_config['password']};"
            f"TrustServerCertificate={trust_str}"
        )
        # Create a connection string
        params = urllib.parse.quote_plus(
            f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={self.db_config['server']};DATABASE={self.db_config['database']};UID={self.db_config['username']};PWD={self.db_config['password']};TrustServerCertificate=yes;")
        connection_string = f'mssql+pyodbc:///?odbc_connect={params}'
        # Create a database engine
        return create_engine(connection_string)

    def check_db_table_exists(self, path: Path) -> bool:
        table_name = path.stem
        conn = self.get_db_connection()
        results = inspect(conn).get_table_names()
        exists = table_name in results
        return exists

    def _append_sql(self, df: pd.DataFrame, path: Path):
        conn = self.get_db_connection()
        table_name = path.stem  # Use the file name (without extension) as the table name
        df['train_date'] = datetime.datetime.now()
        df.to_sql(table_name, conn, if_exists="append", index=False)

    def _read_sql(self, path: Path):
        conn = self.get_db_connection()
        table_name = path.stem  # Use the file name (without extension) as the table name
        return pd.read_sql(f"SELECT * FROM {table_name}", conn)

    def _read_sql_latest_date(self, path: Path):
        conn = self.get_db_connection()
        table_name = path.stem

        if table_name == "t_transformation_report":
            sql = f'''SELECT * FROM [{table_name}]
            WHERE "run_ts" = (SELECT MAX("run_ts") FROM [{table_name}])
            '''
        elif table_name == "t_npl_forecasts":
            sql = f'''SELECT * FROM [{table_name}]
            WHERE "run_ts_utc" = (SELECT MAX("run_ts_utc") FROM [{table_name}])
            '''

        elif table_name == "NPL_raw":
            sql = f'''
                SELECT * FROM [{table_name}]
                WHERE [DATE] = (SELECT MAX([DATE]) FROM [GDP_raw])
            '''
        else:
            sql = f'''SELECT * FROM [{table_name}]
            WHERE "train_date" = (SELECT MAX("train_date") FROM [{table_name}])
            '''
        return pd.read_sql(sql, conn)