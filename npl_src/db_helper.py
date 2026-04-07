import pyodbc
import pandas as pd
import yaml
from pathlib import Path

# Load config.yml (same as other parts of your project)
ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "config.yml").read_text())
DB_CONFIG = CFG["database"]

def get_db_connection(db_config):
    trust_value = db_config.get("trust_server_certificate", False)
    trust_str = "yes" if trust_value else "no"

    conn_str = (
        f"DRIVER={db_config['driver']};"
        f"SERVER={db_config['server']},1433;"
        f"DATABASE={db_config['database']};"
        f"UID={db_config['username']};"
        f"PWD={db_config['password']};"
        f"Encrypt=no;"
        f"TrustServerCertificate={trust_str};"
        f"Connection Timeout=30;"
    )

    return pyodbc.connect(conn_str)