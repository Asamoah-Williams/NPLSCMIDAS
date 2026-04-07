import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from datetime import datetime
from pathlib import Path
import yaml, json
import pandas as pd
import pyodbc
import re
from sqlalchemy import text
import time
from npl_src.npl_dq.db import DatabaseConnection
# Load configs
ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "config.yml").read_text())
with open("config.json", "r") as f:
    config = json.load(f)
GURL = config["GURL"]
HEADERS = config["HEADERS"]
DATA_FOLDER = Path(ROOT / CFG["data_folder"])
DATA_FOLDER.mkdir(parents=True, exist_ok=True)


def download_latest_qgdpe_excel():
    resp = requests.get(GURL, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    # Locate the QGDPE table
    table = soup.find("table", {"id": "myTable"})
    if not table:
        raise ValueError("Could not find QGDPE table on page")

    # First row in tbody
    first_row = table.find("tbody").find("tr")
    cols = first_row.find_all("td")

    # File title (e.g. QGDPE_Expenditure Approach_June 2025_GSS)
    excel_title = cols[0].get_text(strip=True)

    # File link
    a_tag = first_row.find("a", href=True)
    excel_link = urljoin(GURL, a_tag["href"])

    if not excel_link:
        raise ValueError("No valid QGDPE Excel link found in first row.")

    # --- Check if already downloaded ---
    safe_title = excel_title.replace(" ", "_").replace("–", "-")
    existing_files = list(DATA_FOLDER.glob(f"{safe_title}*.xlsx"))
    if existing_files:
        print(f"Already downloaded: {excel_title}")
        return existing_files[0]

    # Download Excel
    resp_file = requests.get(excel_link, headers=HEADERS, timeout=60)
    resp_file.raise_for_status()

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = DATA_FOLDER / f"{safe_title}_{timestamp}.xlsx"
    with open(filename, "wb") as f:
        f.write(resp_file.content)

    print(f"Downloaded: {excel_title}")
    print(f"Saved to: {filename}")
    return filename


def clean_quarter_string(q_str):
    if not isinstance(q_str, str):
        return None

    q_str = q_str.replace("*", "").strip()

    q_str = re.sub(r"(\d{4})(Q\d)", r"\1 \2", q_str)
    parts = q_str.split()
    if len(parts) != 2:
        return None
    year, q = parts
    if q not in ["Q1", "Q2", "Q3", "Q4"]:
        return None
    return f"{year} {q}"


# Mapping from quarter → last day of that quarter
q_map = {"Q1": "03-31", "Q2": "06-30", "Q3": "09-30", "Q4": "12-31"}


def quarter_to_date_safe(q_str):
    year, q = q_str.split()
    return pd.to_datetime(f"{year}-{q_map[q]}")


def clean_qgdpe(file_path: str, last_n: int = 3, include_update: bool = True):
    xl = pd.ExcelFile(file_path)

    # Find correct sheet
    target_sheets = [s for s in xl.sheet_names if "qgdp" in s.lower() and "growth" in s.lower()]
    if not target_sheets:
        raise ValueError(f"❌ Could not find a sheet with 'QGDP' and 'Growth' in its name. Found: {xl.sheet_names}")

    target_sheet = target_sheets[0]

    # Load sheet
    df = pd.read_excel(file_path, sheet_name=target_sheet, skiprows=3, engine="openpyxl")

    # Quarter column is always the second one
    quarter_col = df.columns[1]

    # GDP column: must exactly match "Gross Domestic Expenditure - Open Economy"
    gdp_col = "Gross Domestic Expenditure - Open Economy"
    if gdp_col not in df.columns:
        raise ValueError(f"❌ Could not find '{gdp_col}' column in sheet. Available: {df.columns.tolist()}")

    # Clean quarter strings
    df["Quarter_cleaned"] = df[quarter_col].apply(clean_quarter_string)

    # Build cleaned dataframe
    cleaned = pd.DataFrame({
        "DATE": df["Quarter_cleaned"].dropna().apply(quarter_to_date_safe).dt.date,
        "GDP": pd.to_numeric(df[gdp_col], errors="coerce")
    }).dropna()

    # Keep only the last few quarters
    if include_update:
        cleaned = cleaned.tail(last_n + 1)  # e.g. 4 rows
    else:
        cleaned = cleaned.tail(last_n)  # e.g. 3 rows

    cleaned = cleaned.reset_index(drop=True)

    print(cleaned)

    return cleaned


class GDPDatabaseWriter:
    def __init__(self, db_config, table_name="GDP_raw"):
        self.db_config = db_config
        self.table_name = table_name
        self.last_run_date = None

        self.db = DatabaseConnection()

    def get_last_db_date(self):
        """Return most recent DATE in GDP table"""
        query = text(f"SELECT MAX(DATE) FROM {self.table_name}")
        engine = self.db.get_db_connection()
        with engine.connect() as con:
            result = con.execute(query)
            return result.scalar()

    def get_gdp_by_date(self, date_value):
        query = text(f"SELECT GDP FROM {self.table_name} WHERE DATE = :date_value")
        engine = self.db.get_db_connection()
        with engine.connect() as con:
            result = con.execute(query, {"date_value": date_value})
            row = result.fetchone()
            return float(row[0]) if row else None

    def write_gdp_to_db(self, df):
        """
        Insert *new* GDP row (latest quarter) and update *last two quarters* if revised.
        df: pandas DataFrame with columns [DATE, GDP]
        """
        if df.empty:
            print("⚠️ No GDP data to insert or update.")
            return

        df["DATE"] = pd.to_datetime(df["DATE"]).dt.date
        engine = self.db.get_db_connection()

        with engine.begin() as con:  # begin() handles transaction & commit
            # 1) Update last 2 quarters if revised
            last_two = df.tail(2)
            for _, row in last_two.iterrows():
                date_val = row["DATE"]
                gdp_value = float(row["GDP"])

                select_query = text(f"SELECT GDP FROM {self.table_name} WHERE DATE = :date_val")
                result = con.execute(select_query, {"date_val": date_val}).fetchone()
                db_value = float(result[0]) if result else None

                if db_value is None:
                    continue  # will be handled in insert
                elif round(db_value, 2) != round(gdp_value, 2):
                    update_query = text(f"UPDATE {self.table_name} SET GDP = :gdp_value WHERE DATE = :date_val")
                    con.execute(update_query, {"gdp_value": gdp_value, "date_val": date_val})
                    print(f"✏️ Updated {date_val}: {db_value} → {gdp_value}")

            # 2) Determine last DB date
            last_date_query = text(f"SELECT MAX(DATE) FROM {self.table_name}")
            last_row = con.execute(last_date_query).fetchone()
            last_db_date = last_row[0] if last_row and last_row[0] else None
            if last_db_date:
                last_db_date = pd.to_datetime(last_db_date).date()

            # 3) Insert only new rows
            df_to_insert = df
            if last_db_date:
                df_to_insert = df[df["DATE"] > last_db_date]

            if df_to_insert.empty:
                print("✅ No new GDP rows (after checking last 2 revisions).")
                return

            insert_query = text(f"INSERT INTO {self.table_name} (DATE, GDP) VALUES (:date_val, :gdp_value)")
            for _, row in df_to_insert.iterrows():
                con.execute(insert_query, {"date_val": row["DATE"], "gdp_value": float(row["GDP"])})

            print(f"✅ Inserted {len(df_to_insert)} new GDP rows into {self.table_name}.")

    def run(self, clean_gdp_func, file_path):
        """Scheduler: every Monday at 17:00 → run once, then sleep for 7 days"""
        while True:
            now = datetime.now()
            print(f"[{now}] Scheduler waking up...")
            if (self.last_run_date != now.date() and
                    now.weekday() == 0 and now.hour >= 11):
                print("⏰ Monday 17:00 → Running GDP extraction")
                file_path = download_latest_qgdpe_excel()
                df = clean_gdp_func(file_path)
                self.write_gdp_to_db(df)
                self.last_run_date = now.date()
                time.sleep(7 * 24 * 3600)
            elif now.weekday() == 0 and now.hour < 11:
                print("⌛ Monday but before 17:00 → Sleeping 1 hour")
                time.sleep(3600)
            else:
                print("😴 Not Monday → Sleeping 1 day")
                time.sleep(86400)
