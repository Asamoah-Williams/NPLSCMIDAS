import requests
import yaml
from bs4 import BeautifulSoup
from datetime import datetime
import pyodbc
import time
from pathlib import Path
import json
import threading
import re
import calendar
from npl_src.npl_dq.db import DatabaseConnection
from sqlalchemy import text
ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "config.yml").read_text())

class MonthlyUSExchangeRate:
    last_run_date = None


    def __init__(self):
        with open("config.json", "r") as f:
            config = json.load(f)
        self.url = config["EXURL"]
        self.headers = config["HEADERS"]
        self.filename = config["FILENAME"]
        self.db_config = CFG["database"]
    # ---------------------- DB helpers ----------------------
        self.db = DatabaseConnection()

    from sqlalchemy import text
    from datetime import datetime

    def get_last_db_date(self):
        """Return the most recent date in DB as a datetime.date, or None if empty."""
        query = text("SELECT MAX([Date]) AS max_date FROM DEGU_raw")

        # Use SQLAlchemy engine
        engine = self.db.get_db_connection()
        with engine.connect() as conn:
            row = conn.execute(query).fetchone()
            last_date = row[0] if row and row[0] else None
            print(f"[DB] Last date in DB: {last_date}")

            if last_date is None:
                return None

            # --- Normalize type ---
            if isinstance(last_date, datetime):
                return last_date.date()
            elif isinstance(last_date, str):
                for fmt in ("%Y-%m-%d", "%d/%m/%Y"):
                    try:
                        return datetime.strptime(last_date, fmt).date()
                    except ValueError:
                        continue
                # If no format matches, return None
                return None
            else:
                return last_date  # already a date

    def check_last_date(self):
        last_date = self.get_last_db_date()
        today = datetime.now().date().replace(day=1)  # first day of current month
        return (last_date is None) or (last_date < today)

    # ---------------------- Scraper ----------------------
    def _month_str_to_num(self, month_str):
        mapping = {
            "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4,
            "May": 5, "Jun": 6, "Jul": 7, "Aug": 8,
            "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
        }
        return mapping.get(month_str.strip()[:3], None)

    def get_table_data(self):
        html = requests.get(self.url, headers=self.headers, timeout=30).text
        soup = BeautifulSoup(html, "html.parser")

        # Locate table
        table = soup.find("table", {"id": "table_2"})
        if not table:
            raise ValueError("Could not find table with id=table_2")
        print("[SCRAPER] Table located successfully")
        target_row = None
        for tr in table.find("tbody").find_all("tr"):
            cols = [c.get_text(strip=True).replace("\xa0", " ") for c in tr.find_all("td")]
            if len(cols) > 1 and "Month Average" in cols[1] and "US$" in cols[1]:
                target_row = cols
                break

        if not target_row:
            raise ValueError("Could not find 'Inter-Bank Exchange Rate - Month Average (GHC/US$)' row")

        # Format: [Year, Variable, Jan, Feb, ..., Dec]
        year = int(target_row[0])
        values = target_row[2:]

        out = []
        for idx, val in enumerate(values, start=1):
            if not val or re.match(r"^[\.\-]+$", val):
                continue
            try:
                rate = float(val)
            except ValueError:
                continue
            # Use last day of the month
            last_day = calendar.monthrange(year, idx)[1]
            date_val = datetime(year, idx, last_day).date()
            out.append((date_val, rate))  # (Date, DEGU_diff)
        return out



    def write_to_db(self, rows):
        if not rows:
            return

        insert_sql = text("""
            INSERT INTO DEGU_raw ([Date], [DEGU_diff])
            VALUES (:date, :degu_diff)
        """)

        # Convert datetime.date → str (ISO), float → float
        safe_rows = []
        for d, val in rows:
            if isinstance(d, datetime):
                d = d.date()
            if hasattr(d, "isoformat"):  # date object
                d = d.isoformat()
            safe_rows.append({"date": d, "degu_diff": float(val)})

        # --- Use SQLAlchemy engine instead of pyodbc connection ---
        engine = self.db.get_db_connection()
        with engine.connect() as conn:
            for row in safe_rows:
                conn.execute(insert_sql, **row)
            conn.commit()

    def run_scraper(self):
        if not self.check_last_date():
            print("DB already up to date → skipping")
            return

        rows = self.get_table_data()
        last_db = self.get_last_db_date()
        if last_db:
            rows = [(d, r) for d, r in rows if d > last_db]

        if rows:
            self.write_to_db(rows)
            self.last_run_date = datetime.now().date()
            print(f"Inserted {len(rows)} new rows up to {rows[-1][0]}")
        else:
            print("No new rows to insertsss")

    # ---------------------- Scheduler ----------------------
    def run(self):
        while True:
            now = datetime.now()
            print(f"[{now}] Scheduler waking up...")

            if (
                now.weekday() == 0
                and now.hour >= 19
                and self.last_run_date != now.date()
            ):
                print("It's Thursday after 17:00 → Running scrape.")
                self.run_scraper()
                time.sleep(24 * 3600)  # sleep a day

            elif now.weekday() == 0 and now.hour < 19:
                print("It's Thursday but before 17:00 → Sleeping 1 hour.")
                time.sleep(3600)

            else:
                print("Not Thursday → Sleeping 1 day.")
                time.sleep(86400)



