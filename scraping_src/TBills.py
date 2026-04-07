import requests
import yaml
from bs4 import BeautifulSoup
from datetime import datetime, date
import pyodbc
import time
from pathlib import Path
import json
from npl_src.npl_dq.db import DatabaseConnection
from sqlalchemy import text
# Load configuration
ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "config.yml").read_text())
INT = ROOT / CFG["data_folder"] / "interim"
INT.mkdir(parents=True, exist_ok=True)

class TreasuryBillRates:
    last_run_date = None

    def __init__(self):
        with open("config.json", "r") as f:
            config = json.load(f)
        self.url = config["TBILL_URL"]
        self.headers = config["HEADERS"]
        self.filename = config["FILENAME"]
        self.db_config = CFG["database"]
        self.last_run_date = None
        self.db = DatabaseConnection()

    def write_tb_to_db(self, rows):
        """
        Insert Treasury Bills rows into t_TreasuryBills using SQLAlchemy.
        Each row must have 5 columns:
        [Issue Date, Tender, Security Type, Discount Rate, Interest Rate]
        """
        if not rows:
            return

        # Validate rows
        if not all(isinstance(r, (list, tuple)) and len(r) == 5 for r in rows):
            raise ValueError(f"Each row must have 5 columns; sample: {rows[0] if rows else rows}")

        # Prepare SQLAlchemy insert with named parameters
        insert_sql = text("""
            INSERT INTO t_TreasuryBills (
                [Issue Date],
                Tender,
                [Security Type],
                [Discount Rate],
                [Interest Rate]
            )
            VALUES (
                :issue_date,
                :tender,
                :security_type,
                :discount_rate,
                :interest_rate
            )
        """)

        # Convert rows to dicts with proper ISO dates
        new_rows = []
        for row in rows:
            date_value = row[0]

            if isinstance(date_value, (datetime, date)):
                iso_date = date_value.isoformat()
            else:
                iso_date = datetime.strptime(date_value, "%d %b %Y").date().isoformat()

            new_rows.append({
                "issue_date": iso_date,
                "tender": row[1],
                "security_type": row[2],
                "discount_rate": row[3],
                "interest_rate": row[4],
            })

        # Use engine instead of raw connection
        engine = self.db.get_db_connection()
        with engine.begin() as conn:  # auto commit/rollback
            conn.execute(insert_sql, new_rows)

        print("✅ Inserted rows:")
        for r in new_rows:
            print(f"   {r['issue_date']} → {r['security_type']}")

    def get_tbill_rates(self):
        print(f"[{datetime.now()}] Checking for new T-bill rates...")

        # --- Only run on Thursdays (original code had weekday=3) ---
        if datetime.now().weekday() != 0:  # Thursday = 3
            print("Not Thursday, skipping.")
            return

        # --- Fetch HTML ---
        html = requests.get(self.url, headers=self.headers, timeout=30).text
        soup = BeautifulSoup(html, 'html.parser')
        table = soup.find('table', id='table_1')
        if not table:
            raise ValueError("No table_1 found in response")

        # --- Parse rows ---
        rows = []
        tbody = table.find('tbody')
        trs = tbody.find_all('tr') if tbody else table.find_all('tr')

        for tr in trs:
            cells = [c.get_text(strip=True) for c in tr.find_all(['td', 'th'])]
            if not cells:
                continue
            if cells[0].lower().startswith('issue') or cells[0].lower() == 'issue date':
                continue
            cells = (cells + [''] * 5)[:5]  # ensure 5 columns
            rows.append(cells)

        # --- Deduplicate by (Issue Date, Security Type) ---
        unique_rows = {(r[0], r[2]): r for r in rows}
        rows = list(unique_rows.values())

        # --- Filter out rows already in DB ---
        final_rows = []
        engine = self.db.get_db_connection()
        with engine.connect() as conn:
            for r in rows:
                issue_date = datetime.strptime(r[0], "%d %b %Y").date()
                sec_type = r[2]

                check_sql = text("""
                    SELECT COUNT(*)
                    FROM t_TreasuryBills
                    WHERE [Issue Date] = :issue_date
                      AND [Security Type] = :security_type
                """)
                count = conn.execute(check_sql,
                                     {"issue_date": issue_date.isoformat(), "security_type": sec_type}).scalar()

                if count == 0:
                    final_rows.append([issue_date, *r[1:]])

        # --- Insert new rows ---
        if final_rows:
            print("🆕 New rows to insert:")
            for r in final_rows:
                print(f"   {r[0]} → {r[2]}")
            self.write_tb_to_db(final_rows)
            self.last_run_date = datetime.now().date()
            print(f"✅ Success: Inserted(Tbill) {len(final_rows)} new unique rows")
        else:
            print("✅ No new rows to insert (DB up to date)")

    def run(self):
        while True:
            now = datetime.now()
            print(f"[{now}] Scheduler waking up...")

            if self.last_run_date != now.date() and now.weekday() == 0 and now.hour >= 19:
                print("⏰ It's Friday after 15:00 → Running scrape.")
                self.get_tbill_rates()
                time.sleep(7 * 24 * 3600)

            elif now.weekday() == 0 and now.hour < 19:
                print("⌛ It's fRida but before 15:00 → Sleeping 1 hour.")
                time.sleep(3600)

            else:
                print("😴 Not friday → Sleeping 1 day.")
                time.sleep(86400)


