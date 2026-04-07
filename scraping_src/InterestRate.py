from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import yaml
from bs4 import BeautifulSoup
from datetime import datetime, date, timedelta
import pyodbc
import time
from pathlib import Path
import json
import re
from npl_src.npl_dq.db import DatabaseConnection
from sqlalchemy import text
# === config & paths (same style as your T-bill code) ===
ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "config.yml").read_text())


# INT = ROOT / CFG["data_folder"] / "interim"
# INT.mkdir(parents=True, exist_ok=True)

class InterbankWeeklyInterestRates:
    last_run_date = None

    def __init__(self):
        with open("config.json", "r") as f:
            config = json.load(f)
        self.url = config["INT_URL"]
        self.headers = config["HEADERS"]
        self.filename = config.get("FILENAME", "interbank_weekly_rates.csv")
        self.db_config = CFG["database"]
        self.last_run_date = None

    # ---------------------- DB helpers ----------------------
        self.db = DatabaseConnection()

    def get_last_db_date(self):
        """Return the most recent [Week Ending] from the DB as a date, or None if empty."""
        query = text("""
            SELECT MAX(CAST([Week Ending] AS date)) AS max_date
            FROM t_macro_econs_WeeklyInterestRates
        """)

        # Get engine from DatabaseConnection
        engine = self.db.get_db_connection()
        with engine.connect() as conn:
            result = conn.execute(query)
            return result.scalar()

    # keep a quick "is today already done?" check like your T-bill code
    def check_last_date(self):
        last_date = self.get_last_db_date()
        if not last_date:
            return True
        return last_date != datetime.now().date()

    def write_weekly_to_db(self, rows):
        """
        rows: list of [weekly_ending_str, average_rate_str_or_float]
        """
        if not rows:
            return

        engine = self.db.get_db_connection()

        # Convert strings → types: date, float
        typed = []
        for dt_str, rate in rows:
            weekly_date = self._parse_date(dt_str)  # -> datetime.date
            rate_val = self._parse_float(rate)
            typed.append([weekly_date, rate_val])

        # Insert into DB
        with engine.begin() as conn:  # begin() handles commit/rollback automatically
            for row in typed:
                conn.execute(
                    text("""
                        INSERT INTO t_macro_econs_WeeklyInterestRates (
                            [Week Ending],
                            [Average Rate (%)]
                        ) VALUES (:week_ending, :avg_rate)
                    """),
                    {"week_ending": row[0], "avg_rate": row[1]}
                )

    def _parse_date(self, s):
        s = s.strip().replace("\xa0", " ")  # clean non-breaking spaces
        # remove ordinals (8th, 1st, 2nd, 3rd)
        s = re.sub(r'(\d{1,2})(st|nd|rd|th)', r'\1', s)

        for fmt in ("%d %b %Y", "%d %B %Y", "%Y-%m-%d"):
            try:
                return datetime.strptime(s, fmt).date()
            except ValueError:
                continue

        raise ValueError(f"Unrecognized date format: {s}")

    def _parse_float(self, s):
        if isinstance(s, (int, float)):
            return float(s)
        return float(str(s).replace(",", "").replace("%", "").strip())

    def _header_index_map(self, headers):
        """
        Build a mapping for 'Week Ending' and 'Average Rate (%)' columns, robust to minor text variations.
        """
        norm = [h.lower().strip() for h in headers]
        idx = {}

        # find weekly ending column
        for i, h in enumerate(norm):
            if ("week" in h and "ending" in h) or h == "weekly ending":
                idx["week_ending"] = i
                break

        # find average rate column
        for i, h in enumerate(norm):
            if "average" in h and "rate" in h:
                idx["avg_rate"] = i
                break

        if "week_ending" not in idx or "avg_rate" not in idx:
            raise ValueError(f"Could not locate required columns in headers: {headers}")
        return idx

    def _extract_weekly_rows(self, soup):
        # Step 1: Find Weekly panel
        label = soup.find("div", class_="jet-tabs__label-text",
                          string=re.compile("Weekly Interest Rates", re.I))
        tab = label.find_parent("div", class_="jet-tabs__control")
        panel_id = tab.get("aria-controls")
        weekly_panel = soup.find("div", id=panel_id)

        tables = weekly_panel.find_all("table")

        for idx, table in enumerate(tables, start=1):
            # get headers
            thead = table.find("thead")
            if thead:
                header_cells = [c.get_text(strip=True) for c in thead.find_all(["th", "td"])]
            else:
                first_tr = table.find("tr")
                header_cells = [c.get_text(strip=True) for c in first_tr.find_all(["th", "td"])] if first_tr else []
            # choose the one with "Week Ending" and "Average Rate"
            if "Week Ending" in header_cells and "Average Rate (%)" in header_cells:
                col_idx = self._header_index_map(header_cells)

                body_rows = table.find("tbody").find_all("tr") if table.find("tbody") else table.find_all("tr")[1:]
                out = []
                for tr in body_rows:
                    cells = [c.get_text(strip=True) for c in tr.find_all(["td", "th"])]
                    if len(cells) < max(col_idx.values()) + 1:
                        continue
                    week_str = cells[col_idx["week_ending"]]
                    avg_str = cells[col_idx["avg_rate"]]
                    if not week_str or week_str.lower().startswith("week"):
                        continue
                    out.append([week_str, avg_str])

                print("DEBUG: Extracted", len(out), "rows from chosen Weekly table")
                for row in out[:5]:
                    print("Sample row:", row)

                return out

        raise ValueError("No suitable Weekly Interest Rates table found inside panel")

    def get_weekly_rates(self):
        print("=" * 60)
        print(f"[{datetime.now()}] Checking Weekly Interbank Rates...")
        if not self.check_last_date():
            print("No update needed: today's date already in DB.")
            return

        if datetime.now().weekday() != 0:
            print("Not Friday, skipping.")
            return

        print(f"Fetching HTML with Selenium from {self.url}")
        options = webdriver.ChromeOptions()
        options.add_argument("--headless=new")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")

        driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()),
            options=options
        )
        driver.get(self.url)
        time.sleep(5)  # allow JS tabs/tables to load
        html = driver.page_source
        driver.quit()
        # === Parse with BeautifulSoup ===
        soup = BeautifulSoup(html, "html.parser")
        rows = self._extract_weekly_rows(soup)
        last_db = self.get_last_db_date()
        if last_db:
            # if it's a string, turn it into a date
            if isinstance(last_db, str):
                last_db = self._parse_date(last_db)
            elif isinstance(last_db, datetime):
                last_db = last_db.date()
            # else if it's already a date, leave it alone

            filtered = []
            for dt_str, rate in rows:
                try:
                    d = self._parse_date(dt_str)  # -> datetime.date
                except ValueError:
                    continue
                if d > last_db:  # now both are date objects
                    filtered.append([dt_str, rate])
            rows = filtered

        # === Write to DB if new rows ===
        if rows:
            self.write_weekly_to_db(rows)
            self.last_run_date = datetime.now().date()
            print(f"Inserted {len(rows)} new weekly rows. Success.")
        else:
            print("No new weekly rows to insert (DB up-to-date).")

    def run(self):
        while True:
            now = datetime.now()
            print(f"[{now}] Scheduler waking up...")

            if self.last_run_date != now.date() and now.weekday() == 0 and now.hour >= 18:
                print("It's Friday after 17:00 → Running scrape.")
                self.get_weekly_rates()
                time.sleep(7 * 24 * 3600)

            elif now.weekday() == 0 and now.hour < 18:
                print("It's Friday but before 17:00 → Sleeping 1 hour.")
                time.sleep(3600)

            else:
                print("Not Friday → Sleeping 1 day.")
                time.sleep(86400)


