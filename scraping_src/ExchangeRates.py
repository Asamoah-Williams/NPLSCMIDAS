import requests
import yaml
from bs4 import BeautifulSoup
from datetime import datetime
import pyodbc
import time
from pathlib import Path
import json
import threading
from sqlalchemy import text
from npl_src.npl_dq.db import DatabaseConnection

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "config.yml").read_text())

class ExchangeRates:
    last_run_date = None

    def __init__(self):
        with open("config.json", "r") as f:
            config = json.load(f)
        self.url = config["URL"]
        self.headers = config["HEADERS"]
        self.filename = config["FILENAME"]
        self.db_config = CFG["database"]

        self.db = DatabaseConnection()

    def get_last_db_date(self):
        query = text("SELECT MAX(CurrentDate) FROM T_EXCHANGERATES")
        engine = self.db.get_db_connection()
        with engine.connect() as con:
            result = con.execute(query)
            return result.scalar()

    def check_last_date(self):
        """Return True if today's date is not yet in DB."""
        last_date = self.get_last_db_date()
        today = datetime.now().date()
        return (last_date is None) or (last_date != today)

    def write_to_db(self, rows):
        query = text("""
                INSERT INTO T_EXCHANGERATES (
                    CurrentDate, Currency, CurrencyPair, BuyingRate, SellingRate, MidRate
                ) VALUES (:CurrentDate, :Currency, :CurrencyPair, :BuyingRate, :SellingRate, :MidRate)
            """)
        data = [
            {
                "CurrentDate": row[0],
                "Currency": row[1],
                "CurrencyPair": row[2],
                "BuyingRate": row[3],
                "SellingRate": row[4],
                "MidRate": row[5]
            } for row in rows
        ]

        engine = self.db.get_db_connection()
        with engine.begin() as con:  # begin() automatically commits
            con.execute(query, data)

    def get_exchange_rates(self):
        html = requests.get(self.url, headers=self.headers, timeout=30).text
        soup = BeautifulSoup(html, 'html.parser')
        table = soup.find(name='table', id='table_2')
        if not table:
            raise ValueError("No table found in response")

        rows = []
        for tr in table.find('tbody').find_all('tr'):
            row = [td.get_text(strip=True) for td in tr.find_all('td')]
            rows.append(row)

        self.write_to_db(rows)
        self.last_run_date = datetime.now().date()
        print(f"Inserted {len(rows)} rows for {self.last_run_date}")

    def run(self):
        while True:
            now = datetime.now()
            print(f"[{now}] Scheduler waking up...")

            # Only run if it’s after 5pm, not already run, and today not in DB
            if now.hour >= 19 and self.check_last_date() and self.last_run_date != now.date():
                print("Conditions met → Running scrape.")
                self.get_exchange_rates()
                time.sleep(24 * 3600)  # sleep a day after successful run
            elif now.hour < 19:
                print("Before 5pm → Sleep 1 hour.")
                time.sleep(3600)
            else:
                print("Already updated today → Sleep 6 hours.")
                time.sleep(6 * 3600)



