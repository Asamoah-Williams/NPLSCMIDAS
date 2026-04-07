import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from datetime import datetime
from pathlib import Path
import yaml, json
import pdfplumber
import pandas as pd
from datetime import datetime
import pyodbc
import re
import time
from npl_src.npl_dq.db import DatabaseConnection
from sqlalchemy import text
ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "config.yml").read_text())
with open("config.json", "r") as f:
    config = json.load(f)
SURL = config["SURL"]
HEADERS = config["HEADERS"]
DATA_FOLDER = Path(ROOT / CFG["data_folder"])
DATA_FOLDER.mkdir(parents=True, exist_ok=True)


def download_latest_summary_pdf():
    print(f"Fetching summary page: {SURL}")
    resp = requests.get(SURL, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    # Find target PDF link
    pdf_link = None
    pdf_title = None
    for a in soup.find_all("a", href=True):
        text = a.get_text(strip=True)
        if "Summary of Economic and Financial Data" in text and "Charts" not in text:
            pdf_link = urljoin(SURL, a["href"])
            pdf_title = text
            break
    if not pdf_link:
        raise ValueError("No valid 'Summary of Economic and Financial Data' PDF link found.")
    # checks if we already have this file ---
    safe_title = pdf_title.replace(" ", "_").replace("–", "-")
    existing_files = list(DATA_FOLDER.glob(f"{safe_title}*.pdf"))
    if existing_files:
        print(f"Already downloaded: {pdf_title}")
        return existing_files[0]
    resp_file = requests.get(pdf_link, headers=HEADERS, timeout=60)
    resp_file.raise_for_status()
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = DATA_FOLDER / f"{safe_title}_{timestamp}.pdf"
    with open(filename, "wb") as f:
        f.write(resp_file.content)
    print(f"Downloaded: {pdf_title}")
    print(f"Saved to: {filename}")
    return filename


def extract_npl_and_mpr_from_pdf(pdf_path: Path):
    results = {}
    with pdfplumber.open(pdf_path) as pdf:
        # --- MPR (page 5) ---
        page = pdf.pages[4]
        text = page.extract_text().splitlines()
        mpr_line = None


        for line in text:
            if "Monetary Policy Rate" in line:
                mpr_line = line
                break
        if mpr_line:

            tokens = mpr_line.replace("Monetary Policy Rate", "").split()
            values = [t for t in tokens if t.replace(".", "", 1).isdigit()]

            month_line = next((line for line in text if re.search(r"\d{4}:\d{2}", line)), None)
            if not month_line:
                raise ValueError("❌ Could not locate month header on page 5.")
            month_tokens = re.findall(r"\d{4}:\d{2}", month_line)
            months = [pd.to_datetime(m.replace(":", "-") + "-01") + pd.offsets.MonthEnd(0) for m in month_tokens]

            if len(months) > len(values):
                months = months[-len(values):]
            elif len(values) > len(months):
                values = values[-len(months):]

            mpr_df = pd.DataFrame({
                "Date": months,
                "MPR": pd.to_numeric(values, errors="coerce")
            })

            results["MPR"] = mpr_df

            print("\n✅ MPR Extracted Dynamically:")
            print(mpr_df.head(13))

        # --- NPL (page 13) ---

        page = pdf.pages[12]
        text = page.extract_text().splitlines()
        npl_line = None

        for line in text:
            if "Non-Performing Loans" in line:
                npl_line = line
                break

        if npl_line:
            # Extract numeric values
            tokens = npl_line.replace("Non-Performing Loans", "").split()
            values = [t for t in tokens if t.replace(".", "", 1).isdigit()]
            print("📊 Parsed NPL Numbers:", values)
            month_line = next((line for line in text if re.search(r"\d{4}:\d{2}", line)), None)
            if not month_line:
                raise ValueError("❌ Could not locate month header on page 13 for NPL.")
            month_tokens = re.findall(r"\d{4}:\d{2}", month_line)
            months = [pd.to_datetime(m.replace(":", "-") + "-01") + pd.offsets.MonthEnd(0) for m in month_tokens]

            if len(months) > len(values):
                months = months[-len(values):]
            elif len(values) > len(months):
                values = values[-len(months):]
            npl_df = pd.DataFrame({
                "Date": months,
                "NPL": pd.to_numeric(values, errors="coerce")
            })
            results["NPL"] = npl_df
            print("\n✅ NPL Extracted Dynamically:")
            print(npl_df.head(13))

        # --- Inflation (page 3) ---

        page = pdf.pages[2]
        text = page.extract_text().splitlines()
        inflation_line = None

        #  Locating "All Consumer Prices" line under "Year-on-Year"
        for i, line in enumerate(text):
            if "Year-on-Year" in line:
                for j in range(i + 1, len(text)):
                    if "All Consumer Prices" in text[j]:
                        inflation_line = text[j]
                        break
                break

        if inflation_line:
            #  Extract numeric values
            tokens = inflation_line.replace("All Consumer Prices", "").split()
            values = [t.strip("%") for t in tokens if t.replace(".", "", 1).isdigit()]
            print("📊 Parsed Inflation Values:", values)


            month_line = next((line for line in text if re.search(r"\d{4}:\d{2}", line)), None)
            if not month_line:
                raise ValueError("❌ Could not locate month header on page 3 for Inflation.")
            print("\n🗓️ Raw Month Header Line (Inflation):")
            print(month_line)
            month_tokens = re.findall(r"\d{4}:\d{2}", month_line)
            months = [pd.to_datetime(m.replace(":", "-") + "-01") + pd.offsets.MonthEnd(0) for m in month_tokens]

            if len(months) > len(values):
                months = months[-len(values):]
            elif len(values) > len(months):
                values = values[-len(months):]


            inflation_df = pd.DataFrame({
                "Date": months,
                "Inflation": pd.to_numeric(values, errors="coerce")
            })
            results["Inflation"] = inflation_df
            print("\n✅ Inflation Extracted Dynamically:")
            print(inflation_df.head(13))

        # --- CBLR (page 5) ---
        page = pdf.pages[4]
        text = page.extract_text().splitlines()
        lending_line = None

        #  Locatingg the "Average Lending Rate" line under "Credit Market"
        for i, line in enumerate(text):
            if "Credit Market" in line:
                print(f"🔍 Found section header: {line}")
                for j in range(i + 1, len(text)):
                    if "Average Lending Rate" in text[j]:
                        lending_line = text[j]
                        break
                break

        if lending_line:
            #  Extract numeric values
            tokens = lending_line.replace("Average Lending Rate", "").split()
            values = [t.strip("%") for t in tokens if t.replace(".", "", 1).isdigit()]
            print("📊 Parsed Lending Rate Values:", values)

            month_line = next((line for line in text if re.search(r"\d{4}:\d{2}", line)), None)
            if not month_line:
                raise ValueError("❌ Could not locate month header on page 5 for Lending Rate.")


            month_tokens = re.findall(r"\d{4}:\d{2}", month_line)
            months = [pd.to_datetime(m.replace(":", "-") + "-01") + pd.offsets.MonthEnd(0) for m in month_tokens]


            if len(months) > len(values):
                months = months[-len(values):]
            elif len(values) > len(months):
                values = values[-len(months):]

            lending_df = pd.DataFrame({
                "Date": months,
                "LendingRate": pd.to_numeric(values, errors="coerce")
            })

            results["LendingRate"] = lending_df
            print("\n✅ Lending Rate Extracted Dynamically:")
            print(lending_df.head(13))

        # --- GLA page 13 ---
        page = pdf.pages[12]
        text = page.extract_text().splitlines()
        gla_line = None

        #  Locating the "Annual Growth (%)" line under "Total Advances"
        for i, line in enumerate(text):
            if "Total Advances" in line:
                for j in range(i + 1, len(text)):
                    if "Annual Growth (%)" in text[j]:
                        gla_line = text[j]
                        break
                break

        if gla_line:
            #  Extract numeric values
            tokens = gla_line.replace("Annual Growth (%)", "").split()
            values = [t.strip("%") for t in tokens if t.replace(".", "", 1).isdigit()]
            print("📊 Parsed GLA Growth Values:", values)


            month_line = next((line for line in text if re.search(r"\d{4}:\d{2}", line)), None)
            if not month_line:
                raise ValueError("❌ Could not locate month header on page 13 for GLA Growth.")


            month_tokens = re.findall(r"\d{4}:\d{2}", month_line)
            months = [pd.to_datetime(m.replace(":", "-") + "-01") + pd.offsets.MonthEnd(0) for m in month_tokens]

            if len(months) > len(values):
                months = months[-len(values):]
            elif len(values) > len(months):
                values = values[-len(months):]

            gla_df = pd.DataFrame({
                "Date": months,
                "GLA": pd.to_numeric(values, errors="coerce")
            })
            results["GLA"] = gla_df
            print("\n✅ GLA Growth Extracted Dynamically:")
            print(gla_df.head(13))

        # --- loanAdvances page 13 ---
        page = pdf.pages[12]
        text = page.extract_text().splitlines()

        # Step 1: find month labels
        month_line = next((line for line in text if re.search(r"\d{4}:\d{2}", line)), None)
        if not month_line:
            raise ValueError("❌ Could not locate month header on page 13.")

        month_tokens = re.findall(r"\d{4}:\d{2}", month_line)
        months = [pd.to_datetime(m.replace(":", "-") + "-01") + pd.offsets.MonthEnd(0) for m in month_tokens]
        print("\n🔎 Raw Month Tokens Found:", month_tokens)

        # Step 2: find Total Advances (billion GHC)
        loan_adv_line = next((line for line in text if "Total Advances (billion GHC)" in line), None)
        if not loan_adv_line:
            raise ValueError("❌ Could not locate 'Total Advances (billion GHC)' line on page 13.")



        # Step 3: extract numbers
        tokens = loan_adv_line.replace("Total Advances (billion GHC)", "").split()
        values = [t for t in tokens if re.match(r"^\d+(\.\d+)?$", t)]

        # Align lengths
        if len(values) != len(months):
            print(f"⚠️ Length mismatch: {len(values)} values vs {len(months)} months — trimming to match.")
            min_len = min(len(values), len(months))
            values = values[-min_len:]
            months = months[-min_len:]

        # Step 4: Build DataFrame
        loanadv_df = pd.DataFrame({
            "Date": months,
            "LoanAdvances": pd.to_numeric(values, errors="coerce")
        })

        results["LoanAdvances"] = loanadv_df

        print("\n✅ Loan Advances Extracted Dynamically:")
        print(loanadv_df.head(13))

    return results


def clean_npl(npl_df: pd.DataFrame):
    npl_df["DATE"] = pd.to_datetime(npl_df["Date"], format="%Y:%m", errors="coerce") + pd.offsets.MonthEnd(0)
    npl_df["NPL_diff"] = pd.to_numeric(npl_df["NPL"], errors="coerce")
    cleaned = npl_df[["DATE", "NPL_diff"]].dropna().reset_index(drop=True)
    print("\n Cleaned NPL (raw values kept as NPL_diff):")
    print(cleaned.head(13))
    return cleaned


def clean_mpr(mpr_df: pd.DataFrame):
    """
    Clean raw MPR table extracted from PDF.
    Output columns: Year, Month, DATE, MPR
    """
    dates = pd.to_datetime(mpr_df["Date"], format="%Y:%m", errors="coerce") + pd.offsets.MonthEnd(0)
    cleaned = pd.DataFrame({
        "Year": dates.dt.year,
        "Month": dates.dt.month,
        "DATE": dates,
        "MPR": pd.to_numeric(mpr_df["MPR"], errors="coerce")
    })
    print("\n✅ Cleaned MPR (all rows):")
    print(cleaned.head(20))
    return cleaned


def clean_inflation(inflation_df: pd.DataFrame):
    dates = pd.to_datetime(inflation_df["Date"], format="%Y:%m", errors="coerce") + pd.offsets.MonthEnd(0)
    cleaned = pd.DataFrame({
        "Year": dates.dt.year,
        "Month": dates.dt.month,
        "DATE": dates,
        "HeadlineInflation": pd.to_numeric(inflation_df["Inflation"], errors="coerce")
    })
    print("\n✅ Cleaned Inflation (all rows):")
    print(cleaned.head(20))
    return cleaned


def clean_lending(lending_df: pd.DataFrame):
    lending_df["DATE"] = pd.to_datetime(lending_df["Date"], errors="coerce") + pd.offsets.MonthEnd(0)
    lending_df["CBLR_diff"] = pd.to_numeric(lending_df["LendingRate"], errors="coerce")
    cleaned = lending_df[["DATE", "CBLR_diff"]].dropna().reset_index(drop=True)
    print("\n✅ Cleaned Lending Rate (CBLR_diff):")
    print(cleaned.head(13))
    return cleaned


def clean_gla(gla_df: pd.DataFrame):
    gla_df["DATE"] = pd.to_datetime(gla_df["Date"], errors="coerce") + pd.offsets.MonthEnd(0)
    gla_df["GLA_diff"] = pd.to_numeric(gla_df["GLA"], errors="coerce")
    cleaned = gla_df[["DATE", "GLA_diff"]].dropna().reset_index(drop=True)
    print("\n✅ Cleaned GLA Growth (GLA_diff):")
    print(cleaned.head(13))
    return cleaned

def clean_loan_advances(loanadv_df: pd.DataFrame):
    loanadv_df["DATE"] = pd.to_datetime(loanadv_df["Date"], errors="coerce") + pd.offsets.MonthEnd(0)
    loanadv_df["Loan_value"] = pd.to_numeric(loanadv_df["LoanAdvances"], errors="coerce")
    cleaned = loanadv_df[["DATE", "Loan_value"]].dropna().reset_index(drop=True)

    print("\n✅ Cleaned Loa (LoanAdvances_diff):")
    print(cleaned.head(13))
    return cleaned

class DatabaseWriter:
    def __init__(self, db_config, table_name, columns):
        """
        db_config : dict with driver/server/db/user/pass
        table_name : str (e.g., 't_npl_raw_NPL')
        columns : list of str (column names in DB in insert order)
        """
        self.db_config = db_config
        self.table_name = table_name
        self.columns = columns
        self.last_run_date = None

        self.db = DatabaseConnection()



    def get_last_date(self):
        """Get the most recent DATE in this table using SQLAlchemy engine."""
        query = text(f"SELECT MAX(DATE) FROM {self.table_name}")

        engine = self.db.get_db_connection()  # SQLAlchemy engine
        with engine.connect() as conn:
            row = conn.execute(query).fetchone()
            return row[0] if row and row[0] else None

    from sqlalchemy import text

    def insert(self, df: pd.DataFrame):
        """Insert only new rows into the configured table using SQLAlchemy."""
        last_db_date = self.get_last_date()
        print(f"📌 Last DATE in {self.table_name}: {last_db_date}")

        # Ensure DATE is datetime
        df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")

        # Filter newer rows only
        if last_db_date:
            last_db_date = pd.to_datetime(last_db_date)
            df = df[df["DATE"] > last_db_date]

        if df.empty:
            print(f"✅ No new rows to insert into {self.table_name}.")
            return

        # Build SQLAlchemy insert statement with named placeholders
        columns_str = ", ".join(self.columns)
        placeholders_str = ", ".join([f":{col}" for col in self.columns])
        insert_sql = text(f"""
            INSERT INTO {self.table_name} ({columns_str})
            VALUES ({placeholders_str})
        """)

        # Convert dataframe to list of dicts for parameterized insert
        rows_to_insert = df[self.columns].to_dict(orient="records")

        # Use engine instead of raw connection
        engine = self.db.get_db_connection()
        with engine.begin() as conn:  # automatically commits or rollbacks
            conn.execute(insert_sql, rows_to_insert)

        print(f"✅ Inserted {len(df)} new rows into {self.table_name}.")

    def run(self):
        """Scheduler: every Monday at 16:00 → run once, then sleep for 7 days"""
        while True:
            now = datetime.now()
            print(f"[{now}] Summary Scheduler waking up...")

            # Monday = 0, run if it's after 16:00 and not already run today
            if (self.last_run_date != now.date() and
                    now.weekday() == 0 and now.hour >= 19):

                print("⏰ Monday 16:00 → Running Summary PDF extraction")

                # Step 1: download latest Summary PDF
                pdf_path = download_latest_summary_pdf()

                # Step 2: extract from PDF
                extracted = extract_npl_and_mpr_from_pdf(pdf_path)

                db_config = CFG["database"]

                # Step 3: clean + insert
                if "NPL" in extracted:
                    cleaned_npl = clean_npl(extracted["NPL"])
                    DatabaseWriter(db_config, "NPL_raw", ["DATE", "NPL_diff"]).insert(cleaned_npl)

                if "MPR" in extracted:
                    cleaned_mpr = clean_mpr(extracted["MPR"])
                    DatabaseWriter(db_config, "t_macro_econs_MPR", ["Year", "Month", "DATE", "MPR"]).insert(cleaned_mpr)

                if "Inflation" in extracted:
                    cleaned_infl = clean_inflation(extracted["Inflation"])
                    DatabaseWriter(db_config, "t_macro_econs_Inflation",
                                   ["Year", "Month", "DATE", "HeadlineInflation"]).insert(cleaned_infl)

                if "LendingRate" in extracted:
                    cleaned_lending = clean_lending(extracted["LendingRate"])
                    DatabaseWriter(db_config, "CBLR_raw", ["DATE", "CBLR_diff"]).insert(cleaned_lending)

                if "GLA" in extracted:
                    cleaned_gla = clean_gla(extracted["GLA"])
                    DatabaseWriter(db_config, "GLA_raw", ["DATE", "GLA_diff"]).insert(cleaned_gla)

                if "LoanAdvances" in extracted:
                    cleaned_loanadv = clean_loan_advances(extracted["LoanAdvances"])
                    DatabaseWriter(db_config, "t_insightView_LoanPortfolio", ["DATE", "Loan_value"]).insert(cleaned_loanadv)

                self.last_run_date = now.date()
                time.sleep(7 * 24 * 3600)

            elif now.weekday() == 0 and now.hour < 19:
                print("⌛ Monday but before 16:00 → Sleeping 1 hour")
                time.sleep(3600)
            else:
                print("😴 Not Monday → Sleeping 1 day")
                time.sleep(86400)


