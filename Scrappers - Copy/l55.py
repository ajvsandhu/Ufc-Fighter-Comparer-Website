import os
import sqlite3
import time
import random
import re
import logging
from logging import FileHandler, Formatter
from datetime import datetime
from bs4 import BeautifulSoup
from difflib import SequenceMatcher
import requests

###############################################################################
# CONFIG
###############################################################################
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../Api/ufc_fighters.db")) 
LOG_PATH = os.path.join(BASE_DIR, "ufc_scraper.log")
HEADERS = {"User-Agent": "Mozilla/5.0"}
MAX_FIGHTS = 5
RETRY_ATTEMPTS = 3
RETRY_SLEEP = 2
REQUEST_TIMEOUT = 15

###############################################################################
# LOGGING
###############################################################################
logger = logging.getLogger()
while logger.handlers:
    logger.removeHandler(logger.handlers[0])

LOG_PATH = os.path.join(os.path.dirname(__file__), "ufc_scraper.log")
file_handler = FileHandler(LOG_PATH, mode="a", encoding="utf-8")
formatter = Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.setLevel(logging.INFO)

###############################################################################
# PERSISTENT SESSION
###############################################################################
session = requests.Session()
session.headers.update(HEADERS)

###############################################################################
# 1) RECREATE TABLE with AUTOINCREMENT
###############################################################################
def recreate_last5_table() -> None:
    """
    Drop and recreate 'fighter_last_5_fights' with strictly incremental IDs.
    """
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute("DROP TABLE IF EXISTS fighter_last_5_fights")
        cur.execute('''
            CREATE TABLE fighter_last_5_fights (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fighter_name TEXT,
                fight_url TEXT,
                kd TEXT,
                sig_str TEXT,
                sig_str_pct TEXT,
                total_str TEXT,
                td_pct TEXT,
                ctrl TEXT,
                result TEXT,
                fight_date TEXT
            )
        ''')
        conn.commit()
    logger.info("Dropped & recreated 'fighter_last_5_fights' table with AUTOINCREMENT ID.")

###############################################################################
# 2) FETCH FIGHTERS
###############################################################################
def get_fighters_in_db_order():
    """
    Return list of (fighter_name, fighter_url) from 'fighters' table, in rowid order.
    """
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute("SELECT fighter_name, fighter_url FROM fighters ORDER BY rowid")
        return cur.fetchall()

###############################################################################
# 3) HTTP GET with RETRIES
###############################################################################
def fetch_url_quietly(url: str):
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            resp = session.get(url, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            return resp
        except Exception:
            time.sleep(RETRY_SLEEP)
    return None

###############################################################################
# 4) GATHER ALL FIGHTS, THEN TAKE THE MOST RECENT 5
###############################################################################
def parse_date_from_row(text: str) -> str:
    # Adjusted regex to handle both "Mar." and "Mar" formats
    date_regex = re.compile(r"[A-Z][a-z]{2}\.?\.?\s*\d{1,2},\s*\d{4}")
    match = date_regex.search(text)
    return match.group().strip() if match else ""

def get_fight_links_top5(fighter_url: str) -> list:
    """
    Gather all fights from the table, parse their date, ignore future fights,
    sort them by date descending, then return the 5 most recent as a list of:
      [(fight_date_str, fight_url), ...]
    """
    resp = fetch_url_quietly(fighter_url)
    if not resp:
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    table = soup.find("table", class_="b-fight-details__table")
    if not table:
        return []

    tbody = table.find("tbody", class_="b-fight-details__table-body")
    if not tbody:
        return []

    rows = tbody.find_all("tr", class_="b-fight-details__table-row")
    all_fights = []  # will hold tuples of (fight_date_obj, fight_date_str, link)

    for row in rows:
        link = ""
        data_link = row.get("data-link", "").strip()
        if data_link:
            link = data_link
        else:
            onclick_val = row.get("onclick", "").strip()
            if "doNav(" in onclick_val:
                start = onclick_val.find("doNav('") + len("doNav('")
                end = onclick_val.find("')", start)
                link = onclick_val[start:end].strip()
        if not link:
            continue

        row_text = row.get_text(" ", strip=True)
        row_date_str = parse_date_from_row(row_text)
        if not row_date_str:
            continue

        try:
            # Handle dates with or without period in month abbreviation
            date_format = "%b. %d, %Y" if "." in row_date_str else "%b %d, %Y"
            fight_date_obj = datetime.strptime(row_date_str, date_format)
            # skip if future
            if fight_date_obj > datetime.now():
                continue
        except:
            continue

        # Keep the valid fight
        all_fights.append((fight_date_obj, row_date_str, link))

    # Sort by date descending (most recent first)
    all_fights.sort(key=lambda x: x[0], reverse=True)

    # Take top 5
    top_5 = all_fights[:MAX_FIGHTS]

    # Return as list of (row_date_str, fight_url)
    # because we want to pass row_date_str as fallback
    return [(item[1], item[2]) for item in top_5]

###############################################################################
# 5) FUZZY MATCH HELPERS
###############################################################################
def basic_clean(s: str) -> str:
    return re.sub(r'["\'].*?["\']', '', s).lower().strip()

def fuzzy_ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()

def substring_or_fuzzy(db_name: str, site_name: str) -> float:
    c_db = basic_clean(db_name)
    c_site = basic_clean(site_name)
    if c_db in c_site or c_site in c_db:
        return 1.0
    return fuzzy_ratio(c_db, c_site)

def best_fighter_index(p_texts: list, db_fighter_name: str) -> tuple:
    best_i = None
    best_r = -1.0
    for i, txt in enumerate(p_texts):
        r = substring_or_fuzzy(db_fighter_name, txt)
        if r > best_r:
            best_r = r
            best_i = i
    return best_i, best_r

###############################################################################
# 6) SCRAPE SINGLE TOTALS or SUM ROUND-BY-ROUND
###############################################################################
def scrape_single_totals(soup: BeautifulSoup, db_fighter_name: str) -> dict:
    tables = soup.select("table.b-fight-details__table")
    for tbl in tables:
        tbody = tbl.find("tbody", class_="b-fight-details__table-body")
        if not tbody:
            continue
        rows = tbody.find_all("tr", class_="b-fight-details__table-row")
        if len(rows) == 1:
            row = rows[0]
            cols = row.find_all("td", class_="b-fight-details__table-col")
            if len(cols) >= 10:
                p_tags = cols[0].find_all("p", class_="b-fight-details__table-text")
                if len(p_tags) == 2:
                    texts = [p.get_text(strip=True) for p in p_tags]
                    idx, ratio = best_fighter_index(texts, db_fighter_name)
                    if ratio < 0.3:
                        continue

                    def get_col(ci: int) -> str:
                        ps = cols[ci].find_all("p", class_="b-fight-details__table-text")
                        return ps[idx].get_text(strip=True) if len(ps) > idx else "0"

                    return {
                        "kd": get_col(1),
                        "sig_str": get_col(2),
                        "sig_str_pct": get_col(3),
                        "total_str": get_col(4),
                        "td_pct": get_col(6),
                        "ctrl": get_col(9)
                    }
    return None

def scrape_sum_rounds(soup: BeautifulSoup, db_fighter_name: str) -> dict:
    kd_sum = 0
    sig_x, sig_y = 0, 0
    tot_x, tot_y = 0, 0
    td_pct_vals = []
    ctrl_sec = 0
    matched_rows = 0

    tables = soup.select("table.b-fight-details__table")
    for tbl in tables:
        thead = tbl.find("thead", class_="b-fight-details__table-head_rnd")
        if not thead:
            continue
        tbody = tbl.find("tbody", class_="b-fight-details__table-body")
        if not tbody:
            continue
        rows = tbody.find_all("tr", class_="b-fight-details__table-row")
        for row in rows:
            cols = row.find_all("td", class_="b-fight-details__table-col")
            if len(cols) < 10:
                continue
            p_tags = cols[0].find_all("p", class_="b-fight-details__table-text")
            if len(p_tags) != 2:
                continue

            texts = [p.get_text(strip=True) for p in p_tags]
            idx, ratio = best_fighter_index(texts, db_fighter_name)
            if ratio < 0.3:
                continue

            matched_rows += 1
            kd_sum += safe_int(cols[1], idx)
            s_txt = safe_text(cols[2], idx)
            sx, sy = parse_of(s_txt)
            sig_x += sx
            sig_y += sy
            t_txt = safe_text(cols[4], idx)
            tx, ty = parse_of(t_txt)
            tot_x += tx
            tot_y += ty
            td_txt = safe_text(cols[6], idx)
            if td_txt.endswith("%"):
                try:
                    td_pct_vals.append(float(td_txt[:-1]))
                except:
                    pass
            c_txt = safe_text(cols[9], idx)
            ctrl_sec += mmss_to_seconds(c_txt)

    if matched_rows == 0:
        return None

    kd_str = str(kd_sum)
    if sig_y > 0:
        sig_str_str = f"{sig_x} of {sig_y}"
        sig_pct = round((sig_x / sig_y) * 100)
        sig_str_pct_str = f"{sig_pct}%"
    else:
        sig_str_str = "0"
        sig_str_pct_str = "0"

    tot_str_str = f"{tot_x} of {tot_y}" if tot_y > 0 else "0"
    td_pct_str = f"{(sum(td_pct_vals) / len(td_pct_vals)):.0f}%" if td_pct_vals else "0"
    ctrl_str = seconds_to_mmss(ctrl_sec)
    return {
        "kd": kd_str,
        "sig_str": sig_str_str,
        "sig_str_pct": sig_str_pct_str,
        "total_str": tot_str_str,
        "td_pct": td_pct_str,
        "ctrl": ctrl_str
    }

def safe_int(col, idx: int) -> int:
    p_tags = col.find_all("p", class_="b-fight-details__table-text")
    if len(p_tags) > idx:
        try:
            return int(p_tags[idx].get_text(strip=True))
        except:
            return 0
    return 0

def safe_text(col, idx: int) -> str:
    p_tags = col.find_all("p", class_="b-fight-details__table-text")
    if len(p_tags) > idx:
        return p_tags[idx].get_text(strip=True)
    return "0"

def parse_of(txt: str) -> tuple:
    try:
        parts = txt.lower().split("of")
        x = int(parts[0].strip())
        y = int(parts[1].strip())
        return (x, y)
    except:
        return (0, 0)

def mmss_to_seconds(txt: str) -> int:
    try:
        mm, ss = txt.split(":")
        return int(mm) * 60 + int(ss)
    except:
        return 0

def seconds_to_mmss(sec: int) -> str:
    m = sec // 60
    s = sec % 60
    return f"{m}:{s:02d}"

###############################################################################
# 7) PARSE RESULT & DATE FROM FIGHT PAGE
###############################################################################
def parse_result_and_date(soup: BeautifulSoup, fighter_name: str) -> tuple:
    """
    Return (result, fight_date) by scraping the fight detail page.
    If not found, returns ("0", "N/A").
    """
    result = "0"
    fight_date = "N/A"

    # Attempt to parse fight date from <i> elements
    info_items = soup.select("i.b-fight-details__text-item")
    for item in info_items:
        txt = item.get_text(strip=True)
        if txt.startswith("Date:"):
            fight_date = txt.replace("Date:", "").strip()
            break

    # Attempt to parse result by fuzzy-matching fighter name
    persons = soup.select("div.b-fight-details__person")
    best_ratio = -1.0
    best_block = None
    for p in persons:
        name_el = p.select_one("h3.b-fight-details__person-name a")
        if not name_el:
            continue
        block_name = name_el.get_text(strip=True)
        ratio = fuzzy_ratio(basic_clean(block_name), basic_clean(fighter_name))
        if ratio > best_ratio:
            best_ratio = ratio
            best_block = p

    if best_block and best_ratio >= 0.3:
        i_status = best_block.select_one("i.b-fight-details__person-status")
        if i_status:
            status_text = i_status.get_text(strip=True).upper()
            if "W" in status_text:
                result = "W"
            elif "L" in status_text:
                result = "L"
            elif "D" in status_text or "DRAW" in status_text:
                result = "D"
            elif "NC" in status_text or "NO CONTEST" in status_text:
                result = "NC"
            elif "DQ" in status_text or "DISQUALIFICATION" in status_text:
                result = "DQ"
            else:
                result = status_text

    return (result, fight_date)

###############################################################################
# 8) SCRAPE FIGHT PAGE (with fallback date)
###############################################################################
def scrape_fight_page_for_fighter(fight_url: str, fighter_name: str, fallback_date: str) -> dict:
    """
    Return a dict of stats for one fight. If the detail page doesn't have a date,
    we fallback to the date from the row (fallback_date).
    """
    data = {
        "fighter_name": fighter_name,
        "fight_url": fight_url,
        "kd": "0",
        "sig_str": "0",
        "sig_str_pct": "0",
        "total_str": "0",
        "td_pct": "0",
        "ctrl": "0",
        "result": "0",
        "fight_date": fallback_date  # default to fallback
    }
    resp = fetch_url_quietly(fight_url)
    if not resp:
        return data

    soup = BeautifulSoup(resp.text, "html.parser")
    res, page_date = parse_result_and_date(soup, fighter_name)

    # Overwrite if we found a valid date
    if page_date != "N/A":
        data["fight_date"] = page_date
    data["result"] = res

    # Attempt single-totals
    single_data = scrape_single_totals(soup, fighter_name)
    if single_data:
        data.update(single_data)
        return data

    # Fallback: sum of round-by-round
    round_data = scrape_sum_rounds(soup, fighter_name)
    if round_data:
        data.update(round_data)
        return data

    return data

###############################################################################
# 9) STORE FIGHT DATA (Manual check for duplicates)
###############################################################################
def store_fight_data(row_data: dict) -> bool:
    """
    Returns True if we inserted a new row, False if it was skipped (duplicate).
    """
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        # Modify duplicate check to ensure fights with the same URL but different dates are inserted
        cur.execute("""
            SELECT id FROM fighter_last_5_fights
            WHERE fight_url = ? AND fight_date = ? AND fighter_name = ?
        """, (row_data["fight_url"], row_data["fight_date"], row_data["fighter_name"]))
        existing = cur.fetchone()
        if existing:
            logger.info(f"Skipping duplicate fight: {row_data['fight_url']} for {row_data['fighter_name']}")
            return False
        
        # Insert new row
        cur.execute('''
            INSERT INTO fighter_last_5_fights (
                fighter_name, fight_url, kd, sig_str, sig_str_pct,
                total_str, td_pct, ctrl, result, fight_date
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            row_data["fighter_name"],
            row_data["fight_url"],
            row_data["kd"],
            row_data["sig_str"],
            row_data["sig_str_pct"],
            row_data["total_str"],
            row_data["td_pct"],
            row_data["ctrl"],
            row_data["result"],
            row_data["fight_date"]
        ))
        conn.commit()
        return True


###############################################################################
# MAIN
###############################################################################
def main() -> None:
    logger.info("Starting l5fights.py, ensuring fight_date is always populated.")
    recreate_last5_table()

    fighters = get_fighters_in_db_order()
    logger.info(f"Found {len(fighters)} fighters in DB. Processing from top to bottom...")

    for fighter_name, fighter_url in fighters:
        if not fighter_url:
            continue

        # get_fight_links_top5 returns list of (row_date_str, fight_url)
        fight_info = get_fight_links_top5(fighter_url)
        logger.info(f"Fighter: '{fighter_name}' => {fighter_url}, found {len(fight_info)} valid fight(s).")

        inserted_count = 0
        for (row_date_str, link) in fight_info:
            row_data = scrape_fight_page_for_fighter(link, fighter_name, fallback_date=row_date_str)
            if store_fight_data(row_data):
                inserted_count += 1

        logger.info(f"[DONE] Inserted data for {inserted_count} new fight(s) for '{fighter_name}'")

    logger.info("All fighters processed. Script finished successfully.")
    print("Script finished successfully!")

if __name__ == "__main__":
    main()
