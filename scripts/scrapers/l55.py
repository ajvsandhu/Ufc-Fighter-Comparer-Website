import os
import time
import random
import re
import logging
import argparse
from logging import FileHandler, Formatter
from datetime import datetime
from bs4 import BeautifulSoup
from difflib import SequenceMatcher
import requests
from dotenv import load_dotenv
import sys
from typing import List, Dict, Any

# Simple fix - add the project root to Python's path
sys.path.insert(0, '.')

# Load environment variables
load_dotenv()

# Now import from backend modules
from backend.constants import (
    LOG_LEVEL,
    LOG_FORMAT,
    LOG_DATE_FORMAT,
    REQUEST_HEADERS,
    REQUEST_TIMEOUT,
    RETRY_ATTEMPTS,
    RETRY_DELAY,
    MAX_FIGHTS_DISPLAY
)

###############################################################################
# CONFIG
###############################################################################
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
LOG_PATH = os.path.join(BASE_DIR, "ufc_scraper.log")
HEADERS = {"User-Agent": "Mozilla/5.0"}
MAX_FIGHTS = 5
RETRY_ATTEMPTS = 3
RETRY_SLEEP = 2
REQUEST_TIMEOUT = 15
EVENT_URL = "http://ufcstats.com/statistics/events/completed"

# Load environment variables
load_dotenv()

# Import Supabase client 
from backend.supabase_client import (
    supabase,
    test_connection,
    get_fighters,
    get_fighter,
    get_fighter_by_url,
    insert_fighter,
    update_fighter,
    upsert_fighter,
    get_fighter_fights,
    delete_fighter_fights,
    insert_fighter_fight,
    truncate_table
)

# Test Supabase connection
if not test_connection():
    raise ValueError("Could not connect to Supabase. Please check your credentials and network connection.")

###############################################################################
# LOGGING
###############################################################################
logger = logging.getLogger()
while logger.handlers:
    logger.removeHandler(logger.handlers[0])

LOG_PATH = os.path.join(os.path.dirname(__file__), "ufc_scraper.log")
file_handler = FileHandler(LOG_PATH, mode="a", encoding="utf-8")
formatter = Formatter(LOG_FORMAT)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.setLevel(getattr(logging, LOG_LEVEL))

# Also add a console handler for better visibility during execution
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

logger.info("UFC Last 5 Fights Scraper - Supabase Edition")

###############################################################################
# PERSISTENT SESSION
###############################################################################
session = requests.Session()
session.headers.update(REQUEST_HEADERS)

###############################################################################
# 1) RECREATE TABLE with AUTOINCREMENT and reordered columns
###############################################################################
def recreate_last5_table() -> None:
    """Clear all rows from 'fighter_last_5_fights' table and reset sequence."""
    try:
        # First try to truncate the table
        sql = "TRUNCATE TABLE fighter_last_5_fights RESTART IDENTITY CASCADE;"
        supabase.table("fighters").select("*").limit(1).execute()  # Ensure connection
        supabase.postgrest.rpc('raw_sql', {'query': sql}).execute()
        
        # If truncate fails, try dropping and recreating
        sql = """
        DROP TABLE IF EXISTS fighter_last_5_fights;
        CREATE TABLE fighter_last_5_fights (
            id SERIAL PRIMARY KEY,
            fighter_name TEXT,
            fight_url TEXT,
            kd TEXT,
            sig_str TEXT,
            sig_str_pct TEXT,
            total_str TEXT,
            head_str TEXT,
            body_str TEXT,
            leg_str TEXT,
            takedowns TEXT,
            td_pct TEXT,
            ctrl TEXT,
            result TEXT,
            method TEXT,
            opponent TEXT,
            fight_date TEXT,
            event TEXT
        );
        """
        supabase.postgrest.rpc('raw_sql', {'query': sql}).execute()
        logger.info("Cleared 'fighter_last_5_fights' table.")
    except Exception as e:
        logger.error(f"Error clearing fighter_last_5_fights table: {e}")
        # Try direct delete as last resort
        try:
            supabase.table("fighter_last_5_fights").delete().neq("id", 0).execute()
            logger.info("Cleared table using delete operation")
        except Exception as e2:
            logger.error(f"Failed to clear table using delete: {e2}")

###############################################################################
# 2) FETCH FIGHTERS
###############################################################################
def get_fighters_in_db_order():
    """
    Return list of (fighter_name, fighter_url) from 'fighters' table, in id order.
    """
    try:
        all_fighters = []
        page_size = 1000
        start = 0
        
        while True:
            # Get fighters from Supabase, ordered by id, with pagination
            response = supabase.table("fighters") \
                .select("fighter_name, fighter_url") \
                .order("id") \
                .range(start, start + page_size - 1) \
                .execute()
            
            if not response.data or len(response.data) == 0:
                break
                
            all_fighters.extend([(row["fighter_name"], row["fighter_url"]) for row in response.data])
            
            if len(response.data) < page_size:
                break
                
            start += page_size
            
        logger.info(f"Retrieved {len(all_fighters)} fighters total")
        return all_fighters
    except Exception as e:
        logger.error(f"Error getting fighters from Supabase: {e}")
        raise  # Re-raise the exception since we're fully committed to Supabase

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
    date_regex = re.compile(r"[A-Z][a-z]{2}\.?\.?\s*\d{1,2},\s*\d{4}")
    match = date_regex.search(text)
    return match.group().strip() if match else ""

def get_fight_links_top5(fighter_url: str) -> list:
    """
    Gather all fights from the table, parse their date, ignore future fights,
    sort them by date descending, then return the 5 most recent as a list of:
      [(fight_date_str, fight_url), ...]
    """
    logger.info(f"Getting fight links for {fighter_url}")
    resp = fetch_url_quietly(fighter_url)
    if not resp:
        logger.error(f"Failed to fetch URL: {fighter_url}")
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    table = soup.find("table", class_="b-fight-details__table")
    if not table:
        logger.warning(f"No fight table found at {fighter_url}")
        return []

    tbody = table.find("tbody", class_="b-fight-details__table-body")
    if not tbody:
        logger.warning(f"No tbody found in fight table at {fighter_url}")
        return []

    rows = tbody.find_all("tr", class_="b-fight-details__table-row")
    logger.info(f"Found {len(rows)} fight rows for fighter")
    all_fights = []

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
            # Handle both date formats with and without a period
            date_format = "%b. %d, %Y" if "." in row_date_str else "%b %d, %Y"
            fight_date_obj = datetime.strptime(row_date_str, date_format)
            
            # Log the parsed date for debugging
            logger.info(f"Parsed fight date: {row_date_str} -> {fight_date_obj.strftime('%Y-%m-%d')}")
            
            # Allow fights dated in the future (since many events are predated)
            # but not more than a year in the future
            one_year_from_now = datetime.now().replace(year=datetime.now().year + 1)
            if fight_date_obj > one_year_from_now:
                logger.warning(f"Skipping future fight dated {row_date_str} (more than a year in the future)")
                continue
        except Exception as e:
            logger.warning(f"Error parsing date {row_date_str}: {e}")
            continue

        all_fights.append((fight_date_obj, row_date_str, link))

    # Add detailed logging about all fights found
    logger.info(f"Found {len(all_fights)} valid fights for fighter")
    for date_obj, date_str, link in all_fights:
        logger.info(f"Fight: {date_str} ({date_obj.strftime('%Y-%m-%d')}) - {link}")

    # Sort by date (newest first)
    all_fights.sort(key=lambda x: x[0], reverse=True)
    
    # Log the sorted results
    logger.info("Sorted fights (newest first):")
    for date_obj, date_str, link in all_fights:
        logger.info(f"  {date_str} ({date_obj.strftime('%Y-%m-%d')}) - {link}")
    
    top_5 = all_fights[:MAX_FIGHTS]
    date_links = [(item[1], item[2]) for item in top_5]
    logger.info(f"Returning {len(date_links)} fight links after processing")
    
    # Log the final selected fights
    for date_str, link in date_links:
        logger.info(f"Selected fight: {date_str} - {link}")
        
    return date_links

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
    """Scrape totals from the fight page tables, restoring original working logic."""
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

                    ctrl_value = get_col(9)
                    sig_str_value = get_col(2)  # Get sig_str for percentage calculation
                    sig_landed, sig_attempted = parse_of(sig_str_value)
                    sig_pct = str(round((sig_landed / sig_attempted) * 100)) + "%" if sig_attempted > 0 else "0%"
                    return {
                        "kd": get_col(1),
                        "sig_str": get_col(2),
                        "sig_str_pct": sig_pct,  # Fixed percentage calculation
                        "total_str": get_col(4),
                        "takedowns": get_col(5),
                        "td_pct": get_col(6),  # Already correct, verified no +1% issue
                        "ctrl": "0:00" if ctrl_value == "--" else ctrl_value
                    }
    return None

def scrape_strike_details(soup: BeautifulSoup, db_fighter_name: str) -> dict:
    """Scrape head, body, and leg strikes from the significant strikes table."""
    head_landed, head_attempted = 0, 0
    body_landed, body_attempted = 0, 0
    leg_landed, leg_attempted = 0, 0
    matched_rows = 0

    tables = soup.select("table.b-fight-details__table")
    for tbl in tables:
        headers = [th.get_text(strip=True).lower() for th in tbl.select("thead th")]
        if "head" in headers and "body" in headers and "leg" in headers:
            tbody = tbl.find("tbody", class_="b-fight-details__table-body")
            if not tbody:
                continue
            rows = tbody.find_all("tr", class_="b-fight-details__table-row")
            for row in rows:
                cols = row.find_all("td", class_="b-fight-details__table-col")
                if len(cols) < 6:
                    continue
                p_tags = cols[0].find_all("p", class_="b-fight-details__table-text")
                if len(p_tags) != 2:
                    continue

                texts = [p.get_text(strip=True) for p in p_tags]
                idx, ratio = best_fighter_index(texts, db_fighter_name)
                if ratio < 0.3:
                    continue

                matched_rows += 1
                head_txt = safe_text(cols[3], idx)
                body_txt = safe_text(cols[4], idx)
                leg_txt = safe_text(cols[5], idx)
                hl, ha = parse_of(head_txt)
                bl, ba = parse_of(body_txt)
                ll, la = parse_of(leg_txt)
                head_landed += hl
                head_attempted += ha
                body_landed += bl
                body_attempted += ba
                leg_landed += ll
                leg_attempted += la

    if matched_rows == 0:
        return {"head_str": "0", "body_str": "0", "leg_str": "0"}

    head_str = f"{head_landed} of {head_attempted}" if head_attempted > 0 else "0"
    body_str = f"{body_landed} of {body_attempted}" if body_attempted > 0 else "0"
    leg_str = f"{leg_landed} of {leg_attempted}" if leg_attempted > 0 else "0"

    return {
        "head_str": head_str,
        "body_str": body_str,
        "leg_str": leg_str
    }

def scrape_sum_rounds(soup: BeautifulSoup, db_fighter_name: str) -> dict:
    """Fallback: Sum stats across round-by-round tables."""
    kd_sum = 0
    sig_x, sig_y = 0, 0
    tot_x, tot_y = 0, 0
    td_x, td_y = 0, 0
    td_pct_vals = []
    ctrl_sec = 0
    head_x, head_y = 0, 0
    body_x, body_y = 0, 0
    leg_x, leg_y = 0, 0
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
            td_txt = safe_text(cols[5], idx)
            tdx, tdy = parse_of(td_txt)
            td_x += tdx
            td_y += tdy
            td_pct_txt = safe_text(cols[6], idx)
            if td_pct_txt.endswith("%"):
                try:
                    td_pct_vals.append(float(td_pct_txt[:-1]))
                except:
                    pass
            c_txt = safe_text(cols[9], idx)
            ctrl_sec += mmss_to_seconds(c_txt)

            if len(cols) >= 13:  # Check for strike details in round-by-round
                h_txt = safe_text(cols[10], idx)
                b_txt = safe_text(cols[11], idx)
                l_txt = safe_text(cols[12], idx)
                hx, hy = parse_of(h_txt)
                bx, by = parse_of(b_txt)
                lx, ly = parse_of(l_txt)
                head_x += hx
                head_y += hy
                body_x += bx
                body_y += by
                leg_x += lx
                leg_y += ly

    if matched_rows == 0:
        return None

    kd_str = str(kd_sum)
    sig_str_str = f"{sig_x} of {sig_y}" if sig_y > 0 else "0"
    sig_str_pct_str = f"{round((sig_x / sig_y) * 100)}%" if sig_y > 0 else "0%"
    tot_str_str = f"{tot_x} of {tot_y}" if tot_y > 0 else "0"
    td_str = f"{td_x} of {td_y}" if td_y > 0 else "0"
    td_pct_str = f"{(sum(td_pct_vals) / len(td_pct_vals)):.0f}%" if td_pct_vals else "0%"
    ctrl_str = seconds_to_mmss(ctrl_sec)
    head_str_str = f"{head_x} of {head_y}" if head_y > 0 else "0"
    body_str_str = f"{body_x} of {body_y}" if body_y > 0 else "0"
    leg_str_str = f"{leg_x} of {leg_y}" if leg_y > 0 else "0"

    return {
        "kd": kd_str,
        "sig_str": sig_str_str,
        "sig_str_pct": sig_str_pct_str,
        "total_str": tot_str_str,
        "head_str": head_str_str,
        "body_str": body_str_str,
        "leg_str": leg_str_str,
        "takedowns": td_str,
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
# 7) PARSE RESULT, DATE, OPPONENT, EVENT, AND METHOD FROM FIGHT PAGE
###############################################################################
def parse_result_and_date(soup: BeautifulSoup, fighter_name: str) -> tuple:
    result = "N/A"
    fight_date = "N/A"
    opponent_name = "N/A"
    event_name = "N/A"
    method = "N/A"

    event_el = soup.select_one("h2.b-content__title a")
    if event_el:
        event_name = event_el.get_text(strip=True)
    logger.info(f"Extracted event: {event_name}")

    persons = soup.select("div.b-fight-details__person")
    if len(persons) >= 2:
        fighter1_name = persons[0].select_one("h3.b-fight-details__person-name a").get_text(strip=True)
        fighter2_name = persons[1].select_one("h3.b-fight-details__person-name a").get_text(strip=True)
        logger.info(f"Extracted fighter names: {fighter1_name}, {fighter2_name}")

        ratio1 = fuzzy_ratio(basic_clean(fighter_name), basic_clean(fighter1_name))
        ratio2 = fuzzy_ratio(basic_clean(fighter_name), basic_clean(fighter2_name))
        logger.info(f"Similarity ratios: {ratio1:.2f}, {ratio2:.2f}")

        if max(ratio1, ratio2) > 0.7:
            if ratio1 > ratio2:
                our_fighter_index = 0
                opponent_name = fighter2_name
            else:
                our_fighter_index = 1
                opponent_name = fighter1_name
        else:
            opponent_name = "Unknown"
            logger.warning("No clear fighter name match; setting opponent to 'Unknown'")

        if 'our_fighter_index' in locals():
            i_status = persons[our_fighter_index].select_one("i.b-fight-details__person-status")
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

    info_items = soup.select("li.b-fight-details__text-item")
    for item in info_items:
        txt = item.get_text(strip=True)
        if "Date:" in txt:
            fight_date = txt.split("Date:")[1].strip()
            break
    logger.info(f"Extracted fight date: {fight_date}")

    method_container = soup.find("i", class_="b-fight-details__text-item_first")
    if method_container:
        method_el = method_container.find("i", style="font-style: normal")
        if method_el:
            method = method_el.get_text(strip=True)
    logger.info(f"Extracted method: {method}")

    return (result, fight_date, opponent_name, event_name, method)

###############################################################################
# 8) SCRAPE FIGHT PAGE (with fallback date)
###############################################################################
def scrape_fight_page_for_fighter(fight_url: str, fighter_name: str, fallback_date: str) -> dict:
    """
    Return a dict of stats for one fight, combining totals and strike details.
    """
    data = {
        "fighter_name": fighter_name,
        "fight_url": fight_url,
        "kd": "0",
        "sig_str": "0",
        "sig_str_pct": "0",
        "total_str": "0",
        "head_str": "0",
        "body_str": "0",
        "leg_str": "0",
        "takedowns": "0",
        "td_pct": "0",
        "ctrl": "0:00",
        "result": "N/A",
        "method": "N/A",
        "opponent": "N/A",
        "fight_date": fallback_date,
        "event": "N/A"
    }
    resp = fetch_url_quietly(fight_url)
    if not resp:
        return data

    soup = BeautifulSoup(resp.text, "html.parser")
    res, page_date, opponent, event, method = parse_result_and_date(soup, fighter_name)

    data["result"] = res
    if page_date != "N/A":
        data["fight_date"] = page_date
        logger.info(f"Using fight date from page: {page_date}")
    else:
        data["fight_date"] = fallback_date
        logger.info(f"Using fallback fight date: {fallback_date}")
    data["method"] = method
    data["opponent"] = opponent
    data["event"] = event

    # Attempt single-totals (restored original logic)
    single_data = scrape_single_totals(soup, fighter_name)
    if single_data:
        data.update(single_data)
    else:
        # Fallback to round-by-round if totals fail
        round_data = scrape_sum_rounds(soup, fighter_name)
        if round_data:
            data.update(round_data)

    # Get strike details, prioritizing round-by-round summation if needed
    strike_data = scrape_strike_details(soup, fighter_name)
    if all(value == "0" for value in [strike_data["head_str"], strike_data["body_str"], strike_data["leg_str"]]):
        # Fallback to round-by-round summation for strikes if totals are all zeros
        round_strike_data = {"head_str": "0", "body_str": "0", "leg_str": "0"}
        round_data = scrape_sum_rounds(soup, fighter_name)
        if round_data:
            round_strike_data = {
                "head_str": round_data["head_str"],
                "body_str": round_data["body_str"],
                "leg_str": round_data["leg_str"]
            }
        strike_data = round_strike_data
    data.update(strike_data)

    return data

###############################################################################
# 9) STORE FIGHT DATA (Manual check for duplicates)
###############################################################################
def store_fight_data(row_data: dict) -> bool:
    """Store fight data in Supabase, maintaining proper ID ordering (newer fights get lower IDs)."""
    try:
        # Get current fights for this fighter
        current_fights = get_fighter_fights(row_data['fighter_name'])
        
        # Calculate the next ID based on the total number of fights in the table
        response = supabase.table("fighter_last_5_fights").select("id").execute()
        all_fights = response.data if response.data else []
        next_id = len(all_fights) + 1
        
        # Assign ID to the new fight
        row_data['id'] = next_id
        
        # If we already have 5 fights for this fighter, don't add more
        if len(current_fights) >= 5:
            logger.warning(f"Already have {len(current_fights)} fights for {row_data['fighter_name']}, skipping")
            return False
        
        # Ensure all required fields are present
        required_fields = [
            'fighter_name', 'fight_url', 'kd', 'sig_str', 'sig_str_pct',
            'total_str', 'head_str', 'body_str', 'leg_str', 'takedowns',
            'td_pct', 'ctrl', 'result', 'method', 'opponent', 'fight_date', 'event'
        ]
        
        for field in required_fields:
            if field not in row_data:
                row_data[field] = ''  # Set empty string for missing fields
                
        # Insert the fight data
        response = supabase.table("fighter_last_5_fights").insert(row_data).execute()
        
        success = len(response.data) > 0
        if success:
            logger.info(f"Stored fight for {row_data['fighter_name']} with ID {row_data['id']}")
        else:
            logger.warning(f"Failed to store fight for {row_data['fighter_name']}")
        return success
        
    except Exception as e:
        logger.error(f"Error storing fight data: {e}")
        return False

###############################################################################
# RECENT EVENT FUNCTIONS
###############################################################################
def fetch_most_recent_event(max_retries=RETRY_ATTEMPTS):
    """
    Fetch the most recent completed UFC event.
    Returns a tuple of (event_url, event_name, event_date) or None if failed.
    """
    logger.info(f"Fetching most recent event from {EVENT_URL}")
    
    for attempt in range(1, max_retries + 1):
        try:
            time.sleep(random.uniform(1.0, 2.0))
            session = requests.Session()
            session.headers.update(HEADERS)
            resp = session.get(EVENT_URL, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            
            soup = BeautifulSoup(resp.text, "html.parser")
            
            # Look for the events table - first completed events should be at the top
            event_tables = soup.find_all("table", class_="b-statistics__table")
            if not event_tables:
                logger.warning("Could not find main event table, trying alternative approach")
                event_tables = soup.find_all("table")
            
            if not event_tables:
                logger.error("Could not find any tables on the page")
                return None
            
            recent_events = []
            
            # Process each table to find recent events
            for table in event_tables:
                rows = table.find_all("tr")[1:]  # Skip header row
                
                for row in rows:
                    # Skip header rows
                    if "b-statistics__table-row_type_first" in row.get("class", []) or not row.find_all("td"):
                        continue
                    
                    # Find the event link
                    links = row.find_all("a", href=lambda href: href and href.startswith("http://ufcstats.com/event-details/"))
                    if not links:
                        continue
                    
                    event_url = links[0]["href"].strip()
                    event_name = links[0].get_text(strip=True)
                    
                    # Try to find the date
                    date_span = row.find("span", class_="b-statistics__date")
                    if not date_span:
                        # Try other ways to find date
                        date_cell = row.find_all("td")
                        if len(date_cell) > 1:  # Usually second cell has the date
                            date_text = date_cell[1].get_text(strip=True)
                            if re.search(r'\d{4}', date_text):  # If it contains a year
                                event_date = date_text
                            else:
                                continue  # Skip if no valid date found
                        else:
                            continue  # Skip if no date cell found
                    else:
                        event_date = date_span.get_text(strip=True)
                    
                    # Skip future events
                    try:
                        # Parse the date to check if it's in the future
                        date_parts = event_date.split()
                        if len(date_parts) >= 3:
                            month_str = date_parts[0].rstrip('.')
                            month_map = {
                                'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4,
                                'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8,
                                'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
                            }
                            month = month_map.get(month_str, 1)
                            
                            day = int(date_parts[1].rstrip(',').strip())
                            year = int(date_parts[2])
                            
                            event_date_obj = datetime(year, month, day)
                            now = datetime.now()
                            
                            # Skip future events
                            if event_date_obj > now:
                                logger.info(f"Skipping future event: {event_name} - {event_date}")
                                continue
                                
                            # Add this to our candidates
                            recent_events.append((event_url, event_name, event_date, event_date_obj))
                    except Exception as e:
                        logger.warning(f"Error parsing event date '{event_date}': {e}")
                        # Still add it as a candidate with a default old date
                        recent_events.append((event_url, event_name, event_date, datetime(2000, 1, 1)))
            
            # Sort events by date (newest first)
            if recent_events:
                recent_events.sort(key=lambda x: x[3], reverse=True)
                most_recent = recent_events[0]
                logger.info(f"Found most recent event: {most_recent[1]} - {most_recent[2]}")
                return (most_recent[0], most_recent[1], most_recent[2])
            
            logger.error("No recent events found after processing all tables")
            return None
                    
        except Exception as e:
            logger.error(f"Error fetching recent event (attempt {attempt}/{max_retries}): {e}")
            if attempt < max_retries:
                time.sleep(RETRY_SLEEP)
    
    logger.error(f"Failed to fetch recent event after {max_retries} attempts")
    return None

def extract_fighters_from_event(event_url, max_retries=RETRY_ATTEMPTS):
    """
    Extract all fighters from an event page.
    Returns a list of tuples (fighter_name, fighter_url).
    """
    logger.info(f"Extracting fighters from event: {event_url}")
    
    fighters = []
    
    for attempt in range(1, max_retries + 1):
        try:
            time.sleep(random.uniform(1.0, 2.0))
            resp = session.get(event_url, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            
            soup = BeautifulSoup(resp.text, "html.parser")
            
            # Find all table rows containing fighter details
            fight_tables = soup.find_all("tbody", class_="b-fight-details__table-body")
            
            if not fight_tables:
                logger.warning("Could not find fight tables, trying alternative approach")
                # Try to find any tables
                tables = soup.find_all("table")
                fight_tables = [table.find("tbody") for table in tables if table.find("tbody")]
            
            if not fight_tables:
                # Try directly finding the fighter links
                fighter_links = soup.find_all("a", href=lambda href: href and href.startswith("http://ufcstats.com/fighter-details/"))
                for link in fighter_links:
                    fighter_url = link["href"].strip()
                    fighter_name = link.get_text(strip=True)
                    if fighter_name and fighter_url and (fighter_name, fighter_url) not in fighters:
                        fighters.append((fighter_name, fighter_url))
                
                if fighters:
                    logger.info(f"Found {len(fighters)} fighters using direct link approach")
                    return fighters
                else:
                    logger.error("Could not find any fighter information")
                    return []
            
            # Process each fight table
            for tbody in fight_tables:
                rows = tbody.find_all("tr")
                
                for row in rows:
                    # Find all fighter links in this row
                    fighter_links = row.find_all("a", href=lambda href: href and href.startswith("http://ufcstats.com/fighter-details/"))
                    
                    for link in fighter_links:
                        fighter_url = link["href"].strip()
                        fighter_name = link.get_text(strip=True)
                        if fighter_name and fighter_url and (fighter_name, fighter_url) not in fighters:
                            fighters.append((fighter_name, fighter_url))
            
            logger.info(f"Found {len(fighters)} fighters")
            return fighters
            
        except requests.exceptions.Timeout as e:
            logger.warning(f"Timeout extracting fighters, attempt {attempt}/{max_retries}: {e}")
            if attempt < max_retries:
                wait_time = 2 ** attempt
                time.sleep(wait_time)
        except Exception as e:
            logger.error(f"Failed to extract fighters: {e}")
            if attempt < max_retries:
                wait_time = 2 ** attempt
                time.sleep(wait_time)
            else:
                return []
    
    return []

def get_fighter_latest_fights(fighter_url, fighter_name, max_fights=MAX_FIGHTS, recent_only=False):
    """
    Gets either all 5 fights or just the most recent fight from the fighter's page.
    recent_only: If True, returns only the most recent fight. If False, returns up to max_fights.
    """
    logger.info(f"Getting {'most recent fight' if recent_only else 'last 5 fights'} for {fighter_name} from {fighter_url}")
    
    fight_info = get_fight_links_top5(fighter_url)
    if not fight_info or len(fight_info) == 0:
        logger.warning(f"No fights found for {fighter_name}")
        return []
    
    # For recent mode, only take the first (most recent) fight
    if recent_only:
        fight_info = fight_info[:1]
    
    # Get fights (either just most recent or all 5)
    fight_data_list = []
    for row_date_str, link in fight_info:
        # Scrape each fight's data
        row_data = scrape_fight_page_for_fighter(link, fighter_name, row_date_str)
        logger.info(f"Retrieved fight data: {row_data.get('event')} vs {row_data.get('opponent')}")
        fight_data_list.append(row_data)
        
    return fight_data_list

def update_fighter_latest_fight(fighter_name, fighter_url, recent_only=False):
    """Get and store either all 5 or just most recent fight for a fighter"""
    try:
        if recent_only:
            logger.info(f"Updating most recent fight for {fighter_name}")
            # First get their existing fights from the database
            existing_fights = get_fighter_fights(fighter_name)
            logger.info(f"Found {len(existing_fights)} existing fights for {fighter_name}")
            
            # Get just their most recent fight
            new_fights = get_fighter_latest_fights(fighter_url, fighter_name, recent_only=True)
            if not new_fights:
                logger.warning(f"No new fights found for {fighter_name}")
                return False
            
            # Combine existing and new fights
            all_fights = existing_fights + new_fights
            
            # Sort all fights by date, newest first
            all_fights.sort(key=lambda x: datetime.strptime(x['fight_date'], '%b. %d, %Y' if '.' in x['fight_date'] else '%b %d, %Y'), reverse=True)
            
            # Take only the 5 most recent fights
            final_fights = all_fights[:MAX_FIGHTS]
            
            # Delete existing fights and store the new set
            delete_fighter_fights(fighter_name)
            logger.info(f"Deleted existing fights for {fighter_name}")
            
            # Store the final set of fights
            success_count = 0
            for fight_data in final_fights:
                if store_fight_data(fight_data):
                    success_count += 1
                else:
                    logger.warning(f"Failed to store fight vs {fight_data.get('opponent')} for {fighter_name}")
            
            logger.info(f"Stored {success_count}/{len(final_fights)} fights for {fighter_name}")
            return success_count > 0
            
        else:
            # All mode - just get and store all 5 fights
            logger.info(f"Getting all 5 fights for {fighter_name}")
            fight_data_list = get_fighter_latest_fights(fighter_url, fighter_name, recent_only=False)
            
            if not fight_data_list:
                logger.warning(f"No fights found for {fighter_name}")
                return False
                
            # Remove existing fights for this fighter
            delete_fighter_fights(fighter_name)
            logger.info(f"Deleted existing fights for {fighter_name}")
            
            # Store all fight data
            success_count = 0
            for fight_data in fight_data_list:
                if store_fight_data(fight_data):
                    success_count += 1
                else:
                    logger.warning(f"Failed to store fight vs {fight_data.get('opponent')} for {fighter_name}")
            
            logger.info(f"Stored {success_count}/{len(fight_data_list)} fights for {fighter_name}")
            return success_count > 0
        
    except Exception as e:
        logger.error(f"Error updating latest fights for {fighter_name}: {e}")
        import traceback
        traceback.print_exc()
        return False

def process_recent_event():
    """Process fighters from the most recent UFC event."""
    logger.info("Starting to process fighters from the most recent UFC event")
    
    # Fetch the most recent event
    latest_event = fetch_most_recent_event()
    if not latest_event:
        logger.error("Failed to fetch the most recent event")
        return
    
    # Unpack the tuple returned by fetch_most_recent_event
    event_url, event_name, event_date = latest_event
    
    logger.info(f"Processing event: {event_name} ({event_date}) - {event_url}")
    
    # Extract fighters from the event
    fighter_urls = extract_fighters_from_event(event_url)
    if not fighter_urls:
        logger.error("No fighters found in the most recent event")
        return
    
    logger.info(f"Found {len(fighter_urls)} fighters in the event")
    
    # Get existing fighter data to find fighter names from URLs
    try:
        # Get fighters from Supabase
        response = supabase.table("fighters").select("fighter_name, fighter_url").execute()
        fighters_dict = {}
        
        if response.data:
            fighters_dict = {row["fighter_url"]: row["fighter_name"] for row in response.data}
    except Exception as e:
        logger.error(f"Error getting fighters from Supabase: {e}")
        return
    
    # Process each fighter
    processed_count = 0
    
    logger.info(f"Starting to process {len(fighter_urls)} fighters' most recent fights...")
    for fighter_url in fighter_urls:
        # Get fighter name from the dictionary
        fighter_name = fighters_dict.get(fighter_url)
        if not fighter_name:
            logger.warning(f"Fighter URL {fighter_url} not found in database. Skipping.")
            continue
        
        # Update the fighter's most recent fight only
        updated = update_fighter_latest_fight(fighter_name, fighter_url, recent_only=True)
        
        if updated:
            processed_count += 1
            logger.info(f"Successfully updated most recent fight for {fighter_name}")
        else:
            logger.warning(f"Failed to update most recent fight for {fighter_name}")
    
    logger.info(f"Processed {processed_count}/{len(fighter_urls)} fighters from {event_name}")

###############################################################################
# MAIN
###############################################################################
def process_fighter(fighter_name: str, fighter_url: str, mode: str = 'all') -> bool:
    """Process a fighter's fights and update the database."""
    try:
        # Get all fights for the fighter
        fight_links = get_fight_links_top5(fighter_url)
        if not fight_links:
            logger.warning(f"No fights found for {fighter_name}")
            return False
            
        logger.info(f"Found {len(fight_links)} fights for {fighter_name}")
        
        # Delete existing fights first
        delete_fighter_fights(fighter_name)
        
        # Process each fight in order (newest first)
        success_count = 0
        for date_str, fight_url in fight_links:
            fight_data = scrape_fight_page_for_fighter(fight_url, fighter_name, date_str)
            if fight_data and store_fight_data(fight_data):
                success_count += 1
                
        logger.info(f"Successfully processed {success_count}/{len(fight_links)} fights for {fighter_name}")
        return success_count > 0
        
    except Exception as e:
        logger.error(f"Error processing fighter {fighter_name}: {e}")
        return False

def main() -> None:
    """Main function to run the scraper."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="UFC Last 5 Fights Scraper")
    parser.add_argument("--mode", choices=["all", "recent", "fighter"], default="recent",
                      help="Mode: all=all fighters (5 fights), recent=latest event (1 fight), fighter=specific fighter (5 fights)")
    parser.add_argument("--fighter", type=str, default=None,
                      help="Fighter name (required in fighter mode)")
    
    args = parser.parse_args()
    
    # Different modes
    if args.mode == "all":
        logger.info("Running in ALL mode (getting all 5 fights)")
        # First, truncate and reset the fighter_last_5_fights table
        logger.info("Truncating fighter_last_5_fights table and resetting ID sequence before processing")
        recreate_last5_table()
        
        # Get all fighters from database
        fighters = get_fighters_in_db_order()
        if not fighters:
            logger.error("No fighters found in database. Exiting.")
            return
        logger.info(f"Found {len(fighters)} fighters in database")
        
        # Process each fighter
        success_count = 0
        failure_count = 0
        for fighter_name, fighter_url in fighters:
            logger.info(f"Processing fighter: {fighter_name}")
            try:
                # Update all 5 fights for this fighter
                if process_fighter(fighter_name, fighter_url, mode='all'):
                    logger.info(f"Successfully updated {fighter_name}'s last 5 fights")
                    success_count += 1
                else:
                    logger.warning(f"Failed to update {fighter_name}'s last 5 fights")
                    failure_count += 1
            except Exception as e:
                logger.error(f"Error processing fighter {fighter_name}: {e}")
                failure_count += 1
        
        logger.info(f"All fighters processing completed. Success: {success_count}, Failed: {failure_count}")
        
    elif args.mode == "fighter":
        # Process a specific fighter (all 5 fights)
        if not args.fighter:
            logger.error("Fighter name is required in fighter mode. Exiting.")
            return
            
        fighter_name = args.fighter
        logger.info(f"Running in FIGHTER mode for {fighter_name} (getting all 5 fights)")
        
        # Get fighter from database
        fighter_response = supabase.table("fighters").select("*").eq("fighter_name", fighter_name).execute()
        if not fighter_response.data or len(fighter_response.data) == 0:
            logger.error(f"Fighter {fighter_name} not found in database. Exiting.")
            return
            
        fighter = fighter_response.data[0]
        fighter_url = fighter.get("fighter_url")
        
        if not fighter_url:
            logger.error(f"No URL found for fighter {fighter_name}. Exiting.")
            return
            
        # Update all 5 fights for this fighter
        if process_fighter(fighter_name, fighter_url, mode='all'):
            logger.info(f"Successfully updated {fighter_name}'s last 5 fights")
        else:
            logger.error(f"Failed to update {fighter_name}'s last 5 fights")
            
    else:  # recent mode
        # Process fighters from the most recent event (most recent fight only)
        process_recent_event()
        
    logger.info("Script execution completed!")

if __name__ == "__main__":
    main()