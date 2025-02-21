import requests
from bs4 import BeautifulSoup
import pandas as pd
import concurrent.futures
import string
import re
import time
import random
import sqlite3
import os

# Use a persistent session for faster HTTP requests
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/91.0.4472.124 Safari/537.36"
    )
}
session = requests.Session()
session.headers.update(HEADERS)

MAX_WORKERS = 10
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Gets the script's directory
DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../Api/ufc_fighters.db"))

def clean_record_text(rec_txt):
    rec_txt = rec_txt.replace("Record:", "").strip()
    rec_txt = re.sub(r'-\d{4,}$', '', rec_txt).strip()
    return rec_txt

def parse_stats_block(ul_element, stats_dict):
    li_items = ul_element.find_all("li")
    for li in li_items:
        text = li.get_text(separator=" ", strip=True)
        if ":" in text:
            label, value = text.split(":", 1)
            label = label.strip()
            value = value.strip() if value.strip() else "N/A"
            if value == "--":
                value = "N/A"
            if label in stats_dict:
                stats_dict[label] = value

def fetch_listing_page(letter):
    url = f"http://ufcstats.com/statistics/fighters?char={letter}&page=all"
    print(f"[INFO] Fetch listing for '{letter}' -> {url}")
    try:
        time.sleep(random.uniform(0.5, 1.0))
        resp = session.get(url, timeout=15)
        resp.raise_for_status()
        return (letter, resp.text)
    except Exception as e:
        print(f"[ERROR] Could not fetch listing for '{letter}': {e}")
        return (letter, None)

def parse_listing_page(html):
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", class_="b-statistics__table")
    if not table:
        print("[WARN] No table found on listing page.")
        return []
    rows = table.find_all("tr")[1:]  # skip header row
    fighters = []
    for row in rows:
        cols = row.find_all("td")
        if len(cols) < 3:
            continue
        first_name = cols[0].get_text(strip=True)
        last_name = cols[1].get_text(strip=True)
        nickname = cols[2].get_text(strip=True)
        link_tag = cols[0].find("a")
        fighter_url = link_tag["href"].strip() if link_tag else ""
        if nickname:
            fighter_name = f'{first_name} "{nickname}" {last_name}'
        else:
            fighter_name = f"{first_name} {last_name}"
        if fighter_url:
            fighters.append({
                "fighter_name": fighter_name,
                "fighter_url": fighter_url
            })
    return fighters

def scrape_fighter_detail(fighter_url):
    """
    Parse personal stats from a fighter detail page.
    """
    desired_stats = {
        "Record": "N/A",
        "Height": "N/A",
        "Weight": "N/A",
        "Reach": "N/A",
        "STANCE": "N/A",
        "DOB": "N/A",
        "SLpM": "N/A",
        "Str. Acc.": "N/A",
        "SApM": "N/A",
        "Str. Def": "N/A",
        "TD Avg.": "N/A",
        "TD Acc.": "N/A",
        "TD Def.": "N/A",
        "Sub. Avg.": "N/A"
    }
    try:
        time.sleep(random.uniform(0.5, 1.0))
        resp = session.get(fighter_url, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        # 1) record
        rec_span = soup.find("span", class_="b-content__title-record")
        if rec_span:
            raw_text = rec_span.get_text(strip=True)
            cleaned = clean_record_text(raw_text)
            desired_stats["Record"] = cleaned if cleaned else "N/A"
        # 2) parse <ul> blocks
        blocks = soup.find_all("ul", class_="b-list__box-list")
        for blk in blocks:
            parse_stats_block(blk, desired_stats)
        return desired_stats
    except Exception as e:
        print(f"[ERROR] Could not fetch fighter detail: {fighter_url} -> {e}")
        return desired_stats

def process_fighter(fighter_name, fighter_url):
    """
    concurrency step: parse listing data + fighter detail page
    """
    personal_stats = scrape_fighter_detail(fighter_url)
    row = {
        "fighter_name": fighter_name,
        "fighter_url": fighter_url
    }
    row.update(personal_stats)
    return row

def parse_fighter_name(fighter_name):
    no_quotes = fighter_name.replace('"', "")
    parts = no_quotes.split()
    if len(parts) == 1:
        return ("", parts[0])
    elif len(parts) == 2:
        return (parts[0], parts[1])
    else:
        return (parts[0], parts[-1])

def main():
    all_rows = []
    letters = list(string.ascii_lowercase)

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        listing_futures = {
            executor.submit(fetch_listing_page, letter): letter
            for letter in letters
        }
        for future in concurrent.futures.as_completed(listing_futures):
            letter = listing_futures[future]
            listing_result = future.result()
            if not listing_result:
                continue
            _, html = listing_result
            if not html:
                continue
            fighters = parse_listing_page(html)
            print(f"[INFO] letter='{letter}' -> {len(fighters)} fighters")
            # concurrency for detail pages
            detail_futs = {
                executor.submit(process_fighter, f["fighter_name"], f["fighter_url"]): f["fighter_name"]
                for f in fighters
            }
            for dfut in concurrent.futures.as_completed(detail_futs):
                row_data = dfut.result()
                all_rows.append(row_data)

    df = pd.DataFrame(all_rows)
    # sort
    df["first_name_temp"] = ""
    df["last_name_temp"] = ""
    for idx, row in df.iterrows():
        fname, lname = parse_fighter_name(row["fighter_name"])
        df.at[idx, "first_name_temp"] = fname
        df.at[idx, "last_name_temp"] = lname
    df.sort_values(by=["last_name_temp", "first_name_temp"], inplace=True)
    df.drop(columns=["first_name_temp", "last_name_temp"], inplace=True)

    # reorder columns
    # we'll put fighter_url last, record next to fighter_name
    columns = [
        "fighter_name", "Record",
        "Height", "Weight", "Reach", "STANCE", "DOB",
        "SLpM", "Str. Acc.", "SApM", "Str. Def",
        "TD Avg.", "TD Acc.", "TD Def.", "Sub. Avg.",
        "fighter_url"
    ]
    for c in columns:
        if c not in df.columns:
            df[c] = "N/A"
    df = df[columns]

    # store in sqlite
    try:
        conn = sqlite3.connect(DB_PATH)
        df.to_sql("fighters", conn, if_exists="replace", index=False)
        conn.close()
        print(f"[DONE] Inserted {len(df)} fighters (with personal stats) into 'fighters' table in '{DB_PATH}'")
    except Exception as e:
        print(f"[ERROR] Could not write to database: {e}")

if __name__=="__main__":
    main()
