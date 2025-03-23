#!/usr/bin/env python
"""
Optimized Tapology Scraper with anti-blocking measures and multithreading.
Enhanced with adaptive throttling, comprehensive error handling, and ability to run continuously.
"""

import os
import sys
import logging
import requests
from bs4 import BeautifulSoup
from typing import Tuple, Optional, List, Dict
import time
import re
import sqlite3
from urllib.parse import quote_plus
from difflib import SequenceMatcher
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import json
import datetime
import argparse

# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Tapology Fighter Scraper')
    parser.add_argument('--reset', action='store_true', help='Reset progress and start from the beginning')
    parser.add_argument('--start-index', type=int, default=None, help='Start processing from a specific fighter index')
    return parser.parse_args()

# Fix import by correctly adding the project root to sys.path
# Get the absolute path to the project root directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
    print(f"Added {PROJECT_ROOT} to Python path")

# Now import from backend should work
from backend.api.database import get_db_connection

# Parse command line arguments
args = parse_args()

# Setup logging with timestamps
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("tapology_scraper.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# List of User-Agents to rotate
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Safari/605.1.15",
    "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/115.0",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/116.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36 Edg/116.0.1938.69",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36",
]

# Example proxy list (replace with your own working proxies)
PROXIES = [
    # {"http": "http://your_proxy:port", "https": "http://your_proxy:port"},
    # Add more proxies here or fetch from a proxy service
]

# Scraping configuration
CONFIG = {
    "base_delay": 3.0,  # Base delay between requests (increased from 2.0)
    "max_retries": 5,   # Increased max retries
    "backoff_factor": 2.0,
    "max_workers": 2,   # Limiting concurrent workers
    "request_timeout": 15,
    "throttle_threshold": 3,  # After this many consecutive failures, increase cooldown
    "max_cooldown": 1800,     # Maximum cooldown in seconds (30 minutes)
    "min_cooldown": 60,       # Minimum cooldown in seconds
    "session_requests_limit": 50,  # Create a new session after this many requests
    "session_duration_limit": 1800,  # Create a new session after this many seconds (30 min)
    "state_save_interval": 20,  # Save state every N fighters processed
}

# Lock for thread-safe DB updates
db_lock = Lock()

# Tracking for adaptive throttling
request_stats = {
    "consecutive_failures": 0,
    "total_requests": 0,
    "successful_requests": 0,
    "session_start_time": time.time(),
    "session_request_count": 0,
    "progress_file": "tapology_scraper_progress.json",
    "last_fighter_index": 0,
    "last_save_time": time.time()
}

# If reset flag is set, delete the progress file
if args.reset and os.path.exists(request_stats["progress_file"]):
    try:
        os.remove(request_stats["progress_file"])
        logger.info(f"Reset progress: Deleted {request_stats['progress_file']}")
    except Exception as e:
        logger.error(f"Failed to delete progress file: {e}")

def get_random_headers() -> dict:
    """Returns headers with a random User-Agent."""
    return {"User-Agent": random.choice(USER_AGENTS)}

def get_random_proxy() -> Optional[dict]:
    """Returns a random proxy if available, else None."""
    return random.choice(PROXIES) if PROXIES else None

def adaptive_cooldown() -> float:
    """Calculate cooldown time based on failure patterns."""
    base = CONFIG["min_cooldown"]
    if request_stats["consecutive_failures"] > CONFIG["throttle_threshold"]:
        # Exponential backoff based on consecutive failures
        backoff_multiplier = min(2 ** (request_stats["consecutive_failures"] - CONFIG["throttle_threshold"]), 
                               CONFIG["max_cooldown"] / base)
        cooldown = base * backoff_multiplier
        return min(cooldown, CONFIG["max_cooldown"])
    return base + random.uniform(0, 30)  # Base cooldown with some randomness

def should_rotate_session() -> bool:
    """Determine if we should start a new session based on request count and duration."""
    session_duration = time.time() - request_stats["session_start_time"]
    return (request_stats["session_request_count"] >= CONFIG["session_requests_limit"] or 
            session_duration >= CONFIG["session_duration_limit"])

def rotate_session():
    """Reset session counters and apply a cooldown before starting a new session."""
    cooldown = adaptive_cooldown()
    logger.info(f"Rotating session after {request_stats['session_request_count']} requests. "
                f"Cooling down for {cooldown:.2f} seconds...")
    
    # Save progress before cooldown
    save_progress()
    
    time.sleep(cooldown)
    request_stats["session_start_time"] = time.time()
    request_stats["session_request_count"] = 0
    request_stats["consecutive_failures"] = max(0, request_stats["consecutive_failures"] - 1)  # Reduce failure count

def get_with_retries(url: str, max_retries: int = None, backoff_factor: float = None) -> Optional[requests.Response]:
    """Makes a GET request with retries, exponential backoff, and adaptive delay."""
    if max_retries is None:
        max_retries = CONFIG["max_retries"]
    if backoff_factor is None:
        backoff_factor = CONFIG["backoff_factor"]
    
    # Check if we should rotate session before new request
    if should_rotate_session():
        rotate_session()
    
    attempt = 0
    while attempt < max_retries:
        # Use a random delay variation to appear more human-like
        delay_variation = random.uniform(0.5, 1.5)
        current_delay = CONFIG["base_delay"] * delay_variation
        
        headers = get_random_headers()
        proxies = get_random_proxy()
        request_stats["total_requests"] += 1
        request_stats["session_request_count"] += 1
        
        try:
            response = requests.get(url, headers=headers, proxies=proxies, timeout=CONFIG["request_timeout"])
            response.raise_for_status()
            
            # Success! Reset consecutive failures counter
            request_stats["consecutive_failures"] = 0
            request_stats["successful_requests"] += 1
            
            # Add a random delay between requests
            sleep_time = current_delay + random.uniform(0, 2)
            logger.debug(f"Request successful, sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
            return response
        
        except requests.exceptions.HTTPError as e:
            request_stats["consecutive_failures"] += 1
            status_code = e.response.status_code if hasattr(e, 'response') else "Unknown"
            
            if status_code == 503 or status_code == 429:
                # Rate limiting or service unavailable
                cooldown = backoff_factor * (2 ** attempt) + random.uniform(10, 30)
                cooldown = min(cooldown, CONFIG["max_cooldown"])
                logger.warning(f"Rate limit error ({status_code}) for {url}, attempt {attempt + 1}/{max_retries}. "
                              f"Cooling down for {cooldown:.2f} seconds...")
                time.sleep(cooldown)
                attempt += 1
            else:
                logger.error(f"HTTP error {status_code} for {url}: {e}")
                # For other HTTP errors, try again with a shorter delay
                time.sleep(current_delay * 2)
                attempt += 1
                
        except requests.exceptions.ConnectionError as e:
            request_stats["consecutive_failures"] += 1
            logger.warning(f"Connection error for {url}: {e}. Waiting before retry...")
            time.sleep(current_delay * 3)
            attempt += 1
            
        except requests.exceptions.Timeout as e:
            request_stats["consecutive_failures"] += 1
            logger.warning(f"Timeout for {url}: {e}. Waiting before retry...")
            time.sleep(current_delay * 2)
            attempt += 1
            
        except Exception as e:
            request_stats["consecutive_failures"] += 1
            logger.error(f"Unexpected error for {url}: {e}")
            time.sleep(current_delay * 2)
            attempt += 1
    
    # All retries failed - apply adaptive cooldown
    cooldown = adaptive_cooldown()
    logger.error(f"All {max_retries} retries exhausted for {url}. Cooling down for {cooldown:.2f} seconds...")
    time.sleep(cooldown)
    return None

def save_progress():
    """Save current progress to allow resuming later."""
    state = {
        "last_fighter_index": request_stats["last_fighter_index"],
        "total_requests": request_stats["total_requests"],
        "successful_requests": request_stats["successful_requests"],
        "consecutive_failures": request_stats["consecutive_failures"],
        "timestamp": datetime.datetime.now().isoformat()
    }
    try:
        with open(request_stats["progress_file"], 'w') as f:
            json.dump(state, f)
        logger.info(f"Progress saved: {state['last_fighter_index']} fighters processed, "
                  f"{state['successful_requests']}/{state['total_requests']} successful requests")
        request_stats["last_save_time"] = time.time()
    except Exception as e:
        logger.error(f"Failed to save progress: {e}")

def load_progress() -> int:
    """Load progress from file. Returns the last processed fighter index."""
    try:
        if os.path.exists(request_stats["progress_file"]):
            with open(request_stats["progress_file"], 'r') as f:
                state = json.load(f)
            
            # Check if we've reached the end in a previous run
            last_index = state.get("last_fighter_index", 0)
            if last_index >= request_stats.get("total_fighters", float('inf')):
                logger.info("Previous run completed all fighters. Starting from the beginning.")
                return 0
                
            request_stats["last_fighter_index"] = last_index
            request_stats["total_requests"] = state.get("total_requests", 0)
            request_stats["successful_requests"] = state.get("successful_requests", 0)
            logger.info(f"Loaded progress: resuming from fighter #{request_stats['last_fighter_index']}, "
                      f"{state.get('successful_requests', 0)}/{state.get('total_requests', 0)} successful requests")
            return request_stats["last_fighter_index"]
        return 0
    except Exception as e:
        logger.error(f"Failed to load progress, starting from beginning: {e}")
        return 0

def ensure_tap_link_column():
    """Adds tap_link column if it doesn't exist."""
    conn = get_db_connection()
    cur = conn.cursor()
    columns = cur.execute("PRAGMA table_info(fighters)").fetchall()
    if "tap_link" not in [col[1] for col in columns]:
        cur.execute("ALTER TABLE fighters ADD COLUMN tap_link TEXT;")
        conn.commit()
        logger.info("Added 'tap_link' column.")
    conn.close()

def remove_nicknames_and_extras(name: str) -> str:
    """
    Removes any double-quoted or single-quoted nicknames, parentheses,
    and trailing text like "| MMA Fighter Page" from the name.
    """
    name = re.sub(r'"[^"]+"', '', name)
    name = re.sub(r"'[^']+'", '', name)
    name = re.sub(r'\([^)]*\)', '', name)
    name = re.sub(r'\|\s*(MMA\s+)?Fighter Page', '', name)
    return re.sub(r'\s+', ' ', name).strip()

def process_fighter_name(raw_name: str) -> str:
    """Processes the fighter name by cleaning nicknames and extra text."""
    return remove_nicknames_and_extras(raw_name)

def standardize_name(name: str) -> str:
    """Standardize a name by removing extra spaces, handling special characters."""
    if not name:
        return ""
    # Remove extra spaces, standardize to single space
    name = re.sub(r'\s+', ' ', name.strip())
    return name

def calculate_similarity(db_name: str, scraped_name: str) -> float:
    """Calculates the similarity ratio between two names using SequenceMatcher."""
    cleaned_db_name = process_fighter_name(db_name)
    cleaned_scraped_name = process_fighter_name(scraped_name)
    
    # Handle East Asian names where name order might be reversed
    if looks_like_east_asian_name(cleaned_db_name) or looks_like_east_asian_name(cleaned_scraped_name):
        # Try both name orders and use the highest similarity
        db_parts = cleaned_db_name.split()
        scraped_parts = cleaned_scraped_name.split()
        
        # Reverse name if it has at least 2 parts
        if len(db_parts) >= 2:
            reversed_db_name = ' '.join(db_parts[::-1])  # Reverse the order
            direct_sim = SequenceMatcher(None, cleaned_db_name, cleaned_scraped_name).ratio()
            reversed_sim = SequenceMatcher(None, reversed_db_name, cleaned_scraped_name).ratio()
            return max(direct_sim, reversed_sim)
    
    return SequenceMatcher(None, cleaned_db_name, cleaned_scraped_name).ratio()

def looks_like_east_asian_name(name: str) -> bool:
    """Check if a name appears to be of East Asian origin (Chinese, Korean, Japanese)."""
    # Common East Asian surname indicators
    east_asian_surnames = [
        'zhang', 'wang', 'li', 'liu', 'chen', 'yang', 'huang', 'zhao', 'wu', 'zhou',  # Chinese
        'kim', 'lee', 'park', 'choi', 'jung', 'kang', 'cho', 'yoon', 'jang', 'lim',   # Korean
        'sato', 'suzuki', 'takahashi', 'tanaka', 'watanabe', 'ito', 'yamamoto',       # Japanese
        'nakamura', 'kobayashi', 'kato', 'yoshida', 'yamada', 'sasaki', 'yamaguchi', 
        'matsumoto', 'inoue', 'kimura', 'hayashi', 'saito', 'nakajima', 'ikeda'
    ]
    
    # Check for common East Asian surnames
    name_lower = name.lower()
    for surname in east_asian_surnames:
        if name_lower.startswith(surname + ' ') or name_lower.endswith(' ' + surname):
            return True
    
    # Check for single/double syllable names common in East Asian formats (like Xi Jin)
    name_parts = name.split()
    if len(name_parts) == 2 and all(len(part) <= 4 for part in name_parts):
        return True
        
    return False

def tapology_search_url(fighter_name: str) -> str:
    """Constructs Tapology search URL."""
    return f"https://www.tapology.com/search?term={quote_plus(fighter_name)}"

def search_tapology_for_fighter(fighter_name: str) -> str:
    """Searches Tapology for fighter link."""
    url = tapology_search_url(fighter_name)
    logger.info(f"Searching Tapology for '{fighter_name}'")
    response = get_with_retries(url)
    if not response:
        return ""
    soup = BeautifulSoup(response.text, "html.parser")
    result = soup.find("a", href=lambda href: href and "/fighters/" in href)
    return "https://www.tapology.com" + result['href'] if result else ""

def is_valid_fighter_image(image_url: str) -> bool:
    """
    Validates if the URL is likely to be a fighter image and not a logo/icon.
    Prioritizes letterbox images and rejects headshot images.
    
    Returns:
        bool: True if the URL seems to be a valid fighter image, False otherwise.
    """
    if not image_url:
        return False
        
    # PRIORITY: Letterbox images are what we want
    if "letterbox_images" in image_url.lower() and "default" in image_url.lower():
        return True
        
    # REJECT: Explicitly reject headshot images and tiny images
    if "headshot_images" in image_url.lower() or "tiny" in image_url.lower():
        return False
    
    # Common patterns for fighter images on Tapology
    valid_patterns = [
        "letterbox_images",
        "profile_images",
        "fighter_images"
    ]
    
    # Patterns that indicate the image is not a fighter image
    invalid_patterns = [
        "headshot_images",
        "logo_squares",
        "icon_",
        "logo-square",
        "placeholder",
        "default.png",
        "icons/",
        "default-profile",
        "tiny"
    ]
    
    # Check if URL matches any valid pattern
    is_valid = any(pattern in image_url.lower() for pattern in valid_patterns)
    
    # Check if URL matches any invalid pattern
    is_invalid = any(pattern in image_url.lower() for pattern in invalid_patterns)
    
    # If it specifically matches a valid pattern, it's more likely to be correct
    if is_valid and not is_invalid:
        return True
    
    # If it matches an invalid pattern, it's likely not a fighter image
    if is_invalid:
        return False
    
    return False  # Default to False to be more selective

def scrape_tapology_fighter_page(url: str) -> Tuple[Optional[str], Optional[str]]:
    """Scrapes fighter page for name and image URL."""
    logger.info(f"Scraping fighter page: {url}")
    response = get_with_retries(url)
    if not response:
        return None, None
    soup = BeautifulSoup(response.text, "html.parser")

    h1 = soup.select_one("h1.pageTitle, h1.fightCenterHeaderTitle, div.fighterHeader h1")
    fighter_name = h1.get_text(strip=True) if h1 else None
    if not fighter_name:
        title = soup.find("title")
        fighter_name = title.get_text(strip=True).split(" | ")[0] if title else None

    # COMPLETELY REVISED image finding logic to prioritize letterbox images
    image_url = None
    
    # First priority: Look specifically for letterbox images
    letterbox_images = soup.find_all("img", src=lambda s: s and "letterbox_images" in s.lower() and "default" in s.lower())
    if letterbox_images:
        image_url = letterbox_images[0].get("src", "").strip()
        logger.info(f"Found letterbox image for {fighter_name}: {image_url}")
        return fighter_name, image_url
    
    # Second priority: Look for fighter profile images from the main content
    profile_image = soup.select_one("div.fighterImage img, div.fighterImageLarge img, div.details-content img.fighterImage")
    if profile_image and "letterbox_images" in profile_image.get("src", "").lower():
        image_url = profile_image.get("src", "").strip()
        logger.info(f"Found profile image for {fighter_name}: {image_url}")
        return fighter_name, image_url
    
    # Third priority: Look for any image in the fighter gallery or content area
    content_images = soup.select("div.details-content img, div.fighterImageGallery img")
    for img in content_images:
        src = img.get("src", "").strip()
        if src and "letterbox_images" in src.lower():
            image_url = src
            logger.info(f"Found content image for {fighter_name}: {image_url}")
            return fighter_name, image_url
    
    # Fourth priority: Check all images on the page
    all_images = soup.find_all("img", src=lambda s: s and s.strip())
    for img in all_images:
        src = img.get("src", "").strip()
        if is_valid_fighter_image(src):
            image_url = src
            logger.info(f"Found valid image for {fighter_name}: {image_url}")
            return fighter_name, image_url
    
    logger.warning(f"Failed to find valid letterbox image for {fighter_name}")
    return fighter_name, None

def update_fighter_in_db(db_name: str, new_image_url: str, tapology_url: str):
    """Updates fighter record in DB with thread safety."""
    with db_lock:
        conn = get_db_connection()
        cur = conn.cursor()
        try:
            cur.execute(
                "UPDATE fighters SET image_url = ?, tap_link = ? WHERE fighter_name = ?",
                (new_image_url, tapology_url, db_name)
            )
            conn.commit()
            logger.info(f"Updated DB for {db_name}")
        except sqlite3.Error as e:
            logger.error(f"DB update error for {db_name}: {e}")
        finally:
            conn.close()

def process_fighter(fighter_data: Tuple[str, Optional[str], Optional[str]]) -> bool:
    """Processes a single fighter and returns True if updated."""
    fighter_name, image_url, tap_link = fighter_data
    # Only skip if the fighter has BOTH a Tapology image AND a Tapology link
    if (image_url and "tapology" in image_url.lower() and "cbrimages" not in image_url.lower() and tap_link and "tapology" in tap_link.lower()):
        logger.info(f"Skipping '{fighter_name}' - already has Tapology data (image_url: {image_url}, tap_link: {tap_link})")
        return False

    # Try to search for fighter link on Tapology
    try:
        link = search_tapology_for_fighter(fighter_name)
        if not link:
            logger.warning(f"No Tapology link found for '{fighter_name}'")
            # Shorter delay for no results than for errors
            time.sleep(random.uniform(10, 20))
            return False

        # Add random delay to look more human-like
        time.sleep(random.uniform(2, 5))

        # Try to scrape fighter page
        scraped_name, new_image_url = scrape_tapology_fighter_page(link)
        if not scraped_name or not new_image_url:
            logger.warning(f"Failed to extract name or image for '{fighter_name}' from {link}")
            time.sleep(random.uniform(10, 15))
            return False

        # Check name similarity - lowered threshold to 0.6 to handle first/last name order differences
        similarity = calculate_similarity(fighter_name, scraped_name)
        
        # Special check for name order differences (e.g., "First Last" vs "Last First")
        db_name_parts = process_fighter_name(fighter_name).split()
        scraped_name_parts = process_fighter_name(scraped_name).split()
        
        # If we have the same name parts but in different order, consider it a match
        name_parts_match = False
        if len(db_name_parts) >= 2 and len(scraped_name_parts) >= 2:
            # Check if the sets of name parts match regardless of order
            if set(db_name_parts) == set(scraped_name_parts):
                name_parts_match = True
                logger.info(f"Name parts match despite different order: '{fighter_name}' and '{scraped_name}'")
            
            # Special handling for East Asian names where the order might be flipped
            elif looks_like_east_asian_name(fighter_name) or looks_like_east_asian_name(scraped_name):
                # For East Asian names, we'll be more lenient with name order
                # Check if reversing the name order would make it a better match
                if similarity >= 0.4:  # Lower threshold for East Asian names due to order variations
                    logger.info(f"East Asian name format detected with similarity {similarity*100:.2f}%: '{fighter_name}' and '{scraped_name}'")
                    name_parts_match = True
        
        if similarity >= 0.6 or name_parts_match:
            # Verify image before updating
            if not is_valid_fighter_image(new_image_url):
                logger.warning(f"Found invalid image for {fighter_name}: {new_image_url}")
                # Try to re-scrape a better image
                time.sleep(random.uniform(3, 7))
                _, better_image = scrape_tapology_fighter_page(link)
                if better_image and is_valid_fighter_image(better_image):
                    new_image_url = better_image
                    logger.info(f"Found better image on second attempt: {new_image_url}")
            
            # Update database with the fighter info
            update_fighter_in_db(fighter_name, new_image_url, link)
            logger.info(f"Matched '{fighter_name}' with similarity {similarity*100:.2f}% to '{scraped_name}'")
            return True
        else:
            logger.info(f"Name mismatch (similarity {similarity*100:.2f}%): DB='{process_fighter_name(fighter_name)}' vs. Scraped='{process_fighter_name(scraped_name)}'")
            time.sleep(random.uniform(10, 15))
            return False
            
    except Exception as e:
        logger.error(f"Error processing fighter '{fighter_name}': {e}")
        time.sleep(random.uniform(10, 20))
        return False

def scrape_for_db():
    """Main scraping function with multithreading and resume capability."""
    ensure_tap_link_column()
    
    # Get all fighters from database
    conn = get_db_connection()
    cur = conn.cursor()
    fighters = [(row[0], row[1], row[2]) for row in cur.execute("SELECT fighter_name, image_url, tap_link FROM fighters").fetchall()]
    conn.close()

    if not fighters:
        logger.info("No fighters found in DB.")
        return
        
    # Store total fighters count for reference
    request_stats["total_fighters"] = len(fighters)

    # If a specific start index is provided via command line, use it
    if args.start_index is not None:
        if 0 <= args.start_index < len(fighters):
            start_index = args.start_index
            logger.info(f"Using provided start index: {start_index}")
        else:
            logger.warning(f"Provided start index {args.start_index} is out of range (0-{len(fighters)-1}). Using default.")
            start_index = load_progress()
    else:
        # Load progress to resume from where we left off
        start_index = load_progress()
    
    # Restart from beginning if all fighters were processed in a previous run
    if start_index >= len(fighters):
        logger.info("All fighters were processed in a previous run. Restarting from the beginning.")
        start_index = 0
        
    fighters_to_process = fighters[start_index:]
    
    logger.info(f"Starting processing at index {start_index} ({len(fighters_to_process)} fighters remaining of {len(fighters)} total)")
    
    # Fix: Ensure max_workers is at least 1
    max_workers = max(1, min(CONFIG["max_workers"], len(fighters_to_process)))
    updated_count = 0
    processed_count = 0

    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Instead of processing all at once, batch the work
            batch_size = 10  # Process small batches
            for batch_start in range(0, len(fighters_to_process), batch_size):
                batch_end = min(batch_start + batch_size, len(fighters_to_process))
                batch = fighters_to_process[batch_start:batch_end]
                
                logger.info(f"Processing batch of {len(batch)} fighters (overall progress: {start_index + processed_count}/{len(fighters)} - {((start_index + processed_count) / len(fighters) * 100):.1f}%)")
                
                # Process batch in parallel
                future_to_fighter = {executor.submit(process_fighter, fighter): fighter[0] for fighter in batch}
                for future in as_completed(future_to_fighter):
                    fighter_name = future_to_fighter[future]
                    try:
                        if future.result():
                            updated_count += 1
                        processed_count += 1
                        
                        # Update progress
                        request_stats["last_fighter_index"] = start_index + processed_count
                        
                        # Save progress periodically
                        if processed_count % CONFIG["state_save_interval"] == 0 or (time.time() - request_stats["last_save_time"] > 300):
                            save_progress()
                            
                    except Exception as e:
                        logger.error(f"Error handling result for fighter '{fighter_name}': {e}")
                
                # Add a cooldown period between batches
                cooldown = random.uniform(5, 15)
                logger.info(f"Batch complete. Cooling down for {cooldown:.1f} seconds before next batch...")
                time.sleep(cooldown)
    
    except KeyboardInterrupt:
        logger.info("Scraping interrupted by user. Saving progress...")
        save_progress()
        logger.info(f"Processed {processed_count} fighters, updated {updated_count} before interruption.")
        return
    except Exception as e:
        logger.error(f"Unexpected error in main scraping loop: {e}")
        save_progress()
        logger.info(f"Processed {processed_count} fighters, updated {updated_count} before error.")
        return
    
    # Final progress save
    save_progress()
    logger.info(f"Scraping complete. Processed {processed_count} fighters, updated {updated_count}.")

def main():
    """Main entry point with comprehensive error handling."""
    logger.info("Starting Tapology scraper with continuous operation mode")
    try:
        scrape_for_db()
    except KeyboardInterrupt:
        logger.info("Scraper stopped by user")
    except Exception as e:
        logger.error(f"Uncaught exception in main: {e}")
    finally:
        logger.info(f"Scraper finished. Stats: {request_stats['successful_requests']}/{request_stats['total_requests']} successful requests")

if __name__ == "__main__":
    main()