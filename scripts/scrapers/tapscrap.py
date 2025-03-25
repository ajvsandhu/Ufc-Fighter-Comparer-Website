#!/usr/bin/env python
"""
Optimized Tapology Scraper with anti-blocking measures and multithreading.

This module provides functionality to scrape fighter data from Tapology while
implementing various measures to avoid blocking, including:
- Adaptive throttling
- Comprehensive error handling
- Multithreading support
- Session rotation
- Progress tracking and resumption
"""

import argparse
import datetime
import json
import logging
import os
import random
import re
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher
from threading import Lock
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote_plus

import requests
from bs4 import BeautifulSoup

def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description='Tapology Fighter Scraper')
    parser.add_argument(
        '--reset',
        action='store_true',
        help='Reset progress and start from the beginning'
    )
    parser.add_argument(
        '--start-index',
        type=int,
        default=None,
        help='Start processing from a specific fighter index'
    )
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
    ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
     "(KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"),
    ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
     "(KHTML, like Gecko) Version/16.0 Safari/605.1.15"),
    ("Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:109.0) "
     "Gecko/20100101 Firefox/115.0"),
    ("Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) AppleWebKit/605.1.15 "
     "(KHTML, like Gecko) Version/16.0 Mobile/15E148 Safari/604.1"),
    ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
     "(KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"),
    ("Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) "
     "Gecko/20100101 Firefox/116.0"),
    ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
     "(KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36 Edg/116.0.1938.69"),
    ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
     "(KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"),
    ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
     "(KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"),
]

# Example proxy list (replace with your own working proxies)
PROXIES: List[Dict[str, str]] = [
    # {"http": "http://your_proxy:port", "https": "http://your_proxy:port"},
    # Add more proxies here or fetch from a proxy service
]

# Scraping configuration
CONFIG = {
    "base_delay": 3.0,           # Base delay between requests
    "max_retries": 5,            # Maximum number of retry attempts
    "backoff_factor": 2.0,       # Factor to increase delay between retries
    "max_workers": 2,            # Maximum number of concurrent workers
    "request_timeout": 15,       # Request timeout in seconds
    "throttle_threshold": 3,     # Failures before increasing cooldown
    "max_cooldown": 1800,        # Maximum cooldown in seconds (30 minutes)
    "min_cooldown": 60,          # Minimum cooldown in seconds
    "session_requests_limit": 50, # Requests before creating new session
    "session_duration_limit": 1800,  # Session duration limit in seconds (30 min)
    "state_save_interval": 20,   # Save state every N fighters processed
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

def get_random_headers() -> Dict[str, str]:
    """
    Get a random User-Agent header.
    
    Returns:
        Dict[str, str]: Headers dictionary with random User-Agent
    """
    return {"User-Agent": random.choice(USER_AGENTS)}

def get_random_proxy() -> Optional[Dict[str, str]]:
    """
    Get a random proxy from the proxy list.
    
    Returns:
        Optional[Dict[str, str]]: Random proxy configuration or None if no proxies
    """
    return random.choice(PROXIES) if PROXIES else None

def adaptive_cooldown() -> float:
    """
    Calculate adaptive cooldown time based on failure patterns.
    
    Returns:
        float: Cooldown time in seconds
    """
    base = CONFIG["min_cooldown"]
    if request_stats["consecutive_failures"] > CONFIG["throttle_threshold"]:
        # Exponential backoff based on consecutive failures
        backoff_multiplier = min(
            2 ** (request_stats["consecutive_failures"] - CONFIG["throttle_threshold"]),
            CONFIG["max_cooldown"] / base
        )
        cooldown = base * backoff_multiplier
        return min(cooldown, CONFIG["max_cooldown"])
    return base + random.uniform(0, 30)  # Base cooldown with some randomness

def should_rotate_session() -> bool:
    """
    Determine if we should start a new session based on request count and duration.
    
    Returns:
        bool: True if session should be rotated, False otherwise
    """
    session_duration = time.time() - request_stats["session_start_time"]
    return (request_stats["session_request_count"] >= CONFIG["session_requests_limit"] or 
            session_duration >= CONFIG["session_duration_limit"])

def rotate_session() -> None:
    """
    Reset session counters and apply a cooldown before starting a new session.
    
    This function:
    - Saves current progress
    - Applies an adaptive cooldown
    - Resets session counters
    - Reduces consecutive failure count
    """
    cooldown = adaptive_cooldown()
    logger.info(
        f"Rotating session after {request_stats['session_request_count']} requests. "
        f"Cooling down for {cooldown:.2f} seconds..."
    )
    
    # Save progress before cooldown
    save_progress()
    
    time.sleep(cooldown)
    request_stats["session_start_time"] = time.time()
    request_stats["session_request_count"] = 0
    # Reduce failure count on successful rotation
    request_stats["consecutive_failures"] = max(0, request_stats["consecutive_failures"] - 1)

def get_with_retries(
    url: str,
    max_retries: Optional[int] = None,
    backoff_factor: Optional[float] = None
) -> Optional[requests.Response]:
    """
    Makes a GET request with retries, exponential backoff, and adaptive delay.
    
    Args:
        url: The URL to request
        max_retries: Maximum number of retry attempts (defaults to CONFIG value)
        backoff_factor: Factor to increase delay between retries (defaults to CONFIG value)
        
    Returns:
        Optional[requests.Response]: Response object if successful, None otherwise
    """
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
            response = requests.get(
                url,
                headers=headers,
                proxies=proxies,
                timeout=CONFIG["request_timeout"]
            )
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
            
            if status_code in (503, 429):
                # Rate limiting or service unavailable
                cooldown = backoff_factor * (2 ** attempt) + random.uniform(10, 30)
                cooldown = min(cooldown, CONFIG["max_cooldown"])
                logger.warning(
                    f"Rate limit error ({status_code}) for {url}, "
                    f"attempt {attempt + 1}/{max_retries}. "
                    f"Cooling down for {cooldown:.2f} seconds..."
                )
                time.sleep(cooldown)
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
    logger.error(
        f"All {max_retries} retries exhausted for {url}. "
        f"Cooling down for {cooldown:.2f} seconds..."
    )
    time.sleep(cooldown)
    return None

def save_progress() -> None:
    """
    Save current progress to allow resuming later.
    
    Saves the following information:
    - Last processed fighter index
    - Total requests made
    - Successful requests count
    - Consecutive failures count
    - Current timestamp
    """
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
        logger.info(
            f"Progress saved: {state['last_fighter_index']} fighters processed, "
            f"{state['successful_requests']}/{state['total_requests']} successful requests"
        )
        request_stats["last_save_time"] = time.time()
    except Exception as e:
        logger.error(f"Failed to save progress: {e}")

def load_progress() -> int:
    """
    Load progress from file.
    
    Returns:
        int: The last processed fighter index
    """
    try:
        if not os.path.exists(request_stats["progress_file"]):
            logger.info("No progress file found, starting from beginning")
            return 0
            
        with open(request_stats["progress_file"], 'r') as f:
            state = json.load(f)
            
        # Update request stats from saved state
        request_stats.update({
            "last_fighter_index": state.get("last_fighter_index", 0),
            "total_requests": state.get("total_requests", 0),
            "successful_requests": state.get("successful_requests", 0),
            "consecutive_failures": state.get("consecutive_failures", 0)
        })
        
        logger.info(
            f"Loaded progress: last_fighter_index={state.get('last_fighter_index')}, "
            f"total_requests={state.get('total_requests')}"
        )
        return state.get("last_fighter_index", 0)
        
    except Exception as e:
        logger.error(f"Error loading progress: {e}")
        return 0

def ensure_tap_link_column() -> None:
    """
    Ensure the tap_link column exists in the fighters table.
    
    Creates the column if it doesn't exist, preserving existing data.
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            ALTER TABLE fighters 
            ADD COLUMN tap_link TEXT DEFAULT NULL
        """)
        conn.commit()
        logger.info("Added tap_link column to fighters table")
    except sqlite3.OperationalError as e:
        if "duplicate column name" not in str(e).lower():
            logger.error(f"Error adding tap_link column: {e}")
    finally:
        if conn:
            conn.close()

def remove_nicknames_and_extras(name: str) -> str:
    """
    Remove nicknames and extra information from fighter names.
    
    Args:
        name: Raw fighter name potentially containing nicknames/extras
        
    Returns:
        str: Cleaned fighter name
    """
    # Remove content in quotes (nicknames)
    name = re.sub(r'["\'].*?["\']', '', name)
    # Remove content in parentheses
    name = re.sub(r'\(.*?\)', '', name)
    # Remove extra whitespace
    name = ' '.join(name.split())
    return name.strip()

def process_fighter_name(raw_name: str) -> str:
    """
    Process a raw fighter name for standardization.
    
    Args:
        raw_name: Raw fighter name to process
        
    Returns:
        str: Standardized fighter name
    """
    name = remove_nicknames_and_extras(raw_name)
    return standardize_name(name)

def standardize_name(name: str) -> str:
    """
    Standardize a fighter name for consistent matching.
    
    Args:
        name: Fighter name to standardize
        
    Returns:
        str: Standardized fighter name
    """
    # Convert to lowercase and remove extra whitespace
    name = ' '.join(name.lower().split())
    # Remove accents and special characters
    name = re.sub(r'[^\w\s-]', '', name)
    return name.strip()

def calculate_similarity(db_name: str, scraped_name: str) -> float:
    """
    Calculate similarity score between two fighter names.
    
    Uses SequenceMatcher to compute similarity, with additional
    handling for East Asian names that may have different word orders.
    
    Args:
        db_name: Name from database
        scraped_name: Name from scraping
        
    Returns:
        float: Similarity score between 0 and 1
    """
    # Standardize both names
    db_name = standardize_name(db_name)
    scraped_name = standardize_name(scraped_name)
    
    # Base similarity score
    similarity = SequenceMatcher(None, db_name, scraped_name).ratio()
    
    # Special handling for East Asian names
    if looks_like_east_asian_name(db_name) and looks_like_east_asian_name(scraped_name):
        # Try both word orders
        db_parts = db_name.split()
        scraped_parts = scraped_name.split()
        
        if len(db_parts) == len(scraped_parts) == 2:
            # Try reversed order
            reversed_db = f"{db_parts[1]} {db_parts[0]}"
            reversed_similarity = SequenceMatcher(None, reversed_db, scraped_name).ratio()
            similarity = max(similarity, reversed_similarity)
    
    return similarity

def looks_like_east_asian_name(name: str) -> bool:
    """
    Check if a name appears to be East Asian.
    
    Uses common patterns in East Asian names to make the determination:
    - Short (typically 2-3 words)
    - Each word is short (1-2 characters)
    - Contains common East Asian surname patterns
    
    Args:
        name: Name to check
        
    Returns:
        bool: True if name appears to be East Asian
    """
    # Common East Asian surnames
    east_asian_surnames = {
        'kim', 'lee', 'park', 'choi', 'jung', 'kang', 'cho', 'chang',
        'zhang', 'li', 'wang', 'chen', 'liu', 'yang', 'huang', 'zhao',
        'wu', 'zhou', 'xu', 'sun', 'ma', 'zhu', 'hu', 'guo', 'he',
        'gao', 'lin', 'luo', 'zheng', 'liang', 'xie', 'tang', 'xu',
        'sato', 'suzuki', 'takahashi', 'tanaka', 'watanabe', 'ito',
        'yamamoto', 'nakamura', 'kobayashi', 'kato', 'yoshida',
        'yamada', 'sasaki', 'yamaguchi', 'matsumoto', 'inoue',
        'nguyen', 'tran', 'le', 'pham', 'hoang', 'phan', 'vu', 'dang'
    }
    
    # Convert to lowercase and split
    parts = name.lower().strip().split()
    
    # Check name structure
    if not (1 <= len(parts) <= 3):
        return False
        
    # Check if any part is a common East Asian surname
    if not any(part in east_asian_surnames for part in parts):
        return False
        
    # Check if parts are short (typical for East Asian names)
    if not all(len(part) <= 3 for part in parts):
        return False
        
    return True

def tapology_search_url(fighter_name: str) -> str:
    """
    Generate the Tapology search URL for a fighter.
    
    Args:
        fighter_name: Name of the fighter to search for
        
    Returns:
        str: URL for searching the fighter on Tapology
    """
    return f"https://www.tapology.com/search?term={quote_plus(fighter_name)}&search=fighters"

def search_tapology_for_fighter(fighter_name: str) -> Optional[str]:
    """
    Search Tapology for a fighter and return their profile URL.
    
    Args:
        fighter_name: Name of the fighter to search for
        
    Returns:
        Optional[str]: Fighter's Tapology profile URL if found, None otherwise
    """
    search_url = tapology_search_url(fighter_name)
    response = get_with_retries(search_url)
    
    if not response:
        return None
        
    soup = BeautifulSoup(response.text, 'html.parser')
    results = soup.find_all('a', class_='name')
    
    return results[0]['href'] if results else None

def is_valid_fighter_image(image_url: str) -> bool:
    """
    Check if an image URL appears to be a valid fighter photo.
    
    Validates the URL against known patterns for fighter photos and
    checks for common placeholder/default image indicators.
    
    Args:
        image_url: URL of the image to validate
        
    Returns:
        bool: True if the image appears to be a valid fighter photo
    """
    if not image_url:
        return False
        
    # Known placeholder image patterns
    placeholder_patterns = [
        'placeholder',
        'default',
        'no-image',
        'noimage',
        'silhouette',
        'avatar',
        'unknown'
    ]
    
    # Convert URL to lowercase for pattern matching
    url_lower = image_url.lower()
    
    # Check for placeholder patterns
    if any(pattern in url_lower for pattern in placeholder_patterns):
        return False
        
    # Check for common image extensions
    valid_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp']
    if not any(ext in url_lower for ext in valid_extensions):
        return False
        
    # Additional validation could be added here
    # - Check image dimensions
    # - Verify image loads successfully
    # - Check file size
    # - etc.
        
    return True

def scrape_tapology_fighter_page(url: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Scrape a fighter's profile page on Tapology.
    
    Args:
        url: URL of the fighter's Tapology profile
        
    Returns:
        Tuple[Optional[str], Optional[str]]: Tuple containing:
            - Fighter's image URL (or None if not found)
            - Fighter's profile URL (or None if not found)
    """
    try:
        response = get_with_retries(url)
        if not response:
            return None, None
            
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find the fighter's image
        image_url = None
        image_elem = soup.find('img', class_='profile')
        if image_elem and 'src' in image_elem.attrs:
            image_url = image_elem['src']
            if not image_url.startswith('http'):
                image_url = f"https://www.tapology.com{image_url}"
                
        # Validate the image URL
        if not is_valid_fighter_image(image_url):
            image_url = None
            
        return image_url, url
            
    except Exception as e:
        logger.error(f"Error scraping Tapology page {url}: {e}")
        return None, None

def update_fighter_in_db(
    db_name: str,
    new_image_url: str,
    tapology_url: str
) -> bool:
    """
    Update a fighter's image URL and Tapology link in the database.
    
    Args:
        db_name: Name of the fighter to update
        new_image_url: New image URL to set
        tapology_url: Tapology profile URL to set
        
    Returns:
        bool: True if update was successful, False otherwise
    """
    try:
        with db_lock:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                UPDATE fighters 
                SET image_url = ?, tap_link = ?
                WHERE fighter_name = ?
            """, (new_image_url, tapology_url, db_name))
            
            conn.commit()
            return True
            
    except Exception as e:
        logger.error(f"Error updating database for {db_name}: {e}")
        return False
        
    finally:
        if conn:
            conn.close()

def process_fighter(
    fighter_data: Tuple[str, Optional[str], Optional[str]]
) -> bool:
    """
    Process a single fighter's data.
    
    This function:
    1. Searches for the fighter on Tapology
    2. Scrapes their profile page
    3. Updates their information in the database
    
    Args:
        fighter_data: Tuple containing:
            - Fighter's name
            - Current image URL
            - Current Tapology link
            
    Returns:
        bool: True if processing was successful, False otherwise
    """
    try:
        name, current_image, current_tap_link = fighter_data
        
        # Skip if we already have both image and Tapology link
        if current_image and current_tap_link:
            logger.debug(f"Skipping {name} - already has image and Tapology link")
            return True
            
        # Search for fighter on Tapology
        tap_url = search_tapology_for_fighter(name)
        if not tap_url:
            logger.warning(f"Could not find Tapology page for {name}")
            return False
            
        # Scrape fighter's page
        image_url, profile_url = scrape_tapology_fighter_page(tap_url)
        
        # Update database if we found new information
        if image_url or profile_url:
            success = update_fighter_in_db(
                name,
                image_url or current_image,
                profile_url or current_tap_link
            )
            if success:
                logger.info(f"Updated {name} with new data")
            return success
            
        return False
        
    except Exception as e:
        logger.error(f"Error processing fighter {fighter_data[0]}: {e}")
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