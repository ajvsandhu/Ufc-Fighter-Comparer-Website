#!/usr/bin/env python
"""
Tapology Data Scraper

A robust scraper for collecting fighter data from Tapology with built-in rate limiting
and error handling. Features include session management, proxy support, and progress tracking.
"""

import argparse
import datetime
import json
import logging
import os
import random
import re
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
    Parse command line arguments for scraper configuration.
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
    parser.add_argument(
        '--only-need-image',
        action='store_true',
        help='Only process fighters without an image'
    )
    parser.add_argument(
        '--only-need-tap',
        action='store_true',
        help='Only process fighters without a Tapology link'
    )
    parser.add_argument(
        '--include-complete',
        action='store_true',
        help='Include fighters with both image and Tapology link'
    )
    return parser.parse_args()

# Fix import by correctly adding the project root to sys.path
# Get the absolute path to the project root directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
    print(f"Added {PROJECT_ROOT} to Python path")

# Now import from backend should work
from backend.api.database import get_supabase_client

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
    "max_failure_rate": 0.1,     # Maximum allowed failure rate
    "max_consecutive_failures": 5, # Maximum allowed consecutive failures
    "save_interval": 300,         # Save progress every N seconds
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
    Generate random user agent headers to prevent request blocking.
    """
    return {"User-Agent": random.choice(USER_AGENTS)}

def get_random_proxy() -> Optional[Dict[str, str]]:
    """
    Select a random proxy from the available proxy pool.
    """
    return random.choice(PROXIES) if PROXIES else None

def adaptive_cooldown() -> float:
    """
    Calculate dynamic cooldown time based on request patterns.
    """
    base = CONFIG["min_cooldown"]
    if request_stats["consecutive_failures"] > CONFIG["throttle_threshold"]:
        backoff_multiplier = min(
            2 ** (request_stats["consecutive_failures"] - CONFIG["throttle_threshold"]),
            CONFIG["max_cooldown"] / base
        )
        cooldown = base * backoff_multiplier
        return min(cooldown, CONFIG["max_cooldown"])
    return base + random.uniform(0, 30)

def should_rotate_session() -> bool:
    """
    Determine if the current session needs rotation based on request count and duration.
    """
    session_duration = time.time() - request_stats["session_start_time"]
    return (request_stats["session_request_count"] >= CONFIG["session_requests_limit"] or 
            session_duration >= CONFIG["session_duration_limit"])

def rotate_session() -> None:
    """
    Rotate the current session and apply cooldown period.
    """
    cooldown = adaptive_cooldown()
    logger.debug(
        f"Rotating session after {request_stats['session_request_count']} requests, "
        f"cooldown: {cooldown:.2f}s"
    )
    
    save_progress()
    time.sleep(cooldown)
    request_stats["session_start_time"] = time.time()
    request_stats["session_request_count"] = 0
    request_stats["consecutive_failures"] = max(0, request_stats["consecutive_failures"] - 1)

def get_with_retries(
    url: str,
    max_retries: Optional[int] = None,
    backoff_factor: Optional[float] = None
) -> Optional[requests.Response]:
    """
    Execute HTTP GET request with retry logic and rate limiting.
    """
    if max_retries is None:
        max_retries = CONFIG["max_retries"]
    if backoff_factor is None:
        backoff_factor = CONFIG["backoff_factor"]
    
    if should_rotate_session():
        rotate_session()
    
    attempt = 0
    while attempt < max_retries:
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
            
            request_stats["consecutive_failures"] = 0
            request_stats["successful_requests"] += 1
            
            sleep_time = current_delay + random.uniform(0, 2)
            logger.debug(f"Request successful, delay: {sleep_time:.2f}s")
            time.sleep(sleep_time)
            return response
        
        except requests.exceptions.HTTPError as e:
            request_stats["consecutive_failures"] += 1
            status_code = e.response.status_code if hasattr(e, 'response') else "Unknown"
            
            if status_code in (503, 429):
                cooldown = backoff_factor * (2 ** attempt) + random.uniform(10, 30)
                cooldown = min(cooldown, CONFIG["max_cooldown"])
                logger.debug(
                    f"Rate limit ({status_code}), attempt {attempt + 1}/{max_retries}, "
                    f"cooldown: {cooldown:.2f}s"
                )
                time.sleep(cooldown)
            else:
                logger.debug(f"HTTP error {status_code} for {url}")
                time.sleep(current_delay * 3)
            attempt += 1
                
        except requests.exceptions.ConnectionError as e:
            request_stats["consecutive_failures"] += 1
            logger.debug("Connection error, retrying after delay")
            time.sleep(current_delay * 3)
            attempt += 1
            
        except requests.exceptions.Timeout as e:
            request_stats["consecutive_failures"] += 1
            logger.debug("Request timeout, retrying after delay")
            time.sleep(current_delay * 3)
            attempt += 1
            
        except Exception as e:
            request_stats["consecutive_failures"] += 1
            logger.error(f"Unexpected error: {str(e)}")
            time.sleep(current_delay * 3)
            attempt += 1
    
    cooldown = adaptive_cooldown()
    logger.debug(f"Max retries reached, cooldown: {cooldown:.2f}s")
    time.sleep(cooldown)
    return None

def save_progress() -> None:
    """
    Save current scraping progress to disk.
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
        logger.debug(
            f"Progress saved: {state['last_fighter_index']} fighters, "
            f"{state['successful_requests']}/{state['total_requests']} successful"
        )
        request_stats["last_save_time"] = time.time()
    except Exception as e:
        logger.error(f"Failed to save progress: {str(e)}")

def load_progress() -> int:
    """
    Load saved scraping progress from disk.
    """
    try:
        if not os.path.exists(request_stats["progress_file"]):
            logger.debug("No progress file found, starting from beginning")
            return 0
            
        with open(request_stats["progress_file"], 'r') as f:
            state = json.load(f)
            
        request_stats.update({
            "last_fighter_index": state.get("last_fighter_index", 0),
            "total_requests": state.get("total_requests", 0),
            "successful_requests": state.get("successful_requests", 0),
            "consecutive_failures": state.get("consecutive_failures", 0)
        })
        
        logger.debug(
            f"Progress loaded: index={state.get('last_fighter_index')}, "
            f"requests={state.get('total_requests')}"
        )
        return state.get("last_fighter_index", 0)
        
    except Exception as e:
        logger.error(f"Failed to load progress: {str(e)}")
        return 0

def ensure_tap_link_column() -> None:
    """
    Verify Supabase schema includes tap_link column.
    """
    try:
        logger.debug("Verifying tap_link column in schema")
    except Exception as e:
        logger.error(f"Schema verification failed: {str(e)}")

def remove_nicknames_and_extras(name: str) -> str:
    """
    Clean fighter name by removing nicknames and additional information.
    """
    name = re.sub(r'["\'].*?["\']', '', name)
    name = re.sub(r'\(.*?\)', '', name)
    name = ' '.join(name.split())
    return name.strip()

def process_fighter_name(raw_name: str) -> str:
    """
    Standardize fighter name for consistent matching.
    """
    name = remove_nicknames_and_extras(raw_name)
    return standardize_name(name)

def standardize_name(name: str) -> str:
    """
    Convert fighter name to standardized format.
    """
    name = ' '.join(name.lower().split())
    name = re.sub(r'[^\w\s-]', '', name)
    return name.strip()

def calculate_similarity(db_name: str, scraped_name: str) -> float:
    """
    Calculate name similarity score with special handling for East Asian names.
    """
    db_name = standardize_name(db_name)
    scraped_name = standardize_name(scraped_name)
    
    similarity = SequenceMatcher(None, db_name, scraped_name).ratio()
    
    if looks_like_east_asian_name(db_name) and looks_like_east_asian_name(scraped_name):
        db_parts = db_name.split()
        scraped_parts = scraped_name.split()
        
        if len(db_parts) == len(scraped_parts) == 2:
            reversed_db = f"{db_parts[1]} {db_parts[0]}"
            reversed_similarity = SequenceMatcher(None, reversed_db, scraped_name).ratio()
            similarity = max(similarity, reversed_similarity)
    
    return similarity

def looks_like_east_asian_name(name: str) -> bool:
    """
    Check if name matches East Asian naming patterns.
    """
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
    
    parts = name.lower().strip().split()
    
    if not (1 <= len(parts) <= 3):
        return False
        
    if not any(part in east_asian_surnames for part in parts):
        return False
        
    if not all(len(part) <= 3 for part in parts):
        return False
        
    return True

def tapology_search_url(fighter_name: str) -> str:
    """
    Generate search URL for fighter on Tapology.
    """
    cleaned_name = re.sub(r'[^\w\s]', ' ', fighter_name)
    cleaned_name = ' '.join(cleaned_name.split())
    encoded_name = quote_plus(cleaned_name)
    
    return f"https://www.tapology.com/search?term={encoded_name}&search=fighters"

def create_direct_fighter_url(fighter_name: str) -> List[str]:
    """
    Generate potential direct URLs to fighter's Tapology profile.
    """
    urls = []
    
    clean_name = re.sub(r'[^\w\s]', '', fighter_name).lower().strip()
    slug = re.sub(r'\s+', '-', clean_name)
    urls.append(f"https://www.tapology.com/fightcenter/fighters/{slug}")
    
    for i in range(1, 4):
        urls.append(f"https://www.tapology.com/fightcenter/fighters/{slug}-{i}")
    
    name_parts = clean_name.split()
    if len(name_parts) >= 2:
        first_name = name_parts[0]
        last_name = name_parts[-1]
        
        urls.append(f"https://www.tapology.com/fightcenter/fighters/{last_name}-{first_name}")
        
        for i in range(1, 3):
            urls.append(f"https://www.tapology.com/fightcenter/fighters/{last_name}-{first_name}-{i}")
        
        urls.append(f"https://www.tapology.com/fightcenter/fighters/{first_name}-{last_name}")
    
    return urls

def search_tapology_for_fighter(fighter_name: str) -> Optional[str]:
    """
    Search and retrieve fighter's Tapology profile URL.
    """
    search_url = tapology_search_url(fighter_name)
    logger.debug(f"Searching for fighter: {fighter_name}")
    
    response = get_with_retries(search_url)
    
    if not response:
        logger.debug(f"Search failed for {fighter_name}, trying direct URLs")
        direct_urls = create_direct_fighter_url(fighter_name)
        
        for direct_url in direct_urls:
            logger.debug(f"Trying URL: {direct_url}")
            direct_response = get_with_retries(direct_url)
            if direct_response and direct_response.status_code == 200:
                soup = BeautifulSoup(direct_response.text, 'html.parser')
                if soup.select_one('.fighterHeader') or soup.select_one('.fighterRecord'):
                    logger.debug(f"Found profile: {direct_url}")
                    return direct_url
            time.sleep(random.uniform(1.0, 2.0))
        
        return None
        
    soup = BeautifulSoup(response.text, 'html.parser')
    
    search_results = soup.select('.searchResult a.name')
    if search_results and len(search_results) > 0:
        logger.debug(f"Found {len(search_results)} search results")
        href = search_results[0].get('href')
        if href:
            if not href.startswith('http'):
                href = f"https://www.tapology.com{href}"
            logger.debug(f"Selected profile: {href}")
            return href
    
    alt_results = soup.select('.searchResultsFighter a')
    if alt_results and len(alt_results) > 0:
        logger.debug(f"Found {len(alt_results)} alternative results")
        for result in alt_results:
            href = result.get('href')
            if href and "/fightcenter/fighters/" in href:
                if not href.startswith('http'):
                    href = f"https://www.tapology.com{href}"
                logger.debug(f"Selected alternative profile: {href}")
                return href
    
    direct_links = soup.select('a[href*="/fightcenter/fighters/"]')
    if direct_links and len(direct_links) > 0:
        logger.debug(f"Found {len(direct_links)} direct links")
        for link in direct_links:
            href = link.get('href')
            link_text = link.get_text().lower().strip()
            fighter_name_lower = fighter_name.lower().strip()
            
            if fighter_name_lower in link_text or any(part in link_text for part in fighter_name_lower.split()):
                if not href.startswith('http'):
                    href = f"https://www.tapology.com{href}"
                logger.debug(f"Selected direct link: {href}")
                return href
    
    all_links = soup.find_all('a')
    logger.debug(f"Scanning {len(all_links)} links for matches")
    for link in all_links:
        href = link.get('href')
        if href and "/fightcenter/fighters/" in href:
            link_text = link.get_text().lower()
            fighter_name_parts = fighter_name.lower().split()
            
            if any(part in link_text for part in fighter_name_parts):
                if not href.startswith('http'):
                    href = f"https://www.tapology.com{href}"
                logger.debug(f"Found matching link: {href}")
                return href
    
    direct_urls = create_direct_fighter_url(fighter_name)
    
    for direct_url in direct_urls:
        logger.debug(f"Trying final URL: {direct_url}")
        direct_response = get_with_retries(direct_url)
        if direct_response and direct_response.status_code == 200:
            soup = BeautifulSoup(direct_response.text, 'html.parser')
            if soup.select_one('.fighterHeader') or soup.select_one('.fighterRecord'):
                logger.debug(f"Found valid profile: {direct_url}")
                return direct_url
        time.sleep(random.uniform(1.0, 2.0))
    
    logger.debug(f"No profile found for {fighter_name}")
    return None

def is_valid_fighter_image(image_url: str) -> bool:
    """
    Validate fighter image URLs against quality criteria.
    """
    if not image_url:
        logger.debug("Empty image URL provided")
        return False
    
    if 'images.tapology.com/letterbox_images/' in image_url:
        logger.debug(f"Valid Tapology CDN image found: {image_url}")
        return True
        
    placeholder_patterns = [
        'no-picture', 'no_picture', 'default_profile',
        'blank-profile', 'no-avatar', 'noavatar',
        'blank-user', 'blank_user'
    ]
    
    url_lower = image_url.lower()
    
    if any(pattern in url_lower for pattern in placeholder_patterns):
        logger.debug(f"Placeholder image detected: {image_url}")
        return False
        
    valid_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp']
    if not any(ext in url_lower for ext in valid_extensions):
        logger.debug(f"Invalid image extension: {image_url}")
        return False
        
    if 'tapology.com' in url_lower:
        logger.debug(f"Valid Tapology domain image: {image_url}")
        return True
    
    logger.debug(f"Image validation passed: {image_url}")
    return True

def scrape_tapology_fighter_page(url: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract fighter's image URL and profile URL from their Tapology page.
    """
    try:
        response = get_with_retries(url)
        if not response:
            return None, None
            
        soup = BeautifulSoup(response.text, 'html.parser')
        image_url = None
        
        all_images = soup.find_all('img')
        for img in all_images:
            src = img.get('src', '')
            if 'images.tapology.com/letterbox_images/' in src:
                image_url = src
                logger.debug(f"Found CDN image: {image_url}")
                break
        
        if not image_url:
            for elem in soup.find_all(lambda tag: tag.has_attr('style')):
                style = elem.get('style', '')
                if 'background-image' in style and 'images.tapology.com/letterbox_images/' in style:
                    match = re.search(r"url\(['\"]?(https?://images\.tapology\.com/letterbox_images/[^)'\"]*)['\"]\)?", style)
                    if match:
                        image_url = match.group(1)
                        logger.debug(f"Found CDN background image: {image_url}")
                        break
        
        if not image_url:
            for img in all_images:
                src = img.get('src', '')
                if 'tapology.com' in src and ('images/' in src or 'letterbox' in src):
                    image_url = src
                    logger.debug(f"Found Tapology image: {image_url}")
                    break
        
        if image_url and not image_url.startswith('http'):
            if image_url.startswith('//'):
                image_url = 'https:' + image_url
            else:
                image_url = f"https://www.tapology.com{image_url}"
            
        if not image_url:
            logger.debug(f"No image found on page: {url}")
                
        if image_url and not image_url.startswith('https://images.tapology.com/'):
            logger.debug(f"Non-CDN image found: {image_url}")
        
        return image_url, url
            
    except Exception as e:
        logger.error(f"Failed to scrape page: {str(e)}")
        return None, None

def update_fighter_in_db(
    db_name: str,
    new_image_url: str,
    tapology_url: str
) -> bool:
    """
    Update fighter's image and Tapology URLs in database.
    """
    try:
        with db_lock:
            supabase = get_supabase_client()
            
            logger.debug(f"Updating fighter: {db_name}")
            logger.debug(f"Image URL: {new_image_url}")
            logger.debug(f"Tapology URL: {tapology_url}")
            
            update_data = {}
            if new_image_url:
                update_data['image_url'] = new_image_url
            if tapology_url:
                update_data['tap_link'] = tapology_url
                
            if not update_data:
                logger.debug(f"No updates needed for {db_name}")
                return False
                
            response = supabase.table('fighters') \
                .update(update_data) \
                .eq('fighter_name', db_name) \
                .execute()
            
            if response.data and len(response.data) > 0:
                updated_fields = []
                if 'image_url' in update_data:
                    updated_fields.append('image_url')
                if 'tap_link' in update_data:
                    updated_fields.append('tap_link')
                
                logger.debug(f"Updated {db_name}: {', '.join(updated_fields)}")
                return True
            else:
                logger.debug(f"Update failed for {db_name}")
                
                check_response = supabase.table('fighters') \
                    .select('fighter_name') \
                    .eq('fighter_name', db_name) \
                    .execute()
                
                if check_response.data and len(check_response.data) > 0:
                    logger.debug(f"Fighter exists but update failed: {db_name}")
                else:
                    logger.debug(f"Fighter not found: {db_name}")
                
                return False
                
    except Exception as e:
        logger.error(f"Database update failed: {str(e)}")
        return False

def process_fighter(
    fighter_data: Tuple[str, Optional[str], Optional[str]]
) -> bool:
    """
    Process fighter data and update their Tapology information.
    """
    try:
        name, current_image, current_tap_link = fighter_data
        
        logger.debug(f"Processing fighter: {name}")
        logger.debug(f"Current image: {current_image}")
        logger.debug(f"Current link: {current_tap_link}")
        
        has_tapology_image = current_image and 'images.tapology.com/letterbox_images/' in current_image
        if has_tapology_image and current_tap_link:
            logger.debug(f"Skipping {name}: already complete")
            return True
            
        tap_url = search_tapology_for_fighter(name)
        
        if not tap_url:
            logger.debug(f"Trying direct URLs for {name}")
            direct_urls = create_direct_fighter_url(name)
            
            for url in direct_urls[:3]:
                logger.debug(f"Checking URL: {url}")
                response = get_with_retries(url)
                if response and response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    if soup.select_one('.fighterHeader, .fighterRecord, .fighterBio'):
                        tap_url = url
                        logger.debug(f"Found valid URL: {tap_url}")
                        break
                time.sleep(random.uniform(1.0, 2.0))
        
        if not tap_url:
            logger.debug(f"No Tapology page found for {name}")
            return False
            
        logger.debug(f"Scraping page: {tap_url}")
        image_url, profile_url = scrape_tapology_fighter_page(tap_url)
        
        logger.debug(f"Found image: {image_url}")
        logger.debug(f"Found profile: {profile_url}")
        
        update_image = False
        if image_url:
            if not current_image:
                update_image = True
                logger.debug(f"Adding new image for {name}")
            elif not has_tapology_image and 'images.tapology.com/letterbox_images/' in image_url:
                update_image = True
                logger.debug(f"Upgrading to CDN image for {name}")
        
        update_link = profile_url and not current_tap_link
        
        if update_image or update_link:
            update_data = {}
            if update_image:
                update_data['image_url'] = image_url
            if update_link:
                update_data['tap_link'] = profile_url
                
            success = update_fighter_in_db(
                name,
                update_data.get('image_url', current_image),
                update_data.get('tap_link', current_tap_link)
            )
            
            if success:
                updates = []
                if update_image:
                    updates.append("image_url")
                if update_link:
                    updates.append("tap_link")
                    
                logger.debug(f"Updated {name}: {', '.join(updates)}")
                return True
            else:
                logger.debug(f"Update failed for {name}")
                return False
        else:
            logger.debug(f"No updates needed for {name}")
            return True
            
    except Exception as e:
        logger.error(f"Failed to process {fighter_data[0]}: {str(e)}")
        return False

def should_continue_processing() -> bool:
    """
    Check if scraping should continue based on performance metrics.
    """
    if request_stats["total_requests"] > 20:
        failure_rate = 1 - (request_stats["successful_requests"] / request_stats["total_requests"])
        if failure_rate > CONFIG["max_failure_rate"]:
            logger.error(
                f"Stopping: failure rate {failure_rate:.2f} exceeds {CONFIG['max_failure_rate']:.2f}"
            )
            return False
            
    if request_stats["consecutive_failures"] > CONFIG["max_consecutive_failures"]:
        logger.error(
            f"Stopping: {request_stats['consecutive_failures']} consecutive failures"
        )
        return False
        
    return True

def main():
    """
    Main scraping process for fighter data collection.
    """
    try:
        ensure_tap_link_column()
        
        start_index = 1
        logger.info(f"Starting from index {start_index}")
        request_stats["last_fighter_index"] = start_index
        
        supabase = get_supabase_client()
        
        page_size = 1000
        all_fighters = []
        page = 0
        
        while True:
            response = supabase.table('fighters') \
                .select('fighter_name, image_url, tap_link') \
                .range(page * page_size, (page + 1) * page_size - 1) \
                .execute()
            
            fighters_page = response.data
            if not fighters_page or len(fighters_page) == 0:
                break
                
            all_fighters.extend(fighters_page)
            
            if len(fighters_page) < page_size:
                break
                
            page += 1
        
        fighters = [(row['fighter_name'], row.get('image_url'), row.get('tap_link')) for row in all_fighters]
        
        if args.only_need_image:
            fighters = [f for f in fighters if not f[1]]
            
        if args.only_need_tap:
            fighters = [f for f in fighters if not f[2]]
            
        if not args.include_complete:
            fighters = [f for f in fighters if not (f[1] and f[2])]
            
        logger.info(f"Found {len(fighters)} fighters to process")
        logger.info(f"Starting from index {start_index}")
        
        i = start_index
        while i < len(fighters) and should_continue_processing():
            fighter = fighters[i]
            logger.info(f"Processing {i+1}/{len(fighters)}: {fighter[0]}")
            
            success = process_fighter(fighter)
            i += 1
            request_stats["last_fighter_index"] = i
            
            if time.time() - request_stats["last_save_time"] > CONFIG["save_interval"]:
                save_progress()
                
            if i % 10 == 0:
                success_rate = request_stats['successful_requests'] / max(1, request_stats['total_requests']) * 100
                logger.info(
                    f"Progress: {i}/{len(fighters)} fighters, "
                    f"{request_stats['successful_requests']}/{request_stats['total_requests']} "
                    f"successful ({success_rate:.1f}%)"
                )
                
        save_progress()
        success_rate = request_stats['successful_requests'] / max(1, request_stats['total_requests']) * 100
        logger.info(
            f"Complete: {i}/{len(fighters)} fighters, "
            f"{request_stats['successful_requests']}/{request_stats['total_requests']} "
            f"successful ({success_rate:.1f}%)"
        )
        
    except Exception as e:
        logger.error(f"Main process failed: {str(e)}")
        save_progress()
    finally:
        save_progress()

if __name__ == "__main__":
    main()