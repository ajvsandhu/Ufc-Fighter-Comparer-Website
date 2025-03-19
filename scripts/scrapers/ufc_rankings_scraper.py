"""
UFC Rankings Scraper

This module fetches fighter rankings from the UFC website and stores them directly in the fighters table.
Fighter rankings are used to improve prediction accuracy by factoring in the strength of
a fighter's competition.
"""

import os
import sys
import requests
from bs4 import BeautifulSoup
import logging
import time
import sqlite3
import re
import json
from datetime import datetime
from unicodedata import normalize
from pathlib import Path

# Fix path for direct script execution
if __name__ == "__main__":
    # Add the project root to the Python path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    sys.path.insert(0, project_root)

# Base Directories and constants
BASE_DIR = Path(project_root) if 'project_root' in locals() else Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATA_DIR = BASE_DIR / "data"
DB_NAME = "ufc_fighters.db"
DB_PATH = DATA_DIR / DB_NAME

# Web Scraping Configuration
REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}
REQUEST_TIMEOUT = 15  # seconds
RETRY_ATTEMPTS = 3
RETRY_DELAY = 2  # seconds

# Logging Configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Try to import from backend if available
try:
    from backend.api.database import get_db_connection
except ImportError:
    # Fallback if can't import
    def get_db_connection():
        try:
            conn = sqlite3.connect(DB_PATH)
            conn.row_factory = sqlite3.Row
            print(f"Connected to database at: {DB_PATH}")
            return conn
        except Exception as e:
            print(f"Error connecting to database: {str(e)}")
            return None

# Set up logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format=LOG_FORMAT,
    datefmt=LOG_DATE_FORMAT
)
logger = logging.getLogger(__name__)

# UFC rankings URL
UFC_RANKINGS_URL = "https://www.ufc.com/rankings"

# Weight class mapping - we'll use this to match divisions to weight classes
WEIGHT_CLASSES = {
    "Flyweight": "125",
    "Bantamweight": "135",
    "Featherweight": "145",
    "Lightweight": "155",
    "Welterweight": "170",
    "Middleweight": "185",
    "Light Heavyweight": "205",
    "Heavyweight": "265",
    "Women's Strawweight": "115",
    "Women's Flyweight": "125",
    "Women's Bantamweight": "135",
    "Women's Featherweight": "145"
}

# Path to cached rankings file
CACHED_RANKINGS_PATH = os.path.join(DATA_DIR, "cached_rankings.json")

def cache_rankings(rankings):
    """Cache the rankings to a file for future use"""
    try:
        if not rankings:
            return False
            
        # Convert the rankings to a serializable format
        serializable_rankings = {}
        for name, data in rankings.items():
            serializable_rankings[name] = data
        
        # Add timestamp
        data_to_save = {
            'timestamp': datetime.now().isoformat(),
            'rankings': serializable_rankings
        }
        
        # Save to file
        with open(CACHED_RANKINGS_PATH, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, indent=2)
            
        logger.info(f"Cached {len(rankings)} rankings to {CACHED_RANKINGS_PATH}")
        return True
    except Exception as e:
        logger.error(f"Error caching rankings: {str(e)}")
        return False

def load_cached_rankings():
    """Load rankings from cache file if it exists"""
    try:
        if not os.path.exists(CACHED_RANKINGS_PATH):
            logger.warning(f"No cached rankings found at {CACHED_RANKINGS_PATH}")
            return {}
            
        with open(CACHED_RANKINGS_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Get the timestamp to check freshness
        timestamp = data.get('timestamp', '')
        if timestamp:
            timestamp_date = datetime.fromisoformat(timestamp)
            age_days = (datetime.now() - timestamp_date).days
            
            if age_days > 7:
                logger.warning(f"Cached rankings are {age_days} days old. Consider refreshing.")
            
            logger.info(f"Loaded cached rankings from {timestamp}")
            
        rankings = data.get('rankings', {})
        logger.info(f"Loaded {len(rankings)} fighter rankings from cache")
        
        # Verify the cache format
        if all(isinstance(v, dict) for v in rankings.values()):
            return rankings
        else:
            logger.error("Invalid cache format")
            return {}
    except Exception as e:
        logger.error(f"Error loading cached rankings: {str(e)}")
        return {}

def update_meta_table(conn, key, value):
    """Update or create a meta table for tracking information like last update timestamps"""
    try:
        cursor = conn.cursor()
        
        # Create meta table if it doesn't exist
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at TEXT
            )
        """)
        
        # Update or insert the key-value pair
        cursor.execute("""
            INSERT OR REPLACE INTO meta (key, value, updated_at)
            VALUES (?, ?, ?)
        """, (key, value, datetime.now().isoformat()))
        
        conn.commit()
        logger.info(f"Updated meta table: {key} = {value}")
        return True
    except Exception as e:
        logger.error(f"Error updating meta table: {str(e)}")
        return False

def normalize_name(name):
    """
    Normalize fighter names to improve matching
    - Remove accents and special characters
    - Convert to lowercase
    - Remove punctuation
    - Remove suffixes like "Jr." or "III"
    """
    # Remove accents
    name = normalize('NFKD', name).encode('ASCII', 'ignore').decode('utf-8')
    # Convert to lowercase
    name = name.lower()
    # Remove punctuation except spaces
    name = re.sub(r'[^\w\s]', '', name)
    # Remove common suffixes
    name = re.sub(r'\bjr\b|\biii\b|\biv\b|\bsr\b', '', name)
    # Remove extra spaces
    name = re.sub(r'\s+', ' ', name).strip()
    return name

def fetch_ufc_rankings():
    """Fetch current UFC rankings from the UFC website"""
    try:
        logger.info(f"Fetching UFC rankings from {UFC_RANKINGS_URL}")
        
        # Send request with headers to mimic a browser
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Referer": "https://www.ufc.com/",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1"
        }
        response = requests.get(UFC_RANKINGS_URL, headers=headers, timeout=30)
        print(f"Response status code: {response.status_code}")
        
        if response.status_code != 200:
            logger.error(f"Got status code {response.status_code}")
            print(f"Response text: {response.text[:500]}...")
            # Try to load cached rankings
            return load_cached_rankings()
        
        # Parse the HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Print some debug info
        page_title = soup.title.text if soup.title else 'No title'
        print(f"Page title: {page_title}")
        
        # Save the HTML for debugging (only when directly running the script)
        if __name__ == "__main__":
            debug_file = os.path.join(DATA_DIR, "ufc_rankings_debug.html")
            with open(debug_file, "w", encoding="utf-8") as f:
                f.write(response.text)
            print(f"Saved HTML to {debug_file} for debugging")
        
        # Dictionary to store fighter rankings by name
        fighter_rankings = {}
        
        # Try various approaches to find the rankings
        ranking_tables = soup.find_all('div', class_='view-grouping')
        print(f"Found {len(ranking_tables)} ranking tables")
        
        if len(ranking_tables) == 0:
            # Try alternative class names
            ranking_tables = soup.find_all('div', class_='category-section')
            print(f"Alternative search found {len(ranking_tables)} ranking tables")
            
            if len(ranking_tables) == 0:
                # More general approach to find divisions
                potential_divisions = soup.find_all(['h2', 'h3', 'h4'], string=lambda text: text and any(div in text.lower() for div in 
                    ['heavyweight', 'light heavyweight', 'middleweight', 'welterweight', 'lightweight', 
                     'featherweight', 'bantamweight', 'flyweight', "women's"]))
                
                print(f"Found {len(potential_divisions)} potential division headers")
                if len(potential_divisions) == 0:
                    logger.error("No ranking tables or division headers found. Trying to load cached rankings.")
                    return load_cached_rankings()
        
        # Process each division
        for table in ranking_tables:
            # Get division name from header
            division_header = table.find(['h4', 'h3', 'h2'], class_='view-grouping-header')
            
            if not division_header:
                # Try alternative class names
                division_header = table.find(['h4', 'h3', 'h2'])
            
            if not division_header:
                # Skip if can't identify division
                continue
            
            division_name = division_header.text.strip()
            print(f"Processing division: {division_name}")
            
            if "pound-for-pound" in division_name.lower():
                print(f"Skipping pound-for-pound rankings")
                continue
            
            # Match division name to weight class
            matched_division = None
            for weight_class_name, weight in WEIGHT_CLASSES.items():
                if weight_class_name.lower() in division_name.lower():
                    matched_division = weight_class_name
                    division_weight = weight
                    break
            
            if not matched_division:
                print(f"Could not match division name: {division_name}")
                continue
            
            print(f"Matched {division_name} to {matched_division} ({division_weight} lbs)")
            
            # Track occupied ranks to avoid duplicates
            occupied_ranks = {}
            
            # Process champion separately
            champion_found = False
            champion_section = table.find('div', class_='champion')
            if not champion_section:
                # Try alternative approaches
                champion_section = table.find('div', class_='view-grouping-content')
                if champion_section:
                    champion_container = champion_section.find('div', class_='views-row')
                    if champion_container and 'champion' in champion_container.text.lower():
                        champion_section = champion_container
            
            if champion_section:
                # Look for champion name
                champion_name_elem = champion_section.find(['h5', 'span', 'div'], class_=['champion-name', 'views-field-title'])
                
                if not champion_name_elem:
                    # More general approach
                    champion_name_elem = champion_section.find(['h5', 'span', 'div'])
                
                if champion_name_elem:
                    raw_champion_text = champion_name_elem.text.strip()
                    # Remove "Champion" text and clean up
                    champion_name = re.sub(r'\s+', ' ', raw_champion_text)
                    champion_name = re.sub(r'champion', '', champion_name, flags=re.IGNORECASE).strip()
                    
                    # Further clean up - if multiple words, assume it's the name
                    champion_name_parts = [part for part in champion_name.split() if len(part) > 1 and not part.isdigit()]
                    if champion_name_parts:
                        champion_name = " ".join(champion_name_parts)
                    
                    if champion_name and len(champion_name) > 3:
                        print(f"Found champion: {champion_name}")
                        normalized_name = normalize_name(champion_name)
                        
                        # Champion is rank 1
                        numeric_rank = 1
                        occupied_ranks[numeric_rank] = normalized_name
                        
                        fighter_rankings[normalized_name] = {
                            'original_name': champion_name,
                            'numeric_rank': numeric_rank,
                            'division_weight': division_weight,
                            'is_champion': True
                        }
                        champion_found = True
            
            # Find ranked fighters
            ranked_fighters_container = None
            
            # Try to find the ranked fighters container
            if table.find('div', class_='view-content'):
                view_contents = table.find_all('div', class_='view-content')
                if len(view_contents) > 1:
                    ranked_fighters_container = view_contents[1]  # Second view-content is usually ranked fighters
            
            if not ranked_fighters_container:
                # Try alternative approach
                ranked_fighters_container = table.find('div', class_='rankings')
            
            if not ranked_fighters_container:
                # More general approach - look for container with multiple numbered items
                potential_containers = table.find_all(['div', 'ul', 'ol', 'table'])
                for container in potential_containers:
                    # Check if container has multiple children with numbers
                    items_with_numbers = [item for item in container.find_all(['div', 'li', 'tr']) 
                                         if re.search(r'\d+', item.text) and not 'champion' in item.text.lower()]
                    if len(items_with_numbers) >= 5:  # At least 5 ranked fighters
                        ranked_fighters_container = container
                        break
            
            if ranked_fighters_container:
                # Look for fighter items in container
                ranked_fighters = ranked_fighters_container.find_all('div', class_='views-row')
                
                if not ranked_fighters:
                    # Try alternative selectors
                    ranked_fighters = ranked_fighters_container.find_all('li')
                
                if not ranked_fighters:
                    # More general approach - find elements with numbers
                    ranked_fighters = []
                    potential_items = ranked_fighters_container.find_all(['div', 'li', 'tr'])
                    for item in potential_items:
                        if re.search(r'^\s*#?\d+\s*', item.text.strip()):
                            ranked_fighters.append(item)
                
                print(f"Found {len(ranked_fighters)} potential ranked fighters in {division_name}")
                
                # Track which ranks have been assigned
                assigned_ranks = set()
                
                # Process each ranked fighter
                for i, item in enumerate(ranked_fighters):
                    # Clean up the raw text
                    raw_text = item.text.strip()
                    
                    # Try to extract rank and name using regex
                    rank_match = re.search(r'^\s*#?(\d+)\s*', raw_text)
                    if rank_match:
                        rank_str = rank_match.group(1)
                        try:
                            # Convert to numeric value (ranked #1 is rank 2, etc)
                            numeric_rank = int(rank_str) + 1
                            
                            # Skip if rank already assigned or not in valid range
                            if numeric_rank in occupied_ranks or numeric_rank < 2 or numeric_rank > 16:
                                continue
                            
                            # Extract name after the rank
                            name_text = re.sub(r'^\s*#?\d+\s*', '', raw_text).strip()
                            
                            # Clean up extra text
                            name_text = re.sub(r'\s*\(.*?\)\s*', ' ', name_text)  # Remove parentheses content
                            name_text = re.sub(r'\s+', ' ', name_text).strip()    # Normalize spaces
                            
                            # Try to find the name element directly
                            name_elem = item.find(['h5', 'span', 'div'], class_=['name', 'views-field-title'])
                            if name_elem:
                                name_text = name_elem.text.strip()
                            
                            # Further cleanup - sometimes there's additional text
                            name_parts = []
                            for part in name_text.split():
                                # Skip common non-name parts
                                if part.lower() in ['up', 'down', 'by', 'increased', 'decreased', 'no', 'change', 'nr', 'nc', '-']:
                                    break
                                name_parts.append(part)
                            
                            if name_parts:
                                name = " ".join(name_parts)
                                
                                # Final validation - name should be reasonable length
                                if len(name) > 3:
                                    normalized_name = normalize_name(name)
                                    
                                    # Skip if fighter already has a better rank
                                    if normalized_name in fighter_rankings and fighter_rankings[normalized_name]['numeric_rank'] < numeric_rank:
                                        continue
                                    
                                    print(f"Found ranked fighter: {name} (#{rank_str}) - Rank {numeric_rank}")
                                    fighter_rankings[normalized_name] = {
                                        'original_name': name,
                                        'numeric_rank': numeric_rank,
                                        'division_weight': division_weight,
                                        'is_champion': False
                                    }
                                    
                                    # Mark this rank as occupied
                                    occupied_ranks[numeric_rank] = normalized_name
                                    assigned_ranks.add(numeric_rank)
                        except (ValueError, TypeError):
                            # Couldn't convert rank to int
                            continue
                    
                # Verify we have consistent rankings (1-16)
                division_max_rank = max(assigned_ranks) if assigned_ranks else 0
                if champion_found:
                    expected_ranks = set(range(1, min(division_max_rank + 1, 17)))
                else:
                    expected_ranks = set(range(2, min(division_max_rank + 1, 17)))
                
                missing_ranks = expected_ranks - assigned_ranks
                if missing_ranks:
                    logger.warning(f"Missing ranks in {division_name}: {missing_ranks}")
                
                # Verify we don't have duplicate ranks
                duplicate_check = {}
                for name, data in fighter_rankings.items():
                    if data['division_weight'] == division_weight:
                        rank = data['numeric_rank']
                        if rank in duplicate_check:
                            logger.warning(f"Duplicate rank {rank} in {division_name}: {duplicate_check[rank]} and {name}")
                        duplicate_check[rank] = name
            
            print(f"Processed division {division_name} with {len(occupied_ranks)} fighters")
        
        # Validate the final rankings
        divisions_processed = set()
        for name, data in fighter_rankings.items():
            division = data['division_weight']
            divisions_processed.add(division)
        
        print(f"Processed {len(divisions_processed)} divisions with {len(fighter_rankings)} fighters")
        
        if len(fighter_rankings) == 0:
            logger.warning("No rankings found. Loading cached rankings.")
            return load_cached_rankings()
        
        # Cache the rankings for future use
        cache_rankings(fighter_rankings)
        return fighter_rankings
        
    except requests.RequestException as e:
        logger.error(f"Error fetching UFC rankings: {str(e)}")
        return load_cached_rankings()
    except Exception as e:
        logger.error(f"Error parsing UFC rankings: {str(e)}")
        import traceback
        traceback.print_exc()
        return load_cached_rankings()

def update_fighter_rankings_in_db(fighter_rankings):
    """Update fighter rankings directly in the fighters table"""
    if not fighter_rankings:
        logger.error("No rankings data to update")
        return False
    
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Ensure fighters table exists
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='fighters'
        """)
        
        if not cursor.fetchone():
            # Create fighters table if it doesn't exist
            logger.info("Creating fighters table")
            cursor.execute("""
                CREATE TABLE fighters (
                    fighter_name TEXT PRIMARY KEY NOT NULL,
                    nickname TEXT,
                    height TEXT,
                    weight TEXT,
                    reach TEXT,
                    stance TEXT,
                    dob TEXT,
                    slpm REAL,
                    sapm REAL,
                    str_acc REAL,
                    str_def REAL,
                    td_avg REAL,
                    td_acc REAL,
                    td_def REAL,
                    sub_avg REAL,
                    weight_class TEXT,
                    record TEXT,
                    image_url TEXT,
                    ranking INTEGER,
                    is_champion INTEGER DEFAULT 0,
                    last_ranking_update TEXT
                )
            """)
            logger.info("Created fighters table")
        
        # Check if fighter_rankings table exists and drop it if it does
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='fighter_rankings'
        """)
        
        if cursor.fetchone():
            logger.info("Dropping fighter_rankings table")
            cursor.execute("DROP TABLE fighter_rankings")
            logger.info("Dropped fighter_rankings table")
        
        # Get the column names from the fighters table
        cursor.execute("PRAGMA table_info(fighters)")
        columns = cursor.fetchall()
        column_names = [column[1] for column in columns]  # column[1] is the column name
        
        print(f"Column names in fighters table: {column_names}")
        
        # Check if ranking and is_champion columns exist
        if 'ranking' not in column_names:
            logger.info("Adding ranking column to fighters table")
            cursor.execute("ALTER TABLE fighters ADD COLUMN ranking INTEGER")
        
        if 'is_champion' not in column_names:
            logger.info("Adding is_champion column to fighters table")
            cursor.execute("ALTER TABLE fighters ADD COLUMN is_champion INTEGER DEFAULT 0")
            
        if 'last_ranking_update' not in column_names:
            logger.info("Adding last_ranking_update column to fighters table")
            cursor.execute("ALTER TABLE fighters ADD COLUMN last_ranking_update TEXT")
        
        # Get all fighters from the database
        cursor.execute("SELECT fighter_name FROM fighters")
        db_fighters = cursor.fetchall()
        
        print(f"Found {len(db_fighters)} fighters in the database")
        
        # Initialize counters for statistics
        total_updated = 0
        total_not_found = 0
        
        # First reset all rankings to unranked (99) before applying new rankings
        cursor.execute("""
            UPDATE fighters 
            SET ranking = 99, is_champion = 0
        """)
        
        # Current timestamp for update tracking
        update_timestamp = datetime.now().isoformat()
        
        # Loop through all fighters in the database
        for row in db_fighters:
            fighter_name = row[0]  # fighter_name is the first column
            
            # Skip if fighter_name is None
            if not fighter_name:
                continue
                
            # Normalize the fighter name for better matching
            normalized_db_name = normalize_name(fighter_name)
            
            # Find the closest match in our rankings data
            best_match = None
            best_match_score = 0
            
            for normalized_ranking_name, ranking_data in fighter_rankings.items():
                # Exact match
                if normalized_db_name == normalized_ranking_name:
                    best_match = ranking_data
                    break
                
                # Partial match (one name is contained within the other)
                if normalized_db_name in normalized_ranking_name or normalized_ranking_name in normalized_db_name:
                    score = len(set(normalized_db_name.split()) & set(normalized_ranking_name.split()))
                    if score > best_match_score:
                        best_match = ranking_data
                        best_match_score = score
            
            if best_match:
                # Update the fighter's ranking
                cursor.execute("""
                    UPDATE fighters 
                    SET ranking = ?, is_champion = ?, last_ranking_update = ? 
                    WHERE fighter_name = ?
                """, (
                    best_match['numeric_rank'],
                    1 if best_match['is_champion'] else 0,
                    update_timestamp,
                    fighter_name
                ))
                total_updated += 1
                
                if best_match['is_champion']:
                    logger.info(f"Updated fighter {fighter_name} as champion (rank 1)")
                else:
                    logger.info(f"Updated fighter {fighter_name} with rank {best_match['numeric_rank']}")
            else:
                # This fighter is not in the rankings
                total_not_found += 1
        
        conn.commit()
        logger.info(f"Successfully updated rankings for {total_updated} fighters")
        logger.info(f"{total_not_found} fighters were not found in rankings data")
        
        # Save the timestamp of when rankings were last updated
        update_meta_table(conn, 'rankings_last_updated', update_timestamp)
        
        return True
    
    except Exception as e:
        logger.error(f"Error updating fighter rankings in database: {str(e)}")
        import traceback
        traceback.print_exc()
        if conn:
            conn.rollback()
        return False
    finally:
        if conn:
            conn.close()

def fetch_and_update_rankings():
    """Fetch UFC rankings and update the database"""
    rankings = fetch_ufc_rankings()
    if rankings:
        return update_fighter_rankings_in_db(rankings)
    return False

def generate_fallback_rankings():
    """Generate minimal fallback rankings data for when scraping fails
    
    Instead of hardcoding specific fighters, this function returns
    an empty structure with a warning message in the logs.
    """
    logger.warning("Using minimal fallback rankings - no fighters will be ranked")
    logger.warning("You should try running the scraper again later to get real rankings")
    
    # Return an empty dictionary - we won't rank any fighters when scraping fails
    # rather than using potentially outdated hardcoded ranking data
    return {}

if __name__ == "__main__":
    # When run directly, fetch and update rankings
    print("Running UFC Rankings Scraper directly...")
    try:
        # Create the database directory if it doesn't exist
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        
        print(f"Database directory: {os.path.dirname(DB_PATH)}")
        
        # Test database connection
        try:
            conn = get_db_connection()
            conn.execute("SELECT 1")
            conn.close()
            print("✅ Database connection successful")
        except Exception as e:
            print(f"❌ Database connection error: {str(e)}")
        
        # Fetch and update rankings
        rankings = fetch_ufc_rankings()
        if rankings:
            # Count fighters by division
            division_counts = {}
            for _, data in rankings.items():
                division = data['division_weight']
                if division not in division_counts:
                    division_counts[division] = 0
                division_counts[division] += 1
            
            print(f"Got rankings for {len(rankings)} fighters across {len(division_counts)} divisions:")
            for division, count in sorted(division_counts.items()):
                # Map weight to division name
                division_name = next((name for name, weight in WEIGHT_CLASSES.items() if weight == division), f"{division}lbs")
                print(f"  - {division_name}: {count} fighters")
            
            result = update_fighter_rankings_in_db(rankings)
            if result:
                print("✅ Successfully updated UFC fighter rankings in the database!")
                
                # Print champion information
                print("\nChampions in the database:")
                try:
                    conn = get_db_connection()
                    cursor = conn.cursor()
                    cursor.execute("SELECT fighter_name, ranking, Weight FROM fighters WHERE is_champion=1 ORDER BY Weight")
                    champions = cursor.fetchall()
                    
                    if champions:
                        for champion in champions:
                            weight_class = champion[2] if len(champion) > 2 and champion[2] else "Unknown"
                            print(f"  - {champion[0]} (Rank {champion[1]}, {weight_class})")
                    else:
                        print("  No champions found in database")
                    
                    # Show counts by division in the database
                    print("\nUpdated fighters by division:")
                    cursor.execute("""
                        SELECT Weight, COUNT(*) 
                        FROM fighters 
                        WHERE ranking < 99 
                        GROUP BY Weight 
                        ORDER BY Weight
                    """)
                    divisions = cursor.fetchall()
                    
                    if divisions:
                        for division in divisions:
                            if division[0]:
                                print(f"  - {division[0]}: {division[1]} fighters")
                    
                    # Check for specific fighter (Brendan Allen)
                    print("\nChecking middleweight fighters:")
                    cursor.execute("SELECT fighter_name, ranking, is_champion FROM fighters WHERE Weight LIKE '%185%' AND ranking < 99 ORDER BY ranking")
                    fighters = cursor.fetchall()
                    
                    if fighters:
                        for fighter in fighters[:10]:  # Show first 10
                            print(f"  - Rank {fighter[1]}: {fighter[0]} (Champion: {fighter[2]})")
                        if len(fighters) > 10:
                            print(f"  - ... and {len(fighters) - 10} more fighters")
                    else:
                        print("  No ranked middleweight fighters found")
                    
                    # Check Brendan Allen specifically
                    cursor.execute("SELECT fighter_name, ranking, is_champion, Weight, last_ranking_update FROM fighters WHERE fighter_name LIKE '%Brendan Allen%'")
                    brendan = cursor.fetchone()
                    if brendan:
                        print(f"\nBrendan Allen: Rank {brendan[1]}, Champion: {brendan[2]}, Weight: {brendan[3]}, Last Updated: {brendan[4]}")
                    else:
                        print("\nBrendan Allen not found in database")
                    
                    # Check when rankings were last updated
                    cursor.execute("SELECT value FROM meta WHERE key='rankings_last_updated'")
                    last_update = cursor.fetchone()
                    if last_update:
                        print(f"\nRankings last updated at: {last_update[0]}")
                    
                    conn.close()
                except Exception as e:
                    print(f"Error checking database: {str(e)}")
            else:
                print("❌ Failed to update rankings in the database.")
        else:
            print("❌ Failed to fetch rankings data.")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc() 