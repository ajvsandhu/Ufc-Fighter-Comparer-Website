"""
UFC Rankings Scraper

This module fetches fighter rankings from the UFC website and stores them directly
in the fighters table. Fighter rankings are used to improve prediction accuracy by
factoring in the strength of a fighter's competition.
"""

import os
import sys
import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from unicodedata import normalize

import requests
from bs4 import BeautifulSoup

# Configure import paths
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    sys.path.insert(0, project_root)

# Directory configuration
BASE_DIR = (Path(project_root) if 'project_root' in locals() 
           else Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
DATA_DIR = BASE_DIR / "data"

# HTTP client configuration
REQUEST_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
}
REQUEST_TIMEOUT = 15
RETRY_ATTEMPTS = 3
RETRY_DELAY = 2

# Logger configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Database client import
try:
    from backend.api.database import get_supabase_client
except ImportError:
    def get_supabase_client():
        """Fallback implementation if actual client is unavailable"""
        raise ImportError("Could not import get_supabase_client from backend.api.database")

# Initialize logger
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format=LOG_FORMAT,
    datefmt=LOG_DATE_FORMAT
)
logger = logging.getLogger(__name__)

# Source data URL
UFC_RANKINGS_URL = "https://www.ufc.com/rankings"

# Weight class mapping for division normalization
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

# Cache file location
CACHED_RANKINGS_PATH = os.path.join(DATA_DIR, "cached_rankings.json")

def cache_rankings(rankings: Dict[str, Any]) -> bool:
    """
    Cache the rankings to a file for future use.
    
    Args:
        rankings: Dictionary containing fighter rankings data
        
    Returns:
        bool: True if caching was successful, False otherwise
    """
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

def update_meta_table(supabase, key, value):
    """Update or create a meta table for tracking information like last update timestamps"""
    try:
        # Update or insert the key-value pair in Supabase
        current_time = datetime.now().isoformat()
        
        # Check if the key exists first
        response = supabase.table('meta') \
            .select('key') \
            .eq('key', key) \
            .execute()
            
        if response.data and len(response.data) > 0:
            # Update existing record
            supabase.table('meta') \
                .update({'value': value, 'updated_at': current_time}) \
                .eq('key', key) \
                .execute()
        else:
            # Insert new record
            supabase.table('meta') \
                .insert({'key': key, 'value': value, 'updated_at': current_time}) \
                .execute()
        
        logger.info(f"Updated meta table: {key} = {value}")
        return True
    except Exception as e:
        logger.error(f"Error updating meta table: {str(e)}")
        return False

def normalize_name(name):
    """
    Normalize a fighter name for consistent matching.
    
    Args:
        name: Raw fighter name
        
    Returns:
        str: Normalized fighter name for matching
    """
    if not name:
        return ""
        
    # Convert to lowercase
    name = name.lower()
    
    # Handle special character replacement
    name = normalize('NFKD', name).encode('ASCII', 'ignore').decode('utf-8')
    
    # Remove extra whitespace
    name = ' '.join(name.split())
    
    # Remove common patterns that might differ between sources
    name = re.sub(r'[\(\)\[\]\{\}\'\"\.,-]', '', name)
    
    # Remove common suffixes that appear in some sources but not others
    name = re.sub(r'\s+(jr|sr|iii|iv|ii|i)$', '', name)
    
    return name.strip()

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
        conn = get_supabase_client()
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

def ensure_fighters_table_has_ranking_column(supabase):
    """
    Ensure the fighters table has a ranking column.
    
    For Supabase, we assume the column exists in the schema.
    If it doesn't, you should update the schema through the Supabase dashboard.
    """
    logger.info("Verifying ranking column exists in Supabase schema")
    # With Supabase, we assume the ranking column exists
    # If it doesn't, your data operations will still work but might not be
    # visible without a schema update through Supabase dashboard
    return True

def update_fighter_ranking(supabase, fighter_name, ranking, is_champion=False, division=None, position=None):
    """
    Update a fighter's ranking in the database.
    
    Args:
        supabase: Supabase client instance
        fighter_name: Name of the fighter to update
        ranking: Current ranking, e.g., "C" for champion, or numeric position
        is_champion: Whether this fighter is a champion
        division: Weight division
        position: Position in rankings (1-15)
        
    Returns:
        bool: True if update was successful, False otherwise
    """
    try:
        # Prepare update data
        update_data = {'ranking': ranking}
        
        if is_champion is not None:
            update_data['is_champion'] = is_champion
            
        if division is not None:
            update_data['weight_class'] = division
            
        # Execute the update
        response = supabase.table('fighters') \
            .update(update_data) \
            .eq('fighter_name', fighter_name) \
            .execute()
            
        # Check if update was successful
        if response.data and len(response.data) > 0:
            logger.debug(f"Updated ranking for {fighter_name}: {ranking}")
            return True
        else:
            logger.warning(f"Failed to update ranking for {fighter_name}: fighter not found")
            return False
    except Exception as e:
        logger.error(f"Error updating ranking for {fighter_name}: {str(e)}")
        return False

def find_fighter_in_db(supabase, fighter_name):
    """
    Find a fighter in the database by name, with fuzzy matching.
    
    Args:
        supabase: Supabase client
        fighter_name: Name to search for
        
    Returns:
        dict: Fighter data if found, None otherwise
    """
    try:
        # Try exact match first
        response = supabase.table('fighters') \
            .select('*') \
            .eq('fighter_name', fighter_name) \
            .execute()
        
        if response.data and len(response.data) > 0:
            return response.data[0]
            
        # If not found, try normalized name
        normalized = normalize_name(fighter_name)
        
        # Get all fighters with pagination to handle large datasets
        page_size = 1000
        all_fighters = []
        page = 0
        
        while True:
            # Fetch a page of fighters
            response = supabase.table('fighters') \
                .select('fighter_name') \
                .range(page * page_size, (page + 1) * page_size - 1) \
                .execute()
            
            # Add fighters to our list
            fighters_page = response.data
            all_fighters.extend(fighters_page)
            
            # If we got fewer results than the page size, we've reached the end
            if len(fighters_page) < page_size:
                break
                
            # Move to next page
            page += 1
        
        # Find best match by comparing normalized names
        best_match = None
        best_score = 0
        
        for fighter in all_fighters:
            db_name = fighter['fighter_name']
            db_normalized = normalize_name(db_name)
            
            # Simple scoring: count matching characters
            score = sum(c1 == c2 for c1, c2 in zip(normalized, db_normalized)) / max(len(normalized), len(db_normalized))
            
            if score > 0.85 and score > best_score:
                best_score = score
                best_match = db_name
                
        if best_match:
            # Get the full record for the best match
            response = supabase.table('fighters') \
                .select('*') \
                .eq('fighter_name', best_match) \
                .execute()
                
            if response.data and len(response.data) > 0:
                logger.info(f"Fuzzy matched '{fighter_name}' to '{best_match}' (score: {best_score:.2f})")
                return response.data[0]
                
        return None
    except Exception as e:
        logger.error(f"Error finding fighter in database: {str(e)}")
        return None

def update_fighter_rankings(supabase, rankings_data):
    """
    Update fighter rankings in the database.
    
    Args:
        supabase: Supabase client
        rankings_data: Dictionary of fighter rankings by name
        
    Returns:
        tuple: Count of updates (successful, failed, not_found)
    """
    try:
        # Ensure ranking columns exist
        ensure_fighters_table_has_ranking_column(supabase)
        
        success_count = 0
        failed_count = 0
        not_found_count = 0
        
        # Process each fighter's ranking
        for fighter_name, data in rankings_data.items():
            # Find the fighter in the database
            fighter = find_fighter_in_db(supabase, fighter_name)
            
            if fighter:
                # Update the fighter's ranking
                success = update_fighter_ranking(
                    supabase,
                    fighter['fighter_name'],  # Use the exact name from the database
                    data.get('ranking', ''),
                    data.get('is_champion', False),
                    data.get('division', '')
                )
                
                if success:
                    success_count += 1
                else:
                    failed_count += 1
            else:
                logger.warning(f"Fighter not found in database: {fighter_name}")
                not_found_count += 1
                
        # Update the last rankings update timestamp
        update_meta_table(supabase, 'last_rankings_update', datetime.now().isoformat())
        
        return success_count, failed_count, not_found_count
    except Exception as e:
        logger.error(f"Error updating fighter rankings: {str(e)}")
        return 0, 0, 0

def main():
    """
    Main function for the UFC rankings scraper.
    
    This function:
    1. Fetches rankings from UFC website (or uses cache if recent)
    2. Updates the database with the latest rankings
    3. Logs the results
    """
    logger.info("Starting UFC rankings scraper")
    
    try:
        # Get Supabase connection
        supabase = get_supabase_client()
        
        # Load cached rankings if available and recent
        cached_rankings = load_cached_rankings()
        
        if args.force_refresh or not cached_rankings:
            logger.info("Fetching fresh rankings from UFC website")
            rankings = fetch_ufc_rankings()
            
            if rankings:
                # Cache the new rankings
                cache_rankings(rankings)
            else:
                if cached_rankings:
                    logger.warning("Failed to fetch fresh rankings, using cached data")
                    rankings = cached_rankings
                else:
                    logger.error("Failed to fetch rankings and no cache available")
                    return
        else:
            logger.info("Using cached rankings data")
            rankings = cached_rankings
            
        # Update the database with the rankings
        success, failed, not_found = update_fighter_rankings(supabase, rankings)
        
        logger.info(f"Rankings update complete:")
        logger.info(f" - {success} fighters updated successfully")
        logger.info(f" - {failed} updates failed")
        logger.info(f" - {not_found} fighters not found in database")
        
        # Update timestamp
        update_meta_table(supabase, 'last_rankings_update', datetime.now().isoformat())
        
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")

if __name__ == "__main__":
    # Set up argument parser
    import argparse
    parser = argparse.ArgumentParser(description='UFC Rankings Scraper')
    parser.add_argument('--force-refresh', action='store_true', help='Force refresh of rankings data')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    args = parser.parse_args()
    
    # Set debug level if requested
    if args.debug:
        logger.setLevel(logging.DEBUG)
        
    # When run directly, fetch and update rankings
    main() 