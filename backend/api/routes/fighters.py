from fastapi import APIRouter, Query, HTTPException
from backend.api.database import get_db_connection
import re
from typing import List, Dict
import logging
from urllib.parse import unquote
from backend.constants import (
    API_V1_STR,
    MAX_SEARCH_RESULTS,
    MAX_FIGHTS_DISPLAY,
    DEFAULT_RECORD,
    UNRANKED_VALUE
)
import traceback
from backend.utils import sanitize_json, set_parent_key

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix=API_V1_STR, tags=["Fighters"])

@router.get("/fighters")
def get_fighters(query: str = Query("", min_length=0)):
    """Get all fighters or search for fighters by name."""
    try:
        supabase = get_db_connection()
        if not supabase:
            logger.error("No database connection available")
            raise HTTPException(status_code=500, detail="Database connection error")

        # Initialize an empty fighters list as fallback
        fighters_list = []

        try:
            # Fetch fighters data from Supabase
            if not query:
                # If no query, return all fighters
                try:
                    # First try with nulls_last parameter
                    response = supabase.table('fighters')\
                        .select('fighter_name,Record,ranking,id')\
                        .order('ranking', desc=False, nulls_last=True)\
                        .limit(MAX_SEARCH_RESULTS)\
                        .execute()
                except Exception:
                    # Fall back to simpler ordering if the above fails
                    response = supabase.table('fighters')\
                        .select('fighter_name,Record,ranking,id')\
                        .order('ranking')\
                        .limit(MAX_SEARCH_RESULTS)\
                        .execute()
            else:
                # If query exists, use ilike for case-insensitive search
                response = supabase.table('fighters')\
                    .select('fighter_name,Record,ranking,id')\
                    .ilike('fighter_name', f'%{query}%')\
                    .order('ranking')\
                    .limit(MAX_SEARCH_RESULTS)\
                    .execute()
            
            # Handle case where response or data is None
            if not response or not hasattr(response, 'data') or not response.data:
                logger.info(f"No fighters found for query: {query}")
                return sanitize_json({"fighters": []})
            
            fighter_data = response.data
            logger.info(f"Found {len(fighter_data)} fighters matching query: {query}")
        except Exception as e:
            logger.error(f"Error fetching fighters: {str(e)}")
            return sanitize_json({"fighters": []})  # Return empty array instead of raising error
        
        fighters_list = []
        
        if not query:
            # Return all fighters with record and ranking info
            for fighter in fighter_data:
                # Set parent keys for important fields for enhanced sanitization
                set_parent_key("fighter_name")
                set_parent_key("Record")
                
                # Sanitize fighter data
                sanitized_fighter = sanitize_json(fighter)
                
                # Explicitly ensure fighter_name and record are valid strings
                fighter_name = sanitized_fighter.get('fighter_name', '')
                if fighter_name is None or not isinstance(fighter_name, str):
                    fighter_name = ''  # Double-check fighter_name is a valid string
                
                record = sanitized_fighter.get('Record', DEFAULT_RECORD) 
                if record is None or not isinstance(record, str):
                    record = DEFAULT_RECORD  # Double-check record is a valid string
                
                # Always return as a valid string
                formatted_name = f"{fighter_name} ({record})"
                fighters_list.append(formatted_name)
        else:
            # Improved search logic
            query_parts = query.lower().split()
            
            for fighter in fighter_data:
                # Set parent keys for important fields for enhanced sanitization
                set_parent_key("fighter_name")
                set_parent_key("Record")
                
                # Sanitize fighter data
                sanitized_fighter = sanitize_json(fighter)
                
                # Explicitly ensure fighter_name and record are valid strings
                fighter_name = sanitized_fighter.get('fighter_name', '')
                if fighter_name is None or not isinstance(fighter_name, str):
                    fighter_name = ''  # Double-check fighter_name is a valid string
                
                record = sanitized_fighter.get('Record', DEFAULT_RECORD)
                if record is None or not isinstance(record, str):
                    record = DEFAULT_RECORD  # Double-check record is a valid string
                
                # Only process if fighter_name is not empty
                if fighter_name:
                    ranking = sanitized_fighter.get('ranking')
                    
                    # Split fighter name into parts for matching
                    name_parts = fighter_name.lower().split()
                    
                    # Check for matches:
                    # 1. Full name contains query
                    # 2. Any part of name starts with any query part
                    # 3. Any part of name contains any query part
                    matches = False
                    
                    # Full name contains entire query
                    if query.lower() in fighter_name.lower():
                        matches = True
                    else:
                        # Check if any query part matches start of any name part
                        for q_part in query_parts:
                            for name_part in name_parts:
                                if name_part.startswith(q_part):
                                    matches = True
                                    break
                            if matches:
                                break
                                
                    if matches:
                        # Format name with record and add to results
                        formatted_name = f"{fighter_name} ({record})"
                        fighters_list.append(formatted_name)
        
        # Final safety check - ensure all items are strings
        fighters_list = [str(name) for name in fighters_list if name]
        
        # Return result in expected format
        logger.info(f"Returning {len(fighters_list)} fighters")
        return sanitize_json({"fighters": fighters_list[:MAX_SEARCH_RESULTS]})
    except Exception as e:
        logger.error(f"Unexpected error in get_fighters: {str(e)}")
        # Return empty list instead of error to avoid breaking frontend
        return sanitize_json({"fighters": []})

@router.get("/fighter-stats/{fighter_name}")
def get_fighter_stats(fighter_name: str):
    """Get fighter stats by name."""
    try:
        supabase = get_db_connection()
        if not supabase:
            logger.error("No database connection available")
            raise HTTPException(status_code=500, detail="Database connection error")
        
        # Clean fighter name - remove record if present
        if "(" in fighter_name:
            fighter_name = fighter_name.split("(")[0].strip()
        
        # Fetch fighter stats from Supabase
        response = supabase.table('fighters')\
            .select('*')\
            .eq('fighter_name', fighter_name)\
            .execute()
        
        if not response.data:
            logger.warning(f"Fighter not found: {fighter_name}")
            raise HTTPException(status_code=404, detail=f"Fighter not found: {fighter_name}")
        
        # Return first matching fighter
        fighter_data = response.data[0]
        logger.info(f"Retrieved stats for fighter: {fighter_name}")
        return sanitize_json(fighter_data)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching fighter stats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

def _sanitize_string(value, default=""):
    """Ensure a value is a valid string to prevent frontend errors."""
    if value is None:
        return default
    if not isinstance(value, str):
        try:
            return str(value)
        except:
            return default
    # Return empty string for "null", "undefined" or "None" strings
    if value.lower() in ("null", "undefined", "none"):
        return default
    return value

def _sanitize_fighter_data(fighter_data):
    """Sanitize all string fields in fighter data to prevent frontend errors."""
    if not fighter_data:
        return {}
        
    # For each critical field, set the parent key before sanitizing
    for key in ['fighter_name', 'Record', 'Height', 'Weight', 'Reach', 'STANCE', 'DOB',
               'Str. Acc.', 'Str. Def', 'TD Acc.', 'TD Def.']:
        if key in fighter_data:
            set_parent_key(key)
            
    # Handle last_5_fights if present
    if 'last_5_fights' in fighter_data and fighter_data['last_5_fights']:
        # Make sure it's a list
        if not isinstance(fighter_data['last_5_fights'], list):
            fighter_data['last_5_fights'] = [fighter_data['last_5_fights']]
        
        # Sanitize each fight in the list
        for i, fight in enumerate(fighter_data['last_5_fights']):
            # Set parent keys for important fight fields
            for field in ['opponent_name', 'result', 'method', 'round', 'time', 'date', 'event']:
                if field in fight:
                    set_parent_key(field)
            
            # Convert any None values to appropriate defaults
            if isinstance(fight, dict):
                for field in ['opponent_name', 'result', 'method', 'time', 'date', 'event']:
                    if field not in fight or fight[field] is None:
                        fight[field] = ""
                        
                # Ensure round is a number
                if 'round' not in fight or fight['round'] is None:
                    fight['round'] = 0
            
        # Log the fights for debugging
        logger.info(f"Sanitized {len(fighter_data['last_5_fights'])} fights")
            
    # Apply the enhanced sanitize_json function
    sanitized = sanitize_json(fighter_data)
            
    # Ensure critical fields exist with defaults
    if 'fighter_name' not in sanitized or not sanitized['fighter_name']:
        sanitized['fighter_name'] = ""
    if 'Record' not in sanitized or not sanitized['Record']:
        sanitized['Record'] = "0-0-0"
    if 'Height' not in sanitized or not sanitized['Height']:
        sanitized['Height'] = ""
    if 'Weight' not in sanitized or not sanitized['Weight']:
        sanitized['Weight'] = ""
    if 'Reach' not in sanitized or not sanitized['Reach']:
        sanitized['Reach'] = ""
    if 'STANCE' not in sanitized or not sanitized['STANCE']:
        sanitized['STANCE'] = ""
    if 'DOB' not in sanitized or not sanitized['DOB']:
        sanitized['DOB'] = ""
    if 'image_url' not in sanitized or not sanitized['image_url']:
        sanitized['image_url'] = ""
        
    return sanitized

@router.get("/fighter/{fighter_name}")
def get_fighter(fighter_name: str):
    """Get fighter by name - alias for frontend compatibility."""
    try:
        # URL decode the fighter name and ensure it's a string
        if fighter_name is None:
            logger.warning("Fighter name is None")
            raise HTTPException(status_code=400, detail="Fighter name is required")
            
        try:
            fighter_name = unquote(fighter_name)
        except Exception as e:
            logger.error(f"Error decoding fighter name: {str(e)}")
            # Continue with the original name if decoding fails
        
        # Log the requested fighter name to help with debugging
        logger.info(f"Fighter lookup requested for: {fighter_name}")
        
        supabase = get_db_connection()
        if not supabase:
            logger.error("No database connection available")
            # Return empty fighter with defaults instead of error
            return sanitize_json(_get_default_fighter(fighter_name))
        
        # Clean fighter name - remove record if present
        clean_name = fighter_name
        if "(" in fighter_name:
            clean_name = fighter_name.split("(")[0].strip()
            logger.info(f"Extracted clean name: {clean_name}")
        
        # Try multiple search methods to maximize chances of finding the fighter
        fighter_data = None
        
        # Method 1: Direct match
        response = supabase.table('fighters')\
            .select('*')\
            .eq('fighter_name', clean_name)\
            .execute()
            
        if response and hasattr(response, 'data') and response.data and len(response.data) > 0:
            fighter_data = response.data[0]
            logger.info(f"Found fighter via direct match: {clean_name}")
        
        # Method 2: Case insensitive match if direct match failed
        if not fighter_data:
            response = supabase.table('fighters')\
                .select('*')\
                .ilike('fighter_name', clean_name)\
                .execute()
                
            if response and hasattr(response, 'data') and response.data and len(response.data) > 0:
                fighter_data = response.data[0]
                logger.info(f"Found fighter via case-insensitive match: {clean_name}")
        
        # Method 3: Partial match if previous methods failed
        if not fighter_data:
            response = supabase.table('fighters')\
                .select('*')\
                .ilike('fighter_name', f'%{clean_name}%')\
                .limit(1)\
                .execute()
                
            if response and hasattr(response, 'data') and response.data and len(response.data) > 0:
                fighter_data = response.data[0]
                logger.info(f"Found fighter via partial match: {clean_name}")
        
        # If all methods failed, return a default fighter object instead of 404
        if not fighter_data:
            logger.warning(f"Fighter not found with any method: {clean_name}")
            return sanitize_json(_get_default_fighter(clean_name))
        
        # FIXED: Simplify the query to directly match on fighter_name
        last_5_fights = []
        try:
            logger.info(f"Directly fetching last 5 fights for fighter name: {clean_name}")
            
            # Simple direct query using fighter_name instead of fighter_id
            fights_response = supabase.table('fighter_last_5_fights')\
                .select('*')\
                .eq('fighter_name', clean_name)\
                .limit(MAX_FIGHTS_DISPLAY)\
                .execute()
            
            if fights_response and hasattr(fights_response, 'data') and fights_response.data:
                last_5_fights = fights_response.data
                logger.info(f"Found {len(last_5_fights)} fights for fighter {clean_name}")
            else:
                logger.warning(f"No fights found for fighter {clean_name}")
        except Exception as e:
            logger.error(f"Error fetching fights for fighter {clean_name}: {str(e)}")
            # Continue even if we can't get fights
        
        # Add the last 5 fights to the fighter data
        fighter_data['last_5_fights'] = last_5_fights
        
        # Sanitize all fields to ensure proper string values
        sanitized_data = _sanitize_fighter_data(fighter_data)
        
        logger.info(f"Successfully retrieved fighter: {clean_name} with {len(last_5_fights)} fights")
        return sanitize_json(sanitized_data)
    except Exception as e:
        logger.error(f"Error in get_fighter: {str(e)}")
        logger.error(traceback.format_exc())
        # Return a default fighter object rather than an error
        return sanitize_json(_get_default_fighter(fighter_name if fighter_name else "Unknown Fighter"))

def _get_default_fighter(name):
    """Return a default fighter object with the given name."""
    return {
        "fighter_name": name,
        "Record": "0-0-0",
        "Height": "",
        "Weight": "",
        "Reach": "",
        "STANCE": "",
        "DOB": "",
        "SLpM": 0.0,
        "Str. Acc.": "0%", 
        "SApM": 0.0,
        "Str. Def": "0%",
        "TD Avg.": 0.0,
        "TD Acc.": "0%",
        "TD Def.": "0%",
        "Sub. Avg.": 0.0,
        "image_url": "",
        "ranking": 0,
        "is_champion": False
    }

@router.get("/fighter-details/{fighter_name}")
def get_fighter_details(fighter_name: str):
    """Get detailed fighter information by name."""
    try:
        # Use the same logic as get_fighter_stats but with more detailed logging
        fighter_data = get_fighter_stats(fighter_name)
        logger.info(f"Retrieved detailed information for fighter: {fighter_name}")
        return sanitize_json(fighter_data)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching fighter details: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

@router.get("/fighter-average-stats")
def get_fighter_average_stats():
    """Get average stats across all fighters."""
    try:
        supabase = get_db_connection()
        if not supabase:
            logger.error("No database connection available")
            raise HTTPException(status_code=500, detail="Database connection error")
        
        # Fetch all fighters data from Supabase for calculating averages
        response = supabase.table('fighters').select('*').execute()
        
        if not response.data:
            logger.warning("No fighters found in database")
            raise HTTPException(status_code=404, detail="No fighters found")
        
        fighters = response.data
        
        # Calculate averages of numerical fields
        numeric_fields = [
            'SSLA', 'SApM', 'SSA', 'TDA', 'TDD', 'KD', 'SLPM', 'StrAcc', 'StrDef', 'SUB', 'TD',
            'Height', 'Weight', 'Reach', 'Win', 'Loss', 'Draw', 'winratio'
        ]
        
        # Initialize sums and counts
        sums = {field: 0 for field in numeric_fields}
        counts = {field: 0 for field in numeric_fields}
        
        # Sum up values
        for fighter in fighters:
            for field in numeric_fields:
                if field in fighter and fighter[field] is not None:
                    try:
                        # Convert to float if it's a string
                        value = float(fighter[field]) if isinstance(fighter[field], str) else fighter[field]
                        sums[field] += value
                        counts[field] += 1
                    except (ValueError, TypeError):
                        # Skip if conversion fails
                        pass
        
        # Calculate averages
        averages = {}
        for field in numeric_fields:
            if counts[field] > 0:
                averages[field] = round(sums[field] / counts[field], 2)
            else:
                averages[field] = 0
        
        logger.info("Calculated average stats across all fighters")
        return sanitize_json(averages)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error calculating fighter average stats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

def process_fighter_name(raw_name: str) -> str:
    """
    Keep the nickname if the fighter has at least a first and last name,
    e.g. Israel "The Last Stylebender" Adesanya, but remove the nickname
    if there's only one name, e.g. "The Lion" Kangwang => 'Kangwang'.
    """
    # Check if there is a quoted nickname
    # We look for either single or double quotes
    pattern = r'["\']([^"\']+)["\']'
    match = re.search(pattern, raw_name)
    if not match:
        # No nickname in quotes, just return as-is
        return raw_name.strip()

    # If there is a nickname, split out the quoted part
    nickname = match.group(1)
    # Remove the nickname portion from the full string, leaving the outside
    outside_parts = re.sub(pattern, '', raw_name).strip()

    # Count how many words remain outside the quotes
    word_count = len(outside_parts.split())
    if word_count >= 2:
        # If there are at least two words (e.g. "Israel Adesanya"), keep the nickname
        return raw_name.strip()
    else:
        # Only one name outside the quotes, remove the nickname entirely
        return outside_parts

@router.post("/scrape_and_store_fighters")
def scrape_and_store_fighters(fighters: List[Dict]):
    """
    Endpoint to store fighters in the Supabase database.
    
    This endpoint processes fighter names and stores them in the database.
    """
    try:
        supabase = get_db_connection()
        if not supabase:
            logger.error("No database connection available")
            raise HTTPException(status_code=500, detail="Database connection error")
            
        success_count = 0
        error_count = 0
        
        for fighter in fighters:
            # Process the fighter name if needed
            if "fighter_name" in fighter:
                fighter["fighter_name"] = process_fighter_name(fighter["fighter_name"])
                
            try:
                # Upsert the fighter data to the database
                response = supabase.table("fighters").upsert(fighter, on_conflict="fighter_name").execute()
                if response and hasattr(response, 'data') and response.data:
                    success_count += 1
                else:
                    error_count += 1
                    logger.warning(f"Failed to insert fighter: {fighter.get('fighter_name', 'Unknown')}")
            except Exception as e:
                error_count += 1
                logger.error(f"Error inserting fighter {fighter.get('fighter_name', 'Unknown')}: {str(e)}")
        
        return sanitize_json({
            "status": "success",
            "detail": f"Processed {len(fighters)} fighters. {success_count} succeeded, {error_count} failed."
        })
    except Exception as e:
        logger.error(f"Error in scrape_and_store_fighters: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

