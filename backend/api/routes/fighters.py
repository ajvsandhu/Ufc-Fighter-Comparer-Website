from fastapi import APIRouter, Query, HTTPException
import sqlite3
from backend.api.database import get_db_connection
import re
from typing import List, Dict
import logging
from urllib.parse import unquote
from backend.constants import (
    API_V1_PREFIX,
    MAX_SEARCH_RESULTS,
    MAX_FIGHTS_DISPLAY,
    DEFAULT_RECORD,
    UNRANKED_VALUE
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix=API_V1_PREFIX, tags=["Fighters"])

@router.get("/fighters")
def get_fighters(query: str = Query("", min_length=0)):
    """Get all fighters or search for fighters by name."""
    try:
        supabase = get_db_connection()
        if not supabase:
            logger.error("No database connection available")
            raise HTTPException(status_code=500, detail="Database connection error")

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
            
            if not response.data:
                return {"fighters": []}
            
            fighter_data = response.data
            logger.info(f"Found {len(fighter_data)} fighters matching query: {query}")
        except Exception as e:
            logger.error(f"Error fetching fighters: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
        
        fighters_list = []
        
        if not query:
            # Return all fighters with record and ranking info
            for fighter in fighter_data:
                fighter_name = fighter.get('fighter_name')
                record = fighter.get('Record') if fighter.get('Record') else DEFAULT_RECORD
                
                formatted_name = f"{fighter_name} ({record})"
                fighters_list.append(formatted_name)
        else:
            # Improved search logic
            query_parts = query.lower().split()
            
            for fighter in fighter_data:
                fighter_name = fighter.get('fighter_name')
                record = fighter.get('Record') if fighter.get('Record') else DEFAULT_RECORD
                ranking = fighter.get('ranking')
                
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
        
        # Return result in expected format
        logger.info(f"Returning {len(fighters_list)} fighters")
        return {"fighters": fighters_list[:MAX_SEARCH_RESULTS]}
    except Exception as e:
        logger.error(f"Unexpected error in get_fighters: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

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
        return fighter_data
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching fighter stats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

@router.get("/fighter-details/{fighter_name}")
def get_fighter_details(fighter_name: str):
    """Get detailed fighter information by name."""
    try:
        # Use the same logic as get_fighter_stats but with more detailed logging
        fighter_data = get_fighter_stats(fighter_name)
        logger.info(f"Retrieved detailed information for fighter: {fighter_name}")
        return fighter_data
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
        return averages
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
    Example route to show how you'd store fighters so that the DB ends up
    with the nickname if there's a first+last name, or no nickname if
    there's only one name.
    In your real code, adapt this insertion logic to wherever your
    actual scraper populates the DB.
    """
    conn = get_db_connection()
    cur = conn.cursor()

    for f in fighters:
        # Suppose each 'f' is a dict like:
        # {"fighter_name": "Israel 'The Last Stylebender' Adesanya", "image_url": "...", etc.}
        processed_name = process_fighter_name(f["fighter_name"])
        # Insert into the DB with the processed name
        try:
            cur.execute(
                """
                INSERT INTO fighters (fighter_name, image_url, Record, Height, Weight, Reach,
                                      STANCE, DOB, SLpM, [Str. Acc.], SApM, [Str. Def],
                                      [TD Avg.], [TD Acc.], [TD Def.], [Sub. Avg.], fighter_url, tap_link)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    processed_name,
                    f.get("image_url", ""),
                    f.get("Record", ""),
                    f.get("Height", ""),
                    f.get("Weight", ""),
                    f.get("Reach", ""),
                    f.get("STANCE", ""),
                    f.get("DOB", ""),
                    f.get("SLpM", ""),
                    f.get("Str. Acc.", ""),
                    f.get("SApM", ""),
                    f.get("Str. Def", ""),
                    f.get("TD Avg.", ""),
                    f.get("TD Acc.", ""),
                    f.get("TD Def.", ""),
                    f.get("Sub. Avg.", ""),
                    f.get("fighter_url", ""),
                    f.get("tap_link", "")  # Add tap_link to insertion
                )
            )
        except sqlite3.Error as e:
            logger.error(f"Database insertion error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to insert fighter: {str(e)}")

    conn.commit()
    conn.close()
    return {"detail": "Fighters inserted successfully, with correct nickname handling."}

