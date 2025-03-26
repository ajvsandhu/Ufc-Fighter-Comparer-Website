from fastapi import APIRouter, Query, HTTPException
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

router = APIRouter(prefix=API_V1_PREFIX)

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
        
        return {"fighters": fighters_list[:MAX_SEARCH_RESULTS]}
        
    except Exception as e:
        logger.error(f"Error in get_fighters: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

@router.get("/fighter/{name:path}")
def get_fighter_stats(name: str):
    """Get detailed stats for a specific fighter."""
    logger.info(f"Getting stats for fighter: {name}")
    supabase = get_db_connection()
    if not supabase:
        logger.error("No database connection available")
        raise HTTPException(status_code=500, detail="Database connection error")

    try:
        # Decode URL-encoding and clean the name
        name = unquote(name).strip()
        logger.info(f"Decoded name: {name}")
        
        # Extract base name and record if present
        base_name = name
        record = None
        
        # Check if name contains record in parentheses
        if "(" in name and ")" in name:
            base_name = name.split("(")[0].strip()
            record_part = name.split("(")[1].split(")")[0].strip()
            record = record_part
            logger.info(f"Extracted base_name: {base_name}, record: {record}")

        # Initialize fighter_info to None
        fighter_info = None
        
        # First try exact match with record if available
        if record:
            response = supabase.table('fighters')\
                .select('*')\
                .eq('fighter_name', base_name)\
                .eq('Record', record)\
                .limit(1)\
                .execute()
                
            if response.data and len(response.data) > 0:
                fighter_info = response.data[0]
                logger.info(f"Found fighter by exact name and record match: {fighter_info['fighter_name']}")

        # If no match with record, try just the name
        if not fighter_info:
            response = supabase.table('fighters')\
                .select('*')\
                .ilike('fighter_name', base_name)\
                .limit(1)\
                .execute()
                
            if response.data and len(response.data) > 0:
                fighter_info = response.data[0]
                logger.info(f"Found fighter by name only: {fighter_info['fighter_name']}")

        # If still no match, try fuzzy match
        if not fighter_info:
            response = supabase.table('fighters')\
                .select('*')\
                .ilike('fighter_name', f'%{base_name}%')\
                .limit(1)\
                .execute()
                
            if response.data and len(response.data) > 0:
                fighter_info = response.data[0]
                logger.info(f"Found fighter by fuzzy match: {fighter_info['fighter_name']}")

        if not fighter_info:
            logger.warning(f"Fighter not found: {name}")
            raise HTTPException(status_code=404, detail=f"Fighter not found: {name}")

        # Get the fighter's last 5 fights
        fights_response = supabase.table('fighter_last_5_fights')\
            .select('*')\
            .eq('fighter_name', fighter_info['fighter_name'])\
            .order('id')\
            .limit(MAX_FIGHTS_DISPLAY)\
            .execute()
            
        last_5_fights = fights_response.data if fights_response.data else []

        # Format the response
        return {
            "name": fighter_info['fighter_name'],
            "image_url": fighter_info.get('image_url', ''),
            "record": fighter_info.get('Record', ''),
            "height": fighter_info.get('Height', ''),
            "weight": fighter_info.get('Weight', ''),
            "reach": fighter_info.get('Reach', ''),
            "stance": fighter_info.get('STANCE', ''),
            "dob": fighter_info.get('DOB', ''),
            "slpm": fighter_info.get('SLpM', ''),
            "str_acc": fighter_info.get('Str. Acc.', ''),
            "sapm": fighter_info.get('SApM', ''),
            "str_def": fighter_info.get('Str. Def', ''),
            "td_avg": fighter_info.get('TD Avg.', ''),
            "td_acc": fighter_info.get('TD Acc.', ''),
            "td_def": fighter_info.get('TD Def.', ''),
            "sub_avg": fighter_info.get('Sub. Avg.', ''),
            "fighter_url": fighter_info.get('fighter_url', ''),
            "tap_link": fighter_info.get('tap_link', ''),
            "unique_id": str(fighter_info.get('id', '')),
            "ranking": fighter_info.get('ranking', ''),
            "last_5_fights": last_5_fights
        }

    except Exception as e:
        logger.error(f"Database error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

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
    
    # Check if there are at least two parts to the name outside the nickname
    outside_words = outside_parts.split()
    if len(outside_words) >= 2:
        # Reconstruct name with nickname
        return raw_name.strip()
    else:
        # Just return the outside part without the nickname
        return outside_parts 