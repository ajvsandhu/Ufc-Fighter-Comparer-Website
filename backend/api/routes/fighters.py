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

router = APIRouter(prefix=API_V1_PREFIX)

@router.get("/fighters")
def get_fighters(query: str = Query("", min_length=0)):
    """Get all fighters or search for fighters by name."""
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        try:
            cur.execute("""
                SELECT fighter_name, Record, ranking 
                FROM fighters 
                ORDER BY ranking ASC NULLS LAST, fighter_name ASC
            """)
            
            fighter_data = cur.fetchall()
        except sqlite3.Error as e:
            logger.error(f"SQL error fetching fighters: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
        
        fighters_list = []
        
        if not query:
            # Return all fighters with record and ranking info
            for row in fighter_data:
                fighter_name = row[0]
                record = row[1] if row[1] else DEFAULT_RECORD
                ranking = row[2]
                
                formatted_name = f"{fighter_name} ({record})"
                fighters_list.append(formatted_name)
        else:
            # Improved search logic
            query_parts = query.lower().split()
            
            for row in fighter_data:
                fighter_name = row[0]
                record = row[1] if row[1] else DEFAULT_RECORD
                ranking = row[2]
                
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
            
            # Sort results by ranking (if available) and then alphabetically
            def sort_key(name):
                # Extract original fighter name from the formatted string
                original_name = name.split(" (")[0]
                # Find the corresponding ranking
                for row in fighter_data:
                    if row[0] == original_name:
                        ranking = row[2]
                        # Return tuple for sorting: (has_ranking, ranking_value, name)
                        return (ranking is not None, ranking if ranking is not None else UNRANKED_VALUE, original_name)
                return (False, UNRANKED_VALUE, original_name)
            
            fighters_list.sort(key=sort_key)
        
        return {"fighters": fighters_list[:MAX_SEARCH_RESULTS]}
        
    except Exception as e:
        logger.error(f"Error in get_fighters: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")
    finally:
        if conn:
            conn.close()

@router.get("/fighter/{name:path}")
def get_fighter_stats(name: str):
    """Get detailed stats for a specific fighter."""
    logger.info(f"Getting stats for fighter: {name}")
    conn = get_db_connection()
    cur = conn.cursor()

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

        # First try exact match with record if available
        if record:
            cur.execute(
                """
                SELECT fighter_name, image_url, Record, Height, Weight, Reach, STANCE, DOB, SLpM, [Str. Acc.], SApM,
                       [Str. Def], [TD Avg.], [TD Acc.], [TD Def.], [Sub. Avg.], fighter_url, tap_link, rowid as unique_id,
                       ranking
                FROM fighters
                WHERE LOWER(TRIM(fighter_name)) = LOWER(TRIM(?))
                AND Record = ?
                """,
                (base_name, record)
            )
            fighter_info = cur.fetchone()
            if fighter_info:
                logger.info(f"Found fighter by exact name and record match: {fighter_info[0]}")
        else:
            fighter_info = None

        # If no match with record, try just the name
        if not fighter_info:
            cur.execute(
                """
                SELECT fighter_name, image_url, Record, Height, Weight, Reach, STANCE, DOB, SLpM, [Str. Acc.], SApM,
                       [Str. Def], [TD Avg.], [TD Acc.], [TD Def.], [Sub. Avg.], fighter_url, tap_link, rowid as unique_id,
                       ranking
                FROM fighters
                WHERE LOWER(TRIM(fighter_name)) = LOWER(TRIM(?))
                """,
                (base_name,)
            )
            fighter_info = cur.fetchone()
            if fighter_info:
                logger.info(f"Found fighter by name only: {fighter_info[0]}")

        # If still no match, try fuzzy match
        if not fighter_info:
            cur.execute(
                """
                SELECT fighter_name, image_url, Record, Height, Weight, Reach, STANCE, DOB, SLpM, [Str. Acc.], SApM,
                       [Str. Def], [TD Avg.], [TD Acc.], [TD Def.], [Sub. Avg.], fighter_url, tap_link, rowid as unique_id,
                       ranking
                FROM fighters
                WHERE LOWER(fighter_name) LIKE LOWER(?)
                """,
                (f"%{base_name}%",)
            )
            fighter_info = cur.fetchone()
            if fighter_info:
                logger.info(f"Found fighter by fuzzy match: {fighter_info[0]}")

        if not fighter_info:
            logger.warning(f"Fighter not found: {name}")
            raise HTTPException(status_code=404, detail=f"Fighter not found: {name}")

        # Get the fighter's last 5 fights
        cur.execute(
            """
            SELECT *
            FROM fighter_last_5_fights
            WHERE fighter_name = ?
            ORDER BY id ASC
            LIMIT ?
            """,
            (fighter_info[0], MAX_FIGHTS_DISPLAY)
        )
        last_5_fights = cur.fetchall()

        # Format the response
        return {
            "name": fighter_info[0],
            "image_url": fighter_info[1],
            "record": fighter_info[2],
            "height": fighter_info[3],
            "weight": fighter_info[4],
            "reach": fighter_info[5],
            "stance": fighter_info[6],
            "dob": fighter_info[7],
            "slpm": fighter_info[8],
            "str_acc": fighter_info[9],
            "sapm": fighter_info[10],
            "str_def": fighter_info[11],
            "td_avg": fighter_info[12],
            "td_acc": fighter_info[13],
            "td_def": fighter_info[14],
            "sub_avg": fighter_info[15],
            "fighter_url": fighter_info[16],
            "tap_link": fighter_info[17],
            "unique_id": str(fighter_info[18]),
            "ranking": fighter_info[19],
            "last_5_fights": [dict(zip([col[0] for col in cur.description], row)) for row in last_5_fights] if last_5_fights else []
        }

    except sqlite3.Error as e:
        logger.error(f"Database error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        if conn:
            conn.close()

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

