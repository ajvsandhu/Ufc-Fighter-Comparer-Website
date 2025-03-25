"""
Supabase client module for UFC Fighter Showdown application.
This module handles the connection to Supabase and provides database operations.
"""

import os
from typing import Dict, List, Any, Optional, Tuple
import logging
import traceback
from supabase import create_client, Client

# Import configuration
from backend.config import SUPABASE_URL, SUPABASE_KEY

# Set up logging
logger = logging.getLogger(__name__)

if not SUPABASE_URL or not SUPABASE_KEY:
    logger.error("Supabase URL or Key not found in environment variables")
    raise ValueError("Supabase URL or Key not found in environment variables. Please check your .env file.")

# Connect to Supabase
try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    logger.info(f"Initialized Supabase client with URL: {SUPABASE_URL}")
except Exception as e:
    logger.error(f"Failed to create Supabase client: {e}")
    raise ValueError(f"Failed to create Supabase client: {e}")

def test_connection() -> bool:
    """Test the connection to Supabase"""
    try:
        # Don't use count(*) - it causes errors
        response = supabase.table("fighters").select("*", count="exact").limit(1).execute()
        count = response.count if hasattr(response, "count") else 0
        logger.info(f"Connected to Supabase - Found {count} fighters")
        return True
    except Exception as e:
        logger.error(f"Failed to connect to Supabase: {e}")
        logger.error(traceback.format_exc())
        return False

def get_fighters() -> List[Dict[str, Any]]:
    """Get all fighters from the database"""
    try:
        response = supabase.table("fighters").select("*").execute()
        logger.info(f"Retrieved {len(response.data) if response.data else 0} fighters")
        return response.data
    except Exception as e:
        logger.error(f"Failed to get fighters: {e}")
        logger.error(traceback.format_exc())
        return []

def get_fighter(fighter_name: str) -> Optional[Dict[str, Any]]:
    """Get a fighter by name"""
    try:
        response = supabase.table("fighters").select("*").eq("fighter_name", fighter_name).execute()
        if response.data and len(response.data) > 0:
            logger.info(f"Found fighter: {fighter_name}")
            return response.data[0]
        logger.warning(f"Fighter not found: {fighter_name}")
        return None
    except Exception as e:
        logger.error(f"Failed to get fighter {fighter_name}: {e}")
        logger.error(traceback.format_exc())
        return None

def get_fighter_by_url(fighter_url: str) -> Optional[Dict[str, Any]]:
    """Get a fighter by URL"""
    try:
        response = supabase.table("fighters").select("*").eq("fighter_url", fighter_url).execute()
        if response.data and len(response.data) > 0:
            return response.data[0]
        return None
    except Exception as e:
        logger.error(f"Failed to get fighter by URL {fighter_url}: {e}")
        logger.error(traceback.format_exc())
        return None

def insert_fighter(fighter_data: Dict[str, Any]) -> bool:
    """Insert a new fighter into the database"""
    try:
        response = supabase.table("fighters").insert(fighter_data).execute()
        success = len(response.data) > 0
        if success:
            logger.info(f"Inserted fighter: {fighter_data.get('fighter_name', 'unknown')}")
        else:
            logger.warning(f"Failed to insert fighter: {fighter_data.get('fighter_name', 'unknown')}")
        return success
    except Exception as e:
        logger.error(f"Failed to insert fighter {fighter_data.get('fighter_name', 'unknown')}: {e}")
        logger.error(traceback.format_exc())
        return False

def update_fighter(fighter_url: str, fighter_data: Dict[str, Any]) -> bool:
    """Update a fighter in the database"""
    try:
        response = supabase.table("fighters").update(fighter_data).eq("fighter_url", fighter_url).execute()
        success = len(response.data) > 0
        if success:
            logger.info(f"Updated fighter: {fighter_data.get('fighter_name', 'unknown')}")
        else:
            logger.warning(f"Failed to update fighter: {fighter_data.get('fighter_name', 'unknown')}")
        return success
    except Exception as e:
        logger.error(f"Failed to update fighter {fighter_data.get('fighter_name', 'unknown')}: {e}")
        logger.error(traceback.format_exc())
        return False

def upsert_fighter(fighter_data: Dict[str, Any]) -> bool:
    """Insert or update a fighter in the database"""
    try:
        response = supabase.table("fighters").upsert(fighter_data, on_conflict="fighter_url").execute()
        success = len(response.data) > 0
        if success:
            logger.info(f"Upserted fighter: {fighter_data.get('fighter_name', 'unknown')}")
        else:
            logger.warning(f"Failed to upsert fighter: {fighter_data.get('fighter_name', 'unknown')}")
        return success
    except Exception as e:
        logger.error(f"Failed to upsert fighter {fighter_data.get('fighter_name', 'unknown')}: {e}")
        logger.error(traceback.format_exc())
        return False

def get_fighter_fights(fighter_name: str) -> List[Dict[str, Any]]:
    """Get all fights for a fighter, ordered by ID ascending."""
    try:
        response = supabase.table("fighter_last_5_fights") \
            .select("*") \
            .eq("fighter_name", fighter_name) \
            .order("id", desc=False) \
            .execute()
        return response.data if response.data else []
    except Exception as e:
        logger.error(f"Error getting fights for {fighter_name}: {e}")
        return []

def update_fighter_all_fights(fighter_name: str, fights: List[Dict[str, Any]]) -> bool:
    """Update all fights for a fighter in chronological order (oldest first)."""
    try:
        # Delete existing fights for this fighter
        supabase.table("fighter_last_5_fights").delete().eq("fighter_name", fighter_name).execute()
        
        # Get the current maximum ID from the table
        response = supabase.table("fighter_last_5_fights").select("id").order("id", desc=True).limit(1).execute()
        next_id = 1 if not response.data else response.data[0]['id'] + 1
        
        # Insert fights in order (oldest to newest)
        for fight in fights:
            fight['id'] = next_id
            next_id += 1
            supabase.table("fighter_last_5_fights").insert(fight).execute()
        
        logger.info(f"Successfully updated {len(fights)} fights for {fighter_name}")
        return True
    except Exception as e:
        logger.error(f"Failed to update all fights for {fighter_name}: {e}")
        return False

def update_fighter_recent_fight(fighter_name: str, new_fight: Dict[str, Any]) -> bool:
    """Update fighter's fights by putting new fight at lowest ID and shifting others up."""
    try:
        # Get current fights
        current_fights = get_fighter_fights(fighter_name)
        
        if not current_fights:
            # If no fights exist, just insert the new one
            return insert_fighter_fight(new_fight)
            
        # Get the base ID (lowest ID in the sequence)
        base_id = current_fights[0]['id']
        
        # Delete the oldest fight if we have 5 fights
        if len(current_fights) >= 5:
            oldest_fight_id = current_fights[-1]['id']
            supabase.table("fighter_last_5_fights").delete().eq("id", oldest_fight_id).execute()
            current_fights = current_fights[:-1]
        
        # Shift existing fights up by one ID
        for i in range(len(current_fights)):
            fight = current_fights[i]
            fight['id'] = base_id + i + 1
            supabase.table("fighter_last_5_fights").upsert(fight).execute()
        
        # Insert new fight at lowest ID
        new_fight['id'] = base_id
        response = supabase.table("fighter_last_5_fights").insert(new_fight).execute()
        
        success = len(response.data) > 0
        if success:
            logger.info(f"Updated recent fight for {fighter_name}")
        else:
            logger.warning(f"Failed to update recent fight for {fighter_name}")
        return success
    except Exception as e:
        logger.error(f"Failed to update recent fight for {fighter_name}: {e}")
        return False

def insert_fighter_fight(fight_data: Dict[str, Any]) -> bool:
    """Insert a fighter fight into the database"""
    try:
        # Get the current maximum ID
        response = supabase.table("fighter_last_5_fights").select("id").order("id", desc=True).limit(1).execute()
        next_id = 1 if not response.data else response.data[0]['id'] + 1
        
        # Set the ID and insert
        fight_data['id'] = next_id
        response = supabase.table("fighter_last_5_fights").insert(fight_data).execute()
        success = len(response.data) > 0
        
        if success:
            logger.info(f"Inserted fight for {fight_data.get('fighter_name', 'unknown')}")
        else:
            logger.warning(f"Failed to insert fight for {fight_data.get('fighter_name', 'unknown')}")
        return success
    except Exception as e:
        logger.error(f"Error storing fight data: {e}")
        return False

def delete_fighter_fights(fighter_name: str) -> bool:
    """Delete all fights for a fighter."""
    try:
        response = supabase.table("fighter_last_5_fights") \
            .delete() \
            .eq("fighter_name", fighter_name) \
            .execute()
        success = len(response.data) > 0
        if success:
            logger.info(f"Deleted all fights for {fighter_name}")
        return success
    except Exception as e:
        logger.error(f"Error deleting fights for {fighter_name}: {e}")
        return False

def truncate_table(table_name: str) -> bool:
    """Truncate a table and reset its sequence."""
    try:
        # Use raw SQL to ensure proper sequence reset
        sql = f"TRUNCATE TABLE {table_name} RESTART IDENTITY CASCADE;"
        supabase.postgrest.rpc('raw_sql', {'query': sql}).execute()
        logger.info(f"Truncated table {table_name} and reset sequence")
        return True
    except Exception as e:
        logger.error(f"Error truncating table {table_name}: {e}")
        return False

# Test the connection when the module is imported
try:
    if test_connection():
        logger.info("Supabase connection test successful")
    else:
        logger.warning("Supabase connection test failed")
except Exception as e:
    logger.error(f"Error testing Supabase connection: {e}")
    logger.error(traceback.format_exc()) 