"""
Supabase client module for UFC Fighter Prediction API.

This module provides the main interface to the Supabase database, handling
connection, authentication, and CRUD operations for fighter data.

IMPORTANT: This file has been made redundant by the get_db_connection function in 
backend/api/database.py. Both solutions work, but we prefer the simplified approach
from database.py for consistency.
"""

import os
import json
import requests
from typing import Dict, List, Any, Optional, Tuple
import logging
import traceback
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logger = logging.getLogger(__name__)

# Get Supabase credentials from environment
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    logger.error("Supabase URL or Key not found in environment variables")
    raise ValueError("Supabase URL or Key not found in environment variables. Please check your .env file.")

# Custom Supabase client implementation using requests
class SupabaseClient:
    def __init__(self, supabase_url, supabase_key):
        self.url = supabase_url.rstrip('/')
        self.key = supabase_key
        self.headers = {
            "apikey": supabase_key,
            "Authorization": f"Bearer {supabase_key}",
            "Content-Type": "application/json"
        }
        
    def table(self, table_name):
        return TableQuery(self, table_name)

    def test_connection(self):
        """Test connection to Supabase"""
        try:
            response = requests.get(
                f"{self.url}/rest/v1/fighters?limit=1&select=*",
                headers=self.headers
            )
            response.raise_for_status()
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Supabase: {e}")
            return False

class TableQuery:
    def __init__(self, client, table_name):
        self.client = client
        self.table_name = table_name
        self.query_params = {}
        self.url = f"{client.url}/rest/v1/{table_name}"
        self.select_query = "*"
        self.filter_conditions = []
        self.order_by = None
        self.order_direction = None
        self.order_nulls_last = False
        self.limit_val = None
        self.count_param = None
        
    def select(self, query="*", count=None):
        self.select_query = query
        self.count_param = count
        return self
        
    def eq(self, column, value):
        self.filter_conditions.append(f"{column}=eq.{value}")
        return self
        
    def neq(self, column, value):
        self.filter_conditions.append(f"{column}=neq.{value}")
        return self
        
    def ilike(self, column, value):
        # Add case-insensitive pattern matching with proper escaping
        safe_value = value.replace("*", "%")  # Replace * wildcards with %
        if not safe_value.startswith("%") and not safe_value.endswith("%"):
            safe_value = f"%{safe_value}%"  # Default to contains
        self.filter_conditions.append(f"{column}=ilike.{safe_value}")
        return self
        
    def order(self, column, desc=False, nulls_last=False):
        self.order_by = column
        self.order_direction = "desc" if desc else "asc"
        self.order_nulls_last = nulls_last
        return self
        
    def limit(self, limit_val):
        self.limit_val = limit_val
        return self
        
    def _build_url(self):
        url = f"{self.url}?select={self.select_query}"
        
        if self.filter_conditions:
            url += "&" + "&".join(self.filter_conditions)
            
        if self.order_by:
            order_param = f"&order={self.order_by}.{self.order_direction}"
            if self.order_nulls_last:
                order_param += ".nullslast"
            url += order_param
            
        if self.limit_val:
            url += f"&limit={self.limit_val}"
            
        if self.count_param:
            url += f"&count={self.count_param}"
            
        return url
        
    def execute(self):
        try:
            url = self._build_url()
            response = requests.get(url, headers=self.client.headers)
            response.raise_for_status()
            data = response.json()
            
            # Handle count parsing safely
            try:
                content_range = response.headers.get('content-range', '0-0/0')
                if '/' in content_range:
                    count_str = content_range.split('/')[1]
                    count = int(count_str) if count_str.isdigit() else 0
                else:
                    count = 0
            except (ValueError, IndexError):
                count = 0
                
            return QueryResponse(data, count)
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            return QueryResponse([], 0)
            
    def insert(self, data):
        try:
            response = requests.post(
                self.url,
                headers=self.client.headers,
                json=data if isinstance(data, list) else [data]
            )
            response.raise_for_status()
            return QueryResponse(response.json(), len(response.json()))
        except Exception as e:
            logger.error(f"Error inserting data: {e}")
            return QueryResponse([], 0)
            
    def update(self, data):
        try:
            url = self.url
            if self.filter_conditions:
                url += "?" + "&".join(self.filter_conditions)
                
            response = requests.patch(
                url,
                headers=self.client.headers,
                json=data
            )
            response.raise_for_status()
            return QueryResponse(response.json(), len(response.json()))
        except Exception as e:
            logger.error(f"Error updating data: {e}")
            return QueryResponse([], 0)
            
    def upsert(self, data, on_conflict=None):
        try:
            url = f"{self.url}"
            if on_conflict:
                url += f"?on_conflict={on_conflict}"
                
            response = requests.post(
                url,
                headers=self.client.headers,
                json=data if isinstance(data, list) else [data],
                params={"upsert": "true"}
            )
            response.raise_for_status()
            return QueryResponse(response.json(), len(response.json()))
        except Exception as e:
            logger.error(f"Error upserting data: {e}")
            return QueryResponse([], 0)
            
    def delete(self):
        try:
            url = self.url
            if self.filter_conditions:
                url += "?" + "&".join(self.filter_conditions)
                
            response = requests.delete(
                url,
                headers=self.client.headers
            )
            response.raise_for_status()
            return QueryResponse(response.json(), len(response.json()))
        except Exception as e:
            logger.error(f"Error deleting data: {e}")
            return QueryResponse([], 0)

class QueryResponse:
    def __init__(self, data, count):
        self.data = data
        self.count = count

# Create a Supabase client
try:
    supabase = SupabaseClient(SUPABASE_URL, SUPABASE_KEY)
    logger.info(f"Initialized custom Supabase client with URL: {SUPABASE_URL}")
except Exception as e:
    logger.error(f"Failed to create Supabase client: {e}")
    raise ValueError(f"Failed to create Supabase client: {e}")

def test_connection() -> bool:
    """Test the connection to Supabase"""
    try:
        if supabase.test_connection():
            logger.info("Connected to Supabase successfully")
            return True
        else:
            logger.error("Failed to connect to Supabase")
            return False
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