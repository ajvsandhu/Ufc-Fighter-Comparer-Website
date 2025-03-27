import logging
import os
import json
import requests
from functools import lru_cache
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleSupabaseClient:
    """A simplified client for Supabase that uses direct HTTP requests instead of SDK."""
    
    def __init__(self, url: str, key: str):
        self.base_url = url
        self.key = key
        self.headers = {
            "apikey": key,
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Prefer": "return=representation"
        }
    
    def _handle_response(self, response):
        """Handle API response and convert to expected format."""
        if not response.ok:
            logger.error(f"Supabase API error: {response.status_code} - {response.text}")
            return None
        
        try:
            data = response.json()
            # Wrap in the expected data property for compatibility
            return type('SupabaseResponse', (), {'data': data})
        except Exception as e:
            logger.error(f"Error parsing response: {str(e)}")
            return None
    
    def table(self, table_name: str):
        """Create a query builder for a table."""
        return QueryBuilder(self, table_name)
    
    def test_connection(self):
        """Test the connection to Supabase."""
        try:
            endpoint = f"{self.base_url}/rest/v1/fighters?limit=1"
            response = requests.get(endpoint, headers=self.headers)
            return response.ok
        except Exception as e:
            logger.error(f"Error testing connection: {str(e)}")
            return False

class QueryBuilder:
    """Simple query builder for Supabase tables."""
    
    def __init__(self, client, table_name):
        self.client = client
        self.table_name = table_name
        self.query_params = {}
        self.filters = []
        self.select_columns = "*"
        self.order_value = None
        self.limit_value = None
        self.offset_value = None
        self.count_option = None
    
    def select(self, columns="*", **kwargs):
        """Select columns to return."""
        self.select_columns = columns
        if kwargs.get('count'):
            self.count_option = kwargs.get('count')
        return self
    
    def eq(self, column, value):
        """Add equals filter."""
        self.filters.append(f"{column}=eq.{value}")
        return self
    
    def ilike(self, column, value):
        """Add case-insensitive LIKE filter."""
        self.filters.append(f"{column}=ilike.{value}")
        return self
    
    def order(self, column, desc=False, nulls_last=False):
        """Add order clause."""
        direction = "desc" if desc else "asc"
        nulls = ".nullslast" if nulls_last else ""
        self.order_value = f"{column}.{direction}{nulls}"
        return self
    
    def limit(self, limit_val):
        """Add limit clause."""
        self.limit_value = limit_val
        return self
    
    def range(self, from_val, to_val):
        """Add range for pagination."""
        self.offset_value = from_val
        self.limit_value = to_val - from_val + 1
        return self
    
    def _build_url(self):
        """Build the URL for the request."""
        url = f"{self.client.base_url}/rest/v1/{self.table_name}"
        
        # Add query parameters
        params = []
        
        # Add select columns
        params.append(f"select={self.select_columns}")
        
        # Add filters
        for f in self.filters:
            params.append(f)
        
        # Add order
        if self.order_value:
            params.append(f"order={self.order_value}")
        
        # Add limit
        if self.limit_value is not None:
            params.append(f"limit={self.limit_value}")
        
        # Add offset
        if self.offset_value is not None:
            params.append(f"offset={self.offset_value}")
            
        # Add count option
        if self.count_option:
            params.append(f"count={self.count_option}")
        
        # Combine parameters
        if params:
            url += "?" + "&".join(params)
        
        return url
    
    def execute(self):
        """Execute the query."""
        try:
            url = self._build_url()
            response = requests.get(url, headers=self.client.headers)
            return self.client._handle_response(response)
        except Exception as e:
            logger.error(f"Error executing query: {str(e)}")
            return None
    
    def upsert(self, data, on_conflict=None):
        """Insert or update data."""
        try:
            url = f"{self.client.base_url}/rest/v1/{self.table_name}"
            
            headers = dict(self.client.headers)
            if on_conflict:
                headers["Prefer"] = f"resolution=merge-duplicates,return=representation,on_conflict={on_conflict}"
            
            response = requests.post(url, headers=headers, json=data if isinstance(data, list) else [data])
            return self.client._handle_response(response)
        except Exception as e:
            logger.error(f"Error upserting data: {str(e)}")
            return None

@lru_cache(maxsize=1)
def get_db_connection():
    """Get a connection to the Supabase database. Uses caching to reuse the connection."""
    try:
        # Get Supabase credentials
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")
        
        if not url or not key:
            logger.error("Supabase credentials not found in environment variables")
            return None
        
        client = SimpleSupabaseClient(url, key)
        logger.info("Successfully created Supabase client")
        return client
    except Exception as e:
        logger.error(f"Error creating Supabase client: {str(e)}")
        return None

def check_database_connection():
    """Test the connection to the Supabase database."""
    try:
        client = get_db_connection()
        if not client:
            logger.error("No database connection available")
            return False
        
        success = client.test_connection()
        if success:
            logger.info("Database connection test successful")
        else:
            logger.error("Database connection test failed")
        
        return success
    except Exception as e:
        logger.error(f"Error checking database connection: {str(e)}")
        return False

# Alias for get_db_connection to maintain compatibility
get_supabase_client = get_db_connection