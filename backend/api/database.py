import os
import logging
from supabase import create_client, Client
from contextlib import contextmanager
from typing import Optional
from backend.constants import LOG_LEVEL, LOG_FORMAT, LOG_DATE_FORMAT
from functools import lru_cache

# Set up logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format=LOG_FORMAT,
    datefmt=LOG_DATE_FORMAT
)
logger = logging.getLogger(__name__)

class DatabaseConnection:
    _instance: Optional[Client] = None
    
    @classmethod
    def initialize(cls) -> None:
        """Initialize the database connection if not already initialized."""
        if cls._instance is None:
            supabase_url = os.getenv("SUPABASE_URL")
            supabase_key = os.getenv("SUPABASE_KEY")
            
            if not supabase_url or not supabase_key:
                raise ValueError("Supabase credentials not found in environment variables")
            
            try:
                cls._instance = create_client(supabase_url, supabase_key)
                logger.info("Supabase connection initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Supabase connection: {str(e)}")
                raise
    
    @classmethod
    def get_client(cls) -> Client:
        """Get the Supabase client instance."""
        if cls._instance is None:
            cls.initialize()
        return cls._instance

@lru_cache()
def get_db_connection():
    """Get a cached Supabase client instance."""
    try:
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_KEY")
        
        if not supabase_url or not supabase_key:
            logger.error("Supabase credentials not found in environment variables")
            return None
            
        client = create_client(supabase_url, supabase_key)
        logger.info("Supabase connection initialized successfully")
        return client
    except Exception as e:
        logger.error(f"Error creating Supabase client: {str(e)}")
        return None

# Alias for get_db_connection to maintain compatibility
get_supabase_client = get_db_connection