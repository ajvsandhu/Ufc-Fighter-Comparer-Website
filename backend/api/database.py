import logging
import os
from dotenv import load_dotenv
from functools import lru_cache
from backend.constants import LOG_LEVEL, LOG_FORMAT, LOG_DATE_FORMAT
from supabase import create_client

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format=LOG_FORMAT,
    datefmt=LOG_DATE_FORMAT
)
logger = logging.getLogger(__name__)

# Cache the database connection to avoid recreating it for each request
@lru_cache(maxsize=1)
def get_db_connection():
    """
    Get a connection to the Supabase database.
    Uses caching to reuse the connection.
    """
    try:
        # Get Supabase credentials from environment
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_KEY")
        
        # Check if credentials are available
        if not supabase_url or not supabase_key:
            logger.error("Supabase credentials not found in environment variables")
            return None
        
        # Create Supabase client
        try:
            supabase = create_client(supabase_url, supabase_key)
            logger.info("Successfully created Supabase client")
            return supabase
        except Exception as e:
            logger.error(f"Error creating Supabase client: {str(e)}")
            return None
    except Exception as e:
        logger.error(f"Unexpected error in get_db_connection: {str(e)}")
        return None

def check_database_connection():
    """
    Test the connection to the Supabase database.
    Returns True if the connection is successful, False otherwise.
    """
    try:
        supabase = get_db_connection()
        if not supabase:
            logger.error("No database connection available")
            return False
            
        # Test a simple query
        try:
            response = supabase.table('fighters').select('count', count='exact').limit(1).execute()
            if hasattr(response, 'data'):
                logger.info("Database connection test successful")
                return True
            else:
                logger.error("Database response invalid")
                return False
        except Exception as e:
            logger.error(f"Error testing database connection: {str(e)}")
            return False
    except Exception as e:
        logger.error(f"Unexpected error checking database connection: {str(e)}")
        return False

# Alias for get_db_connection to maintain compatibility
get_supabase_client = get_db_connection