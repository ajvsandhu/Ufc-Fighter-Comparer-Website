import logging
from supabase import create_client
import os
from dotenv import load_dotenv
from functools import lru_cache
from backend.constants import LOG_LEVEL, LOG_FORMAT, LOG_DATE_FORMAT

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
    """Get a connection to the Supabase database."""
    try:
        # Get Supabase credentials from environment variables
        supabase_url = os.getenv('SUPABASE_URL')
        supabase_key = os.getenv('SUPABASE_KEY')
        
        if not supabase_url or not supabase_key:
            logger.error("Missing Supabase credentials. Please check SUPABASE_URL and SUPABASE_KEY environment variables.")
            return None
        
        # Create Supabase client
        supabase = create_client(supabase_url, supabase_key)
        logger.info("Successfully connected to Supabase")
        return supabase
    
    except Exception as e:
        logger.error(f"Error connecting to Supabase: {str(e)}")
        return None

def check_database_connection():
    """Check if the database connection is working properly."""
    try:
        supabase = get_db_connection()
        if not supabase:
            logger.error("Failed to get Supabase connection")
            return False
        
        # Try a simple query to verify connection
        response = supabase.table('fighters').select('fighter_name').limit(1).execute()
        if response and hasattr(response, 'data'):
            logger.info("Database connection test successful")
            return True
        else:
            logger.error("Database connection test failed - no data returned")
            return False
    
    except Exception as e:
        logger.error(f"Database connection test failed with error: {str(e)}")
        return False

# Alias for get_db_connection to maintain compatibility
get_supabase_client = get_db_connection