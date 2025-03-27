import logging
from backend.supabase_client import supabase, test_connection
from functools import lru_cache
from backend.constants import LOG_LEVEL, LOG_FORMAT, LOG_DATE_FORMAT

# Set up logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format=LOG_FORMAT,
    datefmt=LOG_DATE_FORMAT
)
logger = logging.getLogger(__name__)

@lru_cache()
def get_db_connection():
    """Get a cached Supabase client instance."""
    try:
        if test_connection():
            logger.info("Supabase connection successful")
            return supabase
        else:
            logger.error("Failed to connect to Supabase")
            return None
    except Exception as e:
        logger.error(f"Error getting Supabase connection: {str(e)}")
        return None

# Alias for get_db_connection to maintain compatibility
get_supabase_client = get_db_connection