import os
import sqlite3
import logging
from backend.constants import DB_PATH, LOG_LEVEL, LOG_FORMAT, LOG_DATE_FORMAT

# Set up logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format=LOG_FORMAT,
    datefmt=LOG_DATE_FORMAT
)
logger = logging.getLogger(__name__)

def get_db_connection():
    try:
        logger.info(f"Attempting to connect to database at: {DB_PATH}")
        if not os.path.exists(DB_PATH):
            logger.error(f"Database file not found at: {DB_PATH}")
            raise sqlite3.Error("Database file not found")
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        logger.info("Successfully connected to database")
        return conn
    except sqlite3.Error as e:
        logger.error(f"Database connection error: {str(e)}")
        raise