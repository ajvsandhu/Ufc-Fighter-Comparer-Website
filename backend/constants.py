"""
Constants module for UFC Fighter Showdown application.
This module centralizes all configuration values and constants used throughout the application.
"""

import os
from pathlib import Path

# Base Directories
BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = BASE_DIR / "data"
CONFIG_DIR = DATA_DIR / "config"
MODELS_DIR = DATA_DIR / "models"

# Configuration Files
CONFIG_PATH = CONFIG_DIR / "model_config.json"

# Database
DB_NAME = "ufc_fighters.db"
DB_PATH = DATA_DIR / DB_NAME

# API Configuration
API_V1_PREFIX = "/api/v1"
API_TITLE = "UFC Fighter Showdown API"
API_DESCRIPTION = "API for comparing UFC fighters and predicting fight outcomes"
API_VERSION = "1.0.0"

# Server Configuration
SERVER_HOST = "0.0.0.0"  # Allows external connections
SERVER_PORT = 8000
DEBUG_MODE = True

# CORS Configuration
CORS_ORIGINS = ["*"]  # In production, replace with specific origins
CORS_METHODS = ["*"]
CORS_HEADERS = ["*"]
CORS_CREDENTIALS = True

# Query Limits and Pagination
MAX_FIGHTS_DISPLAY = 5  # Maximum number of fights to display in history
MAX_SEARCH_RESULTS = 5  # Maximum number of search results to return
PAGINATION_PAGE_SIZE = 20  # Number of items per page

# Fighter Constants
UNRANKED_VALUE = 99  # Value used to indicate unranked fighters
DEFAULT_RECORD = "N/A"  # Default value for missing fight records

# Cache Configuration
CACHE_TIMEOUT = 3600  # 1 hour in seconds
RANKINGS_CACHE_TIMEOUT = 86400  # 24 hours in seconds

# Model Configuration
MODEL_VERSION = "1.0"
MODEL_FILENAME = "fighter_prediction_model.pkl"
MODEL_PATH = MODELS_DIR / MODEL_FILENAME
SCALER_FILENAME = "scaler.pkl"
SCALER_PATH = MODELS_DIR / SCALER_FILENAME
FEATURE_NAMES_FILENAME = "feature_names.pkl"
FEATURE_NAMES_PATH = MODELS_DIR / FEATURE_NAMES_FILENAME
MODEL_INFO_FILENAME = "model_info.json"
MODEL_INFO_PATH = MODELS_DIR / MODEL_INFO_FILENAME

# Web Scraping Configuration
REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0"
}
REQUEST_TIMEOUT = 15  # seconds
RETRY_ATTEMPTS = 3
RETRY_DELAY = 2  # seconds

# Logging Configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Create required directories
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CONFIG_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True) 



#testing