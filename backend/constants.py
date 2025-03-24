"""
Constants module for UFC Fighter Showdown application.
This module centralizes all configuration values and constants used throughout the application.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base Directories
BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = Path(os.getenv('DATA_DIR', BASE_DIR / "data"))
CONFIG_DIR = DATA_DIR / "config"
MODELS_DIR = Path(os.getenv('MODEL_DIR', DATA_DIR / "models"))

# Configuration Files
CONFIG_PATH = CONFIG_DIR / "model_config.json"

# Database
DB_NAME = os.getenv('DB_NAME', "ufc_fighters.db")
DB_PATH = Path(os.getenv('DB_PATH', DATA_DIR / DB_NAME))

# API Configuration
API_V1_PREFIX = os.getenv('API_PREFIX', "/api/v1")
API_TITLE = "UFC Fighter Showdown API"
API_DESCRIPTION = "API for comparing UFC fighters and predicting fight outcomes"
API_VERSION = os.getenv('API_VERSION', "1.0.0")

# Server Configuration
SERVER_HOST = os.getenv('API_HOST', "0.0.0.0")
SERVER_PORT = int(os.getenv('API_PORT', 8000))
DEBUG_MODE = os.getenv('DEBUG', 'False').lower() == 'true'

# CORS Configuration
CORS_ORIGINS = os.getenv('CORS_ORIGINS', "http://localhost:3000,http://localhost:8000").split(',')
CORS_METHODS = ["*"]
CORS_HEADERS = ["*"]
CORS_CREDENTIALS = True

# Query Limits and Pagination
MAX_FIGHTS_DISPLAY = int(os.getenv('MAX_FIGHTS_DISPLAY', 5))
MAX_SEARCH_RESULTS = int(os.getenv('MAX_SEARCH_RESULTS', 5))
PAGINATION_PAGE_SIZE = int(os.getenv('PAGINATION_PAGE_SIZE', 20))

# Fighter Constants
UNRANKED_VALUE = 99
DEFAULT_RECORD = "N/A"

# Cache Configuration
CACHE_TIMEOUT = int(os.getenv('CACHE_TIMEOUT', 3600))
RANKINGS_CACHE_TIMEOUT = int(os.getenv('RANKINGS_CACHE_TIMEOUT', 86400))

# Model Configuration
MODEL_VERSION = os.getenv('MODEL_VERSION', "1.0")
MODEL_FILENAME = os.getenv('MODEL_FILENAME', "fighter_prediction_model.pkl")
MODEL_PATH = MODELS_DIR / MODEL_FILENAME
SCALER_FILENAME = os.getenv('SCALER_FILENAME', "scaler.pkl")
SCALER_PATH = MODELS_DIR / SCALER_FILENAME
FEATURE_NAMES_FILENAME = os.getenv('FEATURE_NAMES_FILENAME', "feature_names.pkl")
FEATURE_NAMES_PATH = MODELS_DIR / FEATURE_NAMES_FILENAME
MODEL_INFO_FILENAME = "model_info.json"
MODEL_INFO_PATH = MODELS_DIR / MODEL_INFO_FILENAME

# Web Scraping Configuration
REQUEST_HEADERS = {
    "User-Agent": os.getenv('USER_AGENT', "Mozilla/5.0")
}
REQUEST_TIMEOUT = int(os.getenv('REQUEST_TIMEOUT', 15))
RETRY_ATTEMPTS = int(os.getenv('RETRY_ATTEMPTS', 3))
RETRY_DELAY = int(os.getenv('RETRY_DELAY', 2))

# Logging Configuration
LOG_LEVEL = os.getenv('LOG_LEVEL', "INFO")
LOG_FORMAT = os.getenv('LOG_FORMAT', "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
LOG_DATE_FORMAT = os.getenv('LOG_DATE_FORMAT', "%Y-%m-%d %H:%M:%S")

# Create required directories
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CONFIG_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True) 
