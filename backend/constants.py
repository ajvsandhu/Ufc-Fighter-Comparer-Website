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
APP_TITLE = "UFC Fighter Prediction API"
APP_DESCRIPTION = "API for UFC fighter statistics and fight outcome predictions"
APP_VERSION = "1.0.0"
API_V1_STR = "/api/v1"

# Server Configuration
SERVER_HOST = os.getenv('API_HOST', "0.0.0.0")
SERVER_PORT = int(os.getenv('API_PORT', 8000))
DEBUG_MODE = os.getenv('DEBUG', 'False').lower() == 'true'

# CORS Configuration
CORS_ORIGINS = os.getenv('CORS_ORIGINS', "http://localhost:3000,http://localhost:8000,https://ufc-fighter-data-api.vercel.app,https://ufc-fighter-data-api-git-main.vercel.app,http://ufc-fighter-data-api.vercel.app,https://*.vercel.app").split(',')
CORS_ORIGINS_REGEX = os.getenv("CORS_ORIGINS_REGEX", "")
CORS_METHODS = ["*"]
CORS_HEADERS = ["*"]
CORS_CREDENTIALS = True

# Query Limits and Pagination
MAX_FIGHTS_DISPLAY = int(os.getenv('MAX_FIGHTS_DISPLAY', 5))
MAX_SEARCH_RESULTS = int(os.getenv('MAX_SEARCH_RESULTS', 5))
PAGINATION_PAGE_SIZE = int(os.getenv('PAGINATION_PAGE_SIZE', 20))

# Fighter Constants
UNRANKED_VALUE = 99
DEFAULT_RECORD = "0-0-0"
DEFAULT_CONFIDENCE = 0.5

# Cache Configuration
CACHE_TIMEOUT = int(os.getenv('CACHE_TIMEOUT', 3600))
RANKINGS_CACHE_TIMEOUT = int(os.getenv('RANKINGS_CACHE_TIMEOUT', 86400))

# Model Configuration
MODEL_PATH = os.path.join("backend", "ml", "models", "model.pkl")
SCALER_PATH = os.path.join("backend", "ml", "models", "scaler.pkl")
FEATURES_PATH = os.path.join("backend", "ml", "models", "features.pkl")
FEATURE_NAMES_PATH = FEATURES_PATH
MODEL_INFO_FILENAME = "model_info.json"
MODEL_INFO_PATH = MODELS_DIR / MODEL_INFO_FILENAME
MODEL_VERSION = os.getenv("MODEL_VERSION", APP_VERSION)

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

# Database settings
DATABASE_URL = os.getenv("DATABASE_URL", "")
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")

# Security
JWT_SECRET = os.getenv("JWT_SECRET", "")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))

# Feature engineering
IMPORTANT_FEATURES = [
    "Win", "Loss", "Draw", "Height", "Weight", "Reach", "SLPM", 
    "StrAcc", "SApM", "StrDef", "TD", "TDA", "TDD", "SUB"
]

# Create required directories
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CONFIG_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True) 
