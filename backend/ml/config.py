"""
Configuration module for the ML prediction system.

This module manages the configuration settings for the prediction system,
including model parameters, feature weights, and various prediction settings.
It provides functionality to load, update, and reset configuration values.
"""

import os
import json
import logging
from pathlib import Path
from copy import deepcopy
from backend.constants import (
    CONFIG_DIR,
    CONFIG_PATH,
    LOG_LEVEL,
    LOG_FORMAT,
    LOG_DATE_FORMAT,
    APP_VERSION
)

# Configuration Constants
MODEL_TYPES = {
    'GRADIENT_BOOSTING': 'GradientBoosting',
    'RANDOM_FOREST': 'RandomForest'
}

CONFIDENCE_LEVELS = {
    'MIN': 0.53,
    'LOW': 0.60,
    'MODERATE': 0.70,
    'HIGH': 0.80,
    'MAX': 0.90
}

WEIGHT_CLASSES = {
    'FLYWEIGHT': 125,
    'BANTAMWEIGHT': 135,
    'FEATHERWEIGHT': 145,
    'LIGHTWEIGHT': 155,
    'WELTERWEIGHT': 170,
    'MIDDLEWEIGHT': 185,
    'LIGHT_HEAVYWEIGHT': 205,
    'HEAVYWEIGHT': 240
}

# Set up logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format=LOG_FORMAT,
    datefmt=LOG_DATE_FORMAT
)
logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_CONFIG = {
    "model": {
        "type": "GRADIENT_BOOSTING",
        "params": {
            "n_estimators": 100,
            "max_depth": 5,
            "learning_rate": 0.1,
            "random_state": 42
        }
    },
    "features": {
        "importance_weight": 0.7,
        "normalize": True,
        "scaling": "standard"
    },
    "prediction": {
        "confidence": {
            "min": 0.53,
            "low": 0.60,
            "medium": 0.70,
            "high": 0.80,
            "max": 0.90
        },
        "probability_calibration": True
    }
}

# Current configuration (initialized to default)
_current_config = deepcopy(DEFAULT_CONFIG)

def get_config():
    """Get the current configuration."""
    return deepcopy(_current_config)

def update_config(new_config):
    """Update the configuration with new values."""
    global _current_config
    
    # Deep merge the configs
    for section, values in new_config.items():
        if section in _current_config:
            if isinstance(values, dict) and isinstance(_current_config[section], dict):
                # Merge dictionaries
                _current_config[section].update(values)
            else:
                # Replace value
                _current_config[section] = values
        else:
            # Add new section
            _current_config[section] = values
    
    logger.info("Configuration updated")
    return get_config()

def reset_config():
    """Reset the configuration to defaults."""
    global _current_config
    _current_config = deepcopy(DEFAULT_CONFIG)
    logger.info("Configuration reset to defaults")
    return get_config()

def _save_config() -> None:
    """
    Save the current configuration to file.
    
    This function saves the current configuration to the configured file path,
    creating the necessary directories if they don't exist.
    """
    global _current_config
    
    if _current_config is not None:
        try:
            os.makedirs(CONFIG_DIR, exist_ok=True)
            with open(CONFIG_PATH, 'w') as f:
                json.dump(_current_config, f, indent=4)
            logger.info(f"Configuration saved to {CONFIG_PATH}")
        except Exception as e:
            logger.error(f"Error saving configuration: {str(e)}") 