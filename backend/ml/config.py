"""
Configuration module for UFC fighter prediction system.

This module manages the configuration settings for the prediction system,
including model parameters, feature weights, and various prediction settings.
It provides functionality to load, update, and reset configuration values.
"""

import os
import json
import logging
from typing import Dict, Any, Optional
from backend.constants import (
    CONFIG_DIR,
    CONFIG_PATH,
    LOG_LEVEL,
    LOG_FORMAT,
    LOG_DATE_FORMAT,
    MODEL_VERSION
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
    # Model type and parameters
    "model_type": MODEL_TYPES['GRADIENT_BOOSTING'],
    "n_estimators": 200,
    "max_depth": 12,
    "random_state": 42,
    "learning_rate": 0.05,
    
    # Feature importance weighting
    "feature_weights": {
        "striking": 1.1,
        "grappling": 1.1,
        "recent_fights": 1.6,
        "head_to_head": 2.2,
        "physical": 0.8,
        "quality": 1.5,
        "style": 1.3,
        "versatility": 1.2,
        "experience": 0.7
    },
    
    # Confidence calculation settings
    "confidence_calculation": {
        "use_data_driven_confidence": True,
        "dynamic_probabilities": True,
        "statistical_threshold": 0.03,
        "min_confidence": CONFIDENCE_LEVELS['MIN'],
        "max_confidence": CONFIDENCE_LEVELS['MAX'],
        "avoid_fixed_splits": True,
        "probability_jitter": 0.03,
        "performance_based_confidence": True
    },
    
    # Feature extraction parameters
    "feature_extraction": {
        "recent_fight_weight": 2.5,            # Increased weight for recent fights
        "include_style_matchups": True,        # Consider style matchups (striker vs grappler)
        "normalize_features": True,            # Normalize features before prediction
        "use_advanced_metrics": True,          # NEW: Use advanced metrics like versatility
        "include_quality_metrics": True,       # NEW: Include opponent quality metrics
        "career_progression_weight": 1.5,      # NEW: Weight for career progression metrics
        "detect_inconsistent_fighters": True,  # NEW: Special handling for inconsistent fighters
        "physical_matchup_importance": 0.8     # NEW: Importance of physical attributes
    },
    
    # Matchup-specific adjustments
    "matchup_adjustments": {
        "head_to_head_bonus": 0.15,           # Probability boost for each head-to-head win
        "streak_factor": 0.04,                # Slight reduction for streak adjustments
        "champion_bonus": 0.04,               # Bonus for current/former champions
        "weight_difference_penalty": 0.12,    # Penalty per weight class difference
        "inconsistency_penalty": 0.05,        # NEW: Penalty for inconsistent fighters
        "inactivity_penalty": 0.03,           # NEW: Penalty for inactive fighters
        "style_counters": {
            "wrestler_vs_striker": 0.07,      # Bonus for wrestlers against pure strikers
            "bjj_vs_wrestler": 0.04,          # Bonus for BJJ specialists against wrestlers
            "striker_vs_brawler": 0.05,       # NEW: Technical strikers vs brawlers
            "counter_striker_bonus": 0.05     # NEW: Bonus for counter strikers
        }
    },
    
    # NEW: Probability calibration to avoid stuck predictions
    "probability_calibration": {
        "enabled": True,
        "avoid_60_40_split": True,            # Specifically avoid the 60/40 split issue
        "diversify_close_predictions": True,   # Add diversity to close predictions
        "uncertain_matchup_threshold": 0.55,   # Threshold for considering a matchup uncertain
        "calibration_method": "isotonic",      # Method for calibrating probabilities
        "recalibrate_after_postprocessing": True, # Recalibrate after applying adjustments
        "certainty_thresholds": [0.53, 0.60, 0.70, 0.80, 0.90] # Probability threshold buckets
    }
}

# Global variable to store active configuration
_active_config: Optional[Dict[str, Any]] = None

def get_config() -> Dict[str, Any]:
    """
    Get the current configuration.
    
    This function retrieves the current configuration, loading it from file
    if it hasn't been loaded yet. If no configuration file exists, it uses
    the default configuration.
    
    Returns:
        Dict[str, Any]: The current configuration dictionary
    """
    global _active_config
    
    # If we haven't loaded the config yet, try to load it from file
    if _active_config is None:
        if os.path.exists(CONFIG_PATH):
            try:
                with open(CONFIG_PATH, 'r') as f:
                    _active_config = json.load(f)
                logger.info(f"Loaded configuration from {CONFIG_PATH}")
            except Exception as e:
                logger.error(f"Error loading configuration: {str(e)}")
                _active_config = DEFAULT_CONFIG.copy()
                _save_config()
        else:
            # If no config file exists, use the default
            _active_config = DEFAULT_CONFIG.copy()
            _save_config()
    
    return _active_config

def update_config(new_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update the configuration with new values.
    
    This function updates the current configuration with new values,
    handling nested dictionaries recursively. The updated configuration
    is then saved to disk.
    
    Args:
        new_config: New configuration values to update
        
    Returns:
        Dict[str, Any]: The updated configuration dictionary
    """
    global _active_config
    
    if _active_config is None:
        _active_config = get_config()
    
    # Update configuration recursively
    def update_dict_recursive(d1: Dict[str, Any], d2: Dict[str, Any]) -> None:
        """
        Recursively update dictionary d1 with values from d2.
        
        Args:
            d1: Target dictionary to update
            d2: Source dictionary with new values
        """
        for k, v in d2.items():
            if k in d1 and isinstance(d1[k], dict) and isinstance(v, dict):
                update_dict_recursive(d1[k], v)
            else:
                d1[k] = v
    
    update_dict_recursive(_active_config, new_config)
    _save_config()
    
    return _active_config

def reset_config() -> None:
    """
    Reset the configuration to default values.
    
    This function resets the current configuration to the default values
    defined in DEFAULT_CONFIG and saves the changes to disk.
    """
    global _active_config
    _active_config = DEFAULT_CONFIG.copy()
    _save_config()

def _save_config() -> None:
    """
    Save the current configuration to file.
    
    This function saves the current configuration to the configured file path,
    creating the necessary directories if they don't exist.
    """
    global _active_config
    
    if _active_config is not None:
        try:
            os.makedirs(CONFIG_DIR, exist_ok=True)
            with open(CONFIG_PATH, 'w') as f:
                json.dump(_active_config, f, indent=4)
            logger.info(f"Configuration saved to {CONFIG_PATH}")
        except Exception as e:
            logger.error(f"Error saving configuration: {str(e)}") 