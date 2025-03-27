import os
import pickle
import logging
from typing import Any, Dict, List, Tuple, Optional
import traceback

# Configure logging
logger = logging.getLogger(__name__)

# Global model variable
_loaded_model = None
_loaded_scaler = None
_loaded_features = None

def get_model_path() -> str:
    """Get the path to the pickled model file."""
    model_dir = os.path.join("backend", "ml", "models")
    model_path = os.path.join(model_dir, "model.pkl")
    return model_path

def get_scaler_path() -> str:
    """Get the path to the pickled scaler file."""
    model_dir = os.path.join("backend", "ml", "models")
    scaler_path = os.path.join(model_dir, "scaler.pkl")
    return scaler_path
    
def get_features_path() -> str:
    """Get the path to the pickled features file."""
    model_dir = os.path.join("backend", "ml", "models")
    features_path = os.path.join(model_dir, "features.pkl")
    return features_path

def load_model():
    """Load the prediction model from a pickle file."""
    global _loaded_model, _loaded_scaler, _loaded_features
    
    try:
        # Check if model directory exists, create if not
        model_dir = os.path.join("backend", "ml", "models")
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            logger.info(f"Created model directory: {model_dir}")
        
        # Load model
        model_path = get_model_path()
        if not os.path.exists(model_path):
            logger.warning(f"Model file not found at {model_path}")
            return False
        
        with open(model_path, 'rb') as f:
            _loaded_model = pickle.load(f)
        logger.info(f"Model loaded successfully from {model_path}")
            
        # Load scaler if it exists
        scaler_path = get_scaler_path()
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                _loaded_scaler = pickle.load(f)
            logger.info(f"Scaler loaded successfully from {scaler_path}")
        else:
            logger.warning(f"Scaler file not found at {scaler_path}")
        
        # Load features if they exist
        features_path = get_features_path()
        if os.path.exists(features_path):
            with open(features_path, 'rb') as f:
                _loaded_features = pickle.load(f)
            logger.info(f"Features loaded successfully from {features_path}")
        else:
            logger.warning(f"Features file not found at {features_path}")
        
        return True
    
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def get_loaded_model():
    """Get the loaded model."""
    global _loaded_model
    return _loaded_model

def get_loaded_scaler():
    """Get the loaded scaler."""
    global _loaded_scaler
    return _loaded_scaler

def get_loaded_features():
    """Get the loaded features."""
    global _loaded_features
    return _loaded_features

def save_model(model: Any, scaler: Any, features: List[str]) -> bool:
    """Save the model, scaler, and features to pickle files."""
    try:
        # Check if model directory exists, create if not
        model_dir = os.path.join("backend", "ml", "models")
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            logger.info(f"Created model directory: {model_dir}")
        
        # Save model
        model_path = get_model_path()
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"Model saved successfully to {model_path}")
        
        # Save scaler
        scaler_path = get_scaler_path()
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        logger.info(f"Scaler saved successfully to {scaler_path}")
        
        # Save features
        features_path = get_features_path()
        with open(features_path, 'wb') as f:
            pickle.dump(features, f)
        logger.info(f"Features saved successfully to {features_path}")
        
        return True
    
    except Exception as e:
        logger.error(f"Error saving model: {str(e)}")
        logger.error(traceback.format_exc())
        return False 