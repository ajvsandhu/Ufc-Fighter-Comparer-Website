import os
import pickle
import logging
import joblib
import traceback
import numpy as np
from typing import Any, Dict, List, Tuple, Optional
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

from backend.constants import MODEL_PATH, SCALER_PATH, FEATURES_PATH

# Configure logging
logger = logging.getLogger(__name__)

# Global model variable
_model = None
_scaler = None
_features = None

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
    """
    Load model and associated components.
    
    Returns:
        bool: True if successful, False otherwise
    """
    global _model, _scaler, _features
    
    try:
        if not os.path.exists(MODEL_PATH):
            logger.warning(f"Model file not found at {MODEL_PATH}")
            return False
        
        logger.info(f"Loading model from {MODEL_PATH}")
        try:
            # Try loading with joblib
            model_data = joblib.load(MODEL_PATH)
            
            # Handle either package or direct model format
            if isinstance(model_data, dict):
                _model = model_data.get('model')
                _scaler = model_data.get('scaler')
                _features = model_data.get('feature_names')
                logger.info("Loaded model package format")
            else:
                _model = model_data
                logger.info("Loaded direct model format")
            
            # Load scaler if available and not already loaded
            if _scaler is None and os.path.exists(SCALER_PATH):
                try:
                    _scaler = joblib.load(SCALER_PATH)
                    logger.info("Loaded scaler")
                except Exception as scaler_e:
                    logger.error(f"Error loading scaler: {str(scaler_e)}")
                    # Create a default scaler
                    _scaler = StandardScaler()
                    logger.info("Created default scaler")
            
            # Load features if available and not already loaded
            if _features is None and os.path.exists(FEATURES_PATH):
                try:
                    _features = joblib.load(FEATURES_PATH)
                    logger.info("Loaded feature names")
                except Exception as feature_e:
                    logger.error(f"Error loading features: {str(feature_e)}")
                    # Create default feature names
                    _features = [f"feature_{i}" for i in range(28)]
                    logger.info("Created default feature names")
            
            # Create default scaler if not loaded
            if _scaler is None:
                _scaler = StandardScaler()
                dummy_data = np.array([[0.0] * 28])
                _scaler.fit(dummy_data)
                logger.info("Created new default scaler")
            
            # Create default feature names if not loaded
            if _features is None:
                _features = [f"feature_{i}" for i in range(28)]
                logger.info("Created default feature names list")
            
            # Test if model is usable
            if _model is not None:
                try:
                    # Create dummy test data
                    dummy_data = np.array([[0.0] * 28])
                    test_data = _scaler.transform(dummy_data)
                    
                    # Try to make a prediction
                    if hasattr(_model, 'predict_proba'):
                        _ = _model.predict_proba(test_data)
                    else:
                        _ = _model.predict(test_data)
                        
                    logger.info("Model is functional - passed basic prediction test")
                    return True
                except Exception as predict_e:
                    logger.error(f"Model failed prediction test: {str(predict_e)}")
                    return False
            else:
                logger.error("Failed to load model")
                return False
                
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            logger.error(traceback.format_exc())
            return False
    
    except Exception as e:
        logger.error(f"Unexpected error in load_model: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def get_loaded_model():
    """Get the currently loaded model."""
    global _model
    return _model

def get_loaded_scaler():
    """Get the currently loaded scaler."""
    global _scaler
    return _scaler

def get_loaded_features():
    """Get the currently loaded feature names."""
    global _features
    return _features

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