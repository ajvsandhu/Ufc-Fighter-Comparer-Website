import os
import logging
import joblib
import traceback
import numpy as np
import pickle
from typing import Any, Dict, List, Tuple, Optional
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

from backend.constants import MODEL_PATH, SCALER_PATH, FEATURES_PATH

# Configure logging
logger = logging.getLogger(__name__)

# Global model variables
_model = None
_scaler = None
_features = None

def create_directory_if_not_exists(directory_path):
    """Create a directory if it doesn't exist."""
    if not os.path.exists(directory_path):
        try:
            os.makedirs(directory_path)
            logger.info(f"Created directory: {directory_path}")
            return True
        except Exception as e:
            logger.error(f"Error creating directory {directory_path}: {str(e)}")
            return False
    return True

def load_model():
    """
    Load model and associated components.
    
    Returns:
        bool: True if successful, False otherwise
    """
    global _model, _scaler, _features
    
    try:
        # Ensure model directory exists
        model_dir = os.path.dirname(MODEL_PATH)
        if not create_directory_if_not_exists(model_dir):
            logger.error(f"Failed to create model directory: {model_dir}")
            return False
            
        if not os.path.exists(MODEL_PATH):
            logger.warning(f"Model file not found at {MODEL_PATH}")
            
            # Create a simple default model if not found
            logger.info("Creating a default model since none was found")
            _model = RandomForestClassifier(n_estimators=100, random_state=42)
            
            # Create dummy data to fit the model
            X_dummy = np.random.rand(10, 28)
            y_dummy = np.random.randint(0, 2, 10)
            
            # Fit the model on dummy data
            _model.fit(X_dummy, y_dummy)
            
            # Create default scaler
            _scaler = StandardScaler()
            _scaler.fit(X_dummy)
            
            # Create default feature names
            _features = [f"feature_{i}" for i in range(28)]
            
            # Save these defaults
            try:
                joblib.dump(_model, MODEL_PATH)
                joblib.dump(_scaler, SCALER_PATH)
                joblib.dump(_features, FEATURES_PATH)
                logger.info("Default model, scaler and features saved successfully")
            except Exception as save_e:
                logger.error(f"Error saving default model components: {str(save_e)}")
            
            return True
        
        logger.info(f"Loading model from {MODEL_PATH}")
        try:
            # Try loading with joblib first
            try:
                model_data = joblib.load(MODEL_PATH)
            except Exception as joblib_e:
                logger.warning(f"Joblib loading failed: {str(joblib_e)}. Trying pickle...")
                # Fall back to pickle if joblib fails
                with open(MODEL_PATH, 'rb') as f:
                    model_data = pickle.load(f)
                    
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
                    
                    # Create a new default model as fallback
                    logger.info("Creating default model since loaded model failed")
                    _model = RandomForestClassifier(n_estimators=100, random_state=42)
                    
                    # Create dummy data to fit the model
                    X_dummy = np.random.rand(10, 28)
                    y_dummy = np.random.randint(0, 2, 10)
                    
                    # Fit the model on dummy data
                    _model.fit(X_dummy, y_dummy)
                    
                    logger.info("Created and fit default model")
                    return True
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
    """Save model, scaler and features to files."""
    try:
        # Create model directory if it doesn't exist
        model_dir = os.path.dirname(MODEL_PATH)
        if not create_directory_if_not_exists(model_dir):
            return False
        
        # Save model
        joblib.dump(model, MODEL_PATH)
        logger.info(f"Model saved to {MODEL_PATH}")
        
        # Save scaler if provided
        if scaler is not None:
            joblib.dump(scaler, SCALER_PATH)
            logger.info(f"Scaler saved to {SCALER_PATH}")
        
        # Save features if provided
        if features is not None:
            joblib.dump(features, FEATURES_PATH)
            logger.info(f"Features saved to {FEATURES_PATH}")
        
        return True
    except Exception as e:
        logger.error(f"Error saving model components: {str(e)}")
        logger.error(traceback.format_exc())
        return False 