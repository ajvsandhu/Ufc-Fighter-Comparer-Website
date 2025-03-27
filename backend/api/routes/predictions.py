from fastapi import APIRouter, HTTPException, Request, Body, BackgroundTasks
import logging
import traceback
from backend.ml.predictor import FighterPredictor
from backend.ml.config import get_config, update_config, reset_config
from backend.ml.feature_engineering import extract_recent_fight_stats, check_head_to_head, find_common_opponents
from backend.ml.fight_analysis import generate_matchup_analysis
from scripts.scrapers.ufc_rankings_scraper import fetch_and_update_rankings
from backend.api.database import get_db_connection
from backend.constants import (
    LOG_LEVEL,
    LOG_FORMAT,
    LOG_DATE_FORMAT,
    MODEL_PATH,
    API_V1_STR,
    MAX_FIGHTS_DISPLAY
)
from typing import Dict, List, Optional, Any
import json
import re
from urllib.parse import unquote
from pydantic import BaseModel
from backend.ml.model_loader import get_loaded_model, get_loaded_scaler, get_loaded_features, load_model
from backend.ml.predictor_simple import predict_winner  # Using our simpler implementation
from backend.main import sanitize_json  # Import sanitize_json function

# Set up logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format=LOG_FORMAT,
    datefmt=LOG_DATE_FORMAT
)
logger = logging.getLogger(__name__)

router = APIRouter(prefix=f"{API_V1_STR}/prediction", tags=["Predictions"])

# Create predictor instance
predictor = FighterPredictor()

class FighterInput(BaseModel):
    fighter1_name: str
    fighter2_name: str

class FighterMatchup(BaseModel):
    fighter1: str
    fighter2: str

class ModelInfoResponse(BaseModel):
    model_loaded: bool
    model_type: Optional[str] = None
    feature_count: Optional[int] = None
    important_features: Optional[List[str]] = None

@router.get("/train")
async def train_model():
    """Endpoint to manually trigger model training"""
    try:
        success = predictor.train(force=True)
        if success:
            return sanitize_json({"status": "success", "message": "Model trained successfully"})
        else:
            return sanitize_json({"status": "error", "message": "Failed to train model, see server logs for details"})
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error training model: {str(e)}")

@router.get("/update-rankings")
async def update_rankings():
    """Endpoint to manually trigger fetching and updating fighter rankings"""
    try:
        success = fetch_and_update_rankings()
        if success:
            return sanitize_json({"status": "success", "message": "Fighter rankings updated successfully"})
        else:
            return sanitize_json({"status": "error", "message": "Failed to update fighter rankings, see server logs for details"})
    except Exception as e:
        logger.error(f"Error updating rankings: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error updating rankings: {str(e)}")

@router.post("/config")
async def update_model_config(request: Request):
    """Update the model configuration settings"""
    try:
        config_data = await request.json()
        updated_config = update_config(config_data)
        return sanitize_json({"status": "success", "config": updated_config})
    except Exception as e:
        logger.error(f"Error updating config: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error updating config: {str(e)}")

@router.get("/config")
async def get_model_config():
    """Get the current model configuration settings"""
    try:
        return sanitize_json({"status": "success", "config": get_config()})
    except Exception as e:
        logger.error(f"Error getting config: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error getting config: {str(e)}")

@router.get("/config/reset")
async def reset_model_config():
    """Reset the model configuration to default values"""
    try:
        reset_config()
        return sanitize_json({"status": "success", "message": "Configuration reset to defaults", "config": get_config()})
    except Exception as e:
        logger.error(f"Error resetting config: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error resetting config: {str(e)}")

@router.get("/retrain")
async def retrain_model():
    """Force retrain the model with current scikit-learn version"""
    try:
        logger.info("Starting model retraining...")
        success = predictor.train(force=True)
        if success:
            logger.info("Model retrained successfully")
            return sanitize_json({"status": "success", "message": "Model retrained successfully"})
        else:
            logger.error("Failed to train model")
            return sanitize_json({"status": "error", "message": "Failed to train model"})
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def format_fighter_data(fighter_data: Dict, last_5_fights: List) -> Dict[str, Any]:
    """Format fighter data for consistent structure"""
    # Create a copy to avoid modifying the original
    fighter_dict = dict(fighter_data)
    
    # Ensure name field exists
    if 'fighter_name' in fighter_dict and not fighter_dict.get('name'):
        fighter_dict['name'] = fighter_dict['fighter_name']
    
    # Add last 5 fights
    fighter_dict['last_5_fights'] = [dict(fight) for fight in last_5_fights] if last_5_fights else []
    
    # Convert ID fields to strings for consistency
    if 'id' in fighter_dict:
        fighter_dict['id'] = str(fighter_dict['id'])
        # Also set unique_id for backwards compatibility
        fighter_dict['unique_id'] = str(fighter_dict['id'])
    
    # Extract additional statistics from last 5 fights
    try:
        recent_stats = extract_recent_fight_stats(fighter_dict['last_5_fights'])
        fighter_dict.update(recent_stats)
    except Exception as e:
        logger.warning(f"Error extracting recent fight stats: {e}")
    
    return fighter_dict

def normalize_fighter_name(name: str) -> dict:
    """
    Normalize a fighter name by removing punctuation and extra whitespace,
    and extract record information if present.
    
    Returns a dictionary with 'name' and 'record' keys.
    """
    # Decode URL encoding first
    name = unquote(name)
    
    # Initialize return dictionary
    fighter_info = {
        'name': name,
        'record': None,
        'unique_id': None
    }
    
    # Check if name contains additional info in parentheses
    if "(" in name and ")" in name:
        # Extract base name (everything before the parentheses)
        base_name = name.split("(")[0].strip()
        fighter_info['name'] = base_name
        
        # Extract info from parentheses
        info_part = name.split("(")[1].split(")")[0].strip()
        
        # Handle the new format with just the record in parentheses
        if "Record:" in info_part:
            fighter_info['record'] = info_part.replace("Record:", "").strip()
        elif "," in info_part:
            parts = info_part.split(",")
            if len(parts) >= 2:
                fighter_info['record'] = parts[1].strip()
        else:
            # Just the record itself without any prefix
            fighter_info['record'] = info_part.strip()
    
    # Clean the base name
    base_name = fighter_info['name']
    # Remove quotes and apostrophes
    base_name = base_name.replace("'", "").replace('"', "")
    # Remove other punctuation and normalize whitespace
    base_name = re.sub(r'[^\w\s]', '', base_name).strip()
    # Replace multiple spaces with a single space
    base_name = re.sub(r'\s+', ' ', base_name)
    fighter_info['name'] = base_name.lower()
    
    return fighter_info

def _ensure_string(value, default=""):
    """Helper function to ensure all values are valid strings to prevent frontend errors"""
    if value is None:
        return default
    if not isinstance(value, str):
        try:
            return str(value)
        except:
            return default
    return value

@router.post("/predict")
def predict_fight(fighters: FighterMatchup):
    """Predict the outcome of a fight between two fighters."""
    try:
        # Extract fighter names and normalize
        fighter1_name = _ensure_string(fighters.fighter1, "")
        fighter2_name = _ensure_string(fighters.fighter2, "")
        
        if not fighter1_name.strip() or not fighter2_name.strip():
            logger.error("Invalid fighter names provided")
            raise HTTPException(status_code=400, detail="Both fighter names must be provided")
            
        # Remove record details if present in fighter names
        if "(" in fighter1_name:
            fighter1_name = fighter1_name.split("(")[0].strip()
        if "(" in fighter2_name:
            fighter2_name = fighter2_name.split("(")[0].strip()
            
        logger.info(f"Predicting fight between {fighter1_name} and {fighter2_name}")
        
        # Use predictor to get prediction
        predictor = FighterPredictor()
        
        # Get fighter data
        fighter1_data = predictor._get_fighter_data(fighter1_name)
        fighter2_data = predictor._get_fighter_data(fighter2_name)
        
        if not fighter1_data:
            raise HTTPException(status_code=404, detail=f"Fighter not found: {fighter1_name}")
        if not fighter2_data:
            raise HTTPException(status_code=404, detail=f"Fighter not found: {fighter2_name}")
            
        # Check for head-to-head history
        head_to_head = check_head_to_head(fighter1_name, fighter2_name)
        
        # Find common opponents
        common_opponents = find_common_opponents(fighter1_data, fighter2_data)
        
        # Get prediction
        prediction = predictor.predict_winner(fighter1_data, fighter2_data, head_to_head, common_opponents)
        
        # Check for errors in prediction
        if 'error' in prediction:
            raise HTTPException(status_code=500, detail=prediction['error'])
            
        # Prepare response for API - ensuring all keys are properly formatted
        response = predictor.prepare_prediction_for_api(prediction)
        
        # Additional safeguards for frontend - ensure all fields that might be used with string methods are strings
        if 'fighter1' in response:
            response['fighter1']['name'] = _ensure_string(response['fighter1'].get('name', ''))
            response['fighter1']['win_probability'] = _ensure_string(response['fighter1'].get('win_probability', '0%'))
            response['fighter1']['record'] = _ensure_string(response['fighter1'].get('record', '0-0-0'))
            response['fighter1']['image_url'] = _ensure_string(response['fighter1'].get('image_url', ''))
            
        if 'fighter2' in response:
            response['fighter2']['name'] = _ensure_string(response['fighter2'].get('name', ''))
            response['fighter2']['win_probability'] = _ensure_string(response['fighter2'].get('win_probability', '0%'))
            response['fighter2']['record'] = _ensure_string(response['fighter2'].get('record', '0-0-0'))
            response['fighter2']['image_url'] = _ensure_string(response['fighter2'].get('image_url', ''))
            
        response['winner'] = _ensure_string(response.get('winner', ''))
        response['loser'] = _ensure_string(response.get('loser', ''))
        response['winner_probability'] = _ensure_string(response.get('winner_probability', '0%'))
        
        if 'head_to_head' in response:
            response['head_to_head']['last_winner'] = _ensure_string(response['head_to_head'].get('last_winner', ''))
            response['head_to_head']['last_method'] = _ensure_string(response['head_to_head'].get('last_method', ''))
            response['head_to_head']['last_round'] = _ensure_string(response['head_to_head'].get('last_round', ''))
            response['head_to_head']['last_time'] = _ensure_string(response['head_to_head'].get('last_time', ''))
            
        return sanitize_json(response)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error predicting fight: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

@router.get("/model-info")
async def get_model_info():
    """Get information about the loaded model."""
    try:
        model = get_loaded_model()
        features = get_loaded_features()
        
        if not model:
            return sanitize_json(ModelInfoResponse(model_loaded=False).dict())
        
        # Get model type
        model_type = type(model).__name__
        
        # Get feature count
        feature_count = len(features) if features else 0
        
        # Get important features (if available)
        important_features = None
        if hasattr(model, "feature_importances_") and features:
            # Create a list of (feature_name, importance) tuples
            feature_importances = [(features[i], model.feature_importances_[i]) 
                                  for i in range(len(features))]
            
            # Sort by importance in descending order and take top 10
            feature_importances.sort(key=lambda x: x[1], reverse=True)
            important_features = [f[0] for f in feature_importances[:10]]
        
        return sanitize_json(
            ModelInfoResponse(
                model_loaded=True,
                model_type=model_type,
                feature_count=feature_count,
                important_features=important_features
            ).dict()
        )
    
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        return sanitize_json(ModelInfoResponse(model_loaded=False).dict())

@router.post("/retrain")
async def retrain_model(background_tasks: BackgroundTasks):
    """Retrain the prediction model with the latest data."""
    try:
        # This is a placeholder for a more complex retraining process
        # In a real implementation, we would:
        # 1. Schedule a background task to retrain the model
        # 2. Use the latest fighter data from the database
        # 3. Train a new model and save it
        # 4. Update the model metadata
        
        from backend.ml.train import train_model
        
        background_tasks.add_task(train_model)
        
        return sanitize_json({"message": "Model retraining scheduled"})
    
    except Exception as e:
        logger.error(f"Error scheduling model retraining: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Retraining error: {str(e)}")

@router.get("/status")
async def model_status():
    """Get the status of the prediction model"""
    return sanitize_json({
        "model_loaded": predictor.model is not None,
        "model_path": MODEL_PATH if predictor.model else None,
        "config": get_config()
    }) 