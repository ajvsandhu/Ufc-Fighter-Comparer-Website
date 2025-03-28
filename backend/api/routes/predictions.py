from fastapi import APIRouter, HTTPException, Request, Body, BackgroundTasks, Query, status
from fastapi.responses import Response, JSONResponse
import logging
import traceback
import json
from ...ml.predictor import FighterPredictor
from ...ml.feature_engineering import extract_recent_fight_stats, check_head_to_head, find_common_opponents
from ..database import get_db_connection
from ...constants import (
    LOG_LEVEL,
    LOG_FORMAT,
    LOG_DATE_FORMAT,
    MODEL_PATH,
    API_V1_STR,
    MAX_FIGHTS_DISPLAY
)
from typing import Dict, List, Optional, Any
import re
from urllib.parse import unquote
from pydantic import BaseModel
from ...utils import sanitize_json, set_parent_key

# Set up logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format=LOG_FORMAT,
    datefmt=LOG_DATE_FORMAT
)
logger = logging.getLogger(__name__)

router = APIRouter(prefix=f"{API_V1_STR}/prediction", tags=["Predictions"])

# Create predictor instance and ensure model is loaded
predictor = FighterPredictor()  # Explicitly set train=False to only load the model
if not predictor.model:
    logger.warning("Model not loaded, attempting to load...")
    if not predictor._load_model():
        logger.error("Failed to load model")

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

@router.post("/predict")
async def predict_fight(fight_data: FighterInput):
    try:
        logger.info(f"Received prediction request for {fight_data.fighter1_name} vs {fight_data.fighter2_name}")
        
        # Get database connection
        db = get_db_connection()
        
        # Get fighter data from database
        fighter1_data = db.table("fighters").select("*").eq("fighter_name", fight_data.fighter1_name).execute()
        fighter2_data = db.table("fighters").select("*").eq("fighter_name", fight_data.fighter2_name).execute()
        
        if not fighter1_data.data:
            raise HTTPException(status_code=404, detail=f"Fighter not found: {fight_data.fighter1_name}")
        if not fighter2_data.data:
            raise HTTPException(status_code=404, detail=f"Fighter not found: {fight_data.fighter2_name}")
            
        fighter1 = fighter1_data.data[0]
        fighter2 = fighter2_data.data[0]
        
        # Make prediction
        prediction = predictor.predict_winner(fighter1, fighter2)
        
        if not prediction:
            raise HTTPException(status_code=500, detail="Failed to make prediction")
            
        # Format response
        response = {
            "fighter1": fight_data.fighter1_name,
            "fighter2": fight_data.fighter2_name,
            "predicted_winner": prediction["winner"],
            "confidence": prediction["confidence"],
            "analysis": prediction["analysis"]
        }
        
        logger.info(f"Prediction made successfully: {response}")
        return response
        
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/model-info")
async def get_model_info():
    """Get information about the loaded model."""
    try:
        if not predictor.model:
            return sanitize_json(ModelInfoResponse(model_loaded=False).dict())
        
        # Get model type
        model_type = type(predictor.model).__name__
        
        # Get feature count
        feature_count = len(predictor.feature_names) if predictor.feature_names else 0
        
        # Get important features (if available)
        important_features = None
        if hasattr(predictor.model, "feature_importances_") and predictor.feature_names:
            # Create a list of (feature_name, importance) tuples
            feature_importances = [(predictor.feature_names[i], predictor.model.feature_importances_[i]) 
                                  for i in range(len(predictor.feature_names))]
            
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

@router.get("/status")
async def model_status():
    """Get the status of the prediction model"""
    return sanitize_json({
        "model_loaded": predictor.model is not None,
        "model_path": MODEL_PATH if predictor.model else None
    }) 