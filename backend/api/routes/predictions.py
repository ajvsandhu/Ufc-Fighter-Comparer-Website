from fastapi import APIRouter, HTTPException, Request
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
    API_V1_PREFIX,
    MAX_FIGHTS_DISPLAY
)
from typing import Dict, List, Optional, Any
import json
import re
from urllib.parse import unquote

# Set up logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format=LOG_FORMAT,
    datefmt=LOG_DATE_FORMAT
)
logger = logging.getLogger(__name__)

router = APIRouter(prefix=f"{API_V1_PREFIX}/prediction")

# Create predictor instance
predictor = FighterPredictor()

@router.get("/train")
async def train_model():
    """Endpoint to manually trigger model training"""
    try:
        success = predictor.train(force=True)
        if success:
            return {"status": "success", "message": "Model trained successfully"}
        else:
            return {"status": "error", "message": "Failed to train model, see server logs for details"}
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
            return {"status": "success", "message": "Fighter rankings updated successfully"}
        else:
            return {"status": "error", "message": "Failed to update fighter rankings, see server logs for details"}
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
        return {"status": "success", "config": updated_config}
    except Exception as e:
        logger.error(f"Error updating config: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error updating config: {str(e)}")

@router.get("/config")
async def get_model_config():
    """Get the current model configuration settings"""
    try:
        return {"status": "success", "config": get_config()}
    except Exception as e:
        logger.error(f"Error getting config: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error getting config: {str(e)}")

@router.get("/config/reset")
async def reset_model_config():
    """Reset the model configuration to default values"""
    try:
        reset_config()
        return {"status": "success", "message": "Configuration reset to defaults", "config": get_config()}
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
            return {"status": "success", "message": "Model retrained successfully"}
        else:
            logger.error("Failed to train model")
            return {"status": "error", "message": "Failed to train model"}
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

@router.get("/predict/{fighter1_name}/{fighter2_name}")
async def predict_winner(fighter1_name: str, fighter2_name: str):
    """Predict the winner between two fighters"""
    try:
        logger.info(f"Prediction request for {fighter1_name} vs {fighter2_name}")
        # Get fighter data from database
        supabase = get_db_connection()
        if not supabase:
            logger.error("No database connection available")
            raise HTTPException(status_code=500, detail="Database connection error")
        
        # Normalize fighter names and extract info
        fighter1_info = normalize_fighter_name(fighter1_name)
        fighter2_info = normalize_fighter_name(fighter2_name)
        
        logger.info(f"Normalized fighter 1: {fighter1_info}")
        logger.info(f"Normalized fighter 2: {fighter2_info}")
        
        # Get fighter 1 data using all available information
        fighter1_data = None
        fighter1_found = False
        
        try:
            # If we have weight class and record, try to use them for more precise matching
            if fighter1_info['record']:
                # Search by name and record
                response = supabase.table('fighters')\
                    .select('*')\
                    .ilike('fighter_name', fighter1_info['name'])\
                    .eq('Record', fighter1_info['record'])\
                    .execute()
                
                if response.data and len(response.data) > 0:
                    fighter1_data = response.data[0]
            
            # If we couldn't find with record, or if we didn't have record info
            if not fighter1_data:
                # Try just by name
                response = supabase.table('fighters')\
                    .select('*')\
                    .ilike('fighter_name', fighter1_info['name'])\
                    .execute()
                
                if response.data and len(response.data) > 0:
                    fighter1_data = response.data[0]
            
            # If still not found, try with fuzzy search
            if not fighter1_data:
                response = supabase.table('fighters')\
                    .select('*')\
                    .ilike('fighter_name', f'%{fighter1_info["name"]}%')\
                    .execute()
                
                if response.data and len(response.data) > 0:
                    fighter1_data = response.data[0]
            
            fighter1_found = fighter1_data is not None
            
        except Exception as e:
            logger.error(f"Error getting fighter 1 data: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error getting fighter 1 data: {str(e)}")
        
        if not fighter1_found:
            raise HTTPException(status_code=404, detail=f"Fighter not found: {fighter1_name}")
            
        # Get fighter 2 data using all available information
        fighter2_data = None
        fighter2_found = False
        
        try:
            # If we have weight class and record, try to use them for more precise matching
            if fighter2_info['record']:
                # Search by name and record
                response = supabase.table('fighters')\
                    .select('*')\
                    .ilike('fighter_name', fighter2_info['name'])\
                    .eq('Record', fighter2_info['record'])\
                    .execute()
                
                if response.data and len(response.data) > 0:
                    fighter2_data = response.data[0]
            
            # If we couldn't find with record, or if we didn't have record info
            if not fighter2_data:
                # Try just by name
                response = supabase.table('fighters')\
                    .select('*')\
                    .ilike('fighter_name', fighter2_info['name'])\
                    .execute()
                
                if response.data and len(response.data) > 0:
                    fighter2_data = response.data[0]
            
            # If still not found, try with fuzzy search
            if not fighter2_data:
                response = supabase.table('fighters')\
                    .select('*')\
                    .ilike('fighter_name', f'%{fighter2_info["name"]}%')\
                    .execute()
                
                if response.data and len(response.data) > 0:
                    fighter2_data = response.data[0]
            
            fighter2_found = fighter2_data is not None
            
        except Exception as e:
            logger.error(f"Error getting fighter 2 data: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error getting fighter 2 data: {str(e)}")
        
        if not fighter2_found:
            raise HTTPException(status_code=404, detail=f"Fighter not found: {fighter2_name}")
            
        # Get recent fights for both fighters
        try:
            # Get fighter 1's last 5 fights
            fights_response = supabase.table('fighter_last_5_fights')\
                .select('*')\
                .eq('fighter_name', fighter1_data['fighter_name'])\
                .order('id')\
                .limit(5)\
                .execute()
                
            fighter1_last_5_fights = fights_response.data if fights_response.data else []
            
            # Get fighter 2's last 5 fights
            fights_response = supabase.table('fighter_last_5_fights')\
                .select('*')\
                .eq('fighter_name', fighter2_data['fighter_name'])\
                .order('id')\
                .limit(5)\
                .execute()
                
            fighter2_last_5_fights = fights_response.data if fights_response.data else []
            
        except Exception as e:
            logger.error(f"Error getting fighter fight data: {str(e)}")
            fighter1_last_5_fights = []
            fighter2_last_5_fights = []
        
        # Format fighter data for prediction
        fighter1_dict = format_fighter_data(fighter1_data, fighter1_last_5_fights)
        fighter2_dict = format_fighter_data(fighter2_data, fighter2_last_5_fights)
        
        # Add unique_id to help identify fighters
        fighter1_dict['unique_id'] = str(fighter1_data.get('id', "unknown"))
        fighter2_dict['unique_id'] = str(fighter2_data.get('id', "unknown"))
        
        # Check if the same fighter is selected twice
        if (fighter1_dict['unique_id'] and fighter2_dict['unique_id'] and 
            fighter1_dict['unique_id'] == fighter2_dict['unique_id'] and 
            fighter1_dict['unique_id'] != "unknown" and 
            fighter1_dict['unique_id'] != "" and 
            fighter2_dict['unique_id'] != "unknown" and 
            fighter2_dict['unique_id'] != ""):
            logger.warning(f"Same fighter selected twice with unique_id: {fighter1_dict['unique_id']}")
            raise HTTPException(status_code=400, detail="Cannot predict a fighter against themselves. Please select two different fighters.")
        
        # Additional check - if both are "unknown" IDs, but the normalized names are identical, also block
        if (fighter1_dict['unique_id'] == fighter2_dict['unique_id'] == "unknown" and 
            fighter1_info['name'].lower() == fighter2_info['name'].lower()):
            logger.warning(f"Same fighter selected twice based on identical name: {fighter1_info['name']}")
            raise HTTPException(status_code=400, detail="Cannot predict a fighter against themselves. Please select two different fighters.")
        
        # Check for head-to-head matchups
        h2h_data = {}
        try:
            # Check if fighter1 has fought fighter2
            h2h_response = supabase.table('fighter_last_5_fights')\
                .select('*')\
                .eq('fighter_name', fighter1_data['fighter_name'])\
                .eq('opponent', fighter2_data['fighter_name'])\
                .execute()
            
            f1_vs_f2_fights = h2h_response.data if h2h_response.data else []
            
            # Check if fighter2 has fought fighter1
            h2h_response = supabase.table('fighter_last_5_fights')\
                .select('*')\
                .eq('fighter_name', fighter2_data['fighter_name'])\
                .eq('opponent', fighter1_data['fighter_name'])\
                .execute()
                
            f2_vs_f1_fights = h2h_response.data if h2h_response.data else []
            
            # Process head-to-head data
            if f1_vs_f2_fights or f2_vs_f1_fights:
                h2h_data = {
                    'fighter1_name': fighter1_data['fighter_name'],
                    'fighter2_name': fighter2_data['fighter_name'],
                    'fighter1_wins': 0,
                    'fighter2_wins': 0,
                    'draws': 0,
                    'total_fights': len(f1_vs_f2_fights) + len(f2_vs_f1_fights)
                }
                
                # Count wins for fighter1
                for fight in f1_vs_f2_fights:
                    result = fight.get('result', '').lower()
                    if 'win' in result or result.startswith('w'):
                        h2h_data['fighter1_wins'] += 1
                    elif 'draw' in result or result.startswith('d'):
                        h2h_data['draws'] += 1
                
                # Count wins for fighter2
                for fight in f2_vs_f1_fights:
                    result = fight.get('result', '').lower()
                    if 'win' in result or result.startswith('w'):
                        h2h_data['fighter2_wins'] += 1
                    elif 'draw' in result or result.startswith('d'):
                        h2h_data['draws'] += 1
                
                # Get the most recent fight
                all_h2h_fights = f1_vs_f2_fights + f2_vs_f1_fights
                if all_h2h_fights:
                    # Sort by date (most recent first)
                    all_h2h_fights.sort(key=lambda x: x.get('fight_date', ''), reverse=True)
                    last_fight = all_h2h_fights[0]
                    
                    if last_fight.get('fighter_name') == fighter1_data['fighter_name']:
                        result = last_fight.get('result', '').lower()
                        if 'win' in result or result.startswith('w'):
                            h2h_data['last_winner'] = fighter1_data['fighter_name']
                        else:
                            h2h_data['last_winner'] = fighter2_data['fighter_name']
                    else:
                        result = last_fight.get('result', '').lower()
                        if 'win' in result or result.startswith('w'):
                            h2h_data['last_winner'] = fighter2_data['fighter_name']
                        else:
                            h2h_data['last_winner'] = fighter1_data['fighter_name']
                    
                    h2h_data['last_method'] = last_fight.get('method', 'Decision')
                    h2h_data['last_round'] = last_fight.get('round', 'N/A')
                    h2h_data['last_time'] = last_fight.get('time', 'N/A')
        
        except Exception as e:
            logger.error(f"Error getting head-to-head data: {str(e)}")
            h2h_data = {}
        
        # Find common opponents
        common_opponents = []
        
        # Get ranking information
        fighter1_rank = fighter1_dict.get('ranking', '')
        fighter2_rank = fighter2_dict.get('ranking', '')
        fighter1_is_champion = bool(fighter1_dict.get('is_champion', 0))
        fighter2_is_champion = bool(fighter2_dict.get('is_champion', 0))

        # Get weight class information (if available)
        fighter1_weight_class = ""
        fighter2_weight_class = ""
        
        try:
            # Try to get weight class for fighter 1
            fighter1_weight_class = fighter1_dict.get('weight_class', '')
        except (KeyError, AttributeError, TypeError):
            fighter1_weight_class = ""
            
        try:
            # Try to get weight class for fighter 2
            fighter2_weight_class = fighter2_dict.get('weight_class', '')
        except (KeyError, AttributeError, TypeError):
            fighter2_weight_class = ""
            
        # Add ranking and weight class info to the data
        fighter1_dict['ranking'] = fighter1_rank
        fighter1_dict['is_champion'] = fighter1_is_champion
        fighter1_dict['weight_class'] = fighter1_weight_class
        
        fighter2_dict['ranking'] = fighter2_rank
        fighter2_dict['is_champion'] = fighter2_is_champion
        fighter2_dict['weight_class'] = fighter2_weight_class
        
        # Make prediction
        prediction = predictor.predict_winner(fighter1_dict, fighter2_dict, h2h_data, common_opponents)
        
        if 'error' in prediction:
            logger.error(prediction['error'])
            error_msg = prediction['error']
            
            # Check for feature mismatch error
            if "has features, but StandardScaler is expecting" in error_msg:
                error_msg = "The model needs to be retrained after the recent updates. Please click the 'Train Model' button and try again."
            
            raise HTTPException(status_code=500, detail=error_msg)
        
        # Generate detailed fight analysis
        analysis = generate_matchup_analysis(
            fighter1_dict, 
            fighter2_dict, 
            h2h_data,
            common_opponents,
            prediction
        )
        
        # Add the analysis to the prediction
        prediction['analysis'] = analysis
        
        # Add fighter details to the response
        prediction['fighter1'] = {
            'name': fighter1_dict['fighter_name'],
            'image_url': fighter1_dict.get('image_url', ''),
            'record': fighter1_dict.get('record', ''),
            'stance': fighter1_dict.get('stance', ''),
            'ranking': fighter1_rank,
            'is_champion': fighter1_is_champion,
            'weight_class': fighter1_weight_class,
            'unique_id': fighter1_dict.get('unique_id', '')
        }
        
        prediction['fighter2'] = {
            'name': fighter2_dict['fighter_name'],
            'image_url': fighter2_dict.get('image_url', ''),
            'record': fighter2_dict.get('record', ''),
            'stance': fighter2_dict.get('stance', ''),
            'ranking': fighter2_rank,
            'is_champion': fighter2_is_champion,
            'weight_class': fighter2_weight_class,
            'unique_id': fighter2_dict.get('unique_id', '')
        }
        
        # Add head-to-head information
        prediction['head_to_head'] = {
            'fighter1_wins': h2h_data.get('fighter1_wins', 0),
            'fighter2_wins': h2h_data.get('fighter2_wins', 0),
            'last_winner': h2h_data.get('last_winner'),
            'last_method': h2h_data.get('last_method', ''),
            'last_round': h2h_data.get('last_round', 'N/A'),
            'last_time': h2h_data.get('last_time', 'N/A')
        }
        
        # Format prediction for API response
        result = predictor.prepare_prediction_for_api(prediction)
        return result
            
    except HTTPException as he:
        # Re-raise HTTP exceptions to maintain their status codes
        raise he
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@router.get("/status")
async def model_status():
    """Get the status of the prediction model"""
    return {
        "model_loaded": predictor.model is not None,
        "model_path": MODEL_PATH if predictor.model else None,
        "config": get_config()
    } 