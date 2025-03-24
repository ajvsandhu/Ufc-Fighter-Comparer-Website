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
    API_V1_PREFIX
)
from typing import Dict, List, Optional, Any
import json
import re
from urllib.parse import unquote
import sqlite3

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

def format_fighter_data(fighter_data: Dict, last_5_fights: List) -> Dict[str, Any]:
    """Format fighter data for consistent structure"""
    # Create a copy to avoid modifying the original
    fighter_dict = dict(fighter_data)
    
    # Ensure name field exists
    if 'fighter_name' in fighter_dict and not fighter_dict.get('name'):
        fighter_dict['name'] = fighter_dict['fighter_name']
    
    # Add last 5 fights
    fighter_dict['last_5_fights'] = [dict(fight) for fight in last_5_fights] if last_5_fights else []
    
    # Extract additional statistics from last 5 fights
    recent_stats = extract_recent_fight_stats(fighter_dict['last_5_fights'])
    fighter_dict.update(recent_stats)
    
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
    conn = None
    try:
        logger.info(f"Prediction request for {fighter1_name} vs {fighter2_name}")
        # Get fighter data from database
        conn = get_db_connection()
        
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
                # Since weight_class doesn't exist, just search by record
                cursor = conn.execute(
                    """
                    SELECT *, rowid as unique_id FROM fighters 
                    WHERE LOWER(REPLACE(REPLACE(REPLACE(fighter_name, '"', ''), '''', ''), '  ', ' ')) = ?
                      AND Record = ?
                    """, 
                    (fighter1_info['name'], fighter1_info['record'])
                )
                fighter1_data = cursor.fetchone()
            
            # If we couldn't find with record, or if we didn't have record info
            if not fighter1_data:
                cursor = conn.execute(
                    """
                    SELECT *, rowid as unique_id FROM fighters 
                    WHERE LOWER(REPLACE(REPLACE(REPLACE(fighter_name, '"', ''), '''', ''), '  ', ' ')) = ?
                    """, 
                    (fighter1_info['name'],)
                )
                all_matches = cursor.fetchall()
                
                if all_matches and len(all_matches) > 0:
                    # If we have multiple matches and record info, try to find the best match
                    if len(all_matches) > 1 and fighter1_info['record']:
                        best_match = None
                        for match in all_matches:
                            # Get record index
                            record_idx = None
                            for i, col in enumerate(cursor.description):
                                if col[0].lower() == 'record':
                                    record_idx = i
                                    break
                            
                            if record_idx is not None and fighter1_info['record']:
                                match_record = match[record_idx]
                                if match_record and match_record == fighter1_info['record']:
                                    logger.info(f"Found match by record: {match_record}")
                                    best_match = match
                                    break
                        
                        fighter1_data = best_match if best_match else all_matches[0]
                        
                        if not best_match:
                            logger.warning(f"Using first match found since no better match found: {all_matches[0][0]}")
                    else:
                        # Just use the first match
                        fighter1_data = all_matches[0]
                        logger.info(f"Using single match found: {fighter1_data[0]}")
                else:
                    # If no match found, try fuzzy matching
                    cursor = conn.execute(
                        """
                        SELECT *, rowid as unique_id FROM fighters 
                        WHERE LOWER(REPLACE(REPLACE(REPLACE(fighter_name, '"', ''), '''', ''), '  ', ' ')) LIKE ?
                        """, 
                        (f"%{fighter1_info['name']}%",)
                    )
                    potential_matches = cursor.fetchall()
                    
                    if potential_matches and len(potential_matches) > 0:
                        fighter1_data = potential_matches[0]
                        fighter1_found = True
                    else:
                        logger.warning(f"Fighter not found: {fighter1_name}")
                        raise HTTPException(status_code=404, detail=f"Fighter not found: {fighter1_name}. Please check the spelling or try a different fighter.")
            else:
                fighter1_found = True
        except sqlite3.Error as e:
            logger.error(f"Database error looking up fighter 1: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Database error looking up fighter 1: {str(e)}")
        
        # Validate fighter 1 was found
        if not fighter1_data:
            logger.warning(f"Fighter 1 not found: {fighter1_name}")
            raise HTTPException(status_code=404, detail=f"Fighter not found: {fighter1_name}. Please check the spelling or try a different fighter.")
        
        # Similar approach for fighter 2
        fighter2_data = None
        fighter2_found = False
        
        try:
            # Just search by record since weight_class doesn't exist
            if fighter2_info['record']:
                cursor = conn.execute(
                    """
                    SELECT *, rowid as unique_id FROM fighters 
                    WHERE LOWER(REPLACE(REPLACE(REPLACE(fighter_name, '"', ''), '''', ''), '  ', ' ')) = ?
                      AND Record = ?
                    """, 
                    (fighter2_info['name'], fighter2_info['record'])
                )
                fighter2_data = cursor.fetchone()
            
            # If we couldn't find with record, or if we didn't have that info
            if not fighter2_data:
                cursor = conn.execute(
                    """
                    SELECT *, rowid as unique_id FROM fighters 
                    WHERE LOWER(REPLACE(REPLACE(REPLACE(fighter_name, '"', ''), '''', ''), '  ', ' ')) = ?
                    """, 
                    (fighter2_info['name'],)
                )
                all_matches = cursor.fetchall()
                
                if all_matches and len(all_matches) > 0:
                    # If we have multiple matches and record info, try to find the best match
                    if len(all_matches) > 1 and fighter2_info['record']:
                        best_match = None
                        for match in all_matches:
                            # Get record index
                            record_idx = None
                            for i, col in enumerate(cursor.description):
                                if col[0].lower() == 'record':
                                    record_idx = i
                                    break
                            
                            if record_idx is not None and fighter2_info['record']:
                                match_record = match[record_idx]
                                if match_record and match_record == fighter2_info['record']:
                                    logger.info(f"Found match by record: {match_record}")
                                    best_match = match
                                    break
                        
                        fighter2_data = best_match if best_match else all_matches[0]
                        
                        if not best_match:
                            logger.warning(f"Using first match found since no better match found: {all_matches[0][0]}")
                    else:
                        # Just use the first match
                        fighter2_data = all_matches[0]
                        logger.info(f"Using single match found: {fighter2_data[0]}")
                else:
                    # If no match found, try fuzzy matching
                    cursor = conn.execute(
                        """
                        SELECT *, rowid as unique_id FROM fighters 
                        WHERE LOWER(REPLACE(REPLACE(REPLACE(fighter_name, '"', ''), '''', ''), '  ', ' ')) LIKE ?
                        """, 
                        (f"%{fighter2_info['name']}%",)
                    )
                    potential_matches = cursor.fetchall()
                    
                    if potential_matches and len(potential_matches) > 0:
                        fighter2_data = potential_matches[0]
                        fighter2_found = True
                    else:
                        logger.warning(f"Fighter not found: {fighter2_name}")
                        raise HTTPException(status_code=404, detail=f"Fighter not found: {fighter2_name}. Please check the spelling or try a different fighter.")
            else:
                fighter2_found = True
        except sqlite3.Error as e:
            logger.error(f"Database error looking up fighter 2: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Database error looking up fighter 2: {str(e)}")
        
        # Validate fighter 2 was found
        if not fighter2_data:
            logger.warning(f"Fighter 2 not found: {fighter2_name}")
            raise HTTPException(status_code=404, detail=f"Fighter not found: {fighter2_name}. Please check the spelling or try a different fighter.")
        
        # Get unique_id values for both fighters
        unique_id_idx1 = [i for i, col in enumerate(cursor.description) if col[0].lower() == 'unique_id']
        unique_id_idx2 = unique_id_idx1  # Same column position
        
        unique_id1 = str(fighter1_data[unique_id_idx1[0]]) if unique_id_idx1 else "unknown"
        unique_id2 = str(fighter2_data[unique_id_idx2[0]]) if unique_id_idx2 else "unknown"
        
        logger.info(f"Using fighter1 with unique_id: {unique_id1}")
        logger.info(f"Using fighter2 with unique_id: {unique_id2}")
        
        # Check if the same fighter is selected twice
        if (unique_id1 and unique_id2 and 
            unique_id1 == unique_id2 and 
            unique_id1 != "unknown" and 
            unique_id1 != "" and 
            unique_id2 != "unknown" and 
            unique_id2 != ""):
            logger.warning(f"Same fighter selected twice with unique_id: {unique_id1}")
            raise HTTPException(status_code=400, detail="Cannot predict a fighter against themselves. Please select two different fighters.")
        
        # Additional check - if both are "unknown" IDs, but the normalized names are identical, also block
        if (unique_id1 == unique_id2 == "unknown" and 
            fighter1_info['name'].lower() == fighter2_info['name'].lower()):
            logger.warning(f"Same fighter selected twice based on identical name: {fighter1_info['name']}")
            raise HTTPException(status_code=400, detail="Cannot predict a fighter against themselves. Please select two different fighters.")
        
        # Get fighter 1's last 5 fights using the actual name from the database
        fighter_name_idx = [i for i, col in enumerate(cursor.description) if col[0].lower() == 'fighter_name']
        actual_fighter1_name = fighter1_data[fighter_name_idx[0]] if fighter_name_idx else fighter1_data[0]  # Assuming fighter_name is the first column
        cursor = conn.execute(
            """
            SELECT * FROM fighter_last_5_fights 
            WHERE fighter_name = ? 
            ORDER BY fight_date DESC, id DESC 
            LIMIT 5
            """, 
            (actual_fighter1_name,)
        )
        fighter1_last5 = cursor.fetchall()
        logger.info(f"Found {len(fighter1_last5) if fighter1_last5 else 0} fights for {actual_fighter1_name}")
        
        # Get fighter 2's last 5 fights using the actual name from the database
        actual_fighter2_name = fighter2_data[fighter_name_idx[0]] if fighter_name_idx else fighter2_data[0]  # Assuming fighter_name is the first column
        cursor = conn.execute(
            """
            SELECT * FROM fighter_last_5_fights 
            WHERE fighter_name = ? 
            ORDER BY fight_date DESC, id DESC 
            LIMIT 5
            """, 
            (actual_fighter2_name,)
        )
        fighter2_last5 = cursor.fetchall()
        logger.info(f"Found {len(fighter2_last5) if fighter2_last5 else 0} fights for {actual_fighter2_name}")
        
        # Format fighter data for prediction
        fighter1_dict = format_fighter_data(fighter1_data, fighter1_last5)
        fighter2_dict = format_fighter_data(fighter2_data, fighter2_last5)
        
        # Add unique_id to help identify fighters
        fighter1_dict['unique_id'] = unique_id1
        fighter2_dict['unique_id'] = unique_id2
        
        # Check if they have fought each other before
        head_to_head = check_head_to_head(
            fighter1_dict['last_5_fights'], 
            fighter2_dict['fighter_name'], 
            fighter2_dict['last_5_fights'], 
            fighter1_dict['fighter_name']
        )
        
        # Find common opponents and compare performances
        common_opponents = find_common_opponents(
            fighter1_dict['last_5_fights'], 
            fighter2_dict['last_5_fights']
        )
        
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
        
        if conn:
            conn.close()
            conn = None
        
        # Make prediction
        prediction = predictor.predict_winner(fighter1_dict, fighter2_dict, head_to_head, common_opponents)
        
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
            head_to_head,
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
            'fighter1_wins': head_to_head['fighter1_wins'],
            'fighter2_wins': head_to_head['fighter2_wins'],
            'last_winner': head_to_head.get('last_winner'),
            'last_method': head_to_head.get('last_method', '')
        }
        
        return prediction
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        logger.error(traceback.format_exc())
        if conn:
            conn.close()
        raise HTTPException(status_code=500, detail=f"Error making prediction: {str(e)}")

@router.get("/status")
async def model_status():
    """Get the status of the prediction model"""
    return {
        "model_loaded": predictor.model is not None,
        "model_path": MODEL_PATH if predictor.model else None,
        "config": get_config()
    } 