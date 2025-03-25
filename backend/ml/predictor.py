"""
UFC Fighter Prediction System using Machine Learning.

This module implements a machine learning-based system for predicting UFC fight outcomes.
It uses various features including fighter statistics, recent performance, and physical attributes
to generate predictions with confidence levels.
"""

import os
import json
import pickle
import logging
import math
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
import sqlite3
import random
import joblib

from backend.api.database import get_db_connection
from backend.ml.config import get_config
from backend.constants import (
    MODEL_PATH,
    SCALER_PATH,
    FEATURE_NAMES_PATH,
    MODEL_INFO_PATH,
    MODEL_VERSION,
    LOG_LEVEL,
    LOG_FORMAT,
    LOG_DATE_FORMAT,
    DB_PATH
)
from backend.ml.feature_engineering import (
    safe_convert_to_float, 
    extract_height_in_inches, 
    extract_reach_in_inches,
    extract_record_stats,
    calculate_win_percentage,
    extract_style_features,
    extract_recent_fight_stats,
    extract_advanced_fighter_profile,
    extract_physical_comparisons,
    analyze_opponent_quality,
    find_common_opponents,
    check_head_to_head,
    extract_strikes_landed_attempted
)
from backend.ml.fight_analysis import generate_matchup_analysis

# Set up logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format=LOG_FORMAT,
    datefmt=LOG_DATE_FORMAT
)
logger = logging.getLogger(__name__)

# Utility functions
def parse_record(record_str):
    """
    Parse a record string in the format "W-L-D" into wins, losses, and draws
    """
    try:
        parts = record_str.split('-')
        if len(parts) == 3:
            wins = int(parts[0])
            losses = int(parts[1])
            draws = int(parts[2])
            return wins, losses, draws
        else:
            return 0, 0, 0
    except Exception:
        return 0, 0, 0

def calculate_confidence_level(probability):
    """
    Calculate confidence level based on probability
    """
    if probability >= 0.8:
        return "Very High"
    elif probability >= 0.7:
        return "High"
    elif probability >= 0.6:
        return "Moderate"
    elif probability >= 0.5:
        return "Slight"
    else:
        return "Low"

class FighterPredictor:
    """
    A class for predicting UFC fight outcomes using machine learning.
    
    This class handles model training, prediction, and feature engineering for UFC fighter
    predictions. It uses various statistical and machine learning techniques to generate
    predictions with confidence levels.
    
    Attributes:
        model: The trained machine learning model
        scaler: StandardScaler for feature normalization
        feature_names: List of feature names used in the model
        model_info: Dictionary containing model metadata and performance metrics
        config: Configuration dictionary for model parameters
        logger: Logger instance for the class
    """
    
    def __init__(self) -> None:
        """
        Initialize the predictor with default settings.
        
        Sets up the model, scaler, and configuration. Attempts to load an existing
        trained model if available.
        """
        self.model: Optional[Union[RandomForestClassifier, GradientBoostingClassifier]] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_names: Optional[List[str]] = None
        self.model_info: Dict[str, Any] = {
            'last_trained': None,
            'version': MODEL_VERSION,
            'accuracy': None,
            'sample_size': None,
            'status': 'Not trained',
            'message': 'Model not yet trained'
        }
        self.config: Dict[str, Any] = get_config()
        self.logger: logging.Logger = logging.getLogger(__name__)
        
        # Try to load an existing model if available
        self._load_model()
        
        # Ensure model version is set correctly
        if self.model_info and 'version' in self.model_info:
            self.model_info['version'] = MODEL_VERSION

    def _load_model(self) -> bool:
        """
        Load the trained model with better error handling.
        
        Attempts to load the model from disk, handling different format versions
        and potential errors.
        
        Returns:
            bool: True if model was successfully loaded, False otherwise
        """
        if os.path.exists(MODEL_PATH):
            try:
                with open(MODEL_PATH, 'rb') as f:
                    model_data = pickle.load(f)
                
                # Handle different format versions
                if isinstance(model_data, dict):
                    self.model = model_data.get('model')
                    self.scaler = model_data.get('scaler')
                    self.feature_names = model_data.get('feature_names')
                    
                    # If model info is stored in the model file, load it
                    if 'model_info' in model_data:
                        self.model_info = model_data['model_info']
                else:
                    # Legacy format (just the model)
                    self.model = model_data
                    
                    # Try to load scaler and feature names from separate files
                    if os.path.exists(SCALER_PATH):
                        with open(SCALER_PATH, 'rb') as f:
                            self.scaler = pickle.load(f)
                    else:
                        logger.warning("No scaler file found")
                        
                    if os.path.exists(FEATURE_NAMES_PATH):
                        with open(FEATURE_NAMES_PATH, 'rb') as f:
                            self.feature_names = pickle.load(f)
                    else:
                        logger.warning("No feature names file found")
                
                # If model info file exists, load it
                if os.path.exists(MODEL_INFO_PATH):
                    try:
                        with open(MODEL_INFO_PATH, 'r') as f:
                            self.model_info = json.load(f)
                    except:
                        logger.warning("Error loading model info file")
                
                # Validate loaded components
                if self.model is None:
                    logger.error("Loaded model is None")
                    return False
                
                if self.scaler is None:
                    logger.warning("Loaded scaler is None")
                
                if not self.feature_names:
                    logger.warning("No feature names loaded")
                
                # Update model info
                self.model_info['status'] = 'Loaded'
                
                logger.info(f"Successfully loaded model from {MODEL_PATH}")
                return True
            except Exception as e:
                logger.error(f"Error loading model: {str(e)}")
                # Add traceback for better debugging
                import traceback
                logger.error(traceback.format_exc())
                return False
        else:
            logger.info(f"No model file found at {MODEL_PATH}")
        return False
    
    def _save_model(self) -> bool:
        """
        Save the trained model to disk.
        
        Saves the model, scaler, feature names, and model info to disk.
        
        Returns:
            bool: True if model was successfully saved, False otherwise
        """
        if not self.model:
            self.logger.error("Cannot save model - no model is loaded")
            return False
            
        try:
            # Create model directory if it doesn't exist
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
            
            # Create model package with all necessary components
            model_package = {
                'model': self.model,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'model_info': self.model_info
            }
            
            # Save to disk
            with open(MODEL_PATH, 'wb') as f:
                pickle.dump(model_package, f)
                
            self.logger.info(f"Model saved to {MODEL_PATH}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
    
    def _get_db_connection(self) -> Optional[sqlite3.Connection]:
        """
        Get a connection to the SQLite database.
        
        Returns:
            Optional[sqlite3.Connection]: Database connection if successful, None otherwise
        """
        try:
            if not os.path.exists(DB_PATH):
                self.logger.error(f"Database file not found at: {DB_PATH}")
                return None
            conn = sqlite3.connect(DB_PATH)
            conn.row_factory = sqlite3.Row
            return conn
        except sqlite3.Error as e:
            self.logger.error(f"Database connection error: {str(e)}")
            return None
    
    def _get_fighter_data(self, fighter_name: str) -> Optional[Dict[str, Any]]:
        """
        Get fighter data from database by name.
        
        Args:
            fighter_name: Name of the fighter to retrieve data for
            
        Returns:
            Optional[Dict[str, Any]]: Fighter data if found, None otherwise
        """
        try:
            conn = get_db_connection()
            if not conn:
                logger.error("No database connection available")
                return None
                
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM fighters WHERE fighter_name = ?", (fighter_name,))
            fighter = cursor.fetchone()
            
            if not fighter:
                logger.warning(f"Fighter not found in database: {fighter_name}")
                return None
                
            # Convert row to dictionary
            columns = [col[0] for col in cursor.description]
            fighter_data = dict(zip(columns, fighter))
            
            # Get fighter's recent fights
            cursor.execute("""
                SELECT * FROM fighter_last_5_fights 
                WHERE fighter_name = ?
                ORDER BY fight_date DESC LIMIT 10
            """, (fighter_name,))
            
            fights = cursor.fetchall()
            if fights:
                # Convert rows to dictionaries
                columns = [col[0] for col in cursor.description]
                fights_data = [dict(zip(columns, fight)) for fight in fights]
                fighter_data['recent_fights'] = fights_data
                
            cursor.close()
            return fighter_data
            
        except Exception as e:
            logger.error(f"Error retrieving fighter data for {fighter_name}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def _get_fighter_record(self, fighter_name):
        """Get fighter record from the database"""
        try:
            fighter_data = self._get_fighter_data(fighter_name)
            if not fighter_data:
                return None
                
            record = fighter_data.get('Record', '0-0-0')
            try:
                wins, losses, draws = parse_record(record)
            except:
                wins, losses, draws = 0, 0, 0
                
            # Get recent fight results
            recent_results = []
            for fight in fighter_data.get('recent_fights', [])[:3]:  # Last 3 fights
                result = fight.get('result', '')
                if 'w' in result.lower():
                    recent_results.append('W')
                elif 'l' in result.lower():
                    recent_results.append('L')
                else:
                    recent_results.append('D')
                    
            return {
                'record': record,
                'wins': wins,
                'losses': losses,
                'draws': draws,
                'last_three_results': recent_results
            }
        except Exception as e:
            logger.error(f"Error getting fighter record for {fighter_name}: {str(e)}")
            return None
    
    def _get_fighter_image(self, fighter_name):
        """Get fighter image URL from the database"""
        try:
            fighter_data = self._get_fighter_data(fighter_name)
            if not fighter_data:
                return None
                
            return fighter_data.get('image_url')
        except Exception as e:
            logger.error(f"Error getting fighter image for {fighter_name}: {str(e)}")
            return None 

    def _extract_features_from_fighter(self, fighter_name):
        """
        Extract numerical features from a fighter.
        
        Args:
            fighter_name (str): The name of the fighter
            
        Returns:
            dict: A dictionary of fighter features
        """
        try:
            # Get fighter data
            fighter_data = self._get_fighter_data(fighter_name)
            if not fighter_data:
                self.logger.warning(f"No data found for fighter: {fighter_name}")
                return None
                
            # Initialize empty feature vector
            features = {}
            
            # Helper function to safely convert values to float
            def safe_float(value, default=0.0):
                if value is None or value == 'N/A' or value == '':
                    return default
                try:
                    # Handle percentage strings
                    if isinstance(value, str) and '%' in value:
                        return float(value.strip('%')) / 100
                    return float(value)
                except (ValueError, TypeError):
                    return default
            
            # Extract basic stats
            features['slpm'] = safe_float(fighter_data.get('SLpM', 0))
            features['str_acc'] = safe_float(fighter_data.get('Str. Acc.', 0))
            features['sapm'] = safe_float(fighter_data.get('SApM', 0))
            features['str_def'] = safe_float(fighter_data.get('Str. Def', 0))
            features['td_avg'] = safe_float(fighter_data.get('TD Avg.', 0))
            features['td_acc'] = safe_float(fighter_data.get('TD Acc.', 0))
            features['td_def'] = safe_float(fighter_data.get('TD Def.', 0))
            features['sub_avg'] = safe_float(fighter_data.get('Sub. Avg.', 0))
            
            # Extract weight class and encode numerically
            weight_class = fighter_data.get('Weight', 'Unknown')
            weight_classes = {
                'Heavyweight': 5,
                'Light Heavyweight': 4,
                'Middleweight': 3,
                'Welterweight': 2,
                'Lightweight': 1,
                'Featherweight': 0,
                'Bantamweight': -1,
                'Flyweight': -2,
                'Women\'s Bantamweight': -3,
                'Women\'s Strawweight': -4,
                'Women\'s Flyweight': -5,
                'Women\'s Featherweight': -6
            }
            features['weight_class_encoded'] = weight_classes.get(weight_class, 0)
            
            # Parse record (W-L-D) with improved NC handling
            record = fighter_data.get('Record', '0-0-0')
            try:
                # Handle format like "10-5-2" or "10-5-2 (2 NC)"
                main_record = record.split('(')[0].strip()
                parts = main_record.split('-')
                
                if len(parts) >= 3:
                    # Handle potential NC in record string 
                    wins = int(parts[0])
                    losses = int(parts[1])
                    draws = int(parts[2])
                    
                    # Extract NC (No Contest) if available
                    nc = 0
                    if '(' in record and 'NC' in record:
                        nc_part = record.split('(')[1].split(')')[0]
                        nc_matches = [int(s) for s in nc_part.split() if s.isdigit()]
                        if nc_matches:
                            nc = nc_matches[0]
                    
                    features['wins'] = wins
                    features['losses'] = losses
                    features['draws'] = draws
                    features['nc'] = nc
                    features['total_fights'] = wins + losses + draws + nc
                    
                    # Calculate win percentage (excluding NC)
                    if wins + losses + draws > 0:
                        features['win_percentage'] = wins / (wins + losses + draws)
                    else:
                        features['win_percentage'] = 0
                else:
                    features['wins'] = 0
                    features['losses'] = 0
                    features['draws'] = 0
                    features['nc'] = 0
                    features['total_fights'] = 0
                    features['win_percentage'] = 0
            except Exception as e:
                self.logger.warning(f"Error parsing record for {fighter_name}: {str(e)}")
                features['wins'] = 0
                features['losses'] = 0
                features['draws'] = 0
                features['nc'] = 0
                features['total_fights'] = 0
                features['win_percentage'] = 0
            
            # Extract reach
            reach_str = fighter_data.get('Reach', '0"')
            try:
                if reach_str == 'N/A' or not reach_str:
                    features['reach'] = 0  # Default value when reach is unknown
                else:
                    # Clean the string and extract numeric value
                    reach_clean = reach_str.replace('"', '').strip()
                    features['reach'] = safe_float(reach_clean, 0)
            except Exception as e:
                self.logger.warning(f"Error extracting reach for {fighter_name}: {str(e)}")
                features['reach'] = 0
                
            # Extract height
            height_str = fighter_data.get('Height', "0' 0\"")
            try:
                if height_str == 'N/A' or not height_str:
                    features['height'] = 0  # Default height
                else:
                    # Handle format like "5' 10""
                    if "'" in height_str and '"' in height_str:
                        feet_part, inches_part = height_str.split("'")
                        feet = safe_float(feet_part.strip(), 0)
                        inches = safe_float(inches_part.replace('"', '').strip(), 0)
                        total_inches = (feet * 12) + inches
                        features['height'] = total_inches
                    else:
                        # Try direct conversion if in different format
                        features['height'] = safe_float(height_str.replace('"', '').replace("'", '').strip(), 0)
            except Exception as e:
                self.logger.warning(f"Error extracting height for {fighter_name}: {str(e)}")
                features['height'] = 0
                
            # Calculate age from DOB
            dob = fighter_data.get('DOB', 'N/A')
            try:
                if dob == 'N/A' or not dob:
                    features['age'] = 30  # default age
                else:
                    try:
                        # Try standard format first
                        birth_date = datetime.strptime(dob, '%b %d, %Y')
                    except ValueError:
                        try:
                            # Try alternative format
                            birth_date = datetime.strptime(dob, '%B %d, %Y')
                        except ValueError:
                            # If still failing, extract year if possible
                            year_match = [int(s) for s in dob.split() if s.isdigit() and len(s) == 4]
                            if year_match:
                                features['age'] = datetime.today().year - year_match[0]
                            else:
                                features['age'] = 30  # default age
                            raise  # Re-raise to skip the normal age calculation
                    
                    today = datetime.today()
                    features['age'] = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
            except Exception as e:
                if 'age' not in features:  # Only set if not already set in the except block above
                    self.logger.warning(f"Error calculating age for {fighter_name}: {str(e)}")
                    features['age'] = 30  # default age
                
            # Calculate striking and takedown differentials
            features['striking_differential'] = features['slpm'] - features['sapm']
            features['takedown_differential'] = features['td_avg'] * features['td_acc'] - features['td_avg'] * (1 - features['td_def'])
            
            # Calculate combat effectiveness metric
            features['combat_effectiveness'] = (features['slpm'] * features['str_acc']) + (features['td_avg'] * features['td_acc']) + features['sub_avg'] - features['sapm'] * (1 - features['str_def'])
            
            # Encode stance
            stance = fighter_data.get('Stance', 'Orthodox').lower()
            if stance == 'N/A' or not stance:
                features['stance_encoded'] = 3  # unknown
            elif 'orthodox' in stance:
                features['stance_encoded'] = 0
            elif 'southpaw' in stance:
                features['stance_encoded'] = 1
            elif 'switch' in stance:
                features['stance_encoded'] = 2
            else:
                features['stance_encoded'] = 3  # other/unknown
                
            # Additional style features
            features['is_striker'] = 1 if features['slpm'] > 3.0 and features['str_acc'] > 0.4 else 0
            features['is_grappler'] = 1 if features['td_avg'] > 2.0 or features['sub_avg'] > 0.5 else 0
            
            # Process recent fights
            recent_fights = fighter_data.get('recent_fights', [])[:5]  # Get up to 5 recent fights
            
            # Extract results
            results = []
            for fight in recent_fights:
                result = fight.get('result', '').lower()
                if result:
                    if 'w' in result or 'win' in result:
                        results.append('W')
                    elif 'l' in result or 'loss' in result:
                        results.append('L')
                    elif 'd' in result or 'draw' in result:
                        results.append('D')
                    else:
                        results.append('U')  # Unknown
            
            # Calculate recent fight statistics
            features['recent_win_streak'] = sum(1 for r in results if r == 'W')
            features['recent_loss_streak'] = sum(1 for r in results if r == 'L')
            
            # Advanced fighter profile
            if results:
                features['finish_rate'] = sum(1 for fight in recent_fights if 'ko' in fight.get('method', '').lower() or 'sub' in fight.get('method', '').lower() or 'tko' in fight.get('method', '').lower()) / len(results)
                features['decision_rate'] = sum(1 for fight in recent_fights if 'dec' in fight.get('method', '').lower() or 'decision' in fight.get('method', '').lower()) / len(results)
            else:
                features['finish_rate'] = 0
                features['decision_rate'] = 0
            
            # Career progression metrics
            features['career_length'] = features['total_fights']
            features['experience_factor'] = features['total_fights'] * features['win_percentage']
            
            # Normalize features to avoid extreme values
            for key in features:
                if isinstance(features[key], (int, float)):
                    # Cap extreme values
                    if features[key] > 100:
                        features[key] = 100
                    elif features[key] < -100:
                        features[key] = -100
            
            self.logger.info(f"Successfully extracted {len(features)} features for {fighter_name}")
            return features
            
        except Exception as e:
            self.logger.error(f"Error extracting features for {fighter_name}: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
    
    def _get_training_data(self):
        """
        Retrieve training data from the database.
        
        Returns:
            tuple: (X, y) where X is the feature matrix and y is the target vector
        """
        connection = None
        try:
            connection = self._get_db_connection()
            if not connection:
                self.logger.error("Failed to establish database connection for training data retrieval")
                return None, None
                
            # Fetch all fighters from the database
            cursor = connection.cursor()
            cursor.execute("SELECT * FROM fighters")
            fighter_rows = cursor.fetchall()
            
            if not fighter_rows:
                self.logger.warning("No fighters found in the database")
                return None, None
                
            # Create a mapping of fighter names to their data
            fighter_map = {}
            columns = [col[0] for col in cursor.description]
            
            for fighter_row in fighter_rows:
                fighter_dict = dict(zip(columns, fighter_row))
                fighter_name = fighter_dict.get('fighter_name')
                if fighter_name:
                    fighter_map[fighter_name] = fighter_dict
            
            self.logger.info(f"Loaded {len(fighter_map)} fighters for training")
            
            # Fetch fight data - use batched processing to handle large datasets
            # and optimize the query to better match fight records
            cursor.execute("""
                SELECT f1.* 
                FROM fighter_last_5_fights f1
                WHERE f1.result IS NOT NULL 
                AND f1.result != ''
                AND f1.opponent IS NOT NULL
                AND f1.opponent != ''
                ORDER BY f1.fight_date DESC
            """)
            
            all_fights = cursor.fetchall()
            
            if not all_fights:
                self.logger.warning("No valid fights found in the database for training")
                return None, None
                
            self.logger.info(f"Found {len(all_fights)} total fight records")
            
            # Convert to dictionaries for easier processing
            fight_columns = [col[0] for col in cursor.description]
            all_fight_dicts = [dict(zip(fight_columns, fight)) for fight in all_fights]
            
            # Build a lookup of fights by fighter-opponent pairs
            fight_lookup = {}
            for fight in all_fight_dicts:
                fighter = fight.get('fighter_name')
                opponent = fight.get('opponent')
                if fighter and opponent:
                    key = (fighter, opponent)
                    fight_lookup[key] = fight
            
            # Find matching fight pairs between fighter and opponent
            processed_fights = []
            processed_pairs = set()
            
            for fight in all_fight_dicts:
                fighter1 = fight.get('fighter_name')
                fighter2 = fight.get('opponent')
                
                if not fighter1 or not fighter2:
                        continue
                        
                # Skip if we've already processed this pair
                if (fighter1, fighter2) in processed_pairs or (fighter2, fighter1) in processed_pairs:
                        continue
                        
                # Look for the reverse fight (opponent's perspective)
                reverse_key = (fighter2, fighter1)
                if reverse_key in fight_lookup:
                    fighter1_fight = fight
                    fighter2_fight = fight_lookup[reverse_key]
                    
                    # Add as a matched pair
                    processed_fights.append((fighter1_fight, fighter2_fight))
                    processed_pairs.add((fighter1, fighter2))
                    processed_pairs.add((fighter2, fighter1))
            
            self.logger.info(f"Successfully matched {len(processed_fights)} fight pairs")
            
            if not processed_fights:
                self.logger.error("Could not find any valid fight pairs for training")
                return None, None
                
            # Process fights to extract features and labels
            X = []
            y = []
            processed_count = 0
            skipped_count = 0
            
            # Use batching to prevent memory issues
            batch_size = 500
            for batch_index in range(0, len(processed_fights), batch_size):
                batch_end = min(batch_index + batch_size, len(processed_fights))
                batch = processed_fights[batch_index:batch_end]
                
                self.logger.info(f"Processing batch {batch_index//batch_size + 1} ({batch_index} to {batch_end})")
                
                for i, (fight1_dict, fight2_dict) in enumerate(batch):
                    try:
                        # Get fighter names
                        fighter1_name = fight1_dict.get('fighter_name')
                        fighter2_name = fight1_dict.get('opponent')
                        
                        if not fighter1_name or not fighter2_name:
                            self.logger.warning(f"Missing fighter names in fight {batch_index + i}")
                            skipped_count += 1
                            continue
                                        
                        # Get fight result
                        result = fight1_dict.get('result', '').strip().upper()
                        
                        if not result:
                            self.logger.warning(f"Missing result in fight between {fighter1_name} and {fighter2_name}")
                            skipped_count += 1
                            continue
                        
                        # Extract features for both fighters
                        fighter1_features = self._extract_features_from_fighter(fighter1_name)
                        fighter2_features = self._extract_features_from_fighter(fighter2_name)
                        
                        if not fighter1_features or not fighter2_features:
                            self.logger.warning(f"Could not extract features for {fighter1_name} or {fighter2_name}")
                            skipped_count += 1
                            continue
                        
                        # Ensure both feature sets have the same keys
                        all_keys = set(fighter1_features.keys()).union(set(fighter2_features.keys()))
                        for key in all_keys:
                            if key not in fighter1_features:
                                fighter1_features[key] = 0
                            if key not in fighter2_features:
                                fighter2_features[key] = 0
                        
                        # Create feature vector (difference between fighters)
                        feature_keys = sorted(all_keys)
                        feature_vector = []
                        
                        for key in feature_keys:
                            feature_vector.append(fighter1_features[key] - fighter2_features[key])
                        
                        # Determine label: 1 if fighter1 won, 0 if fighter2 won
                        if 'W' in result:
                            label = 1  # fighter1 won
                        elif 'L' in result:
                            label = 0  # fighter1 lost (fighter2 won)
                        else:
                            # Skip draws or unknown results
                            skipped_count += 1
                            continue
                        
                        X.append(feature_vector)
                        y.append(label)
                        processed_count += 1
                        
                    except Exception as e:
                        self.logger.error(f"Error processing fight {batch_index + i}: {str(e)}")
                        import traceback
                        self.logger.error(traceback.format_exc())
                        skipped_count += 1
            
            self.logger.info(f"Processed {processed_count} fights, skipped {skipped_count} fights")
            
            if processed_count == 0:
                self.logger.error("No training examples could be processed")
                return None, None
            
            # Convert to numpy arrays
            try:
                X = np.array(X)
                y = np.array(y)
                
                # Clean any NaN or infinite values
                X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
                
                self.logger.info(f"Final training data shape: X={X.shape}, y={y.shape}")
                return X, y
            except ValueError as e:
                self.logger.error(f"Error converting to numpy arrays: {str(e)}")
                # Print a sample of feature vector lengths to debug
                if len(X) > 0:
                    lengths = [len(x) for x in X[:10]]
                    self.logger.error(f"Sample feature vector lengths: {lengths}")
                return None, None
                
        except Exception as e:
            self.logger.error(f"Error retrieving training data: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None, None
        finally:
            if connection:
                connection.close()
    
    def train(self, force=False):
        """
        Train the model using fight data from the database.
        
        Args:
            force (bool): If True, train even if a model is already loaded
            
        Returns:
            bool: True if training was successful, False otherwise
        """
        if self.model is not None and not force:
            self.logger.info("Model already exists. Use force=True to retrain")
            return True
        
        self.logger.info("Starting model training")
        
        try:
            # Get training data
            X, y = self._get_training_data()
            
            if X is None or y is None or len(X) < 10:
                self.logger.error("Insufficient training data")
                self.model_info['status'] = 'Error'
                self.model_info['message'] = 'Insufficient training data'
                return False
                
            self.logger.info(f"Training with {len(X)} samples")
            
            # Split data into training and validation sets
            # Use a larger portion for training with more data
            train_size = 0.8 if len(X) > 1000 else 0.7
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=(1.0 - train_size), random_state=42, stratify=y
            )
            
            # Scale features
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Get model configuration with improved defaults
            model_config = self.config.get('model', {})
            model_type = model_config.get('type', 'GradientBoosting')
            
            # Dynamic parameter scaling based on dataset size
            n_estimators = model_config.get('n_estimators', 0)
            if n_estimators <= 0:
                # Automatically scale number of estimators based on dataset size
                if len(X) > 5000:
                    n_estimators = 200
                elif len(X) > 1000:
                    n_estimators = 150
                else:
                    n_estimators = 100
            
            # Other parameters with better defaults
            max_depth = model_config.get('max_depth', 5)  # Deeper trees for more complex patterns
            learning_rate = model_config.get('learning_rate', 0.05)  # Lower learning rate for better generalization
            
            # Create and train model with optimized parameters
            if model_type.lower() in ('gradientboosting', 'gbm'):
                self.logger.info(f"Training GradientBoostingClassifier with {n_estimators} estimators")
                model = GradientBoostingClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    subsample=0.8,  # Use subsampling to reduce overfitting
                    min_samples_split=10,  # Require more samples to split nodes
                    min_samples_leaf=5,  # Require more samples in leaf nodes
                    random_state=42
                )
            elif model_type.lower() in ('randomforest', 'rf'):
                self.logger.info(f"Training RandomForestClassifier with {n_estimators} estimators")
                model = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=10,
                    min_samples_leaf=5,
                    bootstrap=True,
                    class_weight='balanced',  # Better handling of class imbalance
                    random_state=42,
                    n_jobs=-1  # Use all available cores
                )
            else:
                self.logger.warning(f"Unknown model type: {model_type}, defaulting to GradientBoosting")
                model = GradientBoostingClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    random_state=42
                )
            
            # Train the model with progress logging for large datasets
            if len(X_train) > 5000:
                self.logger.info("Large dataset detected, training in stages...")
                batch_size = 1000
                for i in range(0, len(X_train), batch_size):
                    end_idx = min(i + batch_size, len(X_train))
                    self.logger.info(f"Training on batch {i//batch_size + 1}: samples {i} to {end_idx}")
                    
                    # For first batch, use fit; for subsequent batches, use partial_fit if available
                    if i == 0 or not hasattr(model, 'partial_fit'):
                        model.fit(X_train_scaled[i:end_idx], y_train[i:end_idx])
                    else:
                        if hasattr(model, 'partial_fit'):
                            model.partial_fit(X_train_scaled[i:end_idx], y_train[i:end_idx])
                        else:
                            # If partial_fit not available, continue with normal fit
                            model.fit(X_train_scaled[i:end_idx], y_train[i:end_idx])
            else:
                # For smaller datasets, train normally
                model.fit(X_train_scaled, y_train)
            
            # Calibrate probabilities if configured
            if model_config.get('calibrate_probabilities', True):
                self.logger.info("Calibrating probability estimates")
                calibrated_model = CalibratedClassifierCV(
                    model, method='sigmoid', cv='prefit'
                )
                calibrated_model.fit(X_test_scaled, y_test)
                self.model = calibrated_model
            else:
                self.model = model
                
            # Evaluate model
            y_pred = self.model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            self.logger.info(f"Model accuracy: {accuracy:.4f}")
            self.logger.info("\nClassification Report:\n" + 
                               classification_report(y_test, y_pred))
            
            # Extract feature importance if available
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                self.logger.info("Feature importances:")
                for i, importance in enumerate(importances):
                    self.logger.info(f"Feature {i}: {importance:.4f}")
            
            # Update model info
            self.model_info['last_trained'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            self.model_info['accuracy'] = float(accuracy)
            self.model_info['sample_size'] = len(X)
            self.model_info['status'] = 'Trained'
            self.model_info['message'] = f'Model trained with accuracy {accuracy:.4f}'
            self.model_info['model_type'] = model_type
            self.model_info['n_estimators'] = n_estimators
            self.model_info['max_depth'] = max_depth
            
            # Save the model
            self._save_model()
            
            self.logger.info("Model training completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error training model: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            self.model_info['status'] = 'Error'
            self.model_info['message'] = f'Training error: {str(e)}'
            return False
    
    def predict_winner(self, fighter1_data, fighter2_data, head_to_head=None, common_opponents=None):
        """
        Predict the winner between two fighters.
        
        Args:
            fighter1_data (dict): Dictionary containing data for the first fighter
            fighter2_data (dict): Dictionary containing data for the second fighter
            head_to_head (dict, optional): Head-to-head data between the fighters
            common_opponents (list, optional): List of common opponents
            
        Returns:
            dict: Prediction results
        """
        try:
            if self.model is None:
                self.logger.error("Model not loaded")
                return {
                    'error': "Model not loaded. Please train the model first."
                }
                
            fighter1_name = fighter1_data.get('fighter_name') or fighter1_data.get('name')
            fighter2_name = fighter2_data.get('fighter_name') or fighter2_data.get('name')
                
            self.logger.info(f"Predicting winner between {fighter1_name} and {fighter2_name}")
            
            # Extract features for both fighters
            fighter1_features = self._extract_features_from_fighter(fighter1_name)
            fighter2_features = self._extract_features_from_fighter(fighter2_name)
            
            if not fighter1_features:
                self.logger.error(f"Could not extract features for {fighter1_name}")
                return {
                    'error': f'Could not extract features for {fighter1_name}'
                }
                
            if not fighter2_features:
                self.logger.error(f"Could not extract features for {fighter2_name}")
                return {
                    'error': f'Could not extract features for {fighter2_name}'
                }
            
            # Ensure both feature sets have the same keys
            all_keys = set(fighter1_features.keys()).union(set(fighter2_features.keys()))
            
            for key in all_keys:
                if key not in fighter1_features:
                    fighter1_features[key] = 0
                if key not in fighter2_features:
                    fighter2_features[key] = 0
            
            # Create a feature vector (difference between fighters)
            # This matches our training approach where features are fighter1 - fighter2
            feature_keys = sorted(all_keys)
            
            # SOLUTION FOR POSITION BIAS: Make predictions in both directions and average them
            # First direction: fighter1 vs fighter2
            feature_vector_1vs2 = []
            for key in feature_keys:
                feature_vector_1vs2.append(fighter1_features[key] - fighter2_features[key])
            
            # Second direction: fighter2 vs fighter1 (reversed)
            feature_vector_2vs1 = []
            for key in feature_keys:
                feature_vector_2vs1.append(fighter2_features[key] - fighter1_features[key])
            
            # Check for NaN or Inf values in both vectors
            for i, val in enumerate(feature_vector_1vs2):
                if math.isnan(val) or math.isinf(val):
                    feature_vector_1vs2[i] = 0.0
            
            for i, val in enumerate(feature_vector_2vs1):
                if math.isnan(val) or math.isinf(val):
                    feature_vector_2vs1[i] = 0.0
            
            # Convert to numpy arrays and reshape for predictions
            X_1vs2 = np.array([feature_vector_1vs2])
            X_2vs1 = np.array([feature_vector_2vs1])
            
            # Scale features if scaler exists
            if self.scaler:
                X_1vs2_scaled = self.scaler.transform(X_1vs2)
                X_2vs1_scaled = self.scaler.transform(X_2vs1)
            else:
                X_1vs2_scaled = X_1vs2
                X_2vs1_scaled = X_2vs1
                
            # Make predictions in both directions
            if hasattr(self.model, 'predict_proba'):
                # Direction 1: fighter1 vs fighter2
                probas_1vs2 = self.model.predict_proba(X_1vs2_scaled)[0]
                f1_wins_prob = probas_1vs2[1]  # Index 1 is probability of class 1 (fighter1 wins)
                f2_wins_prob = probas_1vs2[0]  # Index 0 is probability of class 0 (fighter2 wins)
                
                # Direction 2: fighter2 vs fighter1 (we need to invert this result)
                probas_2vs1 = self.model.predict_proba(X_2vs1_scaled)[0]
                f2_wins_prob_alt = probas_2vs1[1]  # This is actually fighter2's win probability when in position 1
                f1_wins_prob_alt = probas_2vs1[0]  # This is actually fighter1's win probability when in position 2
                
                # Average the probabilities from both directions to eliminate position bias
                f1_final_prob = (f1_wins_prob + f1_wins_prob_alt) / 2
                f2_final_prob = (f2_wins_prob + f2_wins_prob_alt) / 2
                
                # Normalize to ensure they sum to 1.0
                total_prob = f1_final_prob + f2_final_prob
                if total_prob > 0:
                    f1_final_prob = f1_final_prob / total_prob
                    f2_final_prob = f2_final_prob / total_prob
                else:
                    # Fallback if something went wrong with probability calculation
                    f1_final_prob = 0.5
                    f2_final_prob = 0.5
            else:
                # For models without predict_proba, use basic predict
                pred_1vs2 = self.model.predict(X_1vs2_scaled)[0]
                pred_2vs1 = self.model.predict(X_2vs1_scaled)[0]
                
                # Average the predictions (inverting the second one)
                f1_final_prob = (float(pred_1vs2) + (1.0 - float(pred_2vs1))) / 2
                f2_final_prob = 1.0 - f1_final_prob
            
            # Determine winner and format probabilities based on final probabilities
            if f1_final_prob > f2_final_prob:
                winner = fighter1_name
                winner_prob = f1_final_prob
                loser = fighter2_name
                loser_prob = f2_final_prob
            else:
                winner = fighter2_name
                winner_prob = f2_final_prob
                loser = fighter1_name
                loser_prob = f1_final_prob
                
            # Calculate confidence level - difference between probabilities
            confidence = max(f1_final_prob, f2_final_prob)
            
            # Get fighter records
            fighter1_record = fighter1_data.get('record', 'N/A')
            fighter2_record = fighter2_data.get('record', 'N/A')
            
            # Prepare response
            result = {
                'winner': winner,
                'loser': loser,
                'winner_probability': winner_prob,
                'loser_probability': loser_prob,
                'prediction_confidence': confidence,
                'model_version': self.model_info.get('version', '2.0'),
                'model_accuracy': self.model_info.get('accuracy', 'N/A'),
                'head_to_head': head_to_head or {},
                'fighter1': {
                    'name': fighter1_name,
                    'record': fighter1_record,
                    'image_url': fighter1_data.get('image_url', '')
                },
                'fighter2': {
                    'name': fighter2_name,
                    'record': fighter2_record,
                    'image_url': fighter2_data.get('image_url', '')
                }
            }
            
            self.logger.info(f"Prediction: {winner} to win with {winner_prob:.2f} probability (confidence: {confidence:.2f})")
            return result
            
        except Exception as e:
            self.logger.error(f"Error predicting winner: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {
                'error': f'Error during prediction: {str(e)}'
            }

    def get_model_info(self):
        """
        Get information about the trained model
        """
        # Return a copy of model_info to prevent modification
        info = dict(self.model_info)
        
        # Check if model exists and add more information
        if self.model is not None:
            info['model_type'] = type(self.model).__name__
            
            # Add number of features if available
            if self.feature_names:
                info['num_features'] = len(self.feature_names)
                
            # Check if we have a trained model
            if hasattr(self.model, 'n_estimators'):
                info['n_estimators'] = self.model.n_estimators
            
            # For RandomForest or similar models, add more details
            if hasattr(self.model, 'estimators_') and hasattr(self.model, 'n_estimators'):
                info['num_trees'] = len(self.model.estimators_)
                info['max_depth'] = self.model.max_depth if hasattr(self.model, 'max_depth') else 'N/A'
            
            # For GradientBoosting
            if hasattr(self.model, 'learning_rate'):
                info['learning_rate'] = self.model.learning_rate
            else:
                info['status'] = 'Not trained'
            
        return info
        
    def prepare_prediction_for_api(self, prediction):
        """
        Format the prediction result for the API response
        """
        if 'error' in prediction:
            return {
                'success': False,
                'message': prediction.get('error', 'Unknown error')
            }
            
        # If prediction successful, format the output
        fighter1_name = prediction.get('fighter1', {}).get('name')
        fighter2_name = prediction.get('fighter2', {}).get('name')
        winner_name = prediction.get('winner')
        loser_name = prediction.get('loser')
        
        # Determine which fighter is which
        fighter1_is_winner = fighter1_name == winner_name
        
        result = {
            'success': True,
            'fighter1': {
                'name': fighter1_name,
                'win_probability': f"{int(round(prediction['winner_probability' if fighter1_is_winner else 'loser_probability'] * 100))}%",
                'record': prediction.get('fighter1', {}).get('record', 'N/A'),
                'image_url': prediction.get('fighter1', {}).get('image_url', '')
            },
            'fighter2': {
                'name': fighter2_name,
                'win_probability': f"{int(round(prediction['winner_probability' if not fighter1_is_winner else 'loser_probability'] * 100))}%",
                'record': prediction.get('fighter2', {}).get('record', 'N/A'),
                'image_url': prediction.get('fighter2', {}).get('image_url', '')
            },
            'winner': winner_name,
            'loser': loser_name,
            'winner_probability': f"{int(round(prediction['winner_probability'] * 100))}%",
            'prediction_confidence': prediction.get('prediction_confidence', 0.5),
            'model': {
                'version': prediction.get('model_version', '2.0'),
                'accuracy': prediction.get('model_accuracy', 'N/A'),
                'status': 'Trained'
            },
            'analysis': self._generate_fight_analysis(prediction) if hasattr(self, '_generate_fight_analysis') else None,
            'head_to_head': prediction.get('head_to_head', {})
        }
        
        return result 
        
    def _generate_fight_analysis(self, prediction):
        """
        Generate a textual analysis of the fight prediction
        
        Args:
            prediction (dict): The prediction results
            
        Returns:
            str: A detailed fight analysis
        """
        try:
            # Extract fighter information
            fighter1_name = prediction.get('fighter1', {}).get('name')
            fighter2_name = prediction.get('fighter2', {}).get('name')
            winner_name = prediction.get('winner')
            loser_name = prediction.get('loser')
            win_probability = prediction.get('winner_probability', 0.5)
            
            # Extract rankings if available
            fighter1_rank = prediction.get('fighter1', {}).get('ranking', 'unranked')
            fighter2_rank = prediction.get('fighter2', {}).get('ranking', 'unranked')
            fighter1_is_champion = prediction.get('fighter1', {}).get('is_champion', False)
            fighter2_is_champion = prediction.get('fighter2', {}).get('is_champion', False)
            
            # Determine fighter styles if available
            fighter1_style = prediction.get('fighter1', {}).get('style', 'balanced')
            fighter2_style = prediction.get('fighter2', {}).get('style', 'balanced')
            
            # Format rankings for display
            f1_rank_display = "Champion" if fighter1_is_champion else f"#{fighter1_rank}" if fighter1_rank and fighter1_rank != "unranked" else "Unranked"
            f2_rank_display = "Champion" if fighter2_is_champion else f"#{fighter2_rank}" if fighter2_rank and fighter2_rank != "unranked" else "Unranked"
            
            # Generate intro based on confidence
            confidence_text = ""
            if win_probability > 0.80:
                confidence_text = f"Our model strongly favors {winner_name} with {int(win_probability*100)}% confidence"
            elif win_probability > 0.65:
                confidence_text = f"Our model predicts {winner_name} as the likely winner with {int(win_probability*100)}% confidence"
            else:
                confidence_text = f"In a close matchup, our model narrowly favors {winner_name} with {int(win_probability*100)}% confidence"
            
            # Generate ranking context
            ranking_context = ""
            if fighter1_rank and fighter2_rank and fighter1_rank != "unranked" and fighter2_rank != "unranked":
                try:
                    rank1 = int(fighter1_rank) if not fighter1_is_champion else 0
                    rank2 = int(fighter2_rank) if not fighter2_is_champion else 0
                    rank_diff = abs(rank1 - rank2)
                    
                    if fighter1_is_champion and not fighter2_is_champion:
                        ranking_context = f"The champion {fighter1_name} faces #{fighter2_rank} ranked contender {fighter2_name}."
                    elif fighter2_is_champion and not fighter1_is_champion:
                        ranking_context = f"The champion {fighter2_name} faces #{fighter1_rank} ranked contender {fighter1_name}."
                    elif rank_diff > 10:
                        ranking_context = f"This represents a significant ranking mismatch with {rank_diff} positions separating them."
                    elif rank_diff > 5:
                        ranking_context = f"There's a notable ranking gap of {rank_diff} positions between these fighters."
                    elif rank_diff <= 3:
                        ranking_context = f"This is a closely matched contest between similarly ranked fighters."
                except:
                    # Fall back if we can't convert ranks to numbers
                    ranking_context = f"This matchup pits {f1_rank_display} {fighter1_name} against {f2_rank_display} {fighter2_name}."
            
            # Generate style matchup text
            style_matchup = ""
            if fighter1_style and fighter2_style and fighter1_style != "balanced" and fighter2_style != "balanced":
                if fighter1_style == "striker" and fighter2_style == "grappler":
                    style_matchup = f"Classic striker vs grappler matchup with {fighter1_name}'s striking against {fighter2_name}'s ground game."
                elif fighter1_style == "grappler" and fighter2_style == "striker":
                    style_matchup = f"Classic grappler vs striker matchup with {fighter1_name}'s ground game against {fighter2_name}'s striking."
                elif fighter1_style == fighter2_style:
                    style_matchup = f"Both fighters employ a similar {fighter1_style} style, which could lead to an evenly matched contest."
            
            # Generate head-to-head context if available
            h2h_context = ""
            if prediction.get('head_to_head'):
                h2h = prediction.get('head_to_head', {})
                f1_wins = h2h.get('fighter1_wins', 0)
                f2_wins = h2h.get('fighter2_wins', 0)
                
                if f1_wins > 0 or f2_wins > 0:
                    if f1_wins > f2_wins:
                        h2h_context = f"Historically, {fighter1_name} leads the head-to-head matchup {f1_wins}-{f2_wins}."
                    elif f2_wins > f1_wins:
                        h2h_context = f"Historically, {fighter2_name} leads the head-to-head matchup {f2_wins}-{f1_wins}."
                else:
                    h2h_context = f"Their previous matchups are tied at {f1_wins} wins each."
                
                if h2h.get('last_winner') and h2h.get('last_method'):
                    h2h_context += f" Most recently, {h2h.get('last_winner')} won by {h2h.get('last_method')}."
            
            # Assemble the analysis
            analysis_parts = [
                confidence_text + ".",
                ranking_context,
                style_matchup,
                h2h_context,
                f"According to our advanced statistical model (v{prediction.get('model_version', '2.0')}) with an accuracy of {prediction.get('model_accuracy', '80%')}, {winner_name} has key advantages that should lead to victory against {loser_name}."
            ]
            
            # Filter out empty parts and join with spaces
            analysis = " ".join([part for part in analysis_parts if part])
            
            return analysis
        except Exception as e:
            self.logger.error(f"Error generating fight analysis: {str(e)}")
            return f"Our model predicts {prediction.get('winner', 'Fighter 1')} to win this matchup based on statistical analysis of both fighters' performance data."

    def get_available_fighters(self, search_term=None, limit=50):
        """
        Get a list of fighters available in the database
        Optionally filter by search term
        """
        try:
            conn = get_db_connection()
            if not conn:
                logger.error("No database connection available")
                return []
                
            cursor = conn.cursor()
            
            if search_term:
                # Search with pattern matching
                query = "SELECT fighter_name, Record, Weight, ranking, is_champion FROM fighters WHERE fighter_name LIKE ? ORDER BY ranking, fighter_name LIMIT ?"
                cursor.execute(query, (f'%{search_term}%', limit))
            else:
                # Get all fighters up to limit
                query = "SELECT fighter_name, Record, Weight, ranking, is_champion FROM fighters ORDER BY ranking, fighter_name LIMIT ?"
                cursor.execute(query, (limit,))
                
            fighters = cursor.fetchall()
            
            # Convert to list of dictionaries
            columns = [col[0] for col in cursor.description]
            result = [dict(zip(columns, fighter)) for fighter in fighters]
            
            cursor.close()
            return result
            
        except Exception as e:
            logger.error(f"Error getting available fighters: {str(e)}")
            return [] 