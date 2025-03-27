"""
UFC Fight Predictor

A machine learning system that predicts who's going to win in a UFC fight
Uses fighter stats, recent performance, and physical attributes to make smart predictions.
"""

import os
import json
import pickle
import logging
import math
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
import random
import joblib
import traceback

from backend.api.database import get_db_connection
from backend.ml.config import get_config
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
from backend.constants import (
    MODEL_PATH,
    SCALER_PATH,
    FEATURE_NAMES_PATH,
    MODEL_INFO_PATH,
    MODEL_VERSION,
    LOG_LEVEL,
    LOG_FORMAT,
    LOG_DATE_FORMAT,
    DB_PATH,
    IMPORTANT_FEATURES,
    DEFAULT_CONFIDENCE
)
from backend.ml.model_loader import get_loaded_model, get_loaded_scaler, get_loaded_features

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format=LOG_FORMAT,
    datefmt=LOG_DATE_FORMAT
)
logger = logging.getLogger(__name__)

def parse_record(record_str):
    """
    Takes a fighter's record (like '10-2-1') and breaks it down into wins, losses, and draws.
    """
    try:
        parts = record_str.split('-')
        if len(parts) == 3:
            wins = int(parts[0])
            losses = int(parts[1])
            draws = int(parts[2])
            return wins, losses, draws
        return 0, 0, 0
    except Exception:
        return 0, 0, 0

class FighterPredictor:
    """
    The brain of our UFC prediction system! 🧠

    What it does:
    - Analyzes fighter stats and styles
    - Tracks how well fighters have been doing lately
    - Compares physical attributes (height, reach, etc.)
    - Looks at head-to-head history
    - Calculates who's likely to win
    """
    
    def __init__(self, train=False):
        """
        Initialize predictor with optional training mode.
        """
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.model_info = {
            'version': MODEL_VERSION,
            'accuracy': 0.0,
            'status': 'Initializing',
            'last_trained': None,
            'training_size': 0
        }
        
        if not train:
            self._load_model()
        
        self.logger.debug("Predictor initialized")

    def _load_model(self) -> bool:
        """
        Loads up our trained model so it's ready to make predictions! 🎯
        
        Returns:
            bool: True if everything loaded okay, False if something went wrong
        """
        if not os.path.exists(MODEL_PATH):
            self.logger.debug("No model file found")
            return False
            
        try:
            # Try to safely load the model while handling version incompatibilities
            try:
                model_data = joblib.load(MODEL_PATH)
            except Exception as e:
                self.logger.warning(f"Could not load model directly: {str(e)}")
                self.logger.info("Forcing model retraining with current scikit-learn version")
                return False  # Return false to trigger retraining
            
            if isinstance(model_data, dict):
                self.model = model_data.get('model')
                self.scaler = model_data.get('scaler')
                self.feature_names = model_data.get('feature_names')
                
                if 'model_info' in model_data:
                    self.model_info = model_data['model_info']
            else:
                self.model = model_data
                
                if os.path.exists(SCALER_PATH):
                    try:
                        self.scaler = joblib.load(SCALER_PATH)
                    except Exception:
                        with open(SCALER_PATH, 'rb') as f:
                            self.scaler = pickle.load(f)
                else:
                    self.logger.debug("Creating default scaler")
                    self.scaler = StandardScaler()
                    dummy_data = np.array([[0] * 28])
                    self.scaler.fit(dummy_data)
                    
                if os.path.exists(FEATURE_NAMES_PATH):
                    try:
                        self.feature_names = joblib.load(FEATURE_NAMES_PATH)
                    except Exception:
                        with open(FEATURE_NAMES_PATH, 'rb') as f:
                            self.feature_names = pickle.load(f)
                else:
                    self.logger.debug("Creating default feature names")
                    self.feature_names = [
                        'slpm', 'str_acc', 'sapm', 'str_def', 'td_avg', 'td_acc', 'td_def', 'sub_avg',
                        'reach', 'height', 'weight_class_encoded', 'wins', 'losses', 'draws', 'total_fights', 
                        'win_percentage', 'is_striker', 'is_grappler', 'age', 'striking_differential',
                        'takedown_differential', 'combat_effectiveness', 'stance_encoded', 
                        'recent_win_streak', 'recent_loss_streak', 'finish_rate', 'decision_rate', 'experience_factor'
                    ]
            
            if os.path.exists(MODEL_INFO_PATH):
                try:
                    with open(MODEL_INFO_PATH, 'r') as f:
                        self.model_info = json.load(f)
                except:
                    self.logger.debug("Failed to load model info")
            
            if self.model is None:
                self.logger.error("Model loading failed")
                return False
            
            # Check if model is compatible with current scikit-learn version
            try:
                # Try to make a small prediction to check compatibility
                dummy_data = np.array([[0] * len(self.feature_names)])
                scaled_data = self.scaler.transform(dummy_data)
                _ = self.model.predict_proba(scaled_data)
                self.logger.info("Model compatibility check passed")
            except Exception as e:
                self.logger.error(f"Model compatibility check failed: {str(e)}")
                return False
            
            if self.scaler is None:
                self.logger.debug("Creating default scaler")
                self.scaler = StandardScaler()
                dummy_data = np.array([[0] * 28])
                self.scaler.fit(dummy_data)
            
            if not self.feature_names:
                self.logger.debug("Creating default feature names")
                self.feature_names = [
                    'slpm', 'str_acc', 'sapm', 'str_def', 'td_avg', 'td_acc', 'td_def', 'sub_avg',
                    'reach', 'height', 'weight_class_encoded', 'wins', 'losses', 'draws', 'total_fights', 
                    'win_percentage', 'is_striker', 'is_grappler', 'age', 'striking_differential',
                    'takedown_differential', 'combat_effectiveness', 'stance_encoded', 
                    'recent_win_streak', 'recent_loss_streak', 'finish_rate', 'decision_rate', 'experience_factor'
                ]
            
            self.model_info['status'] = 'Loaded'
            self.logger.debug("Model loaded successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Model loading failed: {str(e)}")
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
            
            # Ensure we have valid feature names
            if not self.feature_names or len(self.feature_names) == 0:
                self.logger.warning("No feature names available, creating dummy feature names")
                self.feature_names = [f"feature_{i}" for i in range(100)]  # Create dummy feature names
                
            # Ensure we have a valid scaler
            if not self.scaler:
                self.logger.warning("No scaler available, creating a default scaler")
                self.scaler = StandardScaler()
            
            # Create model package with all necessary components
            model_package = {
                'model': self.model,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'model_info': self.model_info
            }
            
            # Save to disk using joblib for better compatibility
            joblib.dump(model_package, MODEL_PATH)
                
            self.logger.info(f"Model saved to {MODEL_PATH}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
    
    def _get_db_connection(self):
        """
        Get a connection to the Supabase database.
        
        Returns:
            Optional: Supabase client if successful, None otherwise
        """
        try:
            return get_db_connection()
        except Exception as e:
            self.logger.error(f"Database connection error: {str(e)}")
            return None
    
    def _get_fighter_data(self, fighter_name: str) -> Optional[Dict[str, Any]]:
        """
        Looks up a fighter in our database to get all their info! 🔍
        
        Args:
            fighter_name: Name of the fighter we're looking for
            
        Returns:
            All the fighter's data if we find them, None if we don't
            
        Raises:
            ValueError: If the fighter name is empty or invalid
        """
        if not fighter_name or not isinstance(fighter_name, str):
            self.logger.error("Invalid fighter name provided")
            raise ValueError("Fighter name must be a non-empty string")
        
        try:
            supabase = self._get_db_connection()
            if not supabase:
                self.logger.error("No database connection available")
                return None
            
            try:
                # Get fighter data from Supabase using case-insensitive search
                response = supabase.table('fighters').select('*').ilike('fighter_name', fighter_name).execute()
                
                if not response.data or len(response.data) == 0:
                    # Try again with another approach - exact match case insensitive
                    response = supabase.table('fighters').select('*').eq('fighter_name', fighter_name).execute()
                    
                    if not response.data or len(response.data) == 0:
                        # Try one more approach with pattern matching
                        response = supabase.table('fighters').select('*').ilike('fighter_name', f"%{fighter_name}%").execute()
                        
                        if not response.data or len(response.data) == 0:
                            self.logger.warning(f"Fighter not found in database: {fighter_name}")
                            return None
                
                fighter_data = response.data[0]
                
                # Get fighter's recent fights from Supabase
                fights_response = supabase.table('fighter_last_5_fights').select('*')\
                    .eq('fighter_name', fighter_data.get('fighter_name', fighter_name))\
                    .order('fight_date', desc=True)\
                    .limit(10)\
                    .execute()
                
                if fights_response.data and len(fights_response.data) > 0:
                    fighter_data['recent_fights'] = fights_response.data
                else:
                    fighter_data['recent_fights'] = []
                
                return fighter_data
                
            except Exception as e:
                self.logger.error(f"Database error retrieving fighter data: {str(e)}")
                import traceback
                self.logger.error(traceback.format_exc())
                raise
                
        except Exception as e:
            self.logger.error(f"Error retrieving fighter data for {fighter_name}: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
    
    def _get_fighter_record(self, fighter_name):
        """
        Gets a fighter's win-loss-draw record and recent fight results! 📝
        
        This tells us how many fights they've won, lost, and drawn,
        plus how they did in their last few fights.
        """
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
            self.logger.error(f"Error getting fighter record for {fighter_name}: {str(e)}")
            return None
    
    def _get_fighter_image(self, fighter_name):
        """Get fighter image URL from the database"""
        try:
            fighter_data = self._get_fighter_data(fighter_name)
            if not fighter_data:
                return None
                
            return fighter_data.get('image_url')
        except Exception as e:
            self.logger.error(f"Error getting fighter image for {fighter_name}: {str(e)}")
            return None 

    def _extract_features_from_fighter(self, fighter_name):
        """
        Gets all the important stats and numbers we need from a fighter! 📊
        
        This looks at things like:
        - How many strikes they land per minute
        - How good they are at takedowns
        - Their win/loss record
        - Physical stuff like height and reach
        - And lots more!
        
        Args:
            fighter_name (str): The name of the fighter to analyze
            
        Returns:
            dict: All their stats, ready for our model to use
        """
        try:
            # Get fighter data
            fighter_data = self._get_fighter_data(fighter_name)
            if not fighter_data:
                self.logger.warning(f"No data found for fighter: {fighter_name}")
                return None
                
            # Initialize feature vector with exactly the 28 expected features
            expected_features = [
                'slpm', 'str_acc', 'sapm', 'str_def', 'td_avg', 'td_acc', 'td_def', 'sub_avg',
                'reach', 'height', 'weight_class_encoded', 'wins', 'losses', 'draws', 'total_fights', 
                'win_percentage', 'is_striker', 'is_grappler', 'age', 'striking_differential',
                'takedown_differential', 'combat_effectiveness', 'stance_encoded', 
                'recent_win_streak', 'recent_loss_streak', 'finish_rate', 'decision_rate', 'experience_factor'
            ]
            
            # Create empty features dictionary with the expected keys
            features = {key: 0.0 for key in expected_features}
            
            # Extract basic stats using feature engineering functions
            features['slpm'] = safe_convert_to_float(fighter_data.get('SLpM', 0))
            features['str_acc'] = safe_convert_to_float(fighter_data.get('Str. Acc.', 0))
            features['sapm'] = safe_convert_to_float(fighter_data.get('SApM', 0))
            features['str_def'] = safe_convert_to_float(fighter_data.get('Str. Def', 0))
            features['td_avg'] = safe_convert_to_float(fighter_data.get('TD Avg.', 0))
            features['td_acc'] = safe_convert_to_float(fighter_data.get('TD Acc.', 0))
            features['td_def'] = safe_convert_to_float(fighter_data.get('TD Def.', 0))
            features['sub_avg'] = safe_convert_to_float(fighter_data.get('Sub. Avg.', 0))
            
            # Extract physical attributes
            features['reach'] = extract_reach_in_inches(fighter_data.get('Reach', '0"'))
            features['height'] = extract_height_in_inches(fighter_data.get('Height', "0' 0\""))
            
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
            
            # Extract record stats - fixing the issue where extract_record_stats returns a tuple
            record_str = fighter_data.get('Record', '0-0-0')
            # Use import from feature_engineering instead of our own implementation
            record_stats_dict = {}
            wins, losses, draws = parse_record(record_str)
            record_stats_dict['wins'] = wins
            record_stats_dict['losses'] = losses
            record_stats_dict['draws'] = draws
            record_stats_dict['total_fights'] = wins + losses + draws
            
            features['wins'] = record_stats_dict['wins']
            features['losses'] = record_stats_dict['losses']
            features['draws'] = record_stats_dict['draws']
            features['total_fights'] = record_stats_dict['total_fights']
            features['win_percentage'] = calculate_win_percentage(record_stats_dict)
            
            # Extract fighter style based on stats
            features['is_striker'] = 1 if features['slpm'] > 3.0 and features['str_acc'] > 0.4 else 0
            features['is_grappler'] = 1 if features['td_avg'] > 2.0 or features['sub_avg'] > 0.5 else 0
            
            # Calculate age
            try:
                dob = fighter_data.get('DOB', 'N/A')
                if dob != 'N/A' and dob:
                    try:
                        birth_date = datetime.strptime(dob, '%b %d, %Y')
                    except ValueError:
                        birth_date = datetime.strptime(dob, '%B %d, %Y')
                    today = datetime.today()
                    features['age'] = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
                else:
                    features['age'] = 30  # default age
            except Exception as e:
                self.logger.warning(f"Error calculating age for {fighter_name}: {str(e)}")
                features['age'] = 30  # default age
                
            # Calculate derived metrics
            features['striking_differential'] = features['slpm'] - features['sapm']
            features['takedown_differential'] = features['td_avg'] * features['td_acc'] - features['td_avg'] * (1 - features['td_def'])
            features['combat_effectiveness'] = (features['slpm'] * features['str_acc']) + (features['td_avg'] * features['td_acc']) + features['sub_avg'] - features['sapm'] * (1 - features['str_def'])
            
            # Encode stance
            stance = fighter_data.get('STANCE', 'Orthodox').lower()
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
            
            # Calculate finish rates
            if results:
                features['finish_rate'] = sum(1 for fight in recent_fights if 'ko' in fight.get('method', '').lower() or 'sub' in fight.get('method', '').lower() or 'tko' in fight.get('method', '').lower()) / len(results)
                features['decision_rate'] = sum(1 for fight in recent_fights if 'dec' in fight.get('method', '').lower() or 'decision' in fight.get('method', '').lower()) / len(results)
            else:
                features['finish_rate'] = 0
                features['decision_rate'] = 0
            
            # Calculate experience factor
            features['experience_factor'] = features['total_fights'] * features['win_percentage']
            
            # Validate and normalize features to avoid extreme values
            for key in features:
                if features[key] is None:
                    features[key] = 0.0
                elif isinstance(features[key], (int, float)):
                    # Cap extreme values
                    if features[key] > 100:
                        features[key] = 100.0
                    elif features[key] < -100:
                        features[key] = -100.0
            
            # Ensure we're only returning exactly the 28 features
            # that the model expects in the correct order
            result_features = {}
            for key in expected_features:
                result_features[key] = features.get(key, 0.0)
                
            # Log results
            self.logger.info(f"Successfully extracted {len(result_features)} features for {fighter_name}")
            return result_features
            
        except Exception as e:
            self.logger.error(f"Error extracting features for {fighter_name}: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
    
    def _get_training_data(self):
        """
        Collect all the training data we need to teach our model! 📚
        
        This gathers historical fight results and fighter stats,
        then transforms them into the format our model needs to learn.
        
        Returns:
            tuple: (X, y) - Features and labels for training
        """
        try:
            # Get database connection
            supabase = self._get_db_connection()
            if not supabase:
                self.logger.error("No database connection available")
                return None, None
                
            # Get all fighters from the database
            try:
                # Use a simpler query to avoid errors
                response = supabase.table("fighters").select("fighter_name").execute()
                if not response.data or len(response.data) == 0:
                    self.logger.warning("No fighters found in the database")
                    return None, None
                    
                fighters = response.data
                self.logger.info(f"Retrieved {len(fighters)} fighters for training")
            except Exception as e:
                self.logger.error(f"Error fetching fighters: {str(e)}")
                return None, None
                
            # Process fighters to extract training data
            X = []
            y = []
            processed_count = 0
            skipped_count = 0
            
            # Process fighters in batches
            batch_size = 50
            total_fighters = len(fighters)
            self.logger.info(f"Processing {total_fighters} fighters")
            
            for batch_index in range(0, total_fighters, batch_size):
                batch_end = min(batch_index + batch_size, total_fighters)
                self.logger.info(f"Processing fighter batch {batch_index // batch_size + 1}/{(total_fighters + batch_size - 1) // batch_size}")
                
                # Process each fighter in batch
                for i in range(batch_index, batch_end):
                    try:
                        fighter = fighters[i]
                        fighter_name = fighter.get('fighter_name')
                        
                        if not fighter_name:
                            skipped_count += 1
                            continue
                            
                        self.logger.debug(f"Processing fighter: {fighter_name}")
                        
                        # Get fighter's last 5 fights
                        fights = self._get_fighter_fights(fighter_name)
                        self.logger.debug(f"Found {len(fights)} fights for {fighter_name}")
                        
                        if not fights or len(fights) < 2:  # Need at least 2 fights for meaningful data
                            skipped_count += 1
                            continue
                        
                        # Process each fight to extract features and labels
                        for fight in fights:
                            result = fight.get('result', '').upper()
                            if not result or 'W' not in result and 'L' not in result:
                                # Skip fights with no clear result
                                continue
                                
                            opponent_name = fight.get('opponent', '')
                            if not opponent_name:
                                continue
                            
                            fighter1_name = fighter_name  # Current fighter
                            fighter2_name = opponent_name  # Opponent
                            
                            # Skip if fighter names are the same (data error)
                            if fighter1_name.lower() == fighter2_name.lower():
                                continue
                            
                            self.logger.debug(f"Extracting features for matchup: {fighter1_name} vs {fighter2_name}")
                            
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
                        self.logger.error(f"Error processing fighter {i}: {str(e)}")
                        import traceback
                        self.logger.error(traceback.format_exc())
                        skipped_count += 1
            
            self.logger.info(f"Processed {processed_count} fights, skipped {skipped_count} fights")
            
            if processed_count == 0:
                self.logger.error("No training examples could be processed")
                return None, None
            
            # Convert to numpy arrays
            try:
                # Make sure all feature vectors have the same length
                if X:
                    max_length = max(len(x) for x in X)
                    self.logger.info(f"Maximum feature vector length: {max_length}")
                    X = [x + [0] * (max_length - len(x)) for x in X]  # Pad with zeros if needed
                
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
    
    def train(self, force=False):
        """
        Trains our model to get better at predicting fights! 🎓
        
        This is where the magic happens - we feed the model lots of past fight data
        so it can learn patterns and make smarter predictions.
        
        Args:
            force (bool): Set to True to retrain even if we already have a trained model
            
        Returns:
            bool: True if training went well, False if something went wrong
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
            
            # Store feature names from the training data
            if not self.feature_names or len(self.feature_names) != X.shape[1]:
                self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
                self.logger.info(f"Setting {len(self.feature_names)} feature names based on training data")
            
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
            success = self._save_model()
            if not success:
                self.logger.error("Failed to save the trained model")
            
            self.logger.info("Model training completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error training model: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            self.model_info['status'] = 'Error'
            self.model_info['message'] = f'Training error: {str(e)}'
            return False
    
    def _repair_scaler(self, feature_count):
        """
        Repair the scaler when feature count mismatches occur.
        This is helpful when the model was trained with a different number of features
        than what's currently being extracted.
        
        Args:
            feature_count (int): The current number of features being extracted
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.logger.warning(f"Repairing scaler to handle {feature_count} features")
            
            # Create a new scaler
            new_scaler = StandardScaler()
            
            # Create dummy data with the correct number of features
            dummy_data = np.zeros((1, feature_count))
            
            # Fit the scaler with the dummy data
            new_scaler.fit(dummy_data)
            
            # Replace the old scaler
            self.scaler = new_scaler
            
            self.logger.info(f"Successfully repaired scaler to handle {feature_count} features")
            return True
        except Exception as e:
            self.logger.error(f"Failed to repair scaler: {str(e)}")
            return False
            
    def predict_winner(self, fighter1_data, fighter2_data, head_to_head=None, common_opponents=None):
        """
        The main event! 🥊 Predicts who's going to win between two fighters.
        
        Args:
            fighter1_data (dict): All the stats and info for the first fighter
            fighter2_data (dict): All the stats and info for the second fighter
            head_to_head (dict, optional): History of fights between these two
            common_opponents (list, optional): Fighters they've both faced
            
        Returns:
            dict: Everything about the prediction - who wins, how likely, etc.
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
            expected_features = self.feature_names if (self.feature_names and len(self.feature_names) == 28) else [
                'slpm', 'str_acc', 'sapm', 'str_def', 'td_avg', 'td_acc', 'td_def', 'sub_avg',
                'reach', 'height', 'weight_class_encoded', 'wins', 'losses', 'draws', 'total_fights', 
                'win_percentage', 'is_striker', 'is_grappler', 'age', 'striking_differential',
                'takedown_differential', 'combat_effectiveness', 'stance_encoded', 
                'recent_win_streak', 'recent_loss_streak', 'finish_rate', 'decision_rate', 'experience_factor'
            ]
            
            # Validate that both feature sets have exactly 28 features
            if len(fighter1_features) != 28 or len(fighter2_features) != 28:
                self.logger.warning(f"Feature count mismatch. Fighter1: {len(fighter1_features)}, Fighter2: {len(fighter2_features)}")
                # Ensure we're using exactly 28 features
                for key in expected_features:
                    if key not in fighter1_features:
                        fighter1_features[key] = 0.0
                    if key not in fighter2_features:
                        fighter2_features[key] = 0.0
                
                # Trim extra features if any
                fighter1_features = {k: fighter1_features[k] for k in expected_features}
                fighter2_features = {k: fighter2_features[k] for k in expected_features}
            
            # SOLUTION FOR POSITION BIAS: Make predictions in both directions and average them
            # First direction: fighter1 vs fighter2
            feature_vector_1vs2 = []
            for key in expected_features:
                feature_vector_1vs2.append(fighter1_features.get(key, 0.0) - fighter2_features.get(key, 0.0))
            
            # Second direction: fighter2 vs fighter1 (reversed)
            feature_vector_2vs1 = []
            for key in expected_features:
                feature_vector_2vs1.append(fighter2_features.get(key, 0.0) - fighter1_features.get(key, 0.0))
            
            # Check for NaN or Inf values in both vectors
            for i, val in enumerate(feature_vector_1vs2):
                if math.isnan(val) or math.isinf(val):
                    feature_vector_1vs2[i] = 0.0
            
            for i, val in enumerate(feature_vector_2vs1):
                if math.isnan(val) or math.isinf(val):
                    feature_vector_2vs1[i] = 0.0
            
            # Verify vectors have exactly 28 elements
            if len(feature_vector_1vs2) != 28 or len(feature_vector_2vs1) != 28:
                self.logger.warning(f"Feature vector length mismatch: {len(feature_vector_1vs2)} vs {len(feature_vector_2vs1)}")
                # Pad or trim to exactly 28 features
                feature_vector_1vs2 = feature_vector_1vs2[:28] if len(feature_vector_1vs2) > 28 else feature_vector_1vs2 + [0.0] * (28 - len(feature_vector_1vs2))
                feature_vector_2vs1 = feature_vector_2vs1[:28] if len(feature_vector_2vs1) > 28 else feature_vector_2vs1 + [0.0] * (28 - len(feature_vector_2vs1))
            
            # Convert to numpy arrays and reshape for predictions
            X_1vs2 = np.array([feature_vector_1vs2])
            X_2vs1 = np.array([feature_vector_2vs1])
            
            # Scale features if scaler exists
            try:
                if self.scaler:
                    X_1vs2_scaled = self.scaler.transform(X_1vs2)
                    X_2vs1_scaled = self.scaler.transform(X_2vs1)
                    self.logger.info("Successfully scaled feature vectors")
                else:
                    self.logger.warning("No scaler available, using unscaled features")
                    X_1vs2_scaled = X_1vs2
                    X_2vs1_scaled = X_2vs1
            except Exception as e:
                self.logger.warning(f"Error scaling features: {str(e)}. Attempting to repair scaler.")
                # Try to repair the scaler to match the feature count
                if self._repair_scaler(len(feature_vector_1vs2)):
                    try:
                        # Try scaling again with the repaired scaler
                        X_1vs2_scaled = self.scaler.transform(X_1vs2)
                        X_2vs1_scaled = self.scaler.transform(X_2vs1)
                        self.logger.info("Successfully scaled features with repaired scaler")
                    except Exception as e2:
                        self.logger.warning(f"Still failed to scale features after repair: {str(e2)}. Using unscaled features.")
                        X_1vs2_scaled = X_1vs2
                        X_2vs1_scaled = X_2vs1
                else:
                    # Use unscaled features if repair failed
                    self.logger.warning("Failed to repair scaler. Using unscaled features.")
                    X_1vs2_scaled = X_1vs2
                    X_2vs1_scaled = X_2vs1
                
            # Make predictions in both directions
            try:
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
                    if total_prob > 0 and not math.isnan(total_prob):
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
                
                self.logger.info(f"Successful prediction. Fighter 1 win probability: {f1_final_prob:.2f}, Fighter 2 win probability: {f2_final_prob:.2f}")
            except Exception as e:
                self.logger.error(f"Error making prediction: {str(e)}")
                # Check if this is a feature count mismatch error and try to handle it
                if "features, but" in str(e) and "expecting" in str(e):
                    # Extract the expected feature count from the error message
                    import re
                    expected_count_match = re.search(r'expecting (\d+) features', str(e))
                    if expected_count_match:
                        expected_count = int(expected_count_match.group(1))
                        self.logger.warning(f"Model expects {expected_count} features, trying to adapt")
                        
                        # Trim or pad our vectors to match the expected count
                        if len(feature_vector_1vs2) > expected_count:
                            feature_vector_1vs2 = feature_vector_1vs2[:expected_count]
                            feature_vector_2vs1 = feature_vector_2vs1[:expected_count]
                        else:
                            feature_vector_1vs2 = feature_vector_1vs2 + [0.0] * (expected_count - len(feature_vector_1vs2))
                            feature_vector_2vs1 = feature_vector_2vs1 + [0.0] * (expected_count - len(feature_vector_2vs1))
                        
                        # Try again with the adjusted feature vectors
                        try:
                            X_1vs2 = np.array([feature_vector_1vs2])
                            X_2vs1 = np.array([feature_vector_2vs1])
                            
                            # Scale if possible
                            if self._repair_scaler(expected_count):
                                X_1vs2_scaled = self.scaler.transform(X_1vs2)
                                X_2vs1_scaled = self.scaler.transform(X_2vs1)
                            else:
                                X_1vs2_scaled = X_1vs2
                                X_2vs1_scaled = X_2vs1
                            
                            # Try prediction again
                            if hasattr(self.model, 'predict_proba'):
                                probas_1vs2 = self.model.predict_proba(X_1vs2_scaled)[0]
                                f1_wins_prob = probas_1vs2[1]
                                f2_wins_prob = probas_1vs2[0]
                                
                                probas_2vs1 = self.model.predict_proba(X_2vs1_scaled)[0]
                                f2_wins_prob_alt = probas_2vs1[1]
                                f1_wins_prob_alt = probas_2vs1[0]
                                
                                f1_final_prob = (f1_wins_prob + f1_wins_prob_alt) / 2
                                f2_final_prob = (f2_wins_prob + f2_wins_prob_alt) / 2
                                
                                total_prob = f1_final_prob + f2_final_prob
                                if total_prob > 0 and not math.isnan(total_prob):
                                    f1_final_prob = f1_final_prob / total_prob
                                    f2_final_prob = f2_final_prob / total_prob
                                else:
                                    f1_final_prob = 0.5
                                    f2_final_prob = 0.5
                                    
                                self.logger.info(f"Successfully recovered prediction with adjusted feature count. Fighter 1: {f1_final_prob:.2f}, Fighter 2: {f2_final_prob:.2f}")
                            else:
                                # Fallback to win percentage
                                self.logger.warning("Recovery attempt failed, falling back to win percentage")
                                f1_win_pct = fighter1_features.get('win_percentage', 0.5)
                                f2_win_pct = fighter2_features.get('win_percentage', 0.5)
                                
                                total = f1_win_pct + f2_win_pct
                                if total > 0:
                                    f1_final_prob = f1_win_pct / total
                                    f2_final_prob = f2_win_pct / total
                                else:
                                    f1_final_prob = 0.5
                                    f2_final_prob = 0.5
                        except Exception as recovery_error:
                            self.logger.error(f"Recovery attempt failed: {str(recovery_error)}")
                            # Fallback to win percentage
                            f1_win_pct = fighter1_features.get('win_percentage', 0.5)
                            f2_win_pct = fighter2_features.get('win_percentage', 0.5)
                            
                            total = f1_win_pct + f2_win_pct
                            if total > 0:
                                f1_final_prob = f1_win_pct / total
                                f2_final_prob = f2_win_pct / total
                            else:
                                f1_final_prob = 0.5
                                f2_final_prob = 0.5
                    else:
                        # Fallback to win percentage if we can't determine the expected count
                        f1_win_pct = fighter1_features.get('win_percentage', 0.5)
                        f2_win_pct = fighter2_features.get('win_percentage', 0.5)
                        
                        total = f1_win_pct + f2_win_pct
                        if total > 0:
                            f1_final_prob = f1_win_pct / total
                            f2_final_prob = f2_win_pct / total
                        else:
                            f1_final_prob = 0.5
                            f2_final_prob = 0.5
                else:
                    # Other errors, use win percentage as fallback
                    f1_win_pct = fighter1_features.get('win_percentage', 0.5)
                    f2_win_pct = fighter2_features.get('win_percentage', 0.5)
                    
                    total = f1_win_pct + f2_win_pct
                    if total > 0:
                        f1_final_prob = f1_win_pct / total
                        f2_final_prob = f2_win_pct / total
                    else:
                        f1_final_prob = 0.5
                        f2_final_prob = 0.5
                    
                self.logger.warning(f"Using fallback prediction method. Fighter 1: {f1_final_prob:.2f}, Fighter 2: {f2_final_prob:.2f}")
            
            # Check for NaN values and replace with defaults
            if math.isnan(f1_final_prob) or math.isinf(f1_final_prob):
                f1_final_prob = 0.5
            if math.isnan(f2_final_prob) or math.isinf(f2_final_prob):
                f2_final_prob = 0.5
                
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
        Makes our prediction look nice and clean for sending back to users! ✨
        
        Takes all our prediction data and formats it in a way that's easy to read
        and use in the app. Includes win probabilities, fighter info, and analysis.
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
        
        # Ensure all values exist and have fallbacks
        fighter1_prob = prediction.get('winner_probability' if fighter1_is_winner else 'loser_probability', 0.5)
        fighter2_prob = prediction.get('winner_probability' if not fighter1_is_winner else 'loser_probability', 0.5)
        
        # Format percentages safely - ensure they're never NaN
        try:
            if isinstance(fighter1_prob, (int, float)) and not math.isnan(fighter1_prob):
                fighter1_prob_pct = f"{int(round(fighter1_prob * 100))}%"
            else:
                fighter1_prob_pct = "50%"
        except:
            fighter1_prob_pct = "50%"
            
        try:
            if isinstance(fighter2_prob, (int, float)) and not math.isnan(fighter2_prob):
                fighter2_prob_pct = f"{int(round(fighter2_prob * 100))}%"
            else:
                fighter2_prob_pct = "50%"
        except:
            fighter2_prob_pct = "50%"
            
        try:
            winner_prob = prediction.get('winner_probability', 0.5)
            if isinstance(winner_prob, (int, float)) and not math.isnan(winner_prob):
                winner_prob_pct = f"{int(round(winner_prob * 100))}%"
            else:
                winner_prob_pct = "50%"
        except:
            winner_prob_pct = "50%"
            
        # Format head-to-head data safely
        h2h = prediction.get('head_to_head', {})
        if not h2h:
            h2h = {}
        
        result = {
            'success': True,
            'fighter1': {
                'name': fighter1_name,
                'win_probability': fighter1_prob_pct,
                'record': prediction.get('fighter1', {}).get('record', 'N/A'),
                'image_url': prediction.get('fighter1', {}).get('image_url', '')
            },
            'fighter2': {
                'name': fighter2_name,
                'win_probability': fighter2_prob_pct,
                'record': prediction.get('fighter2', {}).get('record', 'N/A'),
                'image_url': prediction.get('fighter2', {}).get('image_url', '')
            },
            'winner': winner_name,
            'loser': loser_name,
            'winner_probability': winner_prob_pct,
            'prediction_confidence': prediction.get('prediction_confidence', 0.5),
            'model': {
                'version': prediction.get('model_version', '2.0'),
                'accuracy': prediction.get('model_accuracy', 'N/A'),
                'status': 'Trained'
            },
            'analysis': self._generate_fight_analysis(prediction) if hasattr(self, '_generate_fight_analysis') else None,
            'head_to_head': {
                'fighter1_wins': h2h.get('fighter1_wins', 0),
                'fighter2_wins': h2h.get('fighter2_wins', 0),
                'last_winner': h2h.get('last_winner'),
                'last_method': h2h.get('last_method', ''),
                'last_round': h2h.get('last_round', 'N/A'),
                'last_time': h2h.get('last_time', 'N/A')
            }
        }
        
        return result
        
    def _generate_fight_analysis(self, prediction):
        """
        Creates a cool breakdown of why we think a fighter will win! 🎯
        
        Takes all our prediction data and turns it into an easy-to-read explanation
        of who we think will win and why. Includes confidence levels and any
        head-to-head history between the fighters.
        """
        try:
            # Use our own implementation that's proven to work
            fighter1_name = prediction.get('fighter1', {}).get('name')
            fighter2_name = prediction.get('fighter2', {}).get('name')
            winner_name = prediction.get('winner')
            loser_name = prediction.get('loser')
            win_probability = prediction.get('winner_probability', 0.5)
            
            # Format as percentage for display
            if isinstance(win_probability, str) and '%' in win_probability:
                win_pct = win_probability
            else:
                try:
                    win_pct = f"{int(float(win_probability) * 100)}%" if isinstance(win_probability, (int, float)) else "50%"
                except:
                    win_pct = "50%"
            
            # Generate confidence text
            confidence_level = ""
            if isinstance(win_probability, (int, float)):
                if win_probability > 0.7:
                    confidence_level = "strongly"
                elif win_probability > 0.6:
                    confidence_level = "confidently"
                elif win_probability > 0.55:
                    confidence_level = "likely"
                else:
                    confidence_level = "narrowly"
            else:
                confidence_level = "likely"
                
            # Get head-to-head info
            h2h = prediction.get('head_to_head', {})
            h2h_text = ""
            f1_wins = h2h.get('fighter1_wins', 0)
            f2_wins = h2h.get('fighter2_wins', 0)
            
            if f1_wins > 0 or f2_wins > 0:
                if f1_wins > f2_wins:
                    h2h_text = f" In their previous encounters, {fighter1_name} has won {f1_wins} times against {fighter2_name}'s {f2_wins}."
                elif f2_wins > f1_wins:
                    h2h_text = f" In their previous encounters, {fighter2_name} has won {f2_wins} times against {fighter1_name}'s {f1_wins}."
                else:
                    h2h_text = f" Their previous matchups are tied at {f1_wins} wins each."
                    
                if h2h.get('last_winner') and h2h.get('last_method'):
                    h2h_text += f" Most recently, {h2h.get('last_winner')} won by {h2h.get('last_method')}."
            
            # Add model info
            model_info = f" (Model v{prediction.get('model_version', '1.0')})"
            
            # Assemble the final analysis
            analysis = f"Our model {confidence_level} predicts {winner_name} to defeat {loser_name} with {win_pct} probability based on their fighting statistics and performance data.{h2h_text}{model_info}"
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error generating fight analysis: {str(e)}")
            return f"Our model predicts {prediction.get('winner', 'Fighter 1')} to win this matchup based on statistical analysis of both fighters' performance data."

    def get_available_fighters(self, search_term=None, limit=50):
        """
        Gets a list of all the fighters in our database! 📋
        
        Args:
            search_term: If you want to search for specific fighters
            limit: How many fighters to return (max 50 to keep things fast)
            
        Returns:
            A list of fighters with their basic info like record and weight class
        """
        try:
            supabase = self._get_db_connection()
            if not supabase:
                logger.error("No database connection available")
                return []
            
            query = supabase.table('fighters').select('fighter_name,Record,Weight,ranking,is_champion,id')
            
            if search_term:
                # Search with pattern matching (ilike for case-insensitive search)
                query = query.ilike('fighter_name', f'%{search_term}%')
            
            # Apply ordering and limit
            # Note: Sort by ranking numerically, with nulls last
            try:
                # First attempt with proper order syntax 
                response = query.order('ranking', desc=False, nulls_last=True)\
                               .order('fighter_name')\
                               .limit(limit)\
                               .execute()
            except Exception:
                # Fallback if the advanced sorting isn't supported
                logger.warning("Advanced sorting failed, using simpler sorting")
                response = query.order('ranking')\
                               .order('fighter_name')\
                               .limit(limit)\
                               .execute()
                
            if not response.data:
                return []
            
            # Data is already in dictionary format from Supabase
            return response.data
            
        except Exception as e:
            logger.error(f"Error getting available fighters: {str(e)}")
            return []

    def _get_fighter_fights(self, fighter_name: str) -> List[Dict[str, Any]]:
        """Get all fights for a fighter from the database."""
        try:
            supabase = self._get_db_connection()
            if not supabase:
                self.logger.error("No database connection available")
                return []
                
            response = supabase.table("fighter_last_5_fights") \
                .select("*") \
                .eq("fighter_name", fighter_name) \
                .order("id", desc=False) \
                .execute()
                
            return response.data if response.data else []
        except Exception as e:
            self.logger.error(f"Error getting fights for {fighter_name}: {str(e)}")
            self.logger.error(traceback.format_exc())
            return [] 