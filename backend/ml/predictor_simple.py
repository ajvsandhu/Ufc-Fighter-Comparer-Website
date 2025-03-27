import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
import random
import traceback

from backend.ml.model_loader import get_loaded_model, get_loaded_scaler, get_loaded_features
from backend.constants import IMPORTANT_FEATURES, DEFAULT_CONFIDENCE

# Configure logging
logger = logging.getLogger(__name__)

def predict_winner(fighter1_data: Dict[str, Any], fighter2_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Predict the winner between two fighters using the loaded ML model.
    If model is not available, falls back to basic stats comparison.
    """
    try:
        # Get names for output
        fighter1_name = fighter1_data.get('fighter_name', 'Fighter 1')
        fighter2_name = fighter2_data.get('fighter_name', 'Fighter 2')
        
        # Get loaded ML components
        model = get_loaded_model()
        scaler = get_loaded_scaler()
        features = get_loaded_features()
        
        if model and scaler and features:
            # Use ML model for prediction
            return predict_with_model(fighter1_data, fighter2_data, model, scaler, features)
        else:
            # Fall back to basic stats comparison
            logger.warning("ML model components not available, using basic comparison")
            return basic_stats_comparison(fighter1_data, fighter2_data)
    
    except Exception as e:
        logger.error(f"Error in predict_winner: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Return fallback result
        return {
            "winner_name": fighter1_data.get('fighter_name', 'Fighter 1'),
            "winner_idx": 0,
            "confidence": 0.5,
            "probability": 0.5,
            "explanation": f"Error making prediction: {str(e)}"
        }

def predict_with_model(
    fighter1_data: Dict[str, Any], 
    fighter2_data: Dict[str, Any], 
    model: Any, 
    scaler: Any, 
    features: List[str]
) -> Dict[str, Any]:
    """
    Make prediction using the ML model.
    """
    try:
        # Create feature arrays for both fighters
        f1_features = extract_features(fighter1_data, features)
        f2_features = extract_features(fighter2_data, features)
        
        # Create feature differences (fighter1 - fighter2)
        feature_diffs = f1_features - f2_features
        
        # Scale features
        scaled_diffs = scaler.transform(feature_diffs.reshape(1, -1))
        
        # Make prediction
        prob = model.predict_proba(scaled_diffs)[0]
        pred_class = int(prob[1] > 0.5)  # 1 if fighter1 wins, 0 if fighter2 wins
        
        # Determine winner
        if pred_class == 1:
            winner_name = fighter1_data.get('fighter_name', 'Fighter 1')
            winner_idx = 0
            probability = prob[1]
        else:
            winner_name = fighter2_data.get('fighter_name', 'Fighter 2')
            winner_idx = 1
            probability = prob[0]
        
        # Calculate confidence (how far from 0.5 is the probability)
        confidence = abs(probability - 0.5) * 2  # Scale to 0-1
        
        # Generate explanation
        explanation = generate_explanation(
            fighter1_data, fighter2_data, 
            winner_name, confidence, 
            model, features, feature_diffs
        )
        
        # Generate matchup analysis
        matchup_analysis = analyze_matchup(fighter1_data, fighter2_data)
        
        # Get important factors
        important_factors = get_important_factors(fighter1_data, fighter2_data, model, features, feature_diffs)
        
        return {
            "winner_name": winner_name,
            "winner_idx": winner_idx,
            "confidence": float(confidence),
            "probability": float(probability),
            "explanation": explanation,
            "matchup_analysis": matchup_analysis,
            "important_factors": important_factors
        }
    
    except Exception as e:
        logger.error(f"Error in predict_with_model: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Fall back to basic comparison
        return basic_stats_comparison(fighter1_data, fighter2_data)

def basic_stats_comparison(fighter1_data: Dict[str, Any], fighter2_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Simple stats-based comparison when ML model is not available.
    """
    try:
        # Get names
        fighter1_name = fighter1_data.get('fighter_name', 'Fighter 1')
        fighter2_name = fighter2_data.get('fighter_name', 'Fighter 2')
        
        # Compare win percentages
        f1_wins = float(fighter1_data.get('Win', 0) or 0)
        f1_losses = float(fighter1_data.get('Loss', 0) or 0)
        f1_total = f1_wins + f1_losses
        f1_win_pct = f1_wins / f1_total if f1_total > 0 else 0.5
        
        f2_wins = float(fighter2_data.get('Win', 0) or 0)
        f2_losses = float(fighter2_data.get('Loss', 0) or 0)
        f2_total = f2_wins + f2_losses
        f2_win_pct = f2_wins / f2_total if f2_total > 0 else 0.5
        
        # Compare key stats
        f1_points = 0
        f2_points = 0
        
        for stat in ['SLPM', 'StrAcc', 'TD', 'TDA', 'SUB']:
            f1_val = float(fighter1_data.get(stat, 0) or 0)
            f2_val = float(fighter2_data.get(stat, 0) or 0)
            
            if f1_val > f2_val:
                f1_points += 1
            elif f2_val > f1_val:
                f2_points += 1
        
        # Combine win percentage and stats
        f1_score = (f1_win_pct * 0.7) + (f1_points / 10)
        f2_score = (f2_win_pct * 0.7) + (f2_points / 10)
        
        # Determine winner
        if f1_score > f2_score:
            winner_name = fighter1_name
            winner_idx = 0
            probability = 0.5 + (f1_score - f2_score) / 2
        else:
            winner_name = fighter2_name
            winner_idx = 1
            probability = 0.5 + (f2_score - f1_score) / 2
        
        # Ensure probability is in range [0.5, 0.95]
        probability = min(0.95, max(0.5, probability))
        
        # Calculate confidence
        confidence = (probability - 0.5) * 2
        
        # Simple explanation
        if winner_name == fighter1_name:
            explanation = f"{fighter1_name} is favored over {fighter2_name} based on win percentage and key stats."
        else:
            explanation = f"{fighter2_name} is favored over {fighter1_name} based on win percentage and key stats."
        
        return {
            "winner_name": winner_name,
            "winner_idx": winner_idx,
            "confidence": float(confidence),
            "probability": float(probability),
            "explanation": explanation,
            "matchup_analysis": {},
            "important_factors": []
        }
    
    except Exception as e:
        logger.error(f"Error in basic_stats_comparison: {str(e)}")
        
        # Return a random result as absolute fallback
        fighters = [fighter1_data.get('fighter_name', 'Fighter 1'), 
                    fighter2_data.get('fighter_name', 'Fighter 2')]
        winner_idx = random.randint(0, 1)
        
        return {
            "winner_name": fighters[winner_idx],
            "winner_idx": winner_idx,
            "confidence": DEFAULT_CONFIDENCE,
            "probability": 0.5 + (DEFAULT_CONFIDENCE / 2),
            "explanation": "Unable to make prediction due to an error. Random result provided.",
            "matchup_analysis": {},
            "important_factors": []
        }

def extract_features(fighter_data: Dict[str, Any], feature_names: List[str]) -> np.ndarray:
    """
    Extract feature values from fighter data in the correct order.
    """
    features = []
    
    for feature in feature_names:
        try:
            # Try to get the feature value, convert to float
            value = fighter_data.get(feature, 0)
            if value is None:
                value = 0
            features.append(float(value))
        except (ValueError, TypeError):
            # If conversion fails, use 0
            features.append(0.0)
    
    return np.array(features)

def generate_explanation(
    fighter1_data: Dict[str, Any], 
    fighter2_data: Dict[str, Any],
    winner_name: str, 
    confidence: float,
    model: Any, 
    features: List[str], 
    feature_diffs: np.ndarray
) -> str:
    """
    Generate a human-readable explanation for the prediction.
    """
    fighter1_name = fighter1_data.get('fighter_name', 'Fighter 1')
    fighter2_name = fighter2_data.get('fighter_name', 'Fighter 2')
    
    # Confidence levels in text
    if confidence < 0.2:
        confidence_text = "very close"
    elif confidence < 0.4:
        confidence_text = "close"
    elif confidence < 0.6:
        confidence_text = "favored"
    elif confidence < 0.8:
        confidence_text = "strongly favored"
    else:
        confidence_text = "heavily favored"
    
    # Base explanation
    explanation = f"{winner_name} is {confidence_text} to win against "
    explanation += fighter2_name if winner_name == fighter1_name else fighter1_name
    
    # Add key advantages if model has feature importances
    if hasattr(model, "feature_importances_"):
        # Get top features for this prediction
        feature_importance = model.feature_importances_
        
        # Combine feature names, importances, and differences
        feature_info = []
        for i, (name, imp) in enumerate(zip(features, feature_importance)):
            diff = feature_diffs[i]
            # Only consider significant features with non-zero differences
            if imp > 0.02 and abs(diff) > 0.1:
                feature_info.append((name, imp, diff))
        
        # Sort by absolute difference * importance
        feature_info.sort(key=lambda x: abs(x[1] * x[2]), reverse=True)
        
        # Take top 3 features
        top_features = feature_info[:3]
        
        if top_features:
            explanation += ". Key advantages: "
            
            feature_texts = []
            for name, _, diff in top_features:
                # Positive diff means advantage for fighter1
                if diff > 0:
                    if winner_name == fighter1_name:
                        feature_texts.append(f"better {name}")
                    else:
                        feature_texts.append(f"despite worse {name}")
                else:
                    if winner_name == fighter2_name:
                        feature_texts.append(f"better {name}")
                    else:
                        feature_texts.append(f"despite worse {name}")
            
            explanation += ", ".join(feature_texts)
    
    return explanation

def analyze_matchup(fighter1_data: Dict[str, Any], fighter2_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a detailed matchup analysis between the two fighters.
    """
    try:
        # Get basic info
        fighter1_name = fighter1_data.get('fighter_name', 'Fighter 1')
        fighter2_name = fighter2_data.get('fighter_name', 'Fighter 2')
        
        analysis = {
            "striker_advantage": "",
            "grappling_advantage": "",
            "experience_advantage": "",
            "overall_assessment": ""
        }
        
        # Striking comparison
        f1_striking_score = 0
        f2_striking_score = 0
        
        for stat in ['SLPM', 'StrAcc', 'StrDef', 'SApM']:
            f1_val = float(fighter1_data.get(stat, 0) or 0)
            f2_val = float(fighter2_data.get(stat, 0) or 0)
            
            if f1_val > f2_val * 1.1:  # 10% better
                f1_striking_score += 1
            elif f2_val > f1_val * 1.1:
                f2_striking_score += 1
        
        if f1_striking_score > f2_striking_score:
            analysis["striker_advantage"] = f"{fighter1_name} has the striking advantage"
        elif f2_striking_score > f1_striking_score:
            analysis["striker_advantage"] = f"{fighter2_name} has the striking advantage"
        else:
            analysis["striker_advantage"] = "Striking appears evenly matched"
        
        # Grappling comparison
        f1_grappling_score = 0
        f2_grappling_score = 0
        
        for stat in ['TD', 'TDA', 'TDD', 'SUB']:
            f1_val = float(fighter1_data.get(stat, 0) or 0)
            f2_val = float(fighter2_data.get(stat, 0) or 0)
            
            if f1_val > f2_val * 1.1:
                f1_grappling_score += 1
            elif f2_val > f1_val * 1.1:
                f2_grappling_score += 1
        
        if f1_grappling_score > f2_grappling_score:
            analysis["grappling_advantage"] = f"{fighter1_name} has the grappling advantage"
        elif f2_grappling_score > f1_grappling_score:
            analysis["grappling_advantage"] = f"{fighter2_name} has the grappling advantage"
        else:
            analysis["grappling_advantage"] = "Grappling appears evenly matched"
        
        # Experience comparison
        f1_fights = int(fighter1_data.get('Win', 0) or 0) + int(fighter1_data.get('Loss', 0) or 0) + int(fighter1_data.get('Draw', 0) or 0)
        f2_fights = int(fighter2_data.get('Win', 0) or 0) + int(fighter2_data.get('Loss', 0) or 0) + int(fighter2_data.get('Draw', 0) or 0)
        
        if f1_fights > f2_fights * 1.25:
            analysis["experience_advantage"] = f"{fighter1_name} has significantly more experience"
        elif f2_fights > f1_fights * 1.25:
            analysis["experience_advantage"] = f"{fighter2_name} has significantly more experience"
        else:
            analysis["experience_advantage"] = "Both fighters have similar experience levels"
        
        # Overall assessment
        f1_score = f1_striking_score + f1_grappling_score + (1 if f1_fights > f2_fights else 0)
        f2_score = f2_striking_score + f2_grappling_score + (1 if f2_fights > f1_fights else 0)
        
        if f1_score > f2_score + 2:
            analysis["overall_assessment"] = f"{fighter1_name} has clear advantages in multiple areas"
        elif f2_score > f1_score + 2:
            analysis["overall_assessment"] = f"{fighter2_name} has clear advantages in multiple areas"
        elif f1_score > f2_score:
            analysis["overall_assessment"] = f"{fighter1_name} has slight overall advantages"
        elif f2_score > f1_score:
            analysis["overall_assessment"] = f"{fighter2_name} has slight overall advantages"
        else:
            analysis["overall_assessment"] = "This appears to be an evenly matched fight"
        
        return analysis
    
    except Exception as e:
        logger.error(f"Error in analyze_matchup: {str(e)}")
        return {
            "note": "Unable to analyze matchup due to an error"
        }

def get_important_factors(
    fighter1_data: Dict[str, Any], 
    fighter2_data: Dict[str, Any],
    model: Any, 
    features: List[str], 
    feature_diffs: np.ndarray
) -> List[Dict[str, Any]]:
    """
    Identify the most important factors influencing the prediction.
    """
    try:
        if not hasattr(model, "feature_importances_"):
            return []
        
        fighter1_name = fighter1_data.get('fighter_name', 'Fighter 1')
        fighter2_name = fighter2_data.get('fighter_name', 'Fighter 2')
        
        # Get feature importances
        importance = model.feature_importances_
        
        # Create list of (feature, importance, difference, abs_impact)
        feature_impacts = []
        for i, (feature, imp) in enumerate(zip(features, importance)):
            diff = feature_diffs[i]
            abs_impact = abs(imp * diff)  # Higher impact = more important for this prediction
            
            # Only include meaningful impacts
            if abs_impact > 0.001:
                feature_impacts.append({
                    "name": feature,
                    "importance": float(imp),
                    "difference": float(diff),
                    "impact": float(abs_impact),
                    "favors": fighter1_name if diff > 0 else fighter2_name
                })
        
        # Sort by impact
        feature_impacts.sort(key=lambda x: x["impact"], reverse=True)
        
        # Return top factors
        return feature_impacts[:5]
    
    except Exception as e:
        logger.error(f"Error in get_important_factors: {str(e)}")
        return [] 