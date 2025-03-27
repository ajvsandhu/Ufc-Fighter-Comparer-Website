"""
Enhanced fight analysis module for generating detailed, data-driven breakdowns.

This module provides comprehensive analysis of UFC fights, including fighter advantages,
matchup analysis, and detailed breakdowns of various aspects of the fight.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from backend.ml.feature_engineering import safe_convert_to_float

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_fighter_advantages(
    fighter1_data: Dict[str, Any],
    fighter2_data: Dict[str, Any]
) -> Dict[str, Dict[str, Any]]:
    """
    Calculate specific advantages between fighters based on their stats.
    
    This function analyzes various aspects of both fighters and determines
    who has the advantage in different categories including striking, grappling,
    defense, physical attributes, experience, and momentum.
    
    Args:
        fighter1_data: Dictionary containing first fighter's statistics
        fighter2_data: Dictionary containing second fighter's statistics
        
    Returns:
        Dict[str, Dict[str, Any]]: Dictionary containing advantages in different categories
    """
    advantages = {
        'striking': {},
        'grappling': {},
        'defense': {},
        'physical': {},
        'experience': {},
        'momentum': {}
    }
    
    # Striking comparisons
    slpm1 = safe_convert_to_float(fighter1_data.get('slpm', 0))
    slpm2 = safe_convert_to_float(fighter2_data.get('slpm', 0))
    str_acc1 = safe_convert_to_float(fighter1_data.get('str_acc', 0))
    str_acc2 = safe_convert_to_float(fighter2_data.get('str_acc', 0))
    sapm1 = safe_convert_to_float(fighter1_data.get('sapm', 0))
    sapm2 = safe_convert_to_float(fighter2_data.get('sapm', 0))
    
    advantages['striking']['volume'] = {
        'fighter': 'fighter1' if slpm1 > slpm2 else 'fighter2',
        'difference': abs(slpm1 - slpm2),
        'percentage': (abs(slpm1 - slpm2) / max(slpm1, slpm2, 1)) * 100,
        'values': (slpm1, slpm2)
    }
    
    advantages['striking']['accuracy'] = {
        'fighter': 'fighter1' if str_acc1 > str_acc2 else 'fighter2',
        'difference': abs(str_acc1 - str_acc2),
        'percentage': abs(str_acc1 - str_acc2),  # Already a percentage
        'values': (str_acc1, str_acc2)
    }
    
    # Grappling comparisons
    td_avg1 = safe_convert_to_float(fighter1_data.get('td_avg', 0))
    td_avg2 = safe_convert_to_float(fighter2_data.get('td_avg', 0))
    td_acc1 = safe_convert_to_float(fighter1_data.get('td_acc', 0))
    td_acc2 = safe_convert_to_float(fighter2_data.get('td_acc', 0))
    td_def1 = safe_convert_to_float(fighter1_data.get('td_def', 0))
    td_def2 = safe_convert_to_float(fighter2_data.get('td_def', 0))
    sub_avg1 = safe_convert_to_float(fighter1_data.get('sub_avg', 0))
    sub_avg2 = safe_convert_to_float(fighter2_data.get('sub_avg', 0))
    
    advantages['grappling']['takedowns'] = {
        'fighter': 'fighter1' if td_avg1 > td_avg2 else 'fighter2',
        'difference': abs(td_avg1 - td_avg2),
        'percentage': (abs(td_avg1 - td_avg2) / max(td_avg1, td_avg2, 1)) * 100,
        'values': (td_avg1, td_avg2)
    }
    
    advantages['grappling']['takedown_accuracy'] = {
        'fighter': 'fighter1' if td_acc1 > td_acc2 else 'fighter2',
        'difference': abs(td_acc1 - td_acc2),
        'percentage': abs(td_acc1 - td_acc2),  # Already a percentage
        'values': (td_acc1, td_acc2)
    }
    
    advantages['grappling']['takedown_defense'] = {
        'fighter': 'fighter1' if td_def1 > td_def2 else 'fighter2',
        'difference': abs(td_def1 - td_def2),
        'percentage': abs(td_def1 - td_def2),  # Already a percentage
        'values': (td_def1, td_def2)
    }
    
    advantages['grappling']['submissions'] = {
        'fighter': 'fighter1' if sub_avg1 > sub_avg2 else 'fighter2',
        'difference': abs(sub_avg1 - sub_avg2),
        'percentage': (abs(sub_avg1 - sub_avg2) / max(sub_avg1, sub_avg2, 0.1)) * 100,
        'values': (sub_avg1, sub_avg2)
    }
    
    # Defense comparisons
    str_def1 = safe_convert_to_float(fighter1_data.get('str_def', 0))
    str_def2 = safe_convert_to_float(fighter2_data.get('str_def', 0))
    
    advantages['defense']['striking'] = {
        'fighter': 'fighter1' if str_def1 > str_def2 else 'fighter2',
        'difference': abs(str_def1 - str_def2),
        'percentage': abs(str_def1 - str_def2),  # Already a percentage
        'values': (str_def1, str_def2)
    }
    
    # Absorption comparison (lower is better)
    advantages['defense']['absorption'] = {
        'fighter': 'fighter1' if sapm1 < sapm2 else 'fighter2',
        'difference': abs(sapm1 - sapm2),
        'percentage': (abs(sapm1 - sapm2) / max(sapm1, sapm2, 1)) * 100,
        'values': (sapm1, sapm2)
    }
    
    # Physical attributes
    reach1 = safe_convert_to_float(fighter1_data.get('reach_inches', 0))
    reach2 = safe_convert_to_float(fighter2_data.get('reach_inches', 0))
    height1 = safe_convert_to_float(fighter1_data.get('height_inches', 0))
    height2 = safe_convert_to_float(fighter2_data.get('height_inches', 0))
    
    if reach1 and reach2:
        advantages['physical']['reach'] = {
            'fighter': 'fighter1' if reach1 > reach2 else 'fighter2',
            'difference': abs(reach1 - reach2),
            'percentage': (abs(reach1 - reach2) / max(reach1, reach2)) * 100,
            'values': (reach1, reach2)
        }
    
    if height1 and height2:
        advantages['physical']['height'] = {
            'fighter': 'fighter1' if height1 > height2 else 'fighter2',
            'difference': abs(height1 - height2),
            'percentage': (abs(height1 - height2) / max(height1, height2)) * 100,
            'values': (height1, height2)
        }
    
    # Experience and record
    wins1, losses1 = 0, 0
    wins2, losses2 = 0, 0
    
    if 'record' in fighter1_data and isinstance(fighter1_data['record'], str):
        parts = fighter1_data['record'].split('-')
        wins1 = int(parts[0]) if len(parts) > 0 and parts[0].isdigit() else 0
        losses1 = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
    
    if 'record' in fighter2_data and isinstance(fighter2_data['record'], str):
        parts = fighter2_data['record'].split('-')
        wins2 = int(parts[0]) if len(parts) > 0 and parts[0].isdigit() else 0
        losses2 = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
    
    total_fights1 = wins1 + losses1
    total_fights2 = wins2 + losses2
    
    # Win percentage
    win_pct1 = (wins1 / total_fights1 * 100) if total_fights1 > 0 else 0
    win_pct2 = (wins2 / total_fights2 * 100) if total_fights2 > 0 else 0
    
    advantages['experience']['win_pct'] = {
        'fighter': 'fighter1' if win_pct1 > win_pct2 else 'fighter2',
        'difference': abs(win_pct1 - win_pct2),
        'percentage': abs(win_pct1 - win_pct2),  # Already a percentage
        'values': (win_pct1, win_pct2)
    }
    
    advantages['experience']['total_fights'] = {
        'fighter': 'fighter1' if total_fights1 > total_fights2 else 'fighter2',
        'difference': abs(total_fights1 - total_fights2),
        'percentage': (abs(total_fights1 - total_fights2) / max(total_fights1, total_fights2, 1)) * 100,
        'values': (total_fights1, total_fights2)
    }
    
    # Recent performance (if available)
    win_streak1 = safe_convert_to_float(fighter1_data.get('winning_streak', 0))
    win_streak2 = safe_convert_to_float(fighter2_data.get('winning_streak', 0))
    loss_streak1 = safe_convert_to_float(fighter1_data.get('losing_streak', 0))
    loss_streak2 = safe_convert_to_float(fighter2_data.get('losing_streak', 0))
    
    # Determine momentum based on streaks
    fighter1_momentum = win_streak1 - loss_streak1
    fighter2_momentum = win_streak2 - loss_streak2
    
    advantages['momentum']['current_streak'] = {
        'fighter': 'fighter1' if fighter1_momentum > fighter2_momentum else 'fighter2',
        'difference': abs(fighter1_momentum - fighter2_momentum),
        'values': (fighter1_momentum, fighter2_momentum)
    }
    
    # Recent win percentage
    recent_win_pct1 = safe_convert_to_float(fighter1_data.get('recent_win_pct', 0))
    recent_win_pct2 = safe_convert_to_float(fighter2_data.get('recent_win_pct', 0))
    
    advantages['momentum']['recent_win_pct'] = {
        'fighter': 'fighter1' if recent_win_pct1 > recent_win_pct2 else 'fighter2',
        'difference': abs(recent_win_pct1 - recent_win_pct2),
        'percentage': abs(recent_win_pct1 - recent_win_pct2),  # Already a percentage
        'values': (recent_win_pct1, recent_win_pct2)
    }
    
    # Finishing ability
    finish_rate1 = safe_convert_to_float(fighter1_data.get('finish_rate', 0))
    finish_rate2 = safe_convert_to_float(fighter2_data.get('finish_rate', 0))
    
    if finish_rate1 or finish_rate2:
        advantages['experience']['finish_rate'] = {
            'fighter': 'fighter1' if finish_rate1 > finish_rate2 else 'fighter2',
            'difference': abs(finish_rate1 - finish_rate2),
            'percentage': abs(finish_rate1 - finish_rate2),  # Already a percentage
            'values': (finish_rate1, finish_rate2)
        }
    
    return advantages

def generate_matchup_analysis(
    fighter1_data: Dict[str, Any],
    fighter2_data: Dict[str, Any],
    head_to_head: Dict[str, Any] = None,
    common_opponents: List[Dict[str, Any]] = None,
    prediction_data: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Generate a detailed analysis of the matchup between two fighters.
    
    Args:
        fighter1_data: Data for fighter 1
        fighter2_data: Data for fighter 2
        head_to_head: Head-to-head history between fighters
        common_opponents: List of common opponents
        prediction_data: Prediction data including probabilities
        
    Returns:
        Dict with matchup analysis
    """
    fighter1_name = fighter1_data.get('fighter_name', 'Fighter 1')
    fighter2_name = fighter2_data.get('fighter_name', 'Fighter 2')
    
    # Initialize analysis with basic sections
    analysis = {
        "striking_comparison": {},
        "grappling_comparison": {},
        "physical_comparison": {},
        "history": {},
        "prediction_notes": {}
    }
    
    # Striking comparison
    try:
        f1_striking = {}
        f2_striking = {}
        
        # Get striking stats
        for fighter, stats in [(fighter1_name, f1_striking), (fighter2_name, f2_striking)]:
            data = fighter1_data if fighter == fighter1_name else fighter2_data
            stats["slpm"] = float(data.get("SLPM", 0) or 0)
            stats["str_acc"] = float(data.get("StrAcc", 0) or 0)
            stats["str_def"] = float(data.get("StrDef", 0) or 0)
            stats["spm"] = float(data.get("SApM", 0) or 0)
        
        # Determine advantage
        if f1_striking.get("slpm", 0) > f2_striking.get("slpm", 0) * 1.2:
            advantage = f"{fighter1_name} lands significantly more strikes"
        elif f2_striking.get("slpm", 0) > f1_striking.get("slpm", 0) * 1.2:
            advantage = f"{fighter2_name} lands significantly more strikes"
        elif f1_striking.get("str_acc", 0) > f2_striking.get("str_acc", 0) * 1.2:
            advantage = f"{fighter1_name} has better striking accuracy"
        elif f2_striking.get("str_acc", 0) > f1_striking.get("str_acc", 0) * 1.2:
            advantage = f"{fighter2_name} has better striking accuracy"
        elif f1_striking.get("str_def", 0) > f2_striking.get("str_def", 0) * 1.2:
            advantage = f"{fighter1_name} has better striking defense"
        elif f2_striking.get("str_def", 0) > f1_striking.get("str_def", 0) * 1.2:
            advantage = f"{fighter2_name} has better striking defense"
        else:
            advantage = "Striking appears evenly matched"
        
        analysis["striking_comparison"] = {
            "fighter1": f1_striking,
            "fighter2": f2_striking,
            "advantage": advantage
        }
    except Exception as e:
        logger.error(f"Error in striking comparison: {str(e)}")
        analysis["striking_comparison"] = {"error": "Could not analyze striking"}
    
    # Grappling comparison
    try:
        f1_grappling = {}
        f2_grappling = {}
        
        # Get grappling stats
        for fighter, stats in [(fighter1_name, f1_grappling), (fighter2_name, f2_grappling)]:
            data = fighter1_data if fighter == fighter1_name else fighter2_data
            stats["td"] = float(data.get("TD", 0) or 0)
            stats["td_acc"] = float(data.get("TDA", 0) or 0)
            stats["td_def"] = float(data.get("TDD", 0) or 0)
            stats["sub"] = float(data.get("SUB", 0) or 0)
        
        # Determine advantage
        if f1_grappling.get("td", 0) > f2_grappling.get("td", 0) * 1.2:
            advantage = f"{fighter1_name} has better takedowns"
        elif f2_grappling.get("td", 0) > f1_grappling.get("td", 0) * 1.2:
            advantage = f"{fighter2_name} has better takedowns"
        elif f1_grappling.get("td_def", 0) > f2_grappling.get("td_def", 0) * 1.2:
            advantage = f"{fighter1_name} has better takedown defense"
        elif f2_grappling.get("td_def", 0) > f1_grappling.get("td_def", 0) * 1.2:
            advantage = f"{fighter2_name} has better takedown defense"
        elif f1_grappling.get("sub", 0) > f2_grappling.get("sub", 0) * 1.2:
            advantage = f"{fighter1_name} has better submission skills"
        elif f2_grappling.get("sub", 0) > f1_grappling.get("sub", 0) * 1.2:
            advantage = f"{fighter2_name} has better submission skills"
        else:
            advantage = "Grappling appears evenly matched"
        
        analysis["grappling_comparison"] = {
            "fighter1": f1_grappling,
            "fighter2": f2_grappling,
            "advantage": advantage
        }
    except Exception as e:
        logger.error(f"Error in grappling comparison: {str(e)}")
        analysis["grappling_comparison"] = {"error": "Could not analyze grappling"}
    
    # Physical comparison
    try:
        f1_physical = {}
        f2_physical = {}
        
        # Get physical stats
        for fighter, stats in [(fighter1_name, f1_physical), (fighter2_name, f2_physical)]:
            data = fighter1_data if fighter == fighter1_name else fighter2_data
            stats["height"] = float(data.get("Height", 0) or 0)
            stats["weight"] = float(data.get("Weight", 0) or 0)
            stats["reach"] = float(data.get("Reach", 0) or 0)
        
        # Determine advantage
        if abs(f1_physical.get("height", 0) - f2_physical.get("height", 0)) > 2:
            taller = fighter1_name if f1_physical.get("height", 0) > f2_physical.get("height", 0) else fighter2_name
            advantage = f"{taller} has a significant height advantage"
        elif abs(f1_physical.get("reach", 0) - f2_physical.get("reach", 0)) > 2:
            longer = fighter1_name if f1_physical.get("reach", 0) > f2_physical.get("reach", 0) else fighter2_name
            advantage = f"{longer} has a significant reach advantage"
        else:
            advantage = "Fighters are physically similar"
        
        analysis["physical_comparison"] = {
            "fighter1": f1_physical,
            "fighter2": f2_physical,
            "advantage": advantage
        }
    except Exception as e:
        logger.error(f"Error in physical comparison: {str(e)}")
        analysis["physical_comparison"] = {"error": "Could not analyze physical attributes"}
    
    # Head-to-head history
    if head_to_head and head_to_head.get("total_fights", 0) > 0:
        analysis["history"]["head_to_head"] = head_to_head
        
        f1_wins = head_to_head.get("fighter1_wins", 0)
        f2_wins = head_to_head.get("fighter2_wins", 0)
        
        if f1_wins > f2_wins:
            analysis["history"]["h2h_advantage"] = f"{fighter1_name} has won more head-to-head fights"
        elif f2_wins > f1_wins:
            analysis["history"]["h2h_advantage"] = f"{fighter2_name} has won more head-to-head fights"
        else:
            analysis["history"]["h2h_advantage"] = "Head-to-head history is even"
    
    # Common opponents
    if common_opponents and len(common_opponents) > 0:
        analysis["history"]["common_opponents"] = common_opponents
        analysis["history"]["common_opponents_count"] = len(common_opponents)
    
    # Prediction notes
    if prediction_data:
        winner = prediction_data.get("winner_name", "")
        confidence = prediction_data.get("confidence", 0)
        
        if confidence > 0.8:
            confidence_text = "very high"
        elif confidence > 0.6:
            confidence_text = "high"
        elif confidence > 0.4:
            confidence_text = "moderate"
        else:
            confidence_text = "low"
        
        analysis["prediction_notes"]["confidence_level"] = confidence_text
        
        if winner == fighter1_name:
            analysis["prediction_notes"]["favorite"] = fighter1_name
            analysis["prediction_notes"]["underdog"] = fighter2_name
        else:
            analysis["prediction_notes"]["favorite"] = fighter2_name
            analysis["prediction_notes"]["underdog"] = fighter1_name
    
    return analysis

def update_dict_recursive(target_dict: Dict[str, Any], source_dict: Dict[str, Any]) -> None:
    """
    Recursively update dictionary target_dict with values from source_dict.
    
    Args:
        target_dict: Target dictionary to update
        source_dict: Source dictionary with new values
    """
    for key, value in source_dict.items():
        if key in target_dict and isinstance(target_dict[key], dict) and isinstance(value, dict):
            update_dict_recursive(target_dict[key], value)
        else:
            target_dict[key] = value 