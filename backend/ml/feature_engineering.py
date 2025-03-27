"""
Enhanced feature engineering for UFC fighter prediction model.

This module provides comprehensive feature extraction and engineering functions
for the UFC fighter prediction system. It handles various aspects of fighter data
including physical attributes, fight statistics, and performance metrics.
"""

import logging
import re
from typing import Dict, List, Any, Tuple, Optional, Union
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def safe_convert_to_float(value: Any, default: float = 0.0) -> float:
    """
    Safely convert a value to float, handling various formats and errors.
    
    Args:
        value: The value to convert
        default: Default value to return if conversion fails
        
    Returns:
        float: The converted float value or default if conversion fails
    """
    if value is None:
        return default
    
    if isinstance(value, (int, float)):
        return float(value)
    
    if not isinstance(value, str):
        return default
    
    # Remove % symbol and any other non-numeric characters except decimal point
    value = value.replace('%', '').strip()
    
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def extract_record_stats(record: str) -> Tuple[int, int, int]:
    """
    Extract wins, losses, and draws from record string.
    
    Args:
        record: Record string in format "W-L-D" (e.g. "21-3-0")
        
    Returns:
        Tuple[int, int, int]: Tuple containing (wins, losses, draws)
    """
    if not isinstance(record, str):
        return 0, 0, 0
    
    record_parts = record.split('-')
    
    # Safely extract wins, losses, and draws
    wins, losses, draws = 0, 0, 0
    try:
        if len(record_parts) > 0:
            wins = int(record_parts[0]) if record_parts[0].isdigit() else 0
        if len(record_parts) > 1:
            losses = int(record_parts[1]) if record_parts[1].isdigit() else 0
        if len(record_parts) > 2:
            draws = int(record_parts[2]) if record_parts[2].isdigit() else 0
    except (ValueError, IndexError):
        logger.warning(f"Could not parse record '{record}'")
    
    return wins, losses, draws

def calculate_win_percentage(record: str) -> float:
    """
    Calculate win percentage from record string.
    
    Args:
        record: Record string in format "W-L-D"
        
    Returns:
        float: Win percentage as a float between 0 and 100
    """
    wins, losses, draws = extract_record_stats(record)
    if wins + losses + draws > 0:
        return (wins / (wins + losses + draws)) * 100
    return 0

def extract_height_in_inches(height_str: str) -> float:
    """
    Convert height string to inches.
    
    Args:
        height_str: Height string in format "X'Y\"" (e.g. "5'11\"")
        
    Returns:
        float: Height in inches, or 0 if parsing fails
    """
    try:
        # Match feet and inches pattern (e.g., "5'11\"")
        height_parts = re.match(r"(\d+)'(\d+)\"", height_str)
        if height_parts:
            feet = int(height_parts.group(1))
            inches = int(height_parts.group(2))
            return (feet * 12) + inches
        else:
            return 0
    except (ValueError, AttributeError):
        logger.warning(f"Could not parse height '{height_str}'")
        return 0

def extract_reach_in_inches(reach_str: str) -> float:
    """
    Extract reach value in inches from string.
    
    Args:
        reach_str: Reach string (e.g. "72\"")
        
    Returns:
        float: Reach in inches, or 0 if parsing fails
    """
    try:
        # Find digits in the reach string
        reach_digits = re.search(r'(\d+)', reach_str)
        if reach_digits:
            return float(reach_digits.group(1))
        else:
            return 0
    except (ValueError, AttributeError):
        logger.warning(f"Could not parse reach '{reach_str}'")
        return 0

def extract_strikes_landed_attempted(strike_str: str) -> Tuple[int, int]:
    """
    Extract landed and attempted strikes from string.
    
    Args:
        strike_str: Strike string in format "X of Y"
        
    Returns:
        Tuple[int, int]: Tuple containing (landed strikes, attempted strikes)
    """
    try:
        if not isinstance(strike_str, str):
            return 0, 0
        
        parts = strike_str.split(' of ')
        landed = safe_convert_to_float(parts[0]) if len(parts) > 0 else 0
        attempted = safe_convert_to_float(parts[1]) if len(parts) > 1 else 0
        
        return int(landed), int(attempted)
    except Exception:
        return 0, 0

def extract_advanced_fighter_profile(
    fighter_data: Dict[str, Any],
    all_fights: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, float]:
    """
    Extract comprehensive fighter profile with career progression metrics.
    
    This function calculates various advanced metrics including:
    - Career trajectory (improvement/decline metrics)
    - Physical attributes relative to weight class
    - Fight frequency and activity level
    - Overall versatility scores
    
    Args:
        fighter_data: Dictionary containing basic fighter information
        all_fights: Optional list of all fights for career progression analysis
        
    Returns:
        Dict[str, float]: Dictionary of calculated metrics and scores
    """
    profile = {}
    
    try:
        # Basic career stats
        wins, losses, draws = extract_record_stats(fighter_data.get('record', '0-0-0'))
        total_fights = wins + losses + draws
        
        # Calculate career length (if debut date available)
        debut_date = fighter_data.get('debut_date', '')
        career_length_days = 0
        current_date = datetime.now()
        
        if debut_date and isinstance(debut_date, str):
            try:
                debut_datetime = datetime.strptime(debut_date, '%Y-%m-%d')
                career_length_days = (current_date - debut_datetime).days
            except (ValueError, TypeError):
                pass
        
        # Calculate fight frequency (fights per year)
        fight_frequency = 0
        if career_length_days > 0:
            fight_frequency = (total_fights / (career_length_days / 365.0))
        
        profile['total_fights'] = total_fights
        profile['career_length_days'] = career_length_days
        profile['fight_frequency'] = fight_frequency
        
        # Fight rate metrics
        if wins > 0:
            profile['finish_ratio'] = safe_convert_to_float(fighter_data.get('finish_rate', 0)) / 100.0
            profile['ko_finish_ratio'] = safe_convert_to_float(fighter_data.get('ko_rate', 0)) / 100.0
            profile['sub_finish_ratio'] = safe_convert_to_float(fighter_data.get('sub_rate', 0)) / 100.0
        else:
            profile['finish_ratio'] = 0
            profile['ko_finish_ratio'] = 0
            profile['sub_finish_ratio'] = 0
        
        # Physical attributes relative to weight class
        weight_class = fighter_data.get('weight_class', '').lower()
        height_inches = extract_height_in_inches(str(fighter_data.get('height', "0'0\"")))
        reach_inches = extract_reach_in_inches(str(fighter_data.get('reach', '0"')))
        
        # Average height and reach by weight class
        weight_class_avg_height = {
            'flyweight': 65.5,      # 5'5.5"
            'bantamweight': 67.5,   # 5'7.5"
            'featherweight': 69.0,  # 5'9"
            'lightweight': 70.5,    # 5'10.5"
            'welterweight': 72.0,   # 6'0"
            'middleweight': 73.5,   # 6'1.5"
            'light heavyweight': 75.0, # 6'3"
            'heavyweight': 76.5,    # 6'4.5"
        }
        
        weight_class_avg_reach = {
            'flyweight': 66.5,
            'bantamweight': 68.5,
            'featherweight': 70.0,
            'lightweight': 71.5,
            'welterweight': 73.5,
            'middleweight': 75.5,
            'light heavyweight': 77.5,
            'heavyweight': 79.0,
        }
        
        # Default to middleweight if weight class not found
        avg_height = weight_class_avg_height.get(weight_class, 73.5)
        avg_reach = weight_class_avg_reach.get(weight_class, 75.5)
        
        profile['height_advantage'] = height_inches - avg_height if height_inches > 0 else 0
        profile['reach_advantage'] = reach_inches - avg_reach if reach_inches > 0 else 0
        profile['reach_height_ratio'] = reach_inches / height_inches if height_inches > 0 else 0
        
        # Versatility scores (offensive, defensive, grappling, striking)
        # Striking offense metrics
        slpm = safe_convert_to_float(fighter_data.get('slpm', 0))
        str_acc = safe_convert_to_float(fighter_data.get('str_acc', 0)) / 100.0
        
        # Striking defense metrics
        str_def = safe_convert_to_float(fighter_data.get('str_def', 0)) / 100.0
        sapm = safe_convert_to_float(fighter_data.get('sapm', 0))
        
        # Grappling offense metrics
        td_avg = safe_convert_to_float(fighter_data.get('td_avg', 0))
        td_acc = safe_convert_to_float(fighter_data.get('td_acc', 0)) / 100.0
        sub_avg = safe_convert_to_float(fighter_data.get('sub_avg', 0))
        
        # Grappling defense metrics
        td_def = safe_convert_to_float(fighter_data.get('td_def', 0)) / 100.0
        
        # Calculate versatility scores (0-1 scale)
        profile['striking_offense_score'] = min(1.0, (slpm / 8.0) * 0.7 + str_acc * 0.3)
        profile['striking_defense_score'] = min(1.0, str_def * 0.7 + (1.0 - min(1.0, sapm / 8.0)) * 0.3)
        profile['grappling_offense_score'] = min(1.0, (td_avg / 5.0) * 0.5 + td_acc * 0.3 + (sub_avg / 3.0) * 0.2)
        profile['grappling_defense_score'] = min(1.0, td_def)
        
        # Overall versatility (balanced fighter metric)
        profile['overall_versatility'] = (
            profile['striking_offense_score'] * 0.3 +
            profile['striking_defense_score'] * 0.2 +
            profile['grappling_offense_score'] * 0.3 +
            profile['grappling_defense_score'] * 0.2
        )
        
        # Career progression (if we have all fights)
        if all_fights and len(all_fights) >= 3:
            # Sort fights by date (most recent first)
            try:
                sorted_fights = sorted(all_fights, 
                                      key=lambda x: datetime.strptime(x.get('date', '2000-01-01'), '%Y-%m-%d'),
                                      reverse=True)
                
                # Calculate performance metrics for first half vs second half of career
                half_point = max(1, len(sorted_fights) // 2)
                recent_fights = sorted_fights[:half_point]
                early_fights = sorted_fights[half_point:]
                
                # Win rates
                recent_wins = sum(1 for f in recent_fights if f.get('result', '').lower() == 'win')
                early_wins = sum(1 for f in early_fights if f.get('result', '').lower() == 'win')
                
                recent_win_rate = recent_wins / len(recent_fights) if recent_fights else 0
                early_win_rate = early_wins / len(early_fights) if early_fights else 0
                
                # Performance progression
                profile['career_progression'] = recent_win_rate - early_win_rate
            except Exception as e:
                logger.warning(f"Error calculating career progression: {str(e)}")
                profile['career_progression'] = 0
        else:
            profile['career_progression'] = 0
        
        return profile
    
    except Exception as e:
        logger.error(f"Error extracting advanced fighter profile: {str(e)}")
        return {
            'total_fights': 0,
            'career_length_days': 0,
            'fight_frequency': 0,
            'finish_ratio': 0,
            'ko_finish_ratio': 0,
            'sub_finish_ratio': 0,
            'height_advantage': 0,
            'reach_advantage': 0,
            'reach_height_ratio': 0,
            'striking_offense_score': 0,
            'striking_defense_score': 0,
            'grappling_offense_score': 0,
            'grappling_defense_score': 0,
            'overall_versatility': 0,
            'career_progression': 0
        }

def extract_recent_fight_stats(fights: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Extract statistics from recent fights.
    
    Args:
        fights: List of recent fights
        
    Returns:
        Dict with extracted statistics
    """
    if not fights:
        return {
            "recent_win_ratio": 0.5,
            "recent_finish_ratio": 0.0,
            "win_streak": 0,
            "loss_streak": 0
        }
    
    stats = {
        "recent_win_ratio": 0.5,
        "recent_finish_ratio": 0.0,
        "win_streak": 0,
        "loss_streak": 0
    }
    
    wins = 0
    finishes = 0
    
    # Count wins and finishes
    for fight in fights:
        result = fight.get('result', '').lower()
        if 'win' in result:
            wins += 1
            if 'ko' in result or 'tko' in result or 'sub' in result:
                finishes += 1
    
    # Calculate ratios
    if fights:
        stats["recent_win_ratio"] = wins / len(fights)
        stats["recent_finish_ratio"] = finishes / len(fights) if wins > 0 else 0.0
    
    # Calculate streaks
    current_win_streak = 0
    current_loss_streak = 0
    
    for fight in fights:
        result = fight.get('result', '').lower()
        if 'win' in result:
            current_win_streak += 1
            current_loss_streak = 0
        elif 'loss' in result:
            current_loss_streak += 1
            current_win_streak = 0
        else:
            # Draw or no contest
            current_win_streak = 0
            current_loss_streak = 0
    
    stats["win_streak"] = current_win_streak
    stats["loss_streak"] = current_loss_streak
    
    return stats

def extract_style_features(fighter_data: Dict[str, Any]) -> Dict[str, int]:
    """Extract fighting style features based on stats and metadata"""
    features = {
        'is_striker': 0,
        'is_wrestler': 0,
        'is_grappler': 0,
        'is_bjj': 0,
        'is_kickboxer': 0,
        'is_counter_striker': 0
    }
    
    # If stance and stats don't exist, return default features
    if not isinstance(fighter_data, dict):
        return features
    
    try:
        # Extract base stats
        slpm = safe_convert_to_float(fighter_data.get('slpm', 0))
        str_acc = safe_convert_to_float(fighter_data.get('str_acc', 0))
        sapm = safe_convert_to_float(fighter_data.get('sapm', 0))
        str_def = safe_convert_to_float(fighter_data.get('str_def', 0))
        td_avg = safe_convert_to_float(fighter_data.get('td_avg', 0))
        td_acc = safe_convert_to_float(fighter_data.get('td_acc', 0))
        td_def = safe_convert_to_float(fighter_data.get('td_def', 0))
        sub_avg = safe_convert_to_float(fighter_data.get('sub_avg', 0))
        
        # More aggressive thresholds for style classification
        
        # Check bio for style indicators (with higher priority)
        fighter_bio = str(fighter_data.get('fighter_status', '')).lower()
        fighter_name = fighter_data.get('fighter_name', '').lower()
        fighter_nickname = str(fighter_data.get('nickname', '')).lower()
        
        # Look for clear style indicators in nickname and bio
        # BJJ/Grappling indicators
        bjj_terms = ['jiu', 'jits', 'ground', 'gracie', 'bjj', 'grappl', 'submission', 'anaconda', 'choke', 'guard']
        # Wrestling indicators
        wrestling_terms = ['wrestl', 'smash', 'ground', 'pound', 'all-american', 'olymp', 'slam', 'takedown']
        # Striking indicators 
        striking_terms = ['box', 'strik', 'punch', 'kick', 'knock', 'ko ', 'muay thai', 'taekwondo', 'karate']
        
        # Check for explicit style mentions in bio/nickname
        for term in bjj_terms:
            if term in fighter_bio or term in fighter_nickname:
                features['is_bjj'] = 1
                features['is_grappler'] = 1
        
        for term in wrestling_terms:
            if term in fighter_bio or term in fighter_nickname:
                features['is_wrestler'] = 1
        
        for term in striking_terms:
            if term in fighter_bio or term in fighter_nickname:
                features['is_striker'] = 1
                if 'counter' in fighter_bio or 'counter' in fighter_nickname:
                    features['is_counter_striker'] = 1
        
        # Known grapplers
        if any(x in fighter_name.lower() for x in ['makhachev', 'khabib', 'islam', 'chimaev', 'usman', 'cormier', 'nurmagomedov']):
            features['is_wrestler'] = 1
            features['is_grappler'] = 1
        
        # Known strikers
        if any(x in fighter_name.lower() for x in ['adesanya', 'holloway', 'poirier', 'mcgregor', 'masvidal', 'whittaker', 'thompson']):
            features['is_striker'] = 1
        
        # Striker identification (secondary - use stats if no direct style mentions found)
        # High striking volume or accuracy, low takedown attempts
        if features['is_striker'] == 0 and features['is_wrestler'] == 0 and features['is_grappler'] == 0:
            if (slpm > 4.0 or str_acc > 50) and td_avg < 1.5:
                features['is_striker'] = 1
            
            # Counter striker
            # Lower volume, good defense, and differential favors defense
            if slpm < 3.5 and str_def > 55 and sapm < slpm:
                features['is_counter_striker'] = 1
                features['is_striker'] = 1
            
            # Wrestler identification
            # High takedown attempts, good accuracy
            if td_avg > 1.5 or (td_avg > 1.0 and td_acc > 45):
                features['is_wrestler'] = 1
            
            # Grappler identification (includes BJJ)
            # Good submission attempts or high ground control
            if sub_avg > 0.5:
                features['is_grappler'] = 1
                
            # Kickboxer
            # High striking with specific striking patterns
            if slpm > 3.5 and str_acc > 45:
                features['is_kickboxer'] = 1
                features['is_striker'] = 1
        
        # Make sure at least one style is set 
        if (features['is_striker'] + features['is_wrestler'] + features['is_grappler']) == 0:
            # Determine primary style based on stats balance
            if td_avg > slpm / 4:  # Significant takedown attempts relative to strikes
                features['is_wrestler'] = 1
            elif sub_avg > 0.2:  # Any submission attempts
                features['is_grappler'] = 1
                features['is_bjj'] = 1
            else:
                features['is_striker'] = 1  # Default to striker
        
        return features
    
    except Exception as e:
        logger.error(f"Error extracting style features: {str(e)}")
        return features

def check_head_to_head(fighter1_fights: List[Dict[str, Any]], 
                       fighter2_fights: List[Dict[str, Any]],
                       fighter1_name: str,
                       fighter2_name: str) -> Dict[str, Any]:
    """
    Check for head-to-head matchups between two fighters.
    
    Args:
        fighter1_fights: List of fighter1's fights
        fighter2_fights: List of fighter2's fights
        fighter1_name: Name of fighter1
        fighter2_name: Name of fighter2
        
    Returns:
        Dict with head-to-head statistics
    """
    h2h = {
        "total_fights": 0,
        "fighter1_wins": 0,
        "fighter2_wins": 0,
        "draws": 0
    }
    
    # Check fighter1's fights for matches against fighter2
    for fight in fighter1_fights:
        opponent = fight.get('opponent', '')
        if opponent.lower() == fighter2_name.lower():
            h2h["total_fights"] += 1
            result = fight.get('result', '').lower()
            if 'win' in result:
                h2h["fighter1_wins"] += 1
            elif 'loss' in result:
                h2h["fighter2_wins"] += 1
            else:
                # Draw or no contest
                h2h["draws"] += 1
    
    # Check fighter2's fights for matches against fighter1
    for fight in fighter2_fights:
        opponent = fight.get('opponent', '')
        if opponent.lower() == fighter1_name.lower():
            # Only count if we haven't already counted this fight
            # (should be the same fight from the other perspective)
            if h2h["total_fights"] == 0:
                h2h["total_fights"] += 1
                result = fight.get('result', '').lower()
                if 'win' in result:
                    h2h["fighter2_wins"] += 1
                elif 'loss' in result:
                    h2h["fighter1_wins"] += 1
                else:
                    # Draw or no contest
                    h2h["draws"] += 1
    
    return h2h

def find_common_opponents(fighter1_fights: List[Dict[str, Any]], 
                         fighter2_fights: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Find common opponents between two fighters.
    
    Args:
        fighter1_fights: List of fighter1's fights
        fighter2_fights: List of fighter2's fights
        
    Returns:
        List of common opponents and result comparisons
    """
    if not fighter1_fights or not fighter2_fights:
        return []
    
    # Get all opponents for each fighter
    fighter1_opponents = {fight.get('opponent', '').lower(): fight for fight in fighter1_fights}
    fighter2_opponents = {fight.get('opponent', '').lower(): fight for fight in fighter2_fights}
    
    # Find common opponents
    common_opponents = []
    
    for opponent in fighter1_opponents:
        if opponent in fighter2_opponents:
            common_opponents.append({
                "opponent": opponent,
                "fighter1_result": fighter1_opponents[opponent].get('result', ''),
                "fighter2_result": fighter2_opponents[opponent].get('result', '')
            })
    
    return common_opponents

def extract_physical_comparisons(fighter1_data: Dict[str, Any], fighter2_data: Dict[str, Any]) -> Dict[str, float]:
    """
    Extract physical attribute comparisons between fighters
    
    Computes differential metrics for height, reach, age, etc. to quantify
    physical advantages in the matchup.
    """
    comparisons = {}
    
    try:
        # Height comparison
        height1 = extract_height_in_inches(str(fighter1_data.get('height', "0'0\"")))
        height2 = extract_height_in_inches(str(fighter2_data.get('height', "0'0\"")))
        
        # Reach comparison
        reach1 = extract_reach_in_inches(str(fighter1_data.get('reach', '0"')))
        reach2 = extract_reach_in_inches(str(fighter2_data.get('reach', '0"')))
        
        # Age comparison
        age1 = safe_convert_to_float(fighter1_data.get('age', 0))
        age2 = safe_convert_to_float(fighter2_data.get('age', 0))
        
        # Weight comparison
        weight_class1 = fighter1_data.get('weight_class', '').lower()
        weight_class2 = fighter2_data.get('weight_class', '').lower()
        
        # Map weight classes to approximate weights (in pounds)
        weight_class_values = {
            'flyweight': 125,
            'bantamweight': 135,
            'featherweight': 145,
            'lightweight': 155,
            'welterweight': 170,
            'middleweight': 185,
            'light heavyweight': 205,
            'heavyweight': 240  # Average heavyweight, not limit
        }
        
        weight1 = weight_class_values.get(weight_class1, 185)
        weight2 = weight_class_values.get(weight_class2, 185)
        
        # Calculate absolute differences
        height_diff = height1 - height2
        reach_diff = reach1 - reach2
        age_diff = age1 - age2
        weight_diff = weight1 - weight2
        
        # Calculate reach advantage normalized by height
        reach_height_ratio1 = reach1 / height1 if height1 > 0 else 0
        reach_height_ratio2 = reach2 / height2 if height2 > 0 else 0
        reach_advantage = reach_height_ratio1 - reach_height_ratio2
        
        # Calculate physical advantage score (positive means fighter1 has advantage)
        # Height and reach advantages are generally positive
        # Age advantage is generally negative (younger is better)
        physical_advantage = 0.0
        
        # Add height component (each inch worth ~0.5% advantage, capped at 5%)
        physical_advantage += min(0.05, max(-0.05, height_diff * 0.005))
        
        # Add reach component (each inch worth ~1% advantage, capped at 7%)
        physical_advantage += min(0.07, max(-0.07, reach_diff * 0.01))
        
        # Add age component (each year worth ~0.5% advantage to younger, capped at 10%)
        youth_advantage = -age_diff * 0.005  # Negative age diff (fighter1 younger) becomes positive advantage
        physical_advantage += min(0.10, max(-0.10, youth_advantage))
        
        # Add weight component (if there's a mismatch)
        # Each 5 pounds worth ~1% advantage, capped at 10%
        if abs(weight_diff) > 3:  # Only consider significant weight differences
            weight_advantage = weight_diff * 0.002  # Each pound is worth 0.2%
            physical_advantage += min(0.10, max(-0.10, weight_advantage))
        
        # Store all comparisons
        comparisons = {
            'height_diff': height_diff,
            'reach_diff': reach_diff,
            'reach_advantage': reach_advantage,
            'age_diff': age_diff,
            'weight_diff': weight_diff,
            'physical_advantage_score': physical_advantage,
            'fighter1_height': height1,
            'fighter2_height': height2,
            'fighter1_reach': reach1,
            'fighter2_reach': reach2,
            'fighter1_age': age1,
            'fighter2_age': age2,
            'fighter1_weight': weight1,
            'fighter2_weight': weight2
        }
        
        return comparisons
        
    except Exception as e:
        logger.error(f"Error extracting physical comparisons: {str(e)}")
        return {
            'height_diff': 0,
            'reach_diff': 0,
            'reach_advantage': 0,
            'age_diff': 0,
            'weight_diff': 0,
            'physical_advantage_score': 0,
            'fighter1_height': 0,
            'fighter2_height': 0,
            'fighter1_reach': 0, 
            'fighter2_reach': 0,
            'fighter1_age': 0,
            'fighter2_age': 0,
            'fighter1_weight': 0,
            'fighter2_weight': 0
        }

def analyze_opponent_quality(fighter_fights: List[Dict[str, Any]], opponent_data: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze the quality of a fighter's opponents and performance against different types
    
    Args:
        fighter_fights: List of the fighter's past fights
        opponent_data: Dictionary of opponent data, keyed by opponent name
        
    Returns:
        Dictionary of opponent quality metrics
    """
    if not fighter_fights:
        return {
            'avg_opponent_win_pct': 0,
            'avg_opponent_rank': 0,
            'wins_against_ranked': 0,
            'losses_against_ranked': 0,
            'wins_against_strikers': 0,
            'losses_against_strikers': 0,
            'wins_against_grapplers': 0,
            'losses_against_grapplers': 0,
            'quality_score': 0
        }
    
    try:
        total_opponents = 0
        sum_win_pct = 0
        sum_rank = 0
        ranked_count = 0
        
        wins_against_ranked = 0
        losses_against_ranked = 0
        wins_against_strikers = 0
        losses_against_strikers = 0
        wins_against_grapplers = 0
        losses_against_grapplers = 0
        
        for fight in fighter_fights:
            opponent_name = fight.get('opponent', '')
            if not opponent_name or opponent_name not in opponent_data:
                continue
                
            opponent = opponent_data[opponent_name]
            total_opponents += 1
            
            # Add opponent win percentage
            opponent_win_pct = safe_convert_to_float(opponent.get('win_pct', 0))
            sum_win_pct += opponent_win_pct
            
            # Check if opponent was ranked
            opponent_rank = safe_convert_to_float(opponent.get('rank', 0))
            is_ranked = opponent_rank > 0 and opponent_rank <= 15
            if is_ranked:
                ranked_count += 1
                sum_rank += opponent_rank
            
            # Determine result against this opponent
            result = fight.get('result', '').lower()
            is_win = result == 'win'
            
            # Count wins/losses against ranked opponents
            if is_ranked:
                if is_win:
                    wins_against_ranked += 1
                else:
                    losses_against_ranked += 1
            
            # Determine opponent style
            is_striker = opponent.get('is_striker', 0) == 1
            is_grappler = opponent.get('is_grappler', 0) == 1 or opponent.get('is_wrestler', 0) == 1
            
            # Count wins/losses against different styles
            if is_striker:
                if is_win:
                    wins_against_strikers += 1
                else:
                    losses_against_strikers += 1
                    
            if is_grappler:
                if is_win:
                    wins_against_grapplers += 1
                else:
                    losses_against_grapplers += 1
        
        # Calculate averages
        avg_opponent_win_pct = sum_win_pct / total_opponents if total_opponents > 0 else 0
        avg_opponent_rank = sum_rank / ranked_count if ranked_count > 0 else 0
        
        # Calculate a quality score (0-1 scale)
        # Higher quality = higher opponent win %, more ranked opponents, better performance against various styles
        quality_components = []
        
        # Component 1: Average opponent win percentage (normalize to 0-1)
        quality_components.append(min(1.0, avg_opponent_win_pct / 80.0))
        
        # Component 2: Percentage of fights against ranked opponents
        quality_components.append(min(1.0, ranked_count / max(1, len(fighter_fights))))
        
        # Component 3: Performance against ranked opponents
        if wins_against_ranked + losses_against_ranked > 0:
            ranked_win_rate = wins_against_ranked / (wins_against_ranked + losses_against_ranked)
            quality_components.append(ranked_win_rate)
        else:
            quality_components.append(0.5)  # Neutral if no ranked opponents
        
        # Component 4: Versatility against different styles
        style_versatility = 0.5  # Default neutral
        if wins_against_strikers + losses_against_strikers > 0 and wins_against_grapplers + losses_against_grapplers > 0:
            striker_win_rate = wins_against_strikers / (wins_against_strikers + losses_against_strikers)
            grappler_win_rate = wins_against_grapplers / (wins_against_grapplers + losses_against_grapplers)
            # High score if can beat both styles
            style_versatility = (striker_win_rate + grappler_win_rate) / 2.0
        quality_components.append(style_versatility)
        
        # Calculate final quality score with weighted components
        quality_score = (
            quality_components[0] * 0.35 +  # Opponent win %
            quality_components[1] * 0.30 +  # % of ranked opponents
            quality_components[2] * 0.20 +  # Performance vs ranked
            quality_components[3] * 0.15    # Style versatility
        )
        
        return {
            'avg_opponent_win_pct': avg_opponent_win_pct,
            'avg_opponent_rank': avg_opponent_rank,
            'ranked_opponents_pct': ranked_count / max(1, len(fighter_fights)),
            'wins_against_ranked': wins_against_ranked,
            'losses_against_ranked': losses_against_ranked,
            'wins_against_strikers': wins_against_strikers,
            'losses_against_strikers': losses_against_strikers,
            'wins_against_grapplers': wins_against_grapplers,
            'losses_against_grapplers': losses_against_grapplers,
            'quality_score': quality_score
        }
    
    except Exception as e:
        logger.error(f"Error analyzing opponent quality: {str(e)}")
        return {
            'avg_opponent_win_pct': 0,
            'avg_opponent_rank': 0,
            'ranked_opponents_pct': 0,
            'wins_against_ranked': 0,
            'losses_against_ranked': 0,
            'wins_against_strikers': 0,
            'losses_against_strikers': 0,
            'wins_against_grapplers': 0,
            'losses_against_grapplers': 0,
            'quality_score': 0
        } 