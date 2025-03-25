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
    head_to_head: Optional[Dict[str, Any]] = None,
    common_opponents: Optional[Dict[str, Any]] = None,
    prediction_data: Optional[Dict[str, Any]] = None
) -> str:
    """
    Generate a detailed, data-driven analysis of the matchup.
    
    This function creates a comprehensive analysis of the matchup by considering:
    - Fighter advantages in various categories
    - Head-to-head history
    - Common opponents
    - Recent performance
    - Physical attributes
    - Fighting styles
    
    Args:
        fighter1_data: Dictionary containing first fighter's data
        fighter2_data: Dictionary containing second fighter's data
        head_to_head: Optional dictionary containing head-to-head history
        common_opponents: Optional dictionary containing common opponents analysis
        prediction_data: Optional dictionary containing prediction results
        
    Returns:
        str: Detailed analysis of the matchup
    """
    try:
        # Get fighter names
        fighter1_name = fighter1_data.get('fighter_name', '') or fighter1_data.get('name', '')
        fighter2_name = fighter2_data.get('fighter_name', '') or fighter2_data.get('name', '')
        
        if not fighter1_name or not fighter2_name:
            logger.warning("Missing fighter names in analysis generation")
            return "Analysis unavailable due to missing fighter data."
        
        # Calculate advantages
        advantages = calculate_fighter_advantages(fighter1_data, fighter2_data)
        
        # Determine winner and loser based on prediction
        winner_name = prediction_data.get('winner', '')
        loser_name = prediction_data.get('loser', '')
        confidence = prediction_data.get('prediction_confidence', 0.5) * 100
        
        if not winner_name or not loser_name:
            logger.warning("Missing winner/loser in prediction data")
            return "Analysis unavailable due to missing prediction data."
        
        is_fighter1_winner = winner_name == fighter1_name
        winner_data = fighter1_data if is_fighter1_winner else fighter2_data
        loser_data = fighter2_data if is_fighter1_winner else fighter1_data
        
        # Get weight classes
        winner_weight_class = winner_data.get('weight_class', '').lower() 
        loser_weight_class = loser_data.get('weight_class', '').lower()
        
        # Get rankings
        winner_rank = winner_data.get('ranking', '')
        loser_rank = loser_data.get('ranking', '')
        winner_is_champion = bool(winner_data.get('is_champion', 0))
        loser_is_champion = bool(loser_data.get('is_champion', 0))
        
        # Determine fighter styles more accurately
        styles_map = {
            'makhachev': 'grappler',
            'nurmagomedov': 'grappler',
            'chimaev': 'grappler',
            'usman': 'wrestler',
            'covington': 'wrestler',
            'adesanya': 'striker',
            'holloway': 'striker', 
            'mcgregor': 'striker',
            'whittaker': 'striker'
        }
        
        # Check for known fighters and their styles
        winner_style = None
        loser_style = None
        
        for name, style in styles_map.items():
            if name in winner_name.lower():
                winner_style = style
            if name in loser_name.lower():
                loser_style = style
        
        if not winner_style:
            # Determine based on features
            if winner_data.get('is_wrestler', 0) and winner_data.get('is_grappler', 0):
                winner_style = 'grappler'
            elif winner_data.get('is_wrestler', 0):
                winner_style = 'wrestler'
            elif winner_data.get('is_striker', 0):
                winner_style = 'striker'
            else:
                winner_style = 'balanced fighter'
        
        if not loser_style:
            # Determine based on features
            if loser_data.get('is_wrestler', 0) and loser_data.get('is_grappler', 0):
                loser_style = 'grappler'
            elif loser_data.get('is_wrestler', 0):
                loser_style = 'wrestler'
            elif loser_data.get('is_striker', 0):
                loser_style = 'striker'
            else:
                loser_style = 'balanced fighter'
        
        # Build the analysis
        analysis = []
        
        # Check for significant weight class difference
        weight_class_advantage = ''
        
        if winner_weight_class and loser_weight_class and winner_weight_class != loser_weight_class:
            weight_classes = ['flyweight', 'bantamweight', 'featherweight', 'lightweight', 
                             'welterweight', 'middleweight', 'light heavyweight', 'heavyweight']
            
            try:
                winner_idx = weight_classes.index(winner_weight_class)
                loser_idx = weight_classes.index(loser_weight_class)
                diff = abs(winner_idx - loser_idx)
                
                if diff >= 2:
                    heavier_fighter = winner_name if winner_idx > loser_idx else loser_name
                    lighter_fighter = loser_name if winner_idx > loser_idx else winner_name
                    weight_class_advantage = f"There is a significant weight class difference between these fighters, with {heavier_fighter} being {diff} weight classes heavier than {lighter_fighter}. "
                    if heavier_fighter == winner_name:
                        weight_class_advantage += f"This size advantage is a major factor in {winner_name}'s favor. "
            except ValueError:
                # One of the weight classes not found in the list
                pass
        
        # Introduction - incorporate rankings if available
        intro = f"Based on our analysis of statistical metrics, {winner_name} has a {confidence:.0f}% chance of defeating {loser_name}. "
        
        if winner_rank and loser_rank:
            if winner_is_champion:
                intro = f"Champion {winner_name} has a {confidence:.0f}% chance of defeating #{loser_rank} ranked {loser_name}. "
            elif loser_is_champion:
                intro = f"#{winner_rank} ranked {winner_name} has a {confidence:.0f}% chance of upsetting Champion {loser_name}. "
            else:
                intro = f"#{winner_rank} ranked {winner_name} has a {confidence:.0f}% chance of defeating #{loser_rank} ranked {loser_name}. "
        
        analysis.append(intro)
        
        # Add weight class difference if present
        if weight_class_advantage:
            analysis.append(weight_class_advantage)
        
        # Head-to-head history
        if head_to_head and (head_to_head.get('fighter1_wins', 0) > 0 or head_to_head.get('fighter2_wins', 0) > 0):
            f1_wins = head_to_head.get('fighter1_wins', 0)
            f2_wins = head_to_head.get('fighter2_wins', 0)
            
            if f1_wins > 0 and f2_wins > 0:
                analysis.append(f"These fighters have a history together with {fighter1_name} winning {f1_wins} time(s) and {fighter2_name} winning {f2_wins} time(s).")
            elif f1_wins > 0:
                times = "times" if f1_wins > 1 else "time"
                analysis.append(f"{fighter1_name} has already beaten {fighter2_name} {f1_wins} {times}.")
            else:
                times = "times" if f2_wins > 1 else "time"
                analysis.append(f"{fighter2_name} has already beaten {fighter1_name} {f2_wins} {times}.")
            
            last_winner = head_to_head.get('last_winner')
            last_method = head_to_head.get('last_method')
            
            if last_winner and last_method:
                if winner_name == last_winner:
                    analysis.append(f"Their most recent matchup ended with {winner_name} winning by {last_method}, which reinforces our prediction.")
                else:
                    analysis.append(f"Although {loser_name} won their most recent matchup by {last_method}, our model indicates {winner_name} has improved enough to win this time.")
        
        # Common opponents
        if common_opponents and common_opponents.get('common_opponents_count', 0) > 0:
            count = common_opponents.get('common_opponents_count', 0)
            advantage = common_opponents.get('common_opponent_advantage', 0)
            
            if count >= 3:
                analysis.append(f"The fighters have faced {count} common opponents, providing solid comparative data.")
            else:
                analysis.append(f"The fighters have faced {count} common opponent(s).")
                
            if advantage > 1:
                analysis.append(f"{fighter1_name} has performed significantly better against these common opponents, which supports their edge in this matchup.")
            elif advantage < -1:
                analysis.append(f"{fighter2_name} has performed significantly better against these common opponents, which supports their edge in this matchup.")
        
        # Style matchup - improved for better accuracy especially for grapplers
        style_matchup = f"This matchup pits {winner_name} the {winner_style} against {loser_name} the {loser_style}. "
        
        # Add style-specific analysis
        if winner_style == 'grappler' and loser_style == 'striker':
            style_matchup += f"{winner_name}'s grappling-heavy approach is likely to neutralize {loser_name}'s striking advantage. "
            
            # Check takedown stats
            if advantages['grappling']['takedowns']['fighter'] == ('fighter1' if is_fighter1_winner else 'fighter2'):
                style_matchup += f"With superior takedown ability, {winner_name} should be able to control where this fight takes place. "
        
        elif winner_style == 'striker' and loser_style == 'grappler':
            style_matchup += f"While {loser_name} will likely look to take this fight to the ground, "
            
            # Check takedown defense
            if advantages['grappling']['takedown_defense']['fighter'] == ('fighter1' if is_fighter1_winner else 'fighter2'):
                td_def_val = advantages['grappling']['takedown_defense']['values'][0 if is_fighter1_winner else 1]
                style_matchup += f"{winner_name}'s strong takedown defense ({td_def_val:.0f}%) should allow them to keep the fight standing where they have the advantage. "
            else:
                style_matchup += f"{winner_name} will need to use their striking advantage in the moments when the fight stays standing. "
        
        analysis.append(style_matchup)
            
        # Key advantages
        key_advantages = []
        
        # Striking advantages
        if advantages['striking']['volume']['fighter'] == ('fighter1' if is_fighter1_winner else 'fighter2'):
            if advantages['striking']['volume']['percentage'] > 20:
                key_advantages.append(f"{winner_name} has significantly higher striking output ({advantages['striking']['volume']['values'][0 if is_fighter1_winner else 1]:.1f} vs. {advantages['striking']['volume']['values'][1 if is_fighter1_winner else 0]:.1f} strikes per minute)")
        
        # Striking accuracy
        if advantages['striking']['accuracy']['fighter'] == ('fighter1' if is_fighter1_winner else 'fighter2'):
            if advantages['striking']['accuracy']['difference'] > 5:
                key_advantages.append(f"{winner_name} is more accurate with strikes ({advantages['striking']['accuracy']['values'][0 if is_fighter1_winner else 1]:.0f}% vs. {advantages['striking']['accuracy']['values'][1 if is_fighter1_winner else 0]:.0f}%)")
        
        # Grappling advantages
        if advantages['grappling']['takedowns']['fighter'] == ('fighter1' if is_fighter1_winner else 'fighter2'):
            if advantages['grappling']['takedowns']['difference'] > 1:
                key_advantages.append(f"{winner_name} averages more takedowns ({advantages['grappling']['takedowns']['values'][0 if is_fighter1_winner else 1]:.1f} vs. {advantages['grappling']['takedowns']['values'][1 if is_fighter1_winner else 0]:.1f} per 15 minutes)")
        
        # Takedown defense
        if advantages['grappling']['takedown_defense']['fighter'] == ('fighter1' if is_fighter1_winner else 'fighter2'):
            if advantages['grappling']['takedown_defense']['difference'] > 10:
                key_advantages.append(f"{winner_name} has superior takedown defense ({advantages['grappling']['takedown_defense']['values'][0 if is_fighter1_winner else 1]:.0f}% vs. {advantages['grappling']['takedown_defense']['values'][1 if is_fighter1_winner else 0]:.0f}%)")
        
        # Physical advantages
        if 'reach' in advantages['physical']:
            if advantages['physical']['reach']['fighter'] == ('fighter1' if is_fighter1_winner else 'fighter2'):
                if advantages['physical']['reach']['difference'] > 2:
                    key_advantages.append(f"{winner_name} has a reach advantage ({winner_data.get('reach', '')} vs. {loser_data.get('reach', '')})")
        
        # Defense advantages
        if advantages['defense']['striking']['fighter'] == ('fighter1' if is_fighter1_winner else 'fighter2'):
            if advantages['defense']['striking']['difference'] > 5:
                key_advantages.append(f"{winner_name} has better striking defense ({advantages['defense']['striking']['values'][0 if is_fighter1_winner else 1]:.0f}% vs. {advantages['defense']['striking']['values'][1 if is_fighter1_winner else 0]:.0f}%)")
        
        # Momentum advantages
        if advantages['momentum']['current_streak']['fighter'] == ('fighter1' if is_fighter1_winner else 'fighter2'):
            winner_streak = advantages['momentum']['current_streak']['values'][0 if is_fighter1_winner else 1]
            loser_streak = advantages['momentum']['current_streak']['values'][1 if is_fighter1_winner else 0]
            if winner_streak > 0 and loser_streak <= 0:
                key_advantages.append(f"{winner_name} has positive momentum with a {winner_streak}-fight win streak, while {loser_name} has been struggling")
        
        # Include key advantages in analysis
        if key_advantages:
            if len(key_advantages) > 1:
                analysis.append(f"{winner_name} holds key advantages including {', '.join(key_advantages[:2])}.")
            else:
                analysis.append(f"{winner_name} holds a key advantage: {key_advantages[0]}.")
            
        # Check for major advantages for the predicted loser
        loser_advantages = []
        
        # Striking advantages for loser
        if advantages['striking']['volume']['fighter'] == ('fighter2' if is_fighter1_winner else 'fighter1'):
            if advantages['striking']['volume']['percentage'] > 20:
                loser_advantages.append(f"higher striking output ({advantages['striking']['volume']['values'][1 if is_fighter1_winner else 0]:.1f} vs. {advantages['striking']['volume']['values'][0 if is_fighter1_winner else 1]:.1f} strikes per minute)")
        
        # Add loser's advantages to analysis
        if loser_advantages:
            analysis.append(f"Despite the prediction, {loser_name} does have some advantages, including {loser_advantages[0]}.")
        
        # Path to victory for winner - taking style into account
        win_path = []
        
        if winner_style == 'grappler' or winner_style == 'wrestler':
            win_path.append(f"secure takedowns and control {loser_name} on the ground")
            if safe_convert_to_float(winner_data.get('sub_avg', 0)) > 0.5:
                win_path.append(f"look for submission opportunities")
            else:
                win_path.append(f"maintain dominant positions and wear down {loser_name}")
        elif winner_style == 'striker':
            if safe_convert_to_float(winner_data.get('td_def', 0)) > safe_convert_to_float(loser_data.get('td_acc', 0)):
                win_path.append(f"maintain distance and defend takedowns")
            win_path.append(f"utilize their striking advantage")
            if 'reach' in advantages['physical'] and advantages['physical']['reach']['fighter'] == ('fighter1' if is_fighter1_winner else 'fighter2'):
                win_path.append(f"use their reach advantage to keep {loser_name} at bay")
        
        if win_path:
            analysis.append(f"The path to victory for {winner_name} is to {', '.join(win_path)}.")
        
        # Path for the underdog
        lose_path = []
        
        if loser_style == 'grappler' or loser_style == 'wrestler':
            if winner_style == 'striker':
                lose_path.append(f"close the distance quickly and get the fight to the ground")
                lose_path.append(f"avoid striking exchanges at range")
            else:
                lose_path.append(f"push the pace and look for opportunities to establish dominant positions")
        elif loser_style == 'striker':
            if winner_style == 'grappler' or winner_style == 'wrestler':
                lose_path.append(f"defend takedowns and keep the fight standing")
                lose_path.append(f"use footwork to maintain distance")
            else:
                lose_path.append(f"exploit any openings in {winner_name}'s striking defense")
        
        if lose_path:
            analysis.append(f"While {loser_name} still has a {100-confidence:.0f}% chance to win if they can {', '.join(lose_path)}.")
        
        # Conclusion
        if winner_rank and loser_rank and winner_rank != 'C' and loser_rank != 'C':
            # Both fighters are ranked
            rank_diff = abs(int(winner_rank) - int(loser_rank))
            if rank_diff > 5:
                analysis.append(f"Given the {rank_diff} rank difference, this prediction aligns with the UFC's official rankings.")
        
        # Final prediction statement
        if winner_style == 'grappler' or winner_style == 'wrestler':
            if loser_style == 'striker':
                analysis.append(f"Our model predicts that {winner_name}'s grappling advantages will likely neutralize {loser_name}'s striking, leading to victory.")
            else:
                analysis.append(f"Our model predicts that {winner_name}'s superior grappling skills will likely lead to victory.")
        elif winner_style == 'striker':
            if loser_style == 'grappler' or loser_style == 'wrestler':
                analysis.append(f"Our model predicts that {winner_name} will be able to keep the fight standing long enough to utilize their striking advantages for victory.")
            else:
                analysis.append(f"Our model predicts that {winner_name}'s striking advantages will likely lead to victory.")
        else:
            analysis.append(f"Our model predicts that {winner_name}'s advantages in key performance metrics will likely lead to victory.")
        
        return " ".join(analysis)
    except Exception as e:
        logger.error(f"Error generating fight analysis: {str(e)}")
        return f"Analysis unavailable. Error: {str(e)}" 