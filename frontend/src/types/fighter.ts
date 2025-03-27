// Add FighterStats interface before FightHistory interface
export interface FighterStats {
  name: string;
  fighter_name?: string;
  record?: string;
  image_url?: string;
  height?: string;
  weight?: string;
  reach?: string;
  stance?: string;
  dob?: string;
  slpm?: string;
  str_acc?: string;
  sapm?: string;
  str_def?: string;
  td_avg?: string;
  td_acc?: string;
  td_def?: string;
  sub_avg?: string;
  nickname?: string;
  weight_class?: string;
  last_5_fights?: FightHistory[];
  ranking?: string;
  tap_link?: string;
}

// The FightHistory interface matches the actual database schema
// All fields are optional to handle different data formats
export interface FightHistory {
  // Database fields (from screenshot)
  id?: number | string;
  fighter_name?: string;
  fight_url?: string;
  kd?: string;
  sig_str?: string;
  sig_str_pct?: string;
  total_str?: string;
  head_str?: string;
  body_str?: string;
  leg_str?: string;
  takedowns?: string;
  td_pct?: string;
  ctrl?: string;
  result?: string;
  method?: string;
  opponent?: string;
  fight_date?: string;
  event?: string;
  created_at?: string;
  updated_at?: string;
  
  // Legacy fields for backward compatibility
  date?: string;
  opponent_name?: string;
  opponent_display_name?: string;
  round?: number;
  time?: string;
} 