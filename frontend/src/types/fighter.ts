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

export interface FightHistory {
  opponent_name: string;
  opponent?: string;  // Add this field to accommodate alternate data formats
  opponent_display_name?: string;
  result: string;
  method: string;
  round: number;
  time: string;
  event: string;
  date: string;
  opponent_stats?: Record<string, any>; // Change to Record to avoid circular dependency
  // Strike statistics
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
} 