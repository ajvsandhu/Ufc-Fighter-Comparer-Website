// API Configuration
export const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
export const API_VERSION = process.env.NEXT_PUBLIC_API_VERSION || 'v1'

// API Endpoints
export const ENDPOINTS = {
  FIGHTERS_SEARCH: (query: string) => `${API_URL}/api/${API_VERSION}/fighters?query=${encodeURIComponent(query)}`,
  FIGHTER: (name: string) => `${API_URL}/api/${API_VERSION}/fighter/${encodeURIComponent(name)}`,
  PREDICTION: (fighter1: string, fighter2: string) => 
    `${API_URL}/api/${API_VERSION}/prediction/predict/${encodeURIComponent(fighter1)}/${encodeURIComponent(fighter2)}`,
  TRAIN_MODEL: `${API_URL}/api/${API_VERSION}/prediction/train`,
  MODEL_STATUS: `${API_URL}/api/${API_VERSION}/prediction/status`,
  UPDATE_RANKINGS: `${API_URL}/api/${API_VERSION}/prediction/update-rankings`
} 