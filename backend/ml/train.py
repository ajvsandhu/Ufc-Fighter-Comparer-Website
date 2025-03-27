import os
import logging
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import traceback

from backend.api.database import get_db_connection
from backend.ml.model_loader import save_model
from backend.constants import IMPORTANT_FEATURES

# Configure logging
logger = logging.getLogger(__name__)

def train_model():
    """Train a new prediction model using the latest fighter data."""
    try:
        logger.info("Starting model training process...")
        
        # Get Supabase connection
        supabase = get_db_connection()
        if not supabase:
            logger.error("No database connection available for training")
            return False
        
        # Fetch fighter data
        try:
            response = supabase.table('fighters').select('*').execute()
            if not response.data:
                logger.error("No fighter data found for training")
                return False
            
            fighters_data = response.data
            logger.info(f"Retrieved {len(fighters_data)} fighters for training")
        except Exception as e:
            logger.error(f"Error fetching fighter data: {str(e)}")
            return False
        
        # Convert to DataFrame
        df = pd.DataFrame(fighters_data)
        
        # Select important features
        X = df[IMPORTANT_FEATURES].copy()
        
        # Fill missing values
        X = X.fillna(X.mean())
        
        # Create synthetic target variable (for demo)
        # In a real application, you would use historical fight results
        # Here we're just creating a dummy target based on win ratio
        y = (df['Win'] > df['Loss']).astype(int)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        logger.info("Training RandomForest model...")
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        logger.info(f"Model training complete. Test accuracy: {accuracy:.4f}")
        
        # Save model, scaler, and feature names
        features = list(X.columns)
        save_success = save_model(model, scaler, features)
        
        if save_success:
            logger.info("Model, scaler, and features saved successfully")
            return True
        else:
            logger.error("Failed to save model files")
            return False
    
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    # Configure logging for direct execution
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()]
    )
    
    # Train model
    success = train_model()
    if success:
        logger.info("Model training completed successfully")
    else:
        logger.error("Model training failed") 