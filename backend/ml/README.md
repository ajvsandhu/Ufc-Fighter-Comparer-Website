# UFC Fight Predictor ML Module

## Overview
This machine learning module is designed to predict the outcome of UFC fights based on fighter statistics and historical performance. It uses a Random Forest classifier trained on fighter data from the database.

## Features Used in Prediction

The model uses the following features for prediction:

### Fighter Stats
- Significant Strikes Landed per Minute (SLpM)
- Striking Accuracy (str_acc)
- Significant Strikes Absorbed per Minute (SApM)
- Striking Defense (str_def)
- Takedown Average (td_avg)
- Takedown Accuracy (td_acc)
- Takedown Defense (td_def)
- Submission Average (sub_avg)
- Win/Loss Record
- Reach
- Recent Win Percentage (from last 5 fights)

## How It Works

1. **Data Collection**: The model pulls fighter statistics and their last 5 fights from the database.
2. **Feature Extraction**: Numerical features are extracted from fighter data, including converting percentages to floats, parsing records, and calculating recent win rates.
3. **Training**: A Random Forest classifier is trained on the extracted features using scikit-learn.
4. **Prediction**: When a prediction is requested, the model compares the stats of both fighters and predicts the winner along with a confidence score.

## API Endpoints

- **GET /api/prediction/predict/{fighter1_name}/{fighter2_name}**: Predicts the winner between two fighters
- **GET /api/prediction/train**: Manually triggers model training/retraining
- **GET /api/prediction/status**: Returns the current status of the prediction model

## Future Improvements

This is a baseline model that can be enhanced by:

1. Adding more features (e.g., age, fight style matchups)
2. Incorporating time-series data to capture fighter evolution
3. Implementing more sophisticated models like gradient boosting or neural networks
4. Adding feature importance analysis for better explainability
5. Implementing active learning to improve the model with user feedback 