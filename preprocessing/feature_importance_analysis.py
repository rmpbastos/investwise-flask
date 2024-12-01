import pandas as pd
import matplotlib.pyplot as plt
import joblib

# Load the trained model
model = joblib.load('random_forest_model_with_features.joblib')

# Load the dataset to match feature names
data = pd.read_csv('combined_training_data.csv')

# Ensure all required feature engineering steps are included
data['price_volatility'] = data['high'] - data['low']
data['prev_close'] = data['adjusted_close'].shift(1)
data['price_change'] = data['adjusted_close'] - data['prev_close']
data.drop(columns=['prev_close'], inplace=True)
data['5_day_avg_close'] = data['adjusted_close'].rolling(window=5).mean()
data['10_day_avg_close'] = data['adjusted_close'].rolling(window=10).mean()
data['overall_sentiment_score_lag1'] = data['overall_sentiment_score'].shift(1)
data['overall_sentiment_score_lag2'] = data['overall_sentiment_score'].shift(2)
data['ticker_sentiment_score_lag1'] = data['ticker_sentiment_score'].shift(1)
data['ticker_sentiment_score_lag2'] = data['ticker_sentiment_score'].shift(2)
data['5_day_volatility'] = data['adjusted_close'].rolling(window=5).std()
data.dropna(inplace=True)

# Select the features used for training
X = data[['overall_sentiment_score', 'ticker_sentiment_score', 'price_volatility', 'price_change',
          '5_day_avg_close', '10_day_avg_close', 'overall_sentiment_score_lag1', 
          'overall_sentiment_score_lag2', 'ticker_sentiment_score_lag1', 
          'ticker_sentiment_score_lag2', '5_day_volatility']]

# Get feature importances from the model
feature_importances = model.feature_importances_

# Create a DataFrame for feature importance visualization
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

# Plot the feature importances
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
plt.xlabel('Importance')
plt.title('Feature Importance from Random Forest Model')
plt.gca().invert_yaxis()  # Invert y-axis to display the highest importance at the top
plt.show()