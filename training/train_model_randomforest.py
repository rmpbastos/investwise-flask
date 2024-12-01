import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load the training data
data = pd.read_csv('combined_training_data.csv')

# Feature Engineering
# Step 1: Calculate daily price volatility (difference between high and low prices)
data['price_volatility'] = data['high'] - data['low']

# Step 2: Calculate daily price change (difference between today's and previous day's close price)
data['prev_close'] = data['adjusted_close'].shift(1)
data['price_change'] = data['adjusted_close'] - data['prev_close']
data.drop(columns=['prev_close'], inplace=True)  # Drop the temporary column

# Step 3: Rolling Averages for adjusted close (5-day and 10-day)
data['5_day_avg_close'] = data['adjusted_close'].rolling(window=5).mean()
data['10_day_avg_close'] = data['adjusted_close'].rolling(window=10).mean()

# Step 4: Lagged Sentiment Scores (1-day and 2-day lags)
data['overall_sentiment_score_lag1'] = data['overall_sentiment_score'].shift(1)
data['overall_sentiment_score_lag2'] = data['overall_sentiment_score'].shift(2)
data['ticker_sentiment_score_lag1'] = data['ticker_sentiment_score'].shift(1)
data['ticker_sentiment_score_lag2'] = data['ticker_sentiment_score'].shift(2)

# Step 5: 5-Day Rolling Volatility for adjusted close
data['5_day_volatility'] = data['adjusted_close'].rolling(window=5).std()

# Drop any rows with NaN values resulting from rolling and shift operations
data.dropna(inplace=True)

# Prepare features and labels
X = data[['overall_sentiment_score', 'ticker_sentiment_score', 'price_volatility', 'price_change',
          '5_day_avg_close', '10_day_avg_close', 'overall_sentiment_score_lag1', 
          'overall_sentiment_score_lag2', 'ticker_sentiment_score_lag1', 
          'ticker_sentiment_score_lag2', '5_day_volatility']]
y = data['price_movement']

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the parameter grid for RandomizedSearch
param_grid = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2'],
    'bootstrap': [True, False]
}

# Initialize the Random Forest model
rf_model = RandomForestClassifier(random_state=42)

# Perform Randomized Search
grid_search = RandomizedSearchCV(estimator=rf_model, param_distributions=param_grid, n_iter=20, scoring='accuracy', cv=5, random_state=42, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Get the best model from grid search
best_rf_model = grid_search.best_estimator_
y_pred = best_rf_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Random Forest Model accuracy: {accuracy:.2f}")

# Print classification report
print("Classification report:")
print(classification_report(y_test, y_pred))

# Save the trained model
joblib.dump(best_rf_model, 'random_forest_model_with_features.joblib')
print("Random Forest model saved as random_forest_model_with_features.joblib")
