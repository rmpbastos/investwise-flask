import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load the dataset and ensure all necessary features are present
data = pd.read_csv('combined_training_data.csv')

# Feature Engineering for Top Features
# Calculate price change
data['prev_close'] = data['adjusted_close'].shift(1)
data['price_change'] = data['adjusted_close'] - data['prev_close']
data.drop(columns=['prev_close'], inplace=True)

# Lagged Sentiment Scores
data['overall_sentiment_score_lag1'] = data['overall_sentiment_score'].shift(1)
data['ticker_sentiment_score_lag1'] = data['ticker_sentiment_score'].shift(1)
data['overall_sentiment_score_lag2'] = data['overall_sentiment_score'].shift(2)

# Rolling Averages
data['5_day_avg_close'] = data['adjusted_close'].rolling(window=5).mean()
data['10_day_avg_close'] = data['adjusted_close'].rolling(window=10).mean()

# Drop NaN values created by shifting and rolling
data.dropna(inplace=True)

# Select only the top features
X = data[['price_change', 'overall_sentiment_score_lag1', 'ticker_sentiment_score_lag1',
          'overall_sentiment_score_lag2', '5_day_avg_close', '10_day_avg_close']]
y = data['price_movement']

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define parameter grid for Randomized Search
param_grid = {
    'num_leaves': [20, 31, 40, 50],
    'max_depth': [-1, 10, 20, 30],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'n_estimators': [50, 100, 200, 300],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}

# Initialize the LightGBM model
lgb_model = LGBMClassifier(random_state=42)

# Perform Randomized Search
grid_search = RandomizedSearchCV(estimator=lgb_model, param_distributions=param_grid, n_iter=20, scoring='accuracy', cv=5, random_state=42, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Get the best model from grid search
best_lgb_model = grid_search.best_estimator_
y_pred = best_lgb_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"LightGBM Model accuracy: {accuracy:.2f}")

# Print classification report
print("Classification report:")
print(classification_report(y_test, y_pred))

# Save the trained model
joblib.dump(best_lgb_model, 'lightgbm_model_with_top_features.joblib')
print("LightGBM model saved as lightgbm_model_with_top_features.joblib")
