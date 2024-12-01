import pandas as pd
import numpy as np
import os
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.combine import SMOTEENN
from sklearn.preprocessing import StandardScaler
import joblib

# Define file paths
training_data_file = os.path.join("..", "data", "training_data", "combined_training_data.csv")
model_output_file = os.path.join("..", "models", "price_movement_model_xgboost_feature_optimized.joblib")

# Load the combined training data
print("Loading training data...")
data = pd.read_csv(training_data_file)

# Check for NaN values and drop them if present
print("Checking for NaN values in the original data...")
data.dropna(inplace=True)

# Feature Engineering: Adding new volatility indicators (RSI_14 and RSI_30)
print("Adding new volatility indicators...")
data['RSI_14'] = data['RSI'].rolling(window=14).mean()
data['RSI_30'] = data['RSI'].rolling(window=30).mean()

# Drop NaN values introduced by the new features
print("Dropping NaN values introduced by feature engineering...")
data.dropna(inplace=True)

# Define features (X) and target (y)
print("Defining features and target...")
low_importance_features = ['ticker', 'sector', 'date', 'Signal_Line', 'MACD_Histogram', 'Band_Width']
features = data.drop(columns=['price_movement'] + low_importance_features)
target = data['price_movement']

# Standardize the features
print("Standardizing features...")
scaler = StandardScaler()
X = scaler.fit_transform(features)

# Convert target values to integers (0, 1, 2)
y = target.astype(int)

# Print the class distribution before resampling
print("Class distribution before resampling:")
print(y.value_counts())

# Apply combined resampling (SMOTE + Edited Nearest Neighbors)
print("Applying combined resampling...")
smote_enn = SMOTEENN(random_state=42)
X_resampled, y_resampled = smote_enn.fit_resample(X, y)

# Print the class distribution after resampling
print("Class distribution after resampling:")
print(pd.Series(y_resampled).value_counts())

# Split the data into training and testing sets
print("Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)

# Initialize the optimized XGBoost model
print("Initializing optimized XGBoost model...")
model = xgb.XGBClassifier(
    n_estimators=400,
    max_depth=8,
    learning_rate=0.03,
    min_child_weight=3,
    gamma=0.2,
    subsample=0.8,
    colsample_bytree=0.7,
    random_state=42,
    eval_metric='mlogloss'
)

# Train the model
print("Training the optimized XGBoost model...")
model.fit(X_train, y_train)

# Make predictions
print("Making predictions on the test set...")
y_pred = model.predict(X_test)

# Evaluate the model
print("Evaluating the optimized model...")
accuracy = np.mean(y_pred == y_test)
print(f"Model accuracy: {accuracy:.4f}")

# Print the classification report
print("Classification report:")
print(classification_report(y_test, y_pred))

# Print the confusion matrix
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))

# Save the trained model as .joblib
print("Saving the optimized model as .joblib...")
joblib.dump(model, model_output_file)
print(f"Model saved to {model_output_file}")

# Also save the model as .json (portable format)
json_output_file = model_output_file.replace(".joblib", ".json")
model.save_model(json_output_file)
print(f"Model saved to {json_output_file}")