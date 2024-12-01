import pandas as pd
import numpy as np
import os
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.combine import SMOTEENN
from sklearn.preprocessing import StandardScaler
import joblib

# Define file paths
training_data_file = os.path.join("..", "data", "training_data", "combined_training_data.csv")
xgboost_output_file = os.path.join("..", "models", "xgboost_model.joblib")
lightgbm_output_file = os.path.join("..", "models", "lightgbm_model.joblib")
catboost_output_file = os.path.join("..", "models", "catboost_model.joblib")

# Load the combined training data
print("Loading training data...")
data = pd.read_csv(training_data_file)

# Check for NaN values and drop them if present
print("Checking for NaN values...")
data.dropna(inplace=True)

# Define features (X) and target (y)
print("Defining features and target...")
features = data.drop(columns=['date', 'price_movement', 'ticker', 'sector'])
target = data['price_movement']

# Standardize the features
print("Standardizing features...")
scaler = StandardScaler()
X = scaler.fit_transform(features)
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
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
)

# Helper function to evaluate models
def evaluate_model(model, model_name):
    print(f"Making predictions with {model_name}...")
    y_pred = model.predict(X_test)

    # Ensure y_pred is a 1-dimensional array (especially for CatBoost)
    if y_pred.ndim > 1:
        y_pred = y_pred.flatten()

    # Convert predictions to pandas Series and reset the index
    y_pred = pd.Series(y_pred).reset_index(drop=True)
    y_test_series = pd.Series(y_test).reset_index(drop=True)

    # Calculate accuracy
    accuracy = np.mean(y_pred == y_test_series)
    print(f"{model_name} Accuracy: {accuracy:.4f}")

    # Print the classification report
    print("Classification Report:")
    print(classification_report(y_test_series, y_pred))

    # Print the confusion matrix
    print("Confusion Matrix:")
    print(confusion_matrix(y_test_series, y_pred))

# Train and evaluate XGBoost model
print("\nTraining the XGBoost model...")
xgboost_model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=10,
    learning_rate=0.05,
    min_child_weight=3,
    gamma=0.1,
    subsample=0.8,
    colsample_bytree=1.0,
    random_state=42,
    eval_metric='mlogloss'
)
xgboost_model.fit(X_train, y_train)
evaluate_model(xgboost_model, "XGBoost")
joblib.dump(xgboost_model, xgboost_output_file)
print(f"XGBoost model saved to {xgboost_output_file}")

# Train and evaluate LightGBM model
print("\nTraining the LightGBM model...")
lightgbm_model = lgb.LGBMClassifier(
    n_estimators=300,
    max_depth=10,
    learning_rate=0.05,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=1.0,
    random_state=42
)
lightgbm_model.fit(X_train, y_train)
evaluate_model(lightgbm_model, "LightGBM")
joblib.dump(lightgbm_model, lightgbm_output_file)
print(f"LightGBM model saved to {lightgbm_output_file}")

# Train and evaluate CatBoost model
print("\nTraining the CatBoost model...")
catboost_model = cb.CatBoostClassifier(
    iterations=300,
    depth=10,
    learning_rate=0.05,
    random_seed=42,
    verbose=0
)
catboost_model.fit(X_train, y_train)
evaluate_model(catboost_model, "CatBoost")
joblib.dump(catboost_model, catboost_output_file)
print(f"CatBoost model saved to {catboost_output_file}")

