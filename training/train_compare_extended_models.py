import pandas as pd
import numpy as np
import os
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.combine import SMOTEENN
from sklearn.preprocessing import StandardScaler
import joblib

# Define file paths
training_data_file = os.path.join("..", "data", "training_data", "combined_training_data.csv")
xgboost_output_file = os.path.join("..", "models", "xgboost_model.joblib")
lightgbm_output_file = os.path.join("..", "models", "lightgbm_model.joblib")
random_forest_output_file = os.path.join("..", "models", "random_forest_model.joblib")
logistic_regression_output_file = os.path.join("..", "models", "logistic_regression_model.joblib")
svm_output_file = os.path.join("..", "models", "svm_model.joblib")

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
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)

# Helper function to train, predict, and evaluate models
def evaluate_model(model, model_name, output_file):
    print(f"Training the {model_name} model...")
    model.fit(X_train, y_train)
    
    print(f"Making predictions with {model_name}...")
    y_pred = model.predict(X_test)

    print(f"{model_name} Accuracy: {np.mean(y_pred == y_test):.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Save the model
    print(f"Saving the {model_name} model...")
    joblib.dump(model, output_file)
    print(f"{model_name} model saved to {output_file}\n")

# Initialize models
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

lightgbm_model = lgb.LGBMClassifier(
    n_estimators=300,
    max_depth=10,
    learning_rate=0.05,
    random_state=42
)

random_forest_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    random_state=42
)

logistic_regression_model = LogisticRegression(
    max_iter=300,
    random_state=42
)

svm_model = SVC(
    kernel='rbf',
    C=1.0,
    gamma='scale',
    random_state=42
)

# Evaluate all models
evaluate_model(xgboost_model, "XGBoost", xgboost_output_file)
evaluate_model(lightgbm_model, "LightGBM", lightgbm_output_file)
evaluate_model(random_forest_model, "Random Forest", random_forest_output_file)
evaluate_model(logistic_regression_model, "Logistic Regression", logistic_regression_output_file)
evaluate_model(svm_model, "SVM", svm_output_file)