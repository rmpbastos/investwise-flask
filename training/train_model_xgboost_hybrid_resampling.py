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
model_output_file = os.path.join("..", "models", "price_movement_model_xgboost_with_all_features_hybrid_resampling.joblib")

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

# Convert target values to integers (0, 1, 2) if needed
y = target.astype(int)

# Print the class distribution before resampling
print("Class distribution before resampling:")
print(y.value_counts())

# Apply combined resampling (SMOTE + Edited Nearest Neighbors)
print("Applying hybrid resampling...")
smote_enn = SMOTEENN(random_state=42)
X_resampled, y_resampled = smote_enn.fit_resample(X, y)

# Print the class distribution after resampling
print("Class distribution after resampling:")
print(pd.Series(y_resampled).value_counts())

# Split the data into training and testing sets
print("Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)

# Calculate scale_pos_weight for each class
weight_0 = len(y_train) / (3 * (y_train == 0).sum())
weight_1 = len(y_train) / (3 * (y_train == 1).sum())
weight_2 = len(y_train) / (3 * (y_train == 2).sum())

# Initialize the XGBoost model with scale_pos_weight
print("Initializing XGBoost model with scale_pos_weight...")
model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=10,
    learning_rate=0.05,
    min_child_weight=3,
    gamma=0.1,
    subsample=0.8,
    colsample_bytree=1.0,
    random_state=42,
    eval_metric='mlogloss',
    scale_pos_weight=[weight_0, weight_1, weight_2]
)

# Train the model
print("Training the XGBoost model...")
model.fit(X_train, y_train)

# Make predictions
print("Making predictions on the test set...")
y_pred = model.predict(X_test)

# Evaluate the model
print("Evaluating the model...")
accuracy = np.mean(y_pred == y_test)
print(f"Model accuracy: {accuracy:.4f}")

# Print the classification report
print("Classification report:")
print(classification_report(y_test, y_pred))

# Print the confusion matrix
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))

# Save the trained model as .joblib
print("Saving the model as .joblib...")
joblib.dump(model, model_output_file)
print(f"Model saved to {model_output_file}")

# Also save the model as .json
json_output_file = model_output_file.replace(".joblib", ".json")
model.save_model(json_output_file)
print(f"Model saved to {json_output_file}")
