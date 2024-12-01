# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
# import xgboost as xgb
# import joblib
# from imblearn.over_sampling import SMOTE

# # Load the combined dataset
# df = pd.read_csv('../data/training_data/combined_training_data.csv')

# # Separate features and labels
# features = [
#     'overall_sentiment_mean', 'overall_sentiment_min', 'overall_sentiment_max', 'overall_sentiment_std',
#     'ticker_sentiment_mean', 'ticker_sentiment_min', 'ticker_sentiment_max', 'ticker_sentiment_std',
#     'open', 'high', 'low', 'close', 'adjusted_close', 'volume', 'dividend_amount', 'split_coefficient',
#     'SMA10', 'SMA30', 'SMA_diff'
# ]
# X = df[features]
# y = df['price_movement']

# # Map labels: -1 → 0, 0 → 1, 1 → 2
# y = y.map({-1: 0, 0: 1, 1: 2})

# # Split data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Handle class imbalance using SMOTE
# smote = SMOTE(random_state=42)
# X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# # Initialize the XGBoost model for multi-class classification
# model = xgb.XGBClassifier(
#     objective='multi:softmax',
#     num_class=3,
#     eval_metric='mlogloss',
#     subsample=0.8,
#     n_estimators=300,
#     min_child_weight=3,
#     max_depth=10,
#     learning_rate=0.05,
#     gamma=0.1,
#     colsample_bytree=1.0
# )

# # Train the model
# model.fit(X_train_resampled, y_train_resampled)

# # Predict on the test set
# y_pred = model.predict(X_test)

# # Evaluate the model
# print("Model accuracy:", accuracy_score(y_test, y_pred))
# print("Classification report:\n", classification_report(y_test, y_pred))
# print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

# # Save the trained model
# model_filename = '../models/price_movement_model_xgboost_with_features.joblib'
# joblib.dump(model, model_filename)
# print(f"Model saved to {model_filename}")




# import pandas as pd
# import numpy as np
# import xgboost as xgb
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
# from imblearn.over_sampling import ADASYN
# from imblearn.under_sampling import RandomUnderSampler
# from imblearn.pipeline import Pipeline
# import joblib

# # Load the combined training data
# data = pd.read_csv('../data/training_data/combined_training_data.csv')

# # Separate features and labels
# X = data.drop(columns=['date', 'ticker', 'sector', 'price_movement'])
# y = data['price_movement']

# # Check for NaN values and drop them
# X.dropna(inplace=True)
# y = y[X.index]

# # Analyze class distribution
# class_distribution = y.value_counts()
# print("Class distribution before resampling:\n", class_distribution)

# # Split data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# # Combine RandomUnderSampler and ADASYN in a pipeline
# print("Applying combined resampling...")
# under_sampler = RandomUnderSampler(sampling_strategy={1: 4000}, random_state=42)
# adasyn = ADASYN(sampling_strategy={0: 4000, 2: 4000}, random_state=42)

# resampling_pipeline = Pipeline(steps=[
#     ('under', under_sampler),
#     ('over', adasyn)
# ])

# X_train_resampled, y_train_resampled = resampling_pipeline.fit_resample(X_train, y_train)

# # Analyze the new class distribution
# resampled_distribution = pd.Series(y_train_resampled).value_counts()
# print("Class distribution after combined resampling:\n", resampled_distribution)

# # Initialize the XGBoost model with class weights
# model = xgb.XGBClassifier(
#     objective='multi:softmax',
#     num_class=3,
#     eval_metric='mlogloss',
#     use_label_encoder=False,
#     scale_pos_weight=[1, 0.5, 2],  # Adjusted weights for imbalance
#     random_state=42
# )

# # Train the model
# print("Training the XGBoost model...")
# model.fit(X_train_resampled, y_train_resampled)

# # Predict on the test set
# y_pred = model.predict(X_test)

# # Evaluate the model
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Model accuracy: {accuracy:.4f}")
# print("Classification report:\n", classification_report(y_test, y_pred))
# print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

# # Save the trained model
# model_filename = '../models/price_movement_model_xgboost_combined_resampling.joblib'
# joblib.dump(model, model_filename)
# print(f"Model saved to {model_filename}")




# ===================== 0.92 acc =====================
# ===================== 0.92 acc =====================
# ===================== 0.92 acc =====================
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
model_output_file = os.path.join("..", "models", "price_movement_model_xgboost_with_all_features.joblib")

# Load the combined training data
print("Loading training data...")
data = pd.read_csv(training_data_file)

# Check for NaN values and drop them if present
print("Checking for NaN values...")
data.dropna(inplace=True)

# Define features (X) and target (y)
print("Defining features and target...")
# Drop non-numeric columns and use encoded columns instead
features = data.drop(columns=['date', 'price_movement', 'ticker', 'sector'])
target = data['price_movement']

# Standardize the features
print("Standardizing features...")
scaler = StandardScaler()
X = scaler.fit_transform(features)

# Ensure target values are integers (0, 1, 2)
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

# Initialize the XGBoost model
print("Initializing XGBoost model...")
model = xgb.XGBClassifier(
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

# Save the trained model
print("Saving the model...")

# Save the model as .joblib (Python-specific format)
joblib.dump(model, model_output_file)
print(f"Model saved to {model_output_file}")

# Also save the model as .json (portable format)
json_output_file = model_output_file.replace(".joblib", ".json")
model.save_model(json_output_file)
print(f"Model saved to {json_output_file}")