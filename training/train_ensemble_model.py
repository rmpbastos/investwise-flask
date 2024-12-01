import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, balanced_accuracy_score, confusion_matrix
from sklearn.ensemble import VotingClassifier
from imblearn.combine import SMOTEENN
from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
import joblib
from collections import Counter

# Define class names
class_names = ["Down", "Stable", "Up"]

# Load the data
print("Loading training data...")
data_path = os.path.join("..", "data", "training_data", "combined_training_data.csv")
df = pd.read_csv(data_path)

# Check for NaN values
print("Checking for NaN values...")
print(df.isna().sum())

# Encode categorical features
print("Encoding categorical features...")
df["ticker_encoded"] = LabelEncoder().fit_transform(df["ticker"])
df["sector_encoded"] = LabelEncoder().fit_transform(df["sector"])

# Encode target labels
print("Encoding target values...")
label_encoder = LabelEncoder()
df["price_movement"] = label_encoder.fit_transform(df["price_movement"])

# Define features and target
features = [
    "overall_sentiment_mean", "overall_sentiment_min", "overall_sentiment_max", "overall_sentiment_std",
    "ticker_sentiment_mean", "ticker_sentiment_min", "ticker_sentiment_max", "ticker_sentiment_std",
    "RSI", "MACD", "Signal_Line", "MACD_Histogram", "Upper_Band", "Lower_Band", "Band_Width", "ATR",
    "ticker_encoded", "sector_encoded"
]
target = "price_movement"

X = df[features]
y = df[target]

# Standardize features
print("Standardizing features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Set up TimeSeriesSplit
print("Setting up TimeSeriesSplit...")
tscv = TimeSeriesSplit(n_splits=5)

# Initialize base models
xgb_model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=7,
    use_label_encoder=False,
    eval_metric="mlogloss"
)

lgb_model = lgb.LGBMClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=7
)

logistic_model = LogisticRegression(max_iter=1000)

# Initialize ensemble model (Voting Classifier)
ensemble_model = VotingClassifier(
    estimators=[
        ('xgb', xgb_model),
        ('lgb', lgb_model),
        ('logistic', logistic_model)
    ],
    voting='soft'
)

# Cross-validation with ensemble model
balanced_accuracies = []

print("Performing cross-validation with ensemble model...")
for fold, (train_idx, test_idx) in enumerate(tscv.split(X_scaled, y)):
    print(f"\nProcessing fold {fold + 1}...")

    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Apply SMOTEENN for resampling
    print("Applying SMOTEENN to the training set...")
    smoteenn = SMOTEENN()
    X_train_resampled, y_train_resampled = smoteenn.fit_resample(X_train, y_train)
    print(f"Class distribution after resampling: {Counter(y_train_resampled)}")

    # Train the ensemble model
    print("Training the ensemble model...")
    ensemble_model.fit(X_train_resampled, y_train_resampled)

    # Make predictions
    y_pred = ensemble_model.predict(X_test)

    # Evaluate the model
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    balanced_accuracies.append(balanced_acc)
    print(f"Fold {fold + 1} balanced accuracy: {balanced_acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=class_names))

    # Display confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion matrix:")
    print(cm)

# Print average balanced accuracy
avg_balanced_accuracy = np.mean(balanced_accuracies)
print(f"\nAverage balanced cross-validation accuracy: {avg_balanced_accuracy:.4f}")

# Train the final ensemble model on the entire dataset
print("Training the final model on the entire dataset...")
smoteenn = SMOTEENN()
X_resampled, y_resampled = smoteenn.fit_resample(X_scaled, y)
ensemble_model.fit(X_resampled, y_resampled)

# Save the model
model_path = os.path.join("..", "models", "ensemble_model.joblib")
print(f"Saving the model to {model_path}...")
joblib.dump(ensemble_model, model_path)
print("Model saved successfully.")
