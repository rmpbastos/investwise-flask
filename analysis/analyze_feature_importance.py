import joblib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.inspection import permutation_importance
import os

# Define the paths using relative paths
model_path = "../models/ensemble_model.joblib"
data_path = "../data/training_data/combined_training_data.csv"

# Load the ensemble model
print("Loading the ensemble model...")
model = joblib.load(model_path)

# Load the training data for feature names
print("Loading the training data for feature names...")
df = pd.read_csv(data_path)

# Drop non-numeric columns and use only numeric features
print("Dropping non-numeric columns and using encoded features...")
df = df.drop(columns=["date", "ticker", "sector"])  # Drop non-numeric columns

# Ensure the feature set matches what was used during training
X = df.drop(columns=["price_movement"])
y = df["price_movement"]

# Check the data types
print("Checking data types...")
print(X.dtypes)

# Get the expected feature names from the XGBoost model within the ensemble
xgb_model = model.estimators_[0]  # Assuming the first model in the ensemble is XGBoost
expected_features = xgb_model.get_booster().feature_names

# Filter the DataFrame to include only the expected features
print("Filtering the DataFrame to include only the expected features from XGBoost model...")
X = X[expected_features]

# Calculate permutation importance
print("Calculating permutation importance...")
perm_importance = permutation_importance(model, X, y, n_repeats=10, random_state=42)

# Create a DataFrame for permutation importances
importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": perm_importance.importances_mean
})

# Sort the DataFrame by importance
importance_df = importance_df.sort_values(by="Importance", ascending=False)

# Display the top 10 features
print("Top 10 Features by Permutation Importance:")
print(importance_df.head(10))

# Plot permutation importances
plt.figure(figsize=(10, 8))
plt.barh(importance_df["Feature"], importance_df["Importance"], color="skyblue")
plt.gca().invert_yaxis()
plt.xlabel("Importance Score")
plt.title("Permutation Importances of the Ensemble Model")
plt.tight_layout()
plt.show()
