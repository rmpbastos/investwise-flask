import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb

# Define file paths
model_file = os.path.join("..", "models", "price_movement_model_xgboost_with_all_features.joblib")
training_data_file = os.path.join("..", "data", "training_data", "combined_training_data.csv")

# Load the model
print("Loading the trained model...")
model = joblib.load(model_file)

# Load the training data to get feature names
print("Loading training data...")
data = pd.read_csv(training_data_file)

# Define features (drop non-numeric columns and the target column)
features = data.drop(columns=['date', 'price_movement', 'ticker', 'sector'])
feature_names = features.columns

# Check if the model has been loaded correctly
if not isinstance(model, xgb.XGBClassifier):
    raise ValueError("Loaded model is not an instance of XGBClassifier.")

# Get feature importance from the model
print("Extracting feature importance...")
importance = model.feature_importances_

# Create a DataFrame for better visualization
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importance
}).sort_values(by='Importance', ascending=False)

# Display the top 20 most important features
print("Top 20 important features:")
print(importance_df.head(20))

# Plot feature importance
print("Plotting feature importance...")
plt.figure(figsize=(10, 8))
plt.barh(importance_df['Feature'][:20], importance_df['Importance'][:20], color='skyblue')
plt.gca().invert_yaxis()  # Invert y-axis to show the most important feature at the top
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.title('Top 20 Feature Importances in XGBoost Model')
plt.tight_layout()

# Save the plot
plot_output_file = os.path.join("..", "analysis", "feature_importance_plot.png")
plt.savefig(plot_output_file)
print(f"Feature importance plot saved to {plot_output_file}")

# Show the plot
plt.show()
