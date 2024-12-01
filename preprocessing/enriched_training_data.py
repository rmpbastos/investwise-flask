import pandas as pd

# Load the enriched training data
data = pd.read_csv("enriched_training_data.csv")

# Display basic information and check for missing values
print("Data Overview:")
print(data.info())
print("\nFirst few rows:")
print(data.head())

# Check for any missing values
missing_values = data.isnull().sum()
print("\nMissing values in each column:")
print(missing_values)
