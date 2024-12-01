import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv('price_movement_data.csv')

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(df.head())

# Check for any missing values
print("\nChecking for missing values:")
print(df.isnull().sum())

# Display basic statistics of the dataset
print("\nBasic statistics of the dataset:")
print(df.describe())


# Separate features and labels
X = df[['overall_sentiment_score', 'ticker_sentiment_score']]
y = df['price_movement']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Display the shapes of the splits to verify
print("\nTraining and testing set shapes:")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)