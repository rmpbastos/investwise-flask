# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import classification_report, accuracy_score
# import joblib

# # Load the dataset
# df = pd.read_csv('combined_training_data.csv')

# # Separate features and labels
# X = df[['overall_sentiment_score', 'ticker_sentiment_score']]  # Feature columns
# y = df['price_movement']  # Target label

# # Split data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Initialize the Logistic Regression model
# model = LogisticRegression(class_weight='balanced')

# # Train the model
# model.fit(X_train, y_train)

# # Predict on the test set
# y_pred = model.predict(X_test)

# # Evaluate the model
# print("Model accuracy:", accuracy_score(y_test, y_pred))
# print("Classification report:\n", classification_report(y_test, y_pred))

# # Save the trained model to a file
# joblib.dump(model, 'price_movement_model_logreg.joblib')
# print("Model saved to price_movement_model_logreg.joblib")




import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib

# Load the dataset
df = pd.read_csv('../data/training_data/training_data_AAPL.csv')

# Separate features and labels
# Exclude columns that are not predictive features
X = df.drop(columns=['date', 'ticker', 'sector', 'price_movement'])  # Use all relevant features
y = df['price_movement']  # Target label with three classes (-1, 0, 1)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Logistic Regression model for multi-class classification
model = LogisticRegression(class_weight='balanced', multi_class='multinomial', solver='lbfgs', max_iter=1000)

# Train the model
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
print("Model accuracy:", accuracy_score(y_test, y_pred))
print("Classification report:\n", classification_report(y_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

# Save the trained model to a file
model_filename = '../models/price_movement_model_logreg_multiclass.joblib'
joblib.dump(model, model_filename)
print(f"Model saved to {model_filename}")