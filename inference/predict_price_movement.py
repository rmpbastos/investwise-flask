import joblib
import pandas as pd

# Load the saved model
model_filename = 'price_movement_model.joblib'
model = joblib.load(model_filename)

# Example new data to predict, formatted as a DataFrame to match training features
new_data = pd.DataFrame([[0.25, 0.3]], columns=['overall_sentiment_score', 'ticker_sentiment_score'])

# Predict the price movement
prediction = model.predict(new_data)
print("Predicted price movement:", "Increase" if prediction[0] == 1 else "Decrease")
