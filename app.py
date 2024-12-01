# Working
from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained XGBoost model
# model_path = "./models/xgboost_model.joblib"
model_path = "./models/price_movement_model_xgboost_with_all_features.joblib"
print(f"Loading model from: {model_path}")
model = joblib.load(model_path)

# Define the expected feature names (in the same order as used during training)
FEATURE_NAMES = [
    "overall_sentiment_mean", "overall_sentiment_min", "overall_sentiment_max", "overall_sentiment_std",
    "ticker_sentiment_mean", "ticker_sentiment_min", "ticker_sentiment_max", "ticker_sentiment_std",
    "open", "high", "low", "close", "adjusted_close", "volume", "dividend_amount", "split_coefficient",
    "RSI", "MACD", "Signal_Line", "MACD_Histogram",
    "Upper_Band", "Lower_Band", "Band_Width", "ATR",
    "ticker_encoded", "sector_encoded"
]

# Initialize a default feature vector (26 features set to 0)
default_features = np.zeros(len(FEATURE_NAMES))

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from the request
        data = request.get_json()
        print("Received data for prediction:", data)

        # Initialize a new feature vector (avoid reusing global state)
        features = np.zeros(len(FEATURE_NAMES))

        # Update the feature vector with input values
        for i, feature_name in enumerate(FEATURE_NAMES):
            if feature_name in data:
                features[i] = data[feature_name]

        print("Feature vector for prediction:", features)

        # Reshape the feature vector for prediction
        features_array = np.array([features])

        # Make the prediction
        prediction = model.predict(features_array)

        print("Prediction result:", prediction)

        # Map prediction to human-readable format
        prediction_map = {0: "Decrease", 1: "No Movement", 2: "Increase"}
        result = prediction_map.get(prediction[0], "Unknown")

        # Return the prediction result as JSON
        return jsonify({"prediction": result})

    except Exception as e:
        print("Error during prediction:", str(e))
        return jsonify({"error": "Prediction failed", "message": str(e)}), 500


if __name__ == '__main__':
    app.run(port=5001)
    
    
    
    



# FOR TESTING ONLY (RANDOMIZED ADJUSTMENT)
# from flask import Flask, request, jsonify
# import joblib
# import numpy as np

# app = Flask(__name__)

# # Load the trained XGBoost model
# model_path = "./models/xgboost_model.joblib"
# # model_path = "./models/price_movement_model_xgboost_with_all_features.joblib"
# print(f"Loading model from: {model_path}")
# model = joblib.load(model_path)

# # Define the expected feature names (in the same order as used during training)
# FEATURE_NAMES = [
#     "overall_sentiment_mean", "overall_sentiment_min", "overall_sentiment_max", "overall_sentiment_std",
#     "ticker_sentiment_mean", "ticker_sentiment_min", "ticker_sentiment_max", "ticker_sentiment_std",
#     "open", "high", "low", "close", "adjusted_close", "volume", "dividend_amount", "split_coefficient",
#     "RSI", "MACD", "Signal_Line", "MACD_Histogram",
#     "Upper_Band", "Lower_Band", "Band_Width", "ATR",
#     "ticker_encoded", "sector_encoded"
# ]

# # Initialize a default feature vector (26 features set to 0)
# default_features = np.zeros(len(FEATURE_NAMES))

# import random

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         # Get JSON data from the request
#         data = request.get_json()
#         print("Received data for prediction:", data)

#         # Initialize a new feature vector (avoid reusing global state)
#         features = np.zeros(len(FEATURE_NAMES))

#         # Update the feature vector with input values
#         for i, feature_name in enumerate(FEATURE_NAMES):
#             if feature_name in data:
#                 features[i] = data[feature_name]

#         print("Feature vector for prediction:", features)

#         # Reshape the feature vector for prediction
#         features_array = np.array([features])

#         # Make the prediction
#         prediction = model.predict(features_array)

#         # Add randomness for testing
#         prediction[0] = random.choice([0, 1, 2])

#         print("Prediction result:", prediction)

#         # Map prediction to human-readable format
#         prediction_map = {0: "Decrease", 1: "No Movement", 2: "Increase"}
#         result = prediction_map.get(prediction[0], "Unknown")

#         return jsonify({"prediction": result})
#     except Exception as e:
#         print("Error during prediction:", str(e))
#         return jsonify({"error": "Prediction failed", "message": str(e)}), 500




# if __name__ == '__main__':
#     app.run(port=5001)