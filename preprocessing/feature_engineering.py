import pandas as pd

# Load the training data
training_data = pd.read_csv("training_data.csv")

# Step 2: Calculate Rolling Averages and Sentiment Trends

# Calculate the 3-day rolling average for overall_sentiment_score and ticker_sentiment_score
training_data["overall_sentiment_score_3d_avg"] = training_data["overall_sentiment_score"].rolling(window=3).mean()
training_data["ticker_sentiment_score_3d_avg"] = training_data["ticker_sentiment_score"].rolling(window=3).mean()

# Calculate the sentiment trend as the difference between today's score and the 3-day rolling average
training_data["overall_sentiment_trend"] = training_data["overall_sentiment_score"] - training_data["overall_sentiment_score_3d_avg"]
training_data["ticker_sentiment_trend"] = training_data["ticker_sentiment_score"] - training_data["ticker_sentiment_score_3d_avg"]

# Step 3: Fill NaN values that result from rolling averages (for the first few rows)
training_data.fillna(0, inplace=True)

# Step 4: Save the enriched dataset
training_data.to_csv("enriched_training_data.csv", index=False)

print("Feature engineering complete. Enriched data saved to enriched_training_data.csv")
