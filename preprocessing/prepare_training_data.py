"""
prepare_training_data.py

This script processes sentiment and historical price data for a list of stock tickers to prepare
a combined dataset for training machine learning models to predict price movements.

For each ticker:
1. Load the sentiment data and historical price data from respective CSV files in the 'historical_data' folder.
2. Process the sentiment data:
   - Convert the date format of sentiment data to align with the price data.
   - Aggregate sentiment scores by date to calculate the daily average sentiment score.
3. Process the price data:
   - Ensure the date format is consistent with sentiment data for accurate merging.
4. Merge the sentiment and price data on the date column to create a combined dataset.
5. Create a target variable ('price_movement'):
   - Calculate daily price movement by comparing adjusted closing prices.
   - Set the target as 1 if the price increased the next day, or 0 otherwise.
6. Save the prepared data to a new CSV file for each ticker in the 'training_data' folder.

Output:
For each ticker, a CSV file is saved in the 'training_data' folder, containing merged sentiment and price data,
with a target variable for daily price movement prediction.

"""


# import pandas as pd
# import os
# import glob

# # List of tickers to process
# tickers = ["AAPL", "MSFT", "META", "BAC", "WFC", "C", "JNJ", "PFE", "AMZN", "TSLA", 
#            "PG", "KO", "BA", "CAT", "XOM", "CVX", "NEE", "AMT", "DIS", "T"]

# # Define directories for input and output
# historical_data_folder = os.path.join("..", "data", "historical_data")
# training_data_folder = os.path.join("..", "data", "training_data")

# # Ensure the training data directory exists
# os.makedirs(training_data_folder, exist_ok=True)

# # Loop through each ticker and prepare the data
# for ticker in tickers:
#     # Define file paths for sentiment and price data
#     sentiment_file = os.path.join(historical_data_folder, f"historical_sentiment_data_{ticker}.csv")
#     price_file = os.path.join(historical_data_folder, f"historical_price_data_{ticker}.csv")
    
#     # Check if both files exist
#     if not os.path.exists(sentiment_file):
#         print(f"Sentiment data file for {ticker} not found. Skipping.")
#         continue
#     if not os.path.exists(price_file):
#         print(f"Price data file for {ticker} not found. Skipping.")
#         continue
    
#     # Load historical sentiment and price data
#     sentiment_data = pd.read_csv(sentiment_file)
#     price_data = pd.read_csv(price_file)
    
#     # Extract sector information from the sentiment data
#     sector = sentiment_data['sector'].iloc[0] if 'sector' in sentiment_data.columns else "Unknown"
    
#     # Step 1: Process sentiment data
#     sentiment_data['date'] = sentiment_data['time_published'].str[:8]
#     sentiment_data['date'] = pd.to_datetime(sentiment_data['date'], format='%Y%m%d').dt.strftime('%Y-%m-%d')
    
#     # Aggregate sentiment data by date with additional statistics
#     daily_sentiment = sentiment_data.groupby('date').agg({
#         'overall_sentiment_score': ['mean', 'min', 'max', 'std'],
#         'ticker_sentiment_score': ['mean', 'min', 'max', 'std']
#     }).reset_index()

#     # Flatten column names after aggregation
#     daily_sentiment.columns = [
#         'date', 'overall_sentiment_mean', 'overall_sentiment_min', 'overall_sentiment_max', 'overall_sentiment_std',
#         'ticker_sentiment_mean', 'ticker_sentiment_min', 'ticker_sentiment_max', 'ticker_sentiment_std'
#     ]
    
#     # Fill NaN values in standard deviation columns with 0
#     daily_sentiment['overall_sentiment_std'] = daily_sentiment['overall_sentiment_std'].fillna(0)
#     daily_sentiment['ticker_sentiment_std'] = daily_sentiment['ticker_sentiment_std'].fillna(0)


#     # Step 2: Process price data
#     price_data['date'] = pd.to_datetime(price_data['date']).dt.strftime('%Y-%m-%d')

#     # Add moving averages
#     price_data['SMA10'] = price_data['adjusted_close'].rolling(window=10).mean()
#     price_data['SMA30'] = price_data['adjusted_close'].rolling(window=30).mean()

#     # Calculate the difference between SMA10 and SMA30 (trend indicator)
#     price_data['SMA_diff'] = price_data['SMA10'] - price_data['SMA30']

    
#     # Step 3: Merge data on the 'date' column
#     training_data = pd.merge(daily_sentiment, price_data, on='date', how='inner')
    
#     # Step 4: Add the sector information and ticker
#     training_data['ticker'] = ticker
#     training_data['sector'] = sector
    
#     # Step 5: Create labels for price movement with threshold
#     threshold = 0.018  # 1.8% threshold for significant price movement
#     training_data['price_movement'] = training_data['adjusted_close'].pct_change().shift(-1)
#     training_data['price_movement'] = training_data['price_movement'].apply(
#         lambda x: 1 if x > threshold else (-1 if x < -threshold else 0)
#     )
    
#     # Drop rows with NaN values resulting from the diff operation
#     training_data.dropna(inplace=True)
    
#     # Step 6: Reorder columns
#     column_order = [
#         'date', 'ticker', 'sector',
#         'overall_sentiment_mean', 'overall_sentiment_min', 'overall_sentiment_max', 'overall_sentiment_std',
#         'ticker_sentiment_mean', 'ticker_sentiment_min', 'ticker_sentiment_max', 'ticker_sentiment_std',
#         'open', 'high', 'low', 'close', 'adjusted_close', 'volume', 'dividend_amount', 'split_coefficient',
#         'SMA10', 'SMA30', 'SMA_diff',  # Include the new features here
#         'price_movement'
#     ]

#     training_data = training_data[column_order]

    
#     # Save the prepared data to a new CSV file for each ticker
#     output_file = os.path.join(training_data_folder, f"training_data_{ticker}.csv")
#     training_data.to_csv(output_file, index=False)
#     print(f"Training data prepared and saved to {output_file}")

# # Step 7: Concatenate all individual CSV files into one combined dataset
# all_files = glob.glob(os.path.join(training_data_folder, "training_data_*.csv"))
# df_list = [pd.read_csv(file) for file in all_files]
# combined_df = pd.concat(df_list, ignore_index=True)

# # Save the combined dataset
# combined_output_file = os.path.join(training_data_folder, "combined_training_data.csv")
# combined_df.to_csv(combined_output_file, index=False)
# print(f"Combined dataset saved to {combined_output_file}")




# import pandas as pd
# import os
# import glob

# # List of tickers to process
# tickers = ["AAPL", "MSFT", "META", "BAC", "WFC", "C", "JNJ", "PFE", "AMZN", "TSLA",
#            "PG", "KO", "BA", "CAT", "XOM", "CVX", "NEE", "AMT", "DIS", "T"]

# # Define directories for input and output
# historical_data_folder = os.path.join("..", "data", "historical_data")
# training_data_folder = os.path.join("..", "data", "training_data")

# # Ensure the training data directory exists
# os.makedirs(training_data_folder, exist_ok=True)

# for ticker in tickers:
#     sentiment_file = os.path.join(historical_data_folder, f"historical_sentiment_data_{ticker}.csv")
#     price_file = os.path.join(historical_data_folder, f"historical_price_data_{ticker}.csv")

#     if not os.path.exists(sentiment_file) or not os.path.exists(price_file):
#         print(f"Data files for {ticker} not found. Skipping.")
#         continue

#     sentiment_data = pd.read_csv(sentiment_file)
#     price_data = pd.read_csv(price_file)

#     sector = sentiment_data['sector'].iloc[0] if 'sector' in sentiment_data.columns else "Unknown"

#     # Process sentiment data
#     sentiment_data['date'] = pd.to_datetime(sentiment_data['time_published'].str[:8], format='%Y%m%d')
#     daily_sentiment = sentiment_data.groupby('date').agg({
#         'overall_sentiment_score': ['mean', 'min', 'max', 'std'],
#         'ticker_sentiment_score': ['mean', 'min', 'max', 'std']
#     }).reset_index()

#     daily_sentiment.columns = [
#         'date', 'overall_sentiment_mean', 'overall_sentiment_min', 'overall_sentiment_max', 'overall_sentiment_std',
#         'ticker_sentiment_mean', 'ticker_sentiment_min', 'ticker_sentiment_max', 'ticker_sentiment_std'
#     ]
#     daily_sentiment.fillna(0, inplace=True)

#     # Process price data
#     price_data['date'] = pd.to_datetime(price_data['date'])

#     # Calculate technical indicators
#     price_data['RSI'] = price_data['adjusted_close'].rolling(window=14).apply(lambda x: 100 - (100 / (1 + (x.diff().clip(lower=0).mean() / x.diff().clip(upper=0).abs().mean()))))
#     price_data['MACD'] = price_data['adjusted_close'].ewm(span=12, adjust=False).mean() - price_data['adjusted_close'].ewm(span=26, adjust=False).mean()
#     price_data['Signal_Line'] = price_data['MACD'].ewm(span=9, adjust=False).mean()
#     price_data['MACD_Histogram'] = price_data['MACD'] - price_data['Signal_Line']
#     price_data['Upper_Band'] = price_data['adjusted_close'].rolling(window=20).mean() + (price_data['adjusted_close'].rolling(window=20).std() * 2)
#     price_data['Lower_Band'] = price_data['adjusted_close'].rolling(window=20).mean() - (price_data['adjusted_close'].rolling(window=20).std() * 2)
#     price_data['Band_Width'] = price_data['Upper_Band'] - price_data['Lower_Band']
#     price_data['ATR'] = price_data[['high', 'low', 'close']].apply(lambda x: max(x['high'] - x['low'], abs(x['high'] - x['close']), abs(x['low'] - x['close'])), axis=1).rolling(window=14).mean()

#     # Merge sentiment and price data
#     training_data = pd.merge(daily_sentiment, price_data, on='date', how='inner')
#     training_data['ticker'] = ticker
#     training_data['sector'] = sector

#     # Label the price movement with three classes
#     threshold = 0.018  # 1.8% threshold for price movement
#     training_data['price_movement'] = training_data['adjusted_close'].pct_change().shift(-1)
#     training_data['price_movement'] = training_data['price_movement'].apply(
#         lambda x: 2 if x > threshold else (0 if x < -threshold else 1)
#     )

#     training_data.dropna(inplace=True)

#     # Save the prepared data
#     output_file = os.path.join(training_data_folder, f"training_data_{ticker}.csv")
#     training_data.to_csv(output_file, index=False)
#     print(f"Training data saved to {output_file}")

# # Concatenate all individual CSV files into a combined dataset
# all_files = glob.glob(os.path.join(training_data_folder, "training_data_*.csv"))
# df_list = [pd.read_csv(file) for file in all_files]
# combined_df = pd.concat(df_list, ignore_index=True)

# combined_output_file = os.path.join(training_data_folder, "combined_training_data.csv")
# combined_df.to_csv(combined_output_file, index=False)
# print(f"Combined dataset saved to {combined_output_file}")




# import pandas as pd
# import os
# import glob

# # List of tickers to process
# tickers = ["AAPL", "MSFT", "META", "BAC", "WFC", "C", "JNJ", "PFE", "AMZN", "TSLA",
#            "PG", "KO", "BA", "CAT", "XOM", "CVX", "NEE", "AMT", "DIS", "T"]

# # Define directories for input and output
# historical_data_folder = os.path.join("..", "data", "historical_data")
# training_data_folder = os.path.join("..", "data", "training_data")
# os.makedirs(training_data_folder, exist_ok=True)

# # Loop through each ticker and prepare the data
# for ticker in tickers:
#     sentiment_file = os.path.join(historical_data_folder, f"historical_sentiment_data_{ticker}.csv")
#     price_file = os.path.join(historical_data_folder, f"historical_price_data_{ticker}.csv")

#     if not os.path.exists(sentiment_file) or not os.path.exists(price_file):
#         print(f"Data files for {ticker} not found. Skipping.")
#         continue

#     # Load sentiment and price data
#     sentiment_data = pd.read_csv(sentiment_file)
#     price_data = pd.read_csv(price_file)

#     # Extract sector information
#     sector = sentiment_data['sector'].iloc[0] if 'sector' in sentiment_data.columns else "Unknown"

#     # Process sentiment data
#     sentiment_data['date'] = pd.to_datetime(sentiment_data['time_published'].str[:8], format='%Y%m%d')
#     daily_sentiment = sentiment_data.groupby('date').agg({
#         'overall_sentiment_score': ['mean', 'min', 'max', 'std'],
#         'ticker_sentiment_score': ['mean', 'min', 'max', 'std']
#     }).reset_index()

#     daily_sentiment.columns = [
#         'date', 'overall_sentiment_mean', 'overall_sentiment_min', 'overall_sentiment_max', 'overall_sentiment_std',
#         'ticker_sentiment_mean', 'ticker_sentiment_min', 'ticker_sentiment_max', 'ticker_sentiment_std'
#     ]
#     daily_sentiment.fillna(0, inplace=True)

#     # Process price data
#     price_data['date'] = pd.to_datetime(price_data['date'])

#     # Calculate technical indicators
#     price_data['RSI'] = price_data['adjusted_close'].rolling(window=14).apply(
#         lambda x: 100 - (100 / (1 + (x.diff().clip(lower=0).mean() / x.diff().clip(upper=0).abs().mean())))
#     )
#     price_data['MACD'] = price_data['adjusted_close'].ewm(span=12).mean() - price_data['adjusted_close'].ewm(span=26).mean()
#     price_data['Signal_Line'] = price_data['MACD'].ewm(span=9).mean()
#     price_data['MACD_Histogram'] = price_data['MACD'] - price_data['Signal_Line']
#     price_data['Upper_Band'] = price_data['adjusted_close'].rolling(window=20).mean() + (price_data['adjusted_close'].rolling(window=20).std() * 2)
#     price_data['Lower_Band'] = price_data['adjusted_close'].rolling(window=20).mean() - (price_data['adjusted_close'].rolling(window=20).std() * 2)
#     price_data['Band_Width'] = price_data['Upper_Band'] - price_data['Lower_Band']
#     price_data['ATR'] = price_data[['high', 'low', 'close']].apply(
#         lambda x: max(x['high'] - x['low'], abs(x['high'] - x['close']), abs(x['low'] - x['close'])), axis=1
#     ).rolling(window=14).mean()

#     # Merge sentiment and price data
#     training_data = pd.merge(daily_sentiment, price_data, on='date', how='inner')
#     training_data['ticker'] = ticker
#     training_data['sector'] = sector

#     # Create labels for price movement
#     threshold = 0.018
#     training_data['price_movement'] = training_data['adjusted_close'].pct_change().shift(-1)
#     training_data['price_movement'] = training_data['price_movement'].apply(
#         lambda x: 2 if x > threshold else (0 if x < -threshold else 1)
#     )

#     training_data.dropna(inplace=True)

#     # Save the prepared data
#     output_file = os.path.join(training_data_folder, f"training_data_{ticker}.csv")
#     training_data.to_csv(output_file, index=False)
#     print(f"Training data saved to {output_file}")

# # Concatenate all individual CSV files
# all_files = glob.glob(os.path.join(training_data_folder, "training_data_*.csv"))
# df_list = [pd.read_csv(file) for file in all_files]
# combined_df = pd.concat(df_list, ignore_index=True)

# # One-hot encode 'ticker' and 'sector'
# combined_df = pd.get_dummies(combined_df, columns=['ticker', 'sector'])

# # Handle missing values in the combined dataset
# combined_df.fillna(0, inplace=True)

# # Save the combined dataset
# combined_output_file = os.path.join(training_data_folder, "combined_training_data.csv")
# combined_df.to_csv(combined_output_file, index=False)
# print(f"Combined dataset saved to {combined_output_file}")




import pandas as pd
import os
import glob
from sklearn.preprocessing import LabelEncoder

# List of tickers to process
tickers = ["AAPL", "MSFT", "META", "BAC", "WFC", "C", "JNJ", "PFE", "AMZN", "TSLA",
           "PG", "KO", "BA", "CAT", "XOM", "CVX", "NEE", "AMT", "DIS", "T"]

# Define directories for input and output
historical_data_folder = os.path.join("..", "data", "historical_data")
training_data_folder = os.path.join("..", "data", "training_data")

# Ensure the training data directory exists
os.makedirs(training_data_folder, exist_ok=True)

# Initialize LabelEncoders for ticker and sector
ticker_encoder = LabelEncoder()
sector_encoder = LabelEncoder()

# Collect all sectors and tickers for encoding
all_sectors = []
all_tickers = []

# Loop through each ticker to gather sector and ticker information
for ticker in tickers:
    sentiment_file = os.path.join(historical_data_folder, f"historical_sentiment_data_{ticker}.csv")
    if os.path.exists(sentiment_file):
        sentiment_data = pd.read_csv(sentiment_file)
        all_sectors.append(sentiment_data['sector'].iloc[0] if 'sector' in sentiment_data.columns else "Unknown")
        all_tickers.append(ticker)

# Fit the encoders
ticker_encoder.fit(all_tickers)
sector_encoder.fit(all_sectors)

# Prepare each ticker's training data
for ticker in tickers:
    sentiment_file = os.path.join(historical_data_folder, f"historical_sentiment_data_{ticker}.csv")
    price_file = os.path.join(historical_data_folder, f"historical_price_data_{ticker}.csv")

    if not os.path.exists(sentiment_file) or not os.path.exists(price_file):
        print(f"Data files for {ticker} not found. Skipping.")
        continue

    sentiment_data = pd.read_csv(sentiment_file)
    price_data = pd.read_csv(price_file)

    sector = sentiment_data['sector'].iloc[0] if 'sector' in sentiment_data.columns else "Unknown"

    # Process sentiment data
    sentiment_data['date'] = pd.to_datetime(sentiment_data['time_published'].str[:8], format='%Y%m%d')
    daily_sentiment = sentiment_data.groupby('date').agg({
        'overall_sentiment_score': ['mean', 'min', 'max', 'std'],
        'ticker_sentiment_score': ['mean', 'min', 'max', 'std']
    }).reset_index()

    daily_sentiment.columns = [
        'date', 'overall_sentiment_mean', 'overall_sentiment_min', 'overall_sentiment_max', 'overall_sentiment_std',
        'ticker_sentiment_mean', 'ticker_sentiment_min', 'ticker_sentiment_max', 'ticker_sentiment_std'
    ]
    daily_sentiment.fillna(0, inplace=True)

    # Process price data
    price_data['date'] = pd.to_datetime(price_data['date'])

    # Calculate technical indicators
    price_data['RSI'] = price_data['adjusted_close'].rolling(window=14).apply(lambda x: 100 - (100 / (1 + (x.diff().clip(lower=0).mean() / x.diff().clip(upper=0).abs().mean()))))
    price_data['MACD'] = price_data['adjusted_close'].ewm(span=12, adjust=False).mean() - price_data['adjusted_close'].ewm(span=26, adjust=False).mean()
    price_data['Signal_Line'] = price_data['MACD'].ewm(span=9, adjust=False).mean()
    price_data['MACD_Histogram'] = price_data['MACD'] - price_data['Signal_Line']
    price_data['Upper_Band'] = price_data['adjusted_close'].rolling(window=20).mean() + (price_data['adjusted_close'].rolling(window=20).std() * 2)
    price_data['Lower_Band'] = price_data['adjusted_close'].rolling(window=20).mean() - (price_data['adjusted_close'].rolling(window=20).std() * 2)
    price_data['Band_Width'] = price_data['Upper_Band'] - price_data['Lower_Band']
    price_data['ATR'] = price_data[['high', 'low', 'close']].apply(lambda x: max(x['high'] - x['low'], abs(x['high'] - x['close']), abs(x['low'] - x['close'])), axis=1).rolling(window=14).mean()

    # Merge sentiment and price data
    training_data = pd.merge(daily_sentiment, price_data, on='date', how='inner')
    training_data['ticker'] = ticker
    training_data['sector'] = sector

    # Encode ticker and sector
    training_data['ticker_encoded'] = ticker_encoder.transform(training_data['ticker'])
    training_data['sector_encoded'] = sector_encoder.transform(training_data['sector'])

    # Label the price movement
    threshold = 0.018  # 1.8% threshold for price movement
    training_data['price_movement'] = training_data['adjusted_close'].pct_change().shift(-1)
    training_data['price_movement'] = training_data['price_movement'].apply(
        lambda x: 2 if x > threshold else (0 if x < -threshold else 1)
    )

    training_data.dropna(inplace=True)

    # Save the prepared data
    output_file = os.path.join(training_data_folder, f"training_data_{ticker}.csv")
    training_data.to_csv(output_file, index=False)
    print(f"Training data saved to {output_file}")

# Concatenate all individual CSV files
all_files = glob.glob(os.path.join(training_data_folder, "training_data_*.csv"))
df_list = [pd.read_csv(file) for file in all_files]
combined_df = pd.concat(df_list, ignore_index=True)

# Save the combined dataset
combined_output_file = os.path.join(training_data_folder, "combined_training_data.csv")
combined_df.to_csv(combined_output_file, index=False)
print(f"Combined dataset saved to {combined_output_file}")