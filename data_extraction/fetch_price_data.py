# import requests
# import pandas as pd
# from dotenv import load_dotenv
# import os
# from datetime import datetime, timedelta
# import time

# # Load environment variables
# load_dotenv()
# api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
# ticker = "AAPL"  # Example ticker; replace with others as needed
# base_url = "https://www.alphavantage.co/query"
# price_data = []

# # Fetch daily adjusted price data
# def fetch_price_data():
#     params = {
#         "function": "TIME_SERIES_DAILY_ADJUSTED",
#         "symbol": ticker,
#         "apikey": api_key,
#         "outputsize": "full"  # Fetch all historical data (last 20+ years)
#     }
    
#     try:
#         response = requests.get(base_url, params=params)
#         response.raise_for_status()
#         data = response.json()
        
#         # Extract daily prices
#         time_series = data.get("Time Series (Daily)", {})
        
#         for date, price_info in time_series.items():
#             # Extract relevant data and include the ticker
#             price_item = {
#                 "ticker": ticker,
#                 "date": date,
#                 "open": float(price_info["1. open"]),
#                 "high": float(price_info["2. high"]),
#                 "low": float(price_info["3. low"]),
#                 "close": float(price_info["4. close"]),
#                 "adjusted_close": float(price_info["5. adjusted close"]),
#                 "volume": int(price_info["6. volume"]),
#                 "dividend_amount": float(price_info["7. dividend amount"]),
#                 "split_coefficient": float(price_info["8. split coefficient"])
#             }
#             price_data.append(price_item)
        
#         print(f"Fetched {len(price_data)} price entries for {ticker}")

#         # Respect API call limits (sleep to avoid hitting the rate limit)
#         time.sleep(1)

#     except requests.exceptions.RequestException as e:
#         print(f"An error occurred: {e}")

# # Run the fetch
# fetch_price_data()

# # Create DataFrame
# df = pd.DataFrame(price_data)

# # Sort by date in ascending order (older dates first)
# df = df.sort_values(by="date")

# # Save DataFrame to CSV file
# df.to_csv("historical_price_data_AAPL.csv", index=False)

# print("Price data saved to historical_price_data.csv")




# import requests
# import pandas as pd
# from dotenv import load_dotenv
# import os
# from datetime import datetime, timedelta
# import time

# # Load environment variables
# load_dotenv()
# api_key = os.getenv("ALPHA_VANTAGE_API_KEY")

# # List of tickers to fetch data for
# # tickers = ["AAPL", "MSFT", "META", "BAC", "WFC", "C", "JNJ", "PFE", "AMZN", "TSLA", "PG", "KO", "BA", "CAT", "XOM", "CVX", "NEE", "AMT", "DIS", "T"]
# tickers = ["AAPL"]

# base_url = "https://www.alphavantage.co/query"

# # Function to fetch price data for each ticker
# def fetch_price_data(ticker):
#     price_data = []

#     params = {
#         "function": "TIME_SERIES_DAILY_ADJUSTED",
#         "symbol": ticker,
#         "apikey": api_key,
#         "outputsize": "full"  # Fetch all historical data (last 20+ years)
#     }
    
#     try:
#         response = requests.get(base_url, params=params)
#         response.raise_for_status()
#         data = response.json()
        
#         # Extract daily prices
#         time_series = data.get("Time Series (Daily)", {})
        
#         for date, price_info in time_series.items():
#             # Extract relevant data and include the ticker
#             price_item = {
#                 "ticker": ticker,
#                 "date": date,
#                 "open": float(price_info["1. open"]),
#                 "high": float(price_info["2. high"]),
#                 "low": float(price_info["3. low"]),
#                 "close": float(price_info["4. close"]),
#                 "adjusted_close": float(price_info["5. adjusted close"]),
#                 "volume": int(price_info["6. volume"]),
#                 "dividend_amount": float(price_info["7. dividend amount"]),
#                 "split_coefficient": float(price_info["8. split coefficient"])
#             }
#             price_data.append(price_item)
        
#         print(f"Fetched {len(price_data)} price entries for {ticker}")

#         # Respect API call limits
#         time.sleep(1)

#     except requests.exceptions.RequestException as e:
#         print(f"An error occurred for {ticker}: {e}")

#     return price_data

# # Run the fetch for each ticker and save results
# for ticker in tickers:
#     print(f"Fetching data for {ticker}...")
#     ticker_price_data = fetch_price_data(ticker)
    
#     # Create DataFrame for each ticker
#     df = pd.DataFrame(ticker_price_data)
    
#     # Check if DataFrame is empty
#     if df.empty:
#         print(f"No price data fetched for {ticker}. Skipping.")
#         continue  # Skip to the next ticker if no data was fetched
    
#     # Sort by date in ascending order (older dates first)
#     df = df.sort_values(by="date")
    
#     # Save DataFrame to a CSV file specific to the ticker
#     file_name = f"historical_price_data_{ticker}.csv"
#     df.to_csv(file_name, index=False)
#     print(f"Price data for {ticker} saved to {file_name}")

# print("Price data fetching completed for all tickers.")




import requests
import pandas as pd
from dotenv import load_dotenv
import os
from datetime import datetime, timedelta
import time

# Load environment variables
load_dotenv()
api_key = os.getenv("ALPHA_VANTAGE_API_KEY")

# List of tickers to fetch data for
tickers = ["AAPL", "MSFT", "META", "BAC", "WFC", "C", "JNJ", "PFE", "AMZN", "TSLA", "PG", "KO", "BA", "CAT", "XOM", "CVX", "NEE", "AMT", "DIS", "T"]
# tickers = ["AAPL"]

base_url = "https://www.alphavantage.co/query"

# Define the output folder path
output_folder = os.path.join("..", "data", "historical_data")

# Ensure the output directory exists
os.makedirs(output_folder, exist_ok=True)

# Function to fetch price data for each ticker
def fetch_price_data(ticker):
    price_data = []

    params = {
        "function": "TIME_SERIES_DAILY_ADJUSTED",
        "symbol": ticker,
        "apikey": api_key,
        "outputsize": "full"  
    }
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        
        # Extract daily prices
        time_series = data.get("Time Series (Daily)", {})
        
        for date, price_info in time_series.items():
            # Extract relevant data and include the ticker
            price_item = {
                "ticker": ticker,
                "date": date,
                "open": float(price_info["1. open"]),
                "high": float(price_info["2. high"]),
                "low": float(price_info["3. low"]),
                "close": float(price_info["4. close"]),
                "adjusted_close": float(price_info["5. adjusted close"]),
                "volume": int(price_info["6. volume"]),
                "dividend_amount": float(price_info["7. dividend amount"]),
                "split_coefficient": float(price_info["8. split coefficient"])
            }
            price_data.append(price_item)
        
        print(f"Fetched {len(price_data)} price entries for {ticker}")

        # Respect API call limits
        # time.sleep(1)

    except requests.exceptions.RequestException as e:
        print(f"An error occurred for {ticker}: {e}")

    return price_data

# Run the fetch for each ticker and save results
for ticker in tickers:
    print(f"Fetching data for {ticker}...")
    ticker_price_data = fetch_price_data(ticker)
    
    # Create DataFrame for each ticker
    df = pd.DataFrame(ticker_price_data)
    
    # Check if DataFrame is empty
    if df.empty:
        print(f"No price data fetched for {ticker}. Skipping.")
        continue  # Skip to the next ticker if no data was fetched
    
    # Sort by date in ascending order (older dates first)
    df = df.sort_values(by="date")
    
    # Save DataFrame to a CSV file specific to the ticker in the historical_data folder
    file_name = os.path.join(output_folder, f"historical_price_data_{ticker}.csv")
    df.to_csv(file_name, index=False)
    print(f"Price data for {ticker} saved to {file_name}")

print("Price data fetching completed for all tickers.")