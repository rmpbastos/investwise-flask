# import requests
# import pandas as pd
# from dotenv import load_dotenv
# import os
# import time
# from datetime import datetime, timedelta

# # Load environment variables
# load_dotenv()
# api_key = os.getenv("ALPHA_VANTAGE_API_KEY")

# # List of tickers from different sectors
# # tickers = ["AAPL", "MSFT", "META", "JNJ", "PFE", "AMZN", "TSLA", "PG", "KO", "BA", "CAT", "CVX", "NEE", "AMT", "T"]
# tickers = ["AAPL"]

# base_url = "https://www.alphavantage.co/query"

# # Set date range: Start 3 years back from today
# end_date = datetime.now()
# start_date = end_date - timedelta(days=1095)  # 3 years ago

# # Function to fetch sentiment data for each ticker
# def fetch_sentiment_data(ticker):
#     news_data = []
#     fetched_dates = set()
#     current_end_date = end_date

#     while start_date < current_end_date:
#         time_to = current_end_date.strftime('%Y%m%dT%H%M')
#         time_from = (current_end_date - timedelta(weeks=1)).strftime('%Y%m%dT%H%M')
        
#         params = {
#             "function": "NEWS_SENTIMENT",
#             "tickers": ticker,
#             "apikey": api_key,
#             "sort": "EARLIEST",
#             "limit": 1000,
#             "time_from": time_from,
#             "time_to": time_to
#         }
        
#         try:
#             response = requests.get(base_url, params=params)
#             response.raise_for_status()
#             data = response.json()
            
#             # Extract articles and avoid duplicates
#             for article in data.get("feed", []):
#                 timestamp = article.get("time_published")
                
#                 # Skip if the timestamp is already in fetched_dates
#                 if timestamp in fetched_dates:
#                     continue
                
#                 fetched_dates.add(timestamp)
                
#                 # Extract relevant data
#                 news_item = {
#                     "ticker": ticker,
#                     "title": article.get("title"),
#                     "url": article.get("url"),
#                     "time_published": timestamp,
#                     "overall_sentiment_score": article.get("overall_sentiment_score"),
#                     "overall_sentiment_label": article.get("overall_sentiment_label"),
#                     "ticker_sentiment_score": None,
#                     "ticker_sentiment_label": None
#                 }
                
#                 # Extract ticker-specific sentiment
#                 for sentiment in article.get("ticker_sentiment", []):
#                     if sentiment.get("ticker") == ticker:
#                         news_item["ticker_sentiment_score"] = sentiment.get("ticker_sentiment_score")
#                         news_item["ticker_sentiment_label"] = sentiment.get("ticker_sentiment_label")
#                         break
                
#                 news_data.append(news_item)
            
#             # Move the current_end_date one week back
#             current_end_date -= timedelta(weeks=1)
#             print(f"Fetched {len(news_data)} articles for {ticker} so far.")

#             # Respect API call limits
#             # time.sleep(1)

#         except requests.exceptions.RequestException as e:
#             print(f"An error occurred for {ticker}: {e}")
#             break

#     return news_data

# # Run the fetch for each ticker and save results
# for ticker in tickers:
#     print(f"Fetching data for {ticker}...")
#     ticker_news_data = fetch_sentiment_data(ticker)
    
#     # Create DataFrame
#     df = pd.DataFrame(ticker_news_data)
    
#     # Check if DataFrame is empty
#     if df.empty:
#         print(f"No articles fetched for {ticker}. Skipping.")
#         continue  # Skip to the next ticker if no data was fetched
    
#     # Sort by time_published in descending order (newest first)
#     df = df.sort_values(by="time_published", ascending=False)
    
#     # Save DataFrame to a CSV file specific to the ticker
#     file_name = f"historical_sentiment_data_{ticker}.csv"
#     df.to_csv(file_name, index=False)
#     print(f"Sentiment data for {ticker} saved to {file_name}")

# print("Sentiment data fetching completed for all tickers.")




# import requests
# import pandas as pd
# from dotenv import load_dotenv
# import os
# import time
# from datetime import datetime, timedelta

# # Load environment variables
# load_dotenv()
# api_key = os.getenv("ALPHA_VANTAGE_API_KEY")

# # List of tickers from different sectors
# # tickers = ["AAPL", "MSFT", "META", "BAC", "WFC", "C", "JNJ", "PFE", "AMZN", "TSLA", "PG", "KO", "BA", "CAT", "XOM", "CVX", "NEE", "AMT", "DIS", "T"]
# tickers = ["XOM"]

# base_url = "https://www.alphavantage.co/query"

# # Set date range: Start 3 years back from today
# end_date = datetime.now()
# start_date = end_date - timedelta(days=1095)  # 3 years ago

# # Define the output folder path
# output_folder = os.path.join("..", "data", "historical_data")

# # Ensure the output directory exists
# os.makedirs(output_folder, exist_ok=True)

# # Function to fetch sentiment data for each ticker
# def fetch_sentiment_data(ticker):
#     news_data = []
#     fetched_dates = set()
#     current_end_date = end_date

#     while start_date < current_end_date:
#         time_to = current_end_date.strftime('%Y%m%dT%H%M')
#         time_from = (current_end_date - timedelta(weeks=1)).strftime('%Y%m%dT%H%M')
        
#         params = {
#             "function": "NEWS_SENTIMENT",
#             "tickers": ticker,
#             "apikey": api_key,
#             "sort": "EARLIEST",
#             "limit": 1000,
#             "time_from": time_from,
#             "time_to": time_to
#         }
        
#         try:
#             response = requests.get(base_url, params=params)
#             response.raise_for_status()
#             data = response.json()
            
#             # Extract articles and avoid duplicates
#             for article in data.get("feed", []):
#                 timestamp = article.get("time_published")
                
#                 # Skip if the timestamp is already in fetched_dates
#                 if timestamp in fetched_dates:
#                     continue
                
#                 fetched_dates.add(timestamp)
                
#                 # Extract relevant data
#                 news_item = {
#                     "ticker": ticker,
#                     "title": article.get("title"),
#                     "url": article.get("url"),
#                     "time_published": timestamp,
#                     "overall_sentiment_score": article.get("overall_sentiment_score"),
#                     "overall_sentiment_label": article.get("overall_sentiment_label"),
#                     "ticker_sentiment_score": None,
#                     "ticker_sentiment_label": None
#                 }
                
#                 # Extract ticker-specific sentiment
#                 for sentiment in article.get("ticker_sentiment", []):
#                     if sentiment.get("ticker") == ticker:
#                         news_item["ticker_sentiment_score"] = sentiment.get("ticker_sentiment_score")
#                         news_item["ticker_sentiment_label"] = sentiment.get("ticker_sentiment_label")
#                         break
                
#                 news_data.append(news_item)
            
#             # Move the current_end_date one week back
#             current_end_date -= timedelta(weeks=1)
#             print(f"Fetched {len(news_data)} articles for {ticker} so far.")

#             # Respect API call limits
#             # time.sleep(1)

#         except requests.exceptions.RequestException as e:
#             print(f"An error occurred for {ticker}: {e}")
#             break

#     return news_data

# # Run the fetch for each ticker and save results
# for ticker in tickers:
#     print(f"Fetching data for {ticker}...")
#     ticker_news_data = fetch_sentiment_data(ticker)
    
#     # Create DataFrame
#     df = pd.DataFrame(ticker_news_data)
    
#     # Check if DataFrame is empty
#     if df.empty:
#         print(f"No articles fetched for {ticker}. Skipping.")
#         continue  # Skip to the next ticker if no data was fetched
    
#     # Sort by time_published in descending order (newest first)
#     df = df.sort_values(by="time_published", ascending=False)
    
#     # Save DataFrame to a CSV file specific to the ticker in the historical_data folder
#     file_name = os.path.join(output_folder, f"historical_sentiment_data_{ticker}.csv")
#     df.to_csv(file_name, index=False)
#     print(f"Sentiment data for {ticker} saved to {file_name}")

# print("Sentiment data fetching completed for all tickers.")





import requests
import pandas as pd
from dotenv import load_dotenv
import os
import time
from datetime import datetime, timedelta

# Load environment variables
load_dotenv()
api_key = os.getenv("ALPHA_VANTAGE_API_KEY")

# List of tickers from different sectors
# tickers = ["AAPL", "MSFT", "META", "BAC", "WFC", "C", "JNJ", "PFE", "AMZN", "TSLA", "PG", "KO", "BA", "CAT", "XOM", "CVX", "NEE", "AMT", "DIS", "T"]
tickers = ["META"]

# Define sectors for each ticker
sector_mapping = {
    "AAPL": "Technology",
    "MSFT": "Technology",
    "META": "Technology",
    "BAC": "Financials",
    "WFC": "Financials",
    "C": "Financials",
    "JNJ": "Healthcare",
    "PFE": "Healthcare",
    "AMZN": "Consumer Discretionary",
    "TSLA": "Consumer Discretionary",
    "PG": "Consumer Staples",
    "KO": "Consumer Staples",
    "BA": "Industrials",
    "CAT": "Industrials",
    "XOM": "Energy",
    "CVX": "Energy",
    "NEE": "Utilities",
    "AMT": "Real Estate",
    "DIS": "Communication Services",
    "T": "Communication Services"
}

base_url = "https://www.alphavantage.co/query"

# Set date range: Start 3 years back from today
end_date = datetime.now()
start_date = end_date - timedelta(days=1095)  # 3 years ago

# Define the output folder path
output_folder = os.path.join("..", "data", "historical_data")

# Ensure the output directory exists
os.makedirs(output_folder, exist_ok=True)

# Function to fetch sentiment data for each ticker
def fetch_sentiment_data(ticker):
    news_data = []
    fetched_dates = set()
    current_end_date = end_date
    sector = sector_mapping.get(ticker, "Unknown")  # Get sector for the ticker

    while start_date < current_end_date:
        time_to = current_end_date.strftime('%Y%m%dT%H%M')
        time_from = (current_end_date - timedelta(weeks=1)).strftime('%Y%m%dT%H%M')
        
        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": ticker,
            "apikey": api_key,
            "sort": "EARLIEST",
            "limit": 1000,
            "time_from": time_from,
            "time_to": time_to
        }
        
        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            # Extract articles and avoid duplicates
            for article in data.get("feed", []):
                timestamp = article.get("time_published")
                
                # Skip if the timestamp is already in fetched_dates
                if timestamp in fetched_dates:
                    continue
                
                fetched_dates.add(timestamp)
                
                # Extract relevant data
                news_item = {
                    "ticker": ticker,
                    "sector": sector, 
                    "title": article.get("title"),
                    "url": article.get("url"),
                    "time_published": timestamp,
                    "overall_sentiment_score": article.get("overall_sentiment_score"),
                    "overall_sentiment_label": article.get("overall_sentiment_label"),
                    "ticker_sentiment_score": None,
                    "ticker_sentiment_label": None
                }
                
                # Extract ticker-specific sentiment
                for sentiment in article.get("ticker_sentiment", []):
                    if sentiment.get("ticker") == ticker:
                        news_item["ticker_sentiment_score"] = sentiment.get("ticker_sentiment_score")
                        news_item["ticker_sentiment_label"] = sentiment.get("ticker_sentiment_label")
                        break
                
                news_data.append(news_item)
            
            # Move the current_end_date one week back
            current_end_date -= timedelta(weeks=1)
            print(f"Fetched {len(news_data)} articles for {ticker} so far.")

            # Respect API call limits
            # time.sleep(1)

        except requests.exceptions.RequestException as e:
            print(f"An error occurred for {ticker}: {e}")
            break

    return news_data

# Run the fetch for each ticker and save results
for ticker in tickers:
    print(f"Fetching data for {ticker}...")
    ticker_news_data = fetch_sentiment_data(ticker)
    
    # Create DataFrame
    df = pd.DataFrame(ticker_news_data)
    
    # Check if DataFrame is empty
    if df.empty:
        print(f"No articles fetched for {ticker}. Skipping.")
        continue  # Skip to the next ticker if no data was fetched
    
    # Sort by time_published in descending order (newest first)
    df = df.sort_values(by="time_published", ascending=False)
    
    # Save DataFrame to a CSV file specific to the ticker in the historical_data folder
    file_name = os.path.join(output_folder, f"historical_sentiment_data_{ticker}.csv")
    df.to_csv(file_name, index=False)
    print(f"Sentiment data for {ticker} saved to {file_name}")

print("Sentiment data fetching completed for all tickers.")