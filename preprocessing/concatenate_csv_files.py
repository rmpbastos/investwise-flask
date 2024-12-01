import pandas as pd
import os

# List of tickers to process
tickers = ["AAPL", "MSFT", "META", "JNJ", "PFE", "AMZN", "TSLA", "PG", "KO", "BA", "CAT", "CVX", "NEE", "AMT", "T"]

# Initialize an empty list to hold each ticker's data
all_data = []

# Loop through each ticker and read the corresponding CSV file
for ticker in tickers:
    file_name = f"training_data_{ticker}.csv"
    
    # Check if the file exists
    if os.path.exists(file_name):
        # Read the CSV file and append it to the list
        df = pd.read_csv(file_name)
        
        # Optionally, add a column to identify the ticker (useful if you want to know the source ticker in the combined data)
        df['ticker'] = ticker
        
        all_data.append(df)
    else:
        print(f"File {file_name} not found. Skipping.")

# Concatenate all DataFrames in the list into a single DataFrame
combined_data = pd.concat(all_data, ignore_index=True)

# Save the combined data to a new CSV file
combined_data.to_csv("combined_training_data.csv", index=False)
print("All training data combined and saved to combined_training_data.csv")
