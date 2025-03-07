import akshare as ak
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import itertools
import requests
import time
import akshare as ak
from collections import deque
from IPython.display import display, clear_output
from concurrent.futures import ThreadPoolExecutor, as_completed
import concurrent.futures
import math
import glob
import os
import ta
import pyarrow.feather as feather


def get_stock_data():
    """
    Fetches and filters A-share stock data, excluding ST stocks and specific stock codes.
    
    Returns:
        pd.DataFrame: Filtered stock data with a 'stockcode' column.
    """
    stock_df = ak.stock_zh_a_spot()
    stock_df = stock_df[~stock_df['名称'].str.contains("ST")]
    stock_df = stock_df[stock_df['代码'].str.startswith(("sh", "sz"))]
    stock_df = stock_df[~stock_df['代码'].str[2:4].isin(['30', '68'])]
    stock_df['stockcode'] = stock_df['代码'].str[2:]

    return stock_df


def load_latest_data(directory):
    """
    Load the latest valid Feather file from the given directory based on the date in the filename.

    Parameters:
    directory (str): Path to the directory containing Feather files.

    Returns:
    pd.DataFrame: The latest daily data as a DataFrame.
    str: The filename of the loaded file.
    """
    # Get all Feather files in the directory
    feather_files = [f for f in os.listdir(directory) if f.endswith('.feather')]

    valid_files = []

    # Extract valid dates from filenames
    for file in feather_files:
        try:
            date_str = file.split('_')[-1].split('.')[0]  # Extract date from filename
            file_date = datetime.strptime(date_str, "%Y-%m-%d")  # Convert to datetime
            valid_files.append((file_date, file))
        except ValueError:
            continue  # Skip files with invalid date format

    # Load the latest valid file
    if valid_files:
        latest_file = max(valid_files, key=lambda x: x[0])[1]  # Get filename with latest date
        latest_filepath = os.path.join(directory, latest_file)

        # Load the latest daily data
        previous_daily_data = pd.read_feather(latest_filepath)
        previous_daily_data['date'] = pd.to_datetime(previous_daily_data['date'])

        print(f"Loaded data from: {latest_filepath}")
        return previous_daily_data, latest_file

    else:
        print("No valid Feather files found in the directory.")
        return pd.DataFrame(), None  # Return an empty DataFrame if no files are found


def analyze_stock_data(df, roc_period=5, rsi_window=6, drop_percentage=5):
    # 1. Calculate body size, upper wick, and lower wick
    df['body_size'] = abs(df['close'] - df['open'])
    df['upper_wick'] = df['high'] - df[['close', 'open']].max(axis=1)
    df['lower_wick'] = df[['close', 'open']].min(axis=1) - df['low']
    df['long_wick'] = (
    (df['upper_wick'] > 1 * df['body_size']) |  # Upper wick is long
    (df['lower_wick'] > 4 * df['body_size'])    # Lower wick is long
    )   

    # 2. Price downtrend (ROC for close price)
    col_pricetrend = f'price_down_{drop_percentage}'
    df[col_pricetrend] = ta.momentum.ROCIndicator(df['close'], window=roc_period).roc()

    # 3. RSI (Relative Strength Index)
    rsi_column_name = f'rsi_{rsi_window}'
    df[rsi_column_name] = ta.momentum.RSIIndicator(df['close'], window=rsi_window).rsi()

    # 4. Add 'vol_trend' for volume trend (Rate of Change for volume)
    col_voltrend = f'vol_down_{drop_percentage}'
    df[col_voltrend] = ta.momentum.ROCIndicator(df['volume'], window=roc_period).roc()

    # Shift the vol_trend by 1 day to get the previous day's volume trend relative to hammer day
    df['prev_day_vol_trend'] = df[col_voltrend].shift(1)

    # 5. Add 'vol_spikes' for volume spike (ROC for volume)
    df['volspikes'] = ta.momentum.ROCIndicator(df['volume'], window=1).roc()

    # 6. Bullish Engulfing Pattern (body comparison between current and previous day)
    df['body_size_previous'] = abs(df['close'].shift(1) - df['open'].shift(1))

    # Remove rows where 'stockcode' is NaN (if required)
    df = df[df['stockcode'].notna()]

    return df


def condition_stocks(df):
    # Initialize the 'group' column with None
    df['group'] = None

    # Initialize individual condition dataframes
    df_bullish = df.copy()
    df_hammer = df.copy()
    df_insidebar = df.copy()
    df_momentum = df.copy()
    df_multiple_longwick = df.copy()

    # Bullish Engulfing Condition
    conditions_bullish = (
        (df_bullish['price_down_5'] <= -5) & 
        (df_bullish['rsi_6'] <= 39) & 
        (df_bullish['volspikes'] >= 30) &
        (df_bullish['close'].shift(1) < df_bullish['open'].shift(1)) & 
        (df_bullish['close'] > df_bullish['open']) & 
        (df_bullish['open'] < df_bullish['close'].shift(1)) & 
        (df_bullish['close'] > df_bullish['open'].shift(1)) & 
        (df_bullish['body_size'] > 1.3 * df_bullish['body_size'].shift(1))  # Bullish engulfing condition
    )
    df_bullish.loc[conditions_bullish, 'group'] = 'bullishgolf'

    # Hammer Candlestick Condition
    conditions_hammer = (
        (df_hammer['price_down_5'] <= -5) & 
        (df_hammer['rsi_6'] <= 39) & 
        (df_hammer['prev_day_vol_trend'] <= -4) & 
        (df_hammer['volspikes'] >= 50) &
        (df_hammer['lower_wick'] >= 2 * df_hammer['body_size']) & 
        (df_hammer['upper_wick'] <= 0.9 * df_hammer['body_size']) & 
        (df_hammer['body_size'] > 0)  # Hammer candlestick pattern
    )
    df_hammer.loc[conditions_hammer, 'group'] = 'hammer'

    # Inside Bar Condition
    conditions_insidebar = (
        (df_insidebar['price_down_5'] <= -5) & 
        (df_insidebar['rsi_6'] <= 38) &
        (df_insidebar['high'] < df_insidebar['high'].shift(1)) & 
        (df_insidebar['low'] > df_insidebar['low'].shift(1)) &
        (df_insidebar['close'].shift(1) < df_insidebar['open'].shift(1)) &  # Day 1 is bearish
        (df_insidebar['close'] > df_insidebar['open']) &  # Day 2 is bullish
        (df_insidebar['close'] < df_insidebar['open'].shift(1)) &  # Day 2 close < Day 1 open
        (df_insidebar['close'].shift(1) < df_insidebar['open'])  # Day 1 close < Day 2 open
    )
    df_insidebar.loc[conditions_insidebar, 'group'] = 'insidebar'

    # Momentum Candle Condition
    conditions_momentum = (
        (df_momentum['price_down_5'] <= -4) & 
        (df_momentum['rsi_6'] <= 39) & 
        (df_momentum['close'] > df_momentum['open']) & 
        (df_momentum['close'].shift(1) < df_momentum['open'].shift(1)) & 
        (df_momentum['body_size'] > 1.5 * df_momentum['body_size'].shift(1)) & 
        ((df_momentum['close'] - df_momentum['close'].shift(1)) / df_momentum['close'].shift(1) * 100 >= 2.2)  # Momentum candle condition
    )
    df_momentum.loc[conditions_momentum, 'group'] = 'momentum'

    # Multiple Long Wick Condition
    conditions_multiple_long_wick = (
        (df_multiple_longwick['price_down_5'] <= -5) & 
        (df_multiple_longwick['rsi_6'] <= 39) &
        ((df_multiple_longwick['upper_wick'] > 1 * df_multiple_longwick['body_size']) | 
         (df_multiple_longwick['lower_wick'] > 4 * df_multiple_longwick['body_size'])) &  # Long wick condition
        (df_multiple_longwick['long_wick'] & df_multiple_longwick['long_wick'].shift(1) & df_multiple_longwick['long_wick'].shift(2)) &  # Consecutive long wick candles
        ((df_multiple_longwick['close'] / df_multiple_longwick['close'].shift(1) - 1).abs() <= 0.03) &
        ((df_multiple_longwick['close'] / df_multiple_longwick['close'].shift(2) - 1).abs() <= 0.03)  # Price range condition
    )
    df_multiple_longwick.loc[conditions_multiple_long_wick, 'group'] = 'multiplelongwick'

    # Filter only the rows that satisfy each condition and add a new 'condition' column to track
    result_bullish = df_bullish[['date', 'stockcode', 'stockname', 'group']].dropna(subset=['group'])

    result_hammer = df_hammer[['date', 'stockcode', 'stockname', 'group']].dropna(subset=['group'])

    result_insidebar = df_insidebar[['date', 'stockcode', 'stockname', 'group']].dropna(subset=['group'])

    result_momentum = df_momentum[['date', 'stockcode', 'stockname', 'group']].dropna(subset=['group'])

    result_multiple_longwick = df_multiple_longwick[['date', 'stockcode', 'stockname', 'group']].dropna(subset=['group'])

    # Combine all the datasets into one
    combined_result = pd.concat([result_bullish, result_hammer, result_insidebar, result_momentum, result_multiple_longwick], axis=0)

    # Reset the index of the combined result DataFrame
    combined_result = combined_result.reset_index(drop=True)

    return combined_result


def fetch_daily_data(code, name, start_date, end_date, max_retries=4):
    """Fetch daily data for a single stock with retries and exponential backoff."""
    wait_time = 2  # Initial wait time

    for attempt in range(1, max_retries + 1):
        try:
            daily_data = ak.stock_zh_a_daily(symbol=code, start_date=start_date, end_date=end_date)
            if not daily_data.empty:
                daily_data['stockcode'] = code
                daily_data['stockname'] = name
                return daily_data[['stockcode', 'stockname', 'date', 'open', 'close', 'high', 'low', 'volume', 'amount', 'outstanding_share', 'turnover']]
        except (requests.exceptions.ConnectionError, ConnectionResetError) as e:
            print(f"Attempt {attempt} failed for {code} ({name}): {e}")
            time.sleep(wait_time)  # Exponential backoff
            wait_time *= 2  
        except Exception as e:
            print(f"Unexpected error for {code} ({name}): {e}")
            break  

    print(f"Failed to fetch {code} ({name}) after {max_retries} attempts.")
    return None

def get_daily_data(stock_df, start_date, end_date, max_workers=10):
    """Fetch daily data for all stocks concurrently with retries, displaying only the last 5 fetched stocks and count."""
    stock_daily = []
    error_codes = []
    counter = itertools.count(1)  # Thread-safe counter
    recent_stocks = deque(maxlen=5)  # Store only the last 5 fetched stocks

    total_stocks = len(stock_df)  # Total number of stocks to fetch

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_daily_data, row['代码'], row['名称'], start_date, end_date): (row['代码'], row['名称'])
                   for _, row in stock_df.iterrows()}

        for future in as_completed(futures):
            result = future.result()
            code, name = futures[future]
            fetched_count = next(counter)  # Increment fetched count

            if result is not None:
                stock_daily.append(result)
                recent_stocks.append(f"Stock {fetched_count}: {code} ({name})")

            else:
                error_codes.append(code)  

            # Clear previous outputs and print only the latest 5 stocks
            clear_output(wait=True)  # Clears the cell output in Jupyter
            print(f"Fetching stocks: {fetched_count}/{total_stocks}")  # Show fetching progress
            print("Latest 5 fetched stocks:")
            for stock in recent_stocks:
                print(stock)

    # Combine results
    combined_data = pd.concat(stock_daily, ignore_index=True) if stock_daily else pd.DataFrame()

    return combined_data, error_codes

def export_daily_stock_data(previous_daily_data, daily_data_latest, save_directory, end_date):
    """
    Combines previous daily stock data with newly fetched data and exports to a Feather file.

    Parameters:
    - previous_daily_data (pd.DataFrame): Existing daily stock data.
    - daily_data_latest (pd.DataFrame): Newly fetched daily stock data.
    - save_directory (str): Directory to save the Feather file.
    - end_date (str): Date string (YYYY-MM-DD) for the file naming.

    Returns:
    - str: Path to the saved file if successful, else None.
    """
    if daily_data_latest.empty:
        print("No new data fetched. Skipping export.")
        return None
    
    # Combine previous and new daily data
    all_daily_data = pd.concat([previous_daily_data, daily_data_latest], ignore_index=True)

    # Ensure 'date' column is in string format
    all_daily_data['date'] = pd.to_datetime(all_daily_data['date']).astype(str)
    
    # Construct file path with your naming convention
    filename = f"{save_directory}/daily_data_{end_date}.feather"
    
    # Save to Feather format
    feather.write_feather(all_daily_data, filename)
    
    print(f"✅ Saved the combined data to: {filename}")
    return filename
