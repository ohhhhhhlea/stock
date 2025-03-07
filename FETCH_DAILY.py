import akshare as ak
import pandas as pd
import numpy as np
import fastparquet
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
import pyarrow.feather as feather
import FUNCTIONS as ft

directory = "/Users/lulu/Documents/STOCK/DATASET/DAILY/"

# Load previous daily data
previous_daily_data, latest_filename = ft.load_latest_data(directory)

stock_df = ft.get_stock_data()

# Set start and end dates for fetching new data
max_date = previous_daily_data['date'].max()
next_day = pd.to_datetime(max_date) + timedelta(days=1)
start_date = next_day.strftime('%Y-%m-%d')
end_date = datetime.today().strftime('%Y-%m-%d')

# Fetch daily data
daily_data, error_codes = ft.get_daily_data(stock_df, start_date, end_date)

# Check if error_codes is not empty
if error_codes:
    # Filter stock_df for codes in error_codes
    error_daily_df = stock_df[stock_df['代码'].isin(error_codes)]
    
    # Fetch daily data for the stocks in error_codes
    daily_data_error, error_daily_2 = ft.get_daily_data(error_daily_df, start_date, end_date)
else:
    print("No error codes found, skipping data fetch.")

daily_data_latest = pd.concat([daily_data, daily_data_error], ignore_index=True)
daily_data_latest

error_daily_2


# Export stock data
filename = ft.export_daily_stock_data(previous_daily_data, pd.concat([daily_data, daily_data_error], ignore_index=True), directory, end_date)

