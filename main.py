import streamlit as st
from datetime import datetime, timedelta
import FUNCTIONS as ft
import pandas as pd
import os
import pyarrow.feather as feather

# Set wide layout
st.set_page_config(
    page_title="STOCK",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Directory containing daily stock data
DATA_DIR = "/Users/lulu/Documents/STOCK/DATASET/DAILY/"

# Sidebar for date range selection
st.sidebar.title("Select Date Range")
start_date = st.sidebar.date_input("Start Date", datetime.today() - timedelta(days=30))
end_date = st.sidebar.date_input("End Date", datetime.today())

# Fetch button
if st.sidebar.button("Fetch Data"):
    df, filepath = ft.load_latest_data(DATA_DIR)

    if df is not None:
        # Run stock analysis
        analyzed_df = ft.analyze_stock_data(df)
        result = ft.condition_stocks(analyzed_df)

        # Convert date column to datetime format
        result['date'] = pd.to_datetime(result['date'])
        start_date_dt = pd.to_datetime(start_date)
        end_date_dt = pd.to_datetime(end_date)

        # Filter data based on selected date range
        filtered_df = result[(result['date'] >= start_date_dt) & (result['date'] <= end_date_dt)]

        # Store filtered data in session state
        st.session_state['filtered_df'] = filtered_df
        st.session_state['save_success'] = False  # Reset success message state

# Display data and "Save to CSV" button only if data exists
if 'filtered_df' in st.session_state and not st.session_state['filtered_df'].empty:
    save_path = f"/Users/lulu/Documents/STOCK/DAILY_STOCK/STOCK_{datetime.today().strftime('%Y-%m-%d')}.csv"

    # Container for dataset and button
    with st.container():
        st.write("### Filtered Stock Analysis Data")
        st.dataframe(st.session_state['filtered_df'][['date', 'stockcode', 'stockname', 'group']], 
                     use_container_width=True, hide_index=True)

        # Create a bottom-left-aligned button
        col1, _ = st.columns([0.2, 0.8])  # Left column for button, right is empty for alignment

        with col1:
            if st.button("💾 Save to CSV"):
                st.session_state['filtered_df'].to_csv(save_path, index=False, encoding='utf-8-sig')
                st.session_state['save_success'] = True  # Set success flag

    # Full-width success message
    if st.session_state.get('save_success', False):
        st.success(f"✅ Data saved to: {save_path}")