import os
import pandas as pd
from dotenv import load_dotenv
from supabase import create_client
from backend_monitor import get_historic_ticker_data


load_dotenv()

url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")
supabase = create_client(url, key)

def ingest_eod_data(ticker: str, start_date: str) -> None: 
    """
    Fetches historical End-of-Day (EOD) data for a given ticker and upserts 
    it into the Supabase 'market_eod_data' table.

    Parameters:
    -----------
    ticker : str
        The stock symbol to fetch data for (e.g., 'AAPL').
    start_date : str
        The start date for the data fetch, formatted as 'YYYY-MM-DD'.

    Returns:
    --------
    None
        This function performs a database execution and returns nothing.

    Raises:
    -------
    ValueError
        If the data fetching function returns an empty or null DataFrame.
    """
    df = get_historic_ticker_data(ticker, time_unit='day', start_date=start_date)
    if df is None or df.empty:
        raise ValueError(f"No EOD data returned for {ticker} starting {start_date}.")
    
    df.reset_index(inplace=True) #prep for Supabase upload
    #preprocess the timestamp column so 5:00:00 doesn't show up on every value of date when uploading(only doing it for EOD data for 1 month, 1 year and 5 years)
    df['date'] = df['timestamp'].dt.strftime('%Y-%m-%d') 
    final_df = df[['ticker', 'date', 'open', 'high', 'low', 'close', 'volume', 'vwap', 'transactions']] #rename column to match Supabase

    data_payload = final_df.to_dict(orient='records') # converts to list of dict for upload
    
    # Upload
    supabase.table('market_eod_data').upsert(data_payload).execute()


def ingest_minute_data(ticker: str, start_date: str) -> None: 
    """
    Fetches historical minute-by-minute data for a given ticker and upserts 
    it into the Supabase 'market_data_minute' table.

    Parameters:
    -----------
    ticker : str
        The stock symbol to fetch data for (e.g., 'AAPL').
    start_date : str
        The start date for the data fetch, formatted as 'YYYY-MM-DD'.

    Returns:
    --------
    None
        This function performs a database execution and returns nothing.

    Raises:
    -------
    ValueError
        If the data fetching function returns an empty or null DataFrame.
    """
    df = get_historic_ticker_data(ticker, time_unit='minute', start_date=start_date)
    if df is None or df.empty:
        raise ValueError(f"No minute data returned for {ticker} starting {start_date}.")
    
    df.reset_index(inplace=True) #reset timestamp to be just a column to prep for Supabase upload
    #rename column to match Supabase
    final_df = df[['ticker', 'timestamp', 'open', 'high', 'low', 'close', 'volume', 'vwap', 'transactions']] 

    final_df['timestamp'] = final_df['timestamp'].astype(str)

    data_payload = final_df.to_dict(orient='records') # converts to list of dict for upload
    # Upload
    supabase.table('market_data_minute').upsert(data_payload).execute()


def ingest_week_data(ticker, start_date): # start date is written as 'YYYY-MM-DD', only called by frontend as series of weekly for 5 yrs
    df = get_historic_ticker_data(ticker, 'week', start_date=start_date)
    if df is None or df.empty:
        print(f"No data for {ticker}")
        return
    
    df.reset_index(inplace=True) #reset timestamp to be just a column to prep for Supabase upload
    #preprocess the timestamp column so 5:00:00 doesn't show up on every value of date when uploading(only doing it for EOD and weekly data for 1 month and 1 year)
    df['date'] = df['timestamp'].dt.strftime('%Y-%m-%d') 
    final_df = df[['ticker', 'date', 'open', 'high', 'low', 'close', 'volume', 'vwap', 'transactions']] #rename column to match Supabase

    data_payload = final_df.to_dict(orient='records') # converts to list of dict for Supabase upload
    
    # Upload
    #supabase.table('market_data_weely').upsert(data_payload).execute()

