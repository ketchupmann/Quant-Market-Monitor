import pandas as pd
from supabase import create_client
import streamlit as st
import time
from datetime import datetime, timedelta, timezone 
from ingest_data import ingest_eod_data, ingest_minute_data

url = st.secrets["SUPABASE_URL"]
key = st.secrets["SUPABASE_KEY"]
supabase = create_client(url, key)


def get_eod_ticker_data(
    ticker: str, 
    one_month: bool = False, 
    half_yr: bool = False, 
    one_yr: bool = False, 
    five_yrs: bool = True
    ) -> pd.DataFrame:
    """
    Orchestrates the fetching of End-of-Day (EOD) data. It checks Supabase for existing
    records and intelligently triggers the ingestion pipeline only if data gaps 
    (historical or recent) are detected.

    Parameters:
    -----------
    ticker : str
        The stock symbol to fetch.
    one_month, half_yr, one_yr, five_yrs : bool
        Timeframe flags. Defaults to 5 years if no other flag is set.

    Returns:
    --------
    pd.DataFrame
        A DataFrame containing the EOD data, sorted by date. Returns empty DataFrame on error.
    """
    today = datetime.now()
    if one_month:
        start_date = today - pd.DateOffset(months=1)
    elif half_yr:
        start_date = today - pd.DateOffset(months=6)
    elif one_yr:
        start_date = today - pd.DateOffset(years=1)
    else:
        start_date = today - pd.DateOffset(years=5)

    # format as string for Supabase (YYYY-MM-DD)
    start_date_str = start_date.strftime('%Y-%m-%d')
    
    try:
        response = supabase.table('market_eod_data') \
            .select("date, open, high, low, close, volume, vwap, transactions") \
            .eq('ticker', ticker) \
            .gte('date', start_date_str) \
            .order('date') \
            .execute()
        
        df = pd.DataFrame(response.data)
        
        needs_ingestion = False
        fetch_start_date_str = start_date_str 
        
        if df.empty:
            # SCENARIO 1: Ticker has never been searched.
            needs_ingestion = True
            print(f"⚠️ No EOD data for {ticker}. Auto-ingesting from {fetch_start_date_str}...")
        else:
            # SCENARIO 2: Ticker exists, but missing recent data
            oldest_record = pd.to_datetime(df['date']).min()
            newest_record = pd.to_datetime(df['date']).max()
            target_start = pd.to_datetime(start_date_str)
            last_trading_day = (pd.Timestamp.now()).normalize()
            
            if newest_record < last_trading_day:
                needs_ingestion = True
                fetch_start_date_str = newest_record.strftime('%Y-%m-%d') 
                print(f"⚠️ Missing recent data. Updating strictly from {fetch_start_date_str} to today...")

            # SCENARIO 3: Ticker exist but missing historic data 
            # 5-day buffer so weekends/holidays don't trigger infinite re-fetching
            elif oldest_record > target_start + pd.Timedelta(days=5):
                needs_ingestion = True
                fetch_start_date_str = start_date_str 
                print(f"⚠️ Missing historical data for {ticker}. Fetching backwards to {fetch_start_date_str}...")

        if needs_ingestion:
            ingest_eod_data(ticker, fetch_start_date_str)

            response = supabase.table('market_eod_data') \
                .select("date, open, high, low, close, volume, vwap, transactions") \
                .eq('ticker', ticker) \
                .gte('date', start_date_str) \
                .order('date') \
                .execute()
            df = pd.DataFrame(response.data)
            
            if df.empty:
                print(f"❌ Auto-ingest failed or ticker {ticker} invalid.")
                return pd.DataFrame()

        return df.drop_duplicates(subset='date')
    
    except Exception as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame()


def get_minute_ticker_data(ticker: str, one_day: bool = False, one_week: bool = True) -> pd.DataFrame:
    """
    Pulls intraday minute data for a specific ticker from Supabase.
    It automatically fills zero-volume gaps (forward-filling price) to create 
    a continuous chart.

    Parameters:
    -----------
    ticker : str
        Stock symbol.
    one_day : bool
        If True, queries the last 24 hours.
    one_week : bool
        If True (default), queries the last 7 days.

    Returns:
    --------
    pd.DataFrame
        A DataFrame with a DatetimeIndex and continuous 1-minute intervals.
    """
    now = datetime.now(timezone.utc)

    if one_day:
        start_date_obj = now - pd.Timedelta(days=1)
    else:
        start_date_obj = now - pd.Timedelta(days=7)
    
    start_date = start_date_obj.strftime('%Y-%m-%d %H:%M:%S')

    try:
        response = supabase.table('market_data_minute') \
            .select("timestamp, open, high, low, close, volume, vwap, transactions") \
            .eq('ticker', ticker) \
            .gte('timestamp', start_date) \
            .order('timestamp') \
            .limit(100000) \
            .execute()
        
        df = pd.DataFrame(response.data)
        needs_ingestion = False
        fetch_start_date_str = None 
        
        if df.empty:
            needs_ingestion = True
            # always ingest 7 days initially
            fetch_start_date_str = (now - pd.Timedelta(days=7)).strftime('%Y-%m-%d')
            print(f"⚠️ No minute data for {ticker}. Auto-ingesting full week from {fetch_start_date_str}...")
        else:
            newest_record = pd.to_datetime(df['timestamp']).max()
                
            # Check Forward Gap:
            if newest_record < now - pd.Timedelta(minutes=16):
                needs_ingestion = True
                newest_record_est = newest_record.tz_convert('US/Eastern')
                
                fetch_start_date_str = newest_record_est.strftime('%Y-%m-%d')
                print(f"⚠️ Missing recent intraday data. Updating from {fetch_start_date_str}...")

        if needs_ingestion:
            ingest_minute_data(ticker, fetch_start_date_str)
            
            response = supabase.table('market_data_minute') \
                .select("timestamp, open, high, low, close, volume, vwap, transactions") \
                .eq('ticker', ticker) \
                .gte('timestamp', start_date) \
                .order('timestamp') \
                .limit(100000) \
                .execute()
            
            df = pd.DataFrame(response.data)
            
            if df.empty:
                print(f"❌ Auto-ingest failed for intraday {ticker}.")
                return pd.DataFrame()
        
        # clean up dataframe
        df = df.drop_duplicates(subset='timestamp')
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'vwap', 'transactions']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col])

        # fill gaps
        df = df.resample('1min').asfreq()
        df['close'] = df['close'].ffill()
        if 'vwap' in df.columns:
            df['vwap'] = df['vwap'].ffill()

        price_cols = ['open', 'high', 'low']
        for col in price_cols:
            if col in df.columns:
                df[col] = df[col].fillna(df['close'])

        vol_cols = ['volume', 'transactions']
        for col in vol_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0)

        return df

    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()

