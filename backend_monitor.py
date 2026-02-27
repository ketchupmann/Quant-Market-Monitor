from polygon import RESTClient
import pandas as pd
import time
from datetime import datetime, timedelta 
from supabase import create_client, Client
import os
from dotenv import load_dotenv


load_dotenv()
api_key = os.getenv("POLYGON_API_KEY")
client = RESTClient(api_key)
url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(url, key)



def get_ticker_details(ticker: str) -> dict:
    """
    Fetches fundamental company metadata and branding assets from Massive.

    Parameters:
    -----------
    ticker : str
        The stock symbol to query.

    Returns:
    --------
    dict
        A dictionary containing company details such as full name, market cap,
        primary exchange, description, and the URL to the company's logo.
    """
    details = client.get_ticker_details(
	ticker,)
    if details.branding:
        icon = details.branding.icon_url
    else:
        icon = None

    company_full_name = details.name
    currency = details.currency_name
    description = details.description
    homepage_link = details.homepage_url
    location = details.locale
    market = details.market
    market_cap = details.market_cap
    exchange = details.primary_exchange
    
    
    return {
        "ticker": ticker,
        "name": company_full_name,
        "market_cap": market_cap,
        "exchange": exchange,
        "market": market,
        "currency": currency,
        "website": homepage_link,
        "description": description,
        "location": location,
        "icon_url": icon
    }

def get_snapshot_ticker(ticker: str) -> dict:
    """
    Retrieves a real-time (15-minute delayed) snapshot of the current
    trading day for a specific ticker.

    Parameters:
    -----------
    ticker : str
        The stock symbol to query.

    Returns:
    --------
    dict
        A dictionary containing daily performance metrics (open, high, low, close,
        volume, vwap, daily change) and the most recent minute-bar statistics.
    """
    snapshot = client.get_snapshot_ticker("stocks", ticker)
    day = snapshot.day
    latest_min = snapshot.min
    dt_obj = pd.to_datetime(latest_min.timestamp, unit='ms')

    return {'ticker':ticker, 
            'close':day.close,
            'high':day.high,
            'low':day.low,
            'open':day.open,
            'volume':day.volume,
            'vwap':day.vwap,
            'todays_change':snapshot.todays_change,
            'todays_change_percent':snapshot.todays_change_percent,
            'min_close':latest_min.close,
            'min_open':latest_min.open,
            'min_high':latest_min.high,
            'min_low':latest_min.low,
            'min_volume':latest_min.volume,
            'min_vwap':latest_min.vwap,
            'min_transactions':latest_min.transactions,
            'time':dt_obj.floor('min') 
    }

def get_historic_ticker_data(
    ticker: str,
    time_unit: str, 
    start_date: str = None, 
    end_date: str = None, 
    target_rows: int = None
    ) -> pd.DataFrame: #start/end_date is written as 'YYYY-MM-DD'
    """
    Fetches historical aggregate bars (minute or daily) from the Polygon API.
    Includes logic to safely estimate required calendar days if a specific
    target number of trading rows is requested.

    Parameters:
    -----------
    ticker : str
        The stock symbol to query.
    time_unit : str
        The timespan multiplier unit (e.g., 'minute', 'day').
    start_date : str, optional
        The start date for the query formatted as 'YYYY-MM-DD'.
    end_date : str, optional
        The end date for the query formatted as 'YYYY-MM-DD'. Defaults to today.
    target_rows : int, optional
        The specific number of trading bars to return. If provided, dynamically calculates 
        and overrides the start_date to account for weekends and holidays.

    Returns:
    --------
    pd.DataFrame
        A DataFrame indexed by datetime containing OHLCV and VWAP data.
        Returns an empty DataFrame if no data is found or inputs are invalid.
    """
    if end_date is None:
        end_date = pd.Timestamp.now(tz='US/Eastern').strftime('%Y-%m-%d')

    if target_rows:
        # need more calendar days than trading days (weekends/holidays).
        # Multiplier 1.65 is safe (252 trading days * 1.65 = ~415 calendar days)
        buffer_days = int(target_rows * 1.65) + 10 
        start_date = end_date - timedelta(days=buffer_days)
        start_date_str = start_date.strftime('%Y-%m-%d')
    
    else:
        if start_date is None:
            print("Error: Must provide either start_date or target_rows")
            return pd.DataFrame()
        start_date_str = start_date
    if not isinstance(end_date, str):
        end_date_str = end_date.strftime('%Y-%m-%d')
    else:
        end_date_str = end_date

    aggs = []
    results = client.list_aggs(
            ticker=ticker,
            adjusted=True, 
            sort='asc',
            multiplier=1, 
            timespan=time_unit, 
            from_=start_date_str, 
            to=end_date_str, 
            limit=50000)
    for bar in results:
        aggs.append({
            "timestamp": bar.timestamp,
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
            "volume": bar.volume,
            "vwap": bar.vwap,
            "transactions": bar.transactions
        })
    df = pd.DataFrame(aggs)

    #Convert timestamp to readable date and index it
    if not df.empty:
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['ticker'] = ticker
        df.set_index('timestamp', inplace=True)
        df['volume'] = df['volume'].fillna(0).astype(int)
        if target_rows and len(df) > target_rows:
            df = df.tail(target_rows)
        
    return df




def get_financial_statements(ticker, limit=5):
    """
    Fetches the 3 major financial statements (Income, Balance Sheet, Cash Flow)
    and converts them into clean Pandas DataFrames.
    
    Args:
        ticker (str): Stock symbol (e.g., "AAPL")
        limit (int): How many periods to fetch (default 5 quarters/years)
        
    Returns:
        dict: Keys are 'income', 'balance', 'cash'. Values are DataFrames.
    """
    results = client.vx.list_stock_financials(
        ticker=ticker,
        limit=limit,
        timeframe='quarterly', 
        sort='period_of_report_date',
        order='desc' # Newest first
    )
    
    # 2. Containers for our data
    income_data = []
    balance_data = []
    cash_data = []
    
    # 3. Loop through each filing (quarter)
    for filing in results:
        report_date = filing.period_of_report_date
        financials = filing.financials
        
        # Helper to extract value safely
        def get_val(category, key):
            # polygon nests data like: financials.income_statement.revenues.value
            try:
                section = getattr(financials, category, None)
                item = getattr(section, key, None)
                return item.value if item else 0
            except:
                return 0

        # income statement
        income_data.append({
            "Date": report_date,
            "Revenue": get_val('income_statement', 'revenues'),
            "Gross Profit": get_val('income_statement', 'gross_profit'),
            "Op Income": get_val('income_statement', 'operating_expenses'),
            "Net Income": get_val('income_statement', 'net_income_loss_attributable_to_parent'),
            "EPS": get_val('income_statement', 'basic_earnings_per_share')
        })
        
        # balance sheet
        balance_data.append({
            "Date": report_date,
            "Total Assets": get_val('balance_sheet', 'assets'),
            "Current Assets": get_val('balance_sheet', 'current_assets'),
            "Total Liab": get_val('balance_sheet', 'liabilities'),
            "Current Liab": get_val('balance_sheet', 'current_liabilities'),
            "Equity": get_val('balance_sheet', 'equity')
        })
        
        # cash flow
        cash_data.append({
            "Date": report_date,
            "Operating Cash": get_val('cash_flow_statement', 'net_cash_flow_from_operating_activities'),
            "Investing Cash": get_val('cash_flow_statement', 'net_cash_flow_from_investing_activities'),
            "Financing Cash": get_val('cash_flow_statement', 'net_cash_flow_from_financing_activities'),
            "Free Cash Flow": get_val('cash_flow_statement', 'net_cash_flow_from_operating_activities') - 
                              abs(get_val('cash_flow_statement', 'net_cash_flow_from_investing_activities')) 
                              
        })

    # 4. Convert to DataFrames and Transpose for "Column View"
    def clean_df(data_list):
        if not data_list: return pd.DataFrame()
        df = pd.DataFrame(data_list)
        df.set_index("Date", inplace=True)
        return df.T # Transpose so Dates are columns (Better for comparisons)

    return {
        "income": clean_df(income_data),
        "balance": clean_df(balance_data),
        "cash": clean_df(cash_data)
    }


   