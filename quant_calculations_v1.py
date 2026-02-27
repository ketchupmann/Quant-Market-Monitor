import numpy as np
import pandas as pd
from backend_monitor import get_historic_ticker_data

def get_close(ticker_df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Extracts and isolates the 'close' price column from a ticker's DataFrame, 
    renaming it to the ticker symbol for cleaner downstream joining.

    Parameters:
    -----------
    ticker_df : pd.DataFrame
        DataFrame containing historical price data, must include a 'close' column.
    ticker : str
        The stock symbol (e.g., 'AAPL') used to rename the isolated column.

    Returns:
    --------
    pd.DataFrame
        A single-column DataFrame indexed by date, with the column named as the ticker.
    
    Raises:
    -------
    KeyError
        If the 'close' column is missing from the provided DataFrame.
    """
    ticker_df.columns = ticker_df.columns.str.strip()
    
    if 'close' not in ticker_df.columns:
        raise KeyError(f"Column 'close' not found for {ticker}")

    # Isolate the close column as a copy to prevent SettingWithCopy warnings
    close_df = ticker_df[['close']].copy()
    
    # Rename column to ticker symbol
    close_df.columns = [ticker]
    return close_df


def combine_closed_stitcher(ticker1_df: pd.DataFrame, ticker1: str, ticker2_df: pd.DataFrame, ticker2: str) -> pd.DataFrame:
    """
    Extracts the closing prices of two assets and joins them into a single 
    DataFrame, perfectly aligning them by Date/Time index.

    Parameters:
    -----------
    ticker1_df : pd.DataFrame
        Historical price DataFrame for the primary asset.
    ticker1 : str
        Symbol for the primary asset.
    ticker2_df : pd.DataFrame
        Historical price DataFrame for the secondary asset.
    ticker2 : str
        Symbol for the secondary asset.

    Returns:
    --------
    pd.DataFrame
        A combined DataFrame with two columns (ticker1, ticker2) aligned by index.
        
    Raises:
    -------
    ValueError
        If either input DataFrame is None or empty.
    """
    if ticker1_df is None or ticker1_df.empty or ticker2_df is None or ticker2_df.empty:
       raise ValueError("Incomplete data provided. Cannot stitch empty DataFrames.")

    close1_df = get_close(ticker1_df, ticker1)
    close2_df = get_close(ticker2_df, ticker2)
    
    if close1_df is not None and close2_df is not None:
        # join='inner' automatically aligns the Datetime Indexes perfectly and ignores mismatched shapes
        combined_df = pd.concat([close1_df, close2_df], axis=1, join='inner')
        return combined_df
    else:
        return None

def get_period_volatility(df: pd.DataFrame) -> float:
    """
    Calculates the annualized historical volatility of an asset based on daily closing prices.
    
    Formula: $\sigma_{annual} = \sigma_{daily} \times \sqrt{252}$

    Parameters:
    -----------
    df : pd.DataFrame
        Historical price DataFrame containing a 'close' column.

    Returns:
    --------
    float
        The annualized standard deviation of daily returns.
    """
    daily_returns = df['close'].pct_change().dropna()
    return daily_returns.std() * np.sqrt(252)

def get_max_drawdown(df: pd.DataFrame) -> float:
    """
    Calculates the Maximum Drawdown (MDD), representing the largest peak-to-trough 
    percentage drop in the asset's history.

    Parameters:
    -----------
    df : pd.DataFrame
        Historical price DataFrame containing a 'close' column.

    Returns:
    --------
    float
        The maximum drawdown as a negative decimal (e.g., -0.25 represents a 25% drop).
    """
    cum_returns = (1 + df['close'].pct_change().dropna()).cumprod()
    running_max = cum_returns.cummax()
    drawdown = (cum_returns / running_max) - 1
    return drawdown.min()

def calculate_sharpe_ratio(df: pd.DataFrame, risk_free_rate: float = 0.04) -> float:
    """
    Calculates the Annualized Sharpe Ratio to measure risk-adjusted return.
    
    Formula: $Sharpe = \frac{R_p - R_f}{\sigma_p}$

    Parameters:
    -----------
    df : pd.DataFrame
        Historical price DataFrame containing a 'close' column.
    risk_free_rate : float, optional
        The assumed risk-free rate of return (default is 0.04 or 4%).

    Returns:
    --------
    float
        The calculated Sharpe Ratio. Returns 0 if volatility is 0 to prevent ZeroDivisionError.
    """
    returns = df['close'].pct_change().dropna()
    annualized_return = returns.mean() * 252
    annualized_vol = returns.std() * np.sqrt(252)
    
    if annualized_vol == 0:
        return 0.0
    return (annualized_return - risk_free_rate) / annualized_vol

def calculate_beta_alpha(ticker_df: pd.DataFrame, market_df: pd.DataFrame, risk_free_rate: float = 0.04) -> tuple:
    """
    Calculates the CAPM Beta (systematic risk) and Alpha (excess return) of an asset 
    relative to a market benchmark.

    Parameters:
    -----------
    ticker_df : pd.DataFrame
        Historical price DataFrame for the target asset containing a 'close' column.
    market_df : pd.DataFrame
        Historical price DataFrame for the market benchmark (e.g., SPY) containing a 'close' column.
    risk_free_rate : float, optional
        The assumed risk-free rate of return (default is 0.04).

    Returns:
    --------
    tuple (float, float)
        A tuple containing (Beta, Alpha).
    """
    # Inner join aligns the dates perfectly, dropping any mismatched days
    combined = pd.concat([ticker_df['close'], market_df['close']], axis=1, join='inner')
    combined.columns = ['Ticker', 'Market']
    
    returns = combined.pct_change().dropna()
    
    # Beta
    covariance = returns['Ticker'].cov(returns['Market'])
    variance = returns['Market'].var()
    beta = covariance / variance if variance != 0 else 0
    
    # Alpha (CAPM)
    annual_stock_return = returns['Ticker'].mean() * 252
    annual_market_return = returns['Market'].mean() * 252
    expected_return = risk_free_rate + beta * (annual_market_return - risk_free_rate)
    alpha = annual_stock_return - expected_return
    
    return beta, alpha

def calculate_correlation_signal(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    ticker1: str,
    ticker2: str,
    window: int = 50, #for corr
    wide_window: int = 100, #for moving avg corr
    std_multiplier: float = 2.2 # 3.0 used for catching extreme outliers, industry standard. I am using 2.2 because 3.0 is too aggressive -> no sell signals
) -> pd.DataFrame:
    """
    Computes a rolling correlation between two assets and generates statistical arbitrage 
    signals based on standard deviation band breakouts.

    Parameters:
    -----------
    df1 : pd.DataFrame
        Historical price data for the primary asset.
    df2 : pd.DataFrame
        Historical price data for the comparison asset.
    ticker1 : str
        Symbol for the primary asset.
    ticker2 : str
        Symbol for the comparison asset.
    window : int, optional
        Number of periods used for computing the short rolling correlation (default is 50).
    wide_window : int, optional
        Number of periods used for computing the moving average correlation baseline (default is 100).
    std_multiplier : float, optional
        Standard deviation multiplier to set the upper and lower bands. Default is 2.2 
        (adjusted from 3.0 to generate a higher frequency of viable signals).

    Returns:
    --------
    pd.DataFrame
        DataFrame indexed by date containing:
        - 'rolling_corr': Short-window rolling Pearson correlation.
        - 'avg_corr': Wide-window rolling average of the correlation.
        - 'upper_threshold': avg_corr + (std_multiplier * std_corr).
        - 'lower_threshold': avg_corr - (std_multiplier * std_corr).
        - 'signal': 1 (expect upward reversion), -1 (expect downward reversion), or 0.
    """
    close_df = combine_closed_stitcher(df1, ticker1, df2, ticker2)

    if close_df is None or close_df.empty:
        raise ValueError("Stitcher failed to combine DataFrames. Cannot calculate correlation.")

    for col in [ticker1, ticker2]:
        if col not in close_df.columns:
            raise ValueError(f'{col} close data could not be found in DataFrame')

    # get Pearson correlation by default
    close_df['rolling_corr'] = close_df[ticker1].rolling(window=window, min_periods=window).corr(close_df[ticker2])

    close_df['avg_corr'] = close_df['rolling_corr'].rolling(window=wide_window, min_periods=10).mean()
    close_df['std_corr'] = close_df['rolling_corr'].rolling(window=wide_window, min_periods=10).std() 

    close_df['upper_threshold'] = close_df['avg_corr'] + (std_multiplier * close_df['std_corr'])
    close_df['lower_threshold'] = close_df['avg_corr'] - (std_multiplier * close_df['std_corr'])

    # initialize signal to be zero
    close_df['signal'] = 0

    # If correlation is above the upper threshold, set signal to -1 (expecting revert downward)
    close_df.loc[close_df['rolling_corr'] > close_df['upper_threshold'], 'signal'] = -1
    
    # If correlation is below the lower threshold, set signal to 2 (expecting revert upward)
    close_df.loc[close_df['rolling_corr'] < close_df['lower_threshold'], 'signal'] = 1

    return close_df[['rolling_corr', 'avg_corr', 'upper_threshold', 'lower_threshold', 'signal']]
    

def get_correlation_engine(prices_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates a Pearson correlation matrix for a basket of multiple assets, 
    calculated on normalized daily returns rather than raw prices.

    Parameters:
    -----------
    prices_df : pd.DataFrame
        DataFrame of closing prices where the index is Dates and columns are Ticker symbols.

    Returns:
    --------
    pd.DataFrame
        A square correlation matrix where index and columns are both Ticker symbols.
    """
    # calculate Daily Returns (Percentage Change)
    returns_df = prices_df.pct_change().dropna()

    # Generate the Correlation Matrix (Pearson)
    # Result is a square matrix (e.g., NVDA vs TSLA = 0.65)
    corr_matrix = returns_df.corr()
    
    return corr_matrix

def get_rsi(close_df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    Calculates the Relative Strength Index (RSI) momentum oscillator.

    Formula: $RSI = 100 - /frac{100}{1 + RS}$

    Parameters:
    -----------
    close_df : pd.DataFrame
        Historical price DataFrame containing a 'close' column.
    window : int, optional
        The lookback period for the Wilder's Smoothing Moving Average (default is 14).

    Returns:
    --------
    pd.DataFrame
        A DataFrame indexed by date containing a single 'rsi' column.
        
    Raises:
    -------
    ValueError
        If the input DataFrame is empty or missing the 'close' column.
    """
    if close_df is None or close_df.empty:
        raise ValueError("Data empty, cannot calculate rsi.")
    
    if 'close' not in close_df.columns:
        raise ValueError(f'close data could not be found in DataFrame')

    close_df['daily_change'] = close_df['close'].diff()
    
    # Keep positives, turn negatives to 0
    close_df['gain'] = close_df['daily_change'].clip(lower=0)
    
    # Keep negatives, turn positives to 0, then multiply by -1 to ensure positive
    close_df['loss'] = -1 * close_df['daily_change'].clip(upper=0)

    close_df.dropna(subset=['daily_change', 'gain', 'loss'], inplace=True)
    close_df['avg_gain'] = close_df['gain'].ewm(com=window - 1, min_periods=window, adjust=False).mean()
    close_df['avg_loss'] = close_df['loss'].ewm(com=window - 1, min_periods=window, adjust=False).mean()

    close_df['relative_strength'] = close_df['avg_gain'] / close_df['avg_loss']
    close_df['rsi'] = 100 - (100 / (1 + close_df['relative_strength']))
    close_df.dropna(subset='rsi', inplace=True)

    return close_df[['rsi']]

def calculate_ema(prices: pd.Series, span: int) -> pd.Series:
    """
    Calculates the Exponential Moving Average (EMA) for a given price series.

    Parameters:
    -----------
    prices : pd.Series
        A Pandas Series of asset prices.
    span : int
        The time period span for the moving average (e.g., 9 or 21).

    Returns:
    --------
    pd.Series
        A Series representing the calculated EMA, aligned with the original index.
    """
    return prices.ewm(span=span, adjust=False).mean()


