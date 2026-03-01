import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np 
import plotly.figure_factory as ff

# --- SECRETS & SETUP ---
polygon_key = st.secrets["POLYGON_API_KEY"]
supabase_url = st.secrets["SUPABASE_URL"]
supabase_key = st.secrets["SUPABASE_KEY"]

# --- BACKEND FUNCTIONS ---
from backend_monitor import get_ticker_details, get_snapshot_ticker, format_large_number
from pull_supabase_data_v1 import get_eod_ticker_data, get_minute_ticker_data
from quant_calculations_v1 import combine_closed_stitcher, get_period_volatility, get_max_drawdown, calculate_sharpe_ratio, calculate_beta_alpha, calculate_correlation_signal, get_correlation_engine, get_rsi, calculate_ema


# --- CUSTOM CSS FOR COMPACT LAYOUT ---
st.set_page_config(page_title="Market Monitor", layout="wide", page_icon="📈")

st.markdown("""
<style>
/* 1. Reduce padding at the very top of the page */
.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 2rem !important;
}

/* 2. Shrink headers and reduce their bottom margin */
h1 { font-size: 1.8rem !important; padding-bottom: 0rem !important;}
h2 { font-size: 1.3rem !important; padding-bottom: 0rem !important;}
h3 { font-size: 1.1rem !important; padding-bottom: 0rem !important;}

/* 3. Tighten the horizontal divider lines */
hr {
    margin-top: 0.3rem !important;
    margin-bottom: 0.3rem !important;
}

/* 4. Top Live Snapshot Metrics (Small & Compact) */
[data-testid="stMetricValue"] {
    font-size: 1.1rem !important; 
}
[data-testid="stMetricLabel"] {
    font-size: 0.75rem !important;
    margin-bottom: -0.6rem !important;
}
[data-testid="stMetricDelta"] {
    font-size: 0.7rem !important;
}

/* 5. Risk & Performance Metrics (Restored to Original Large Size) */
/* This targets ONLY the metrics inside your bordered container! */
[data-testid="stVerticalBlockBorderWrapper"] [data-testid="stMetricValue"] {
    font-size: 2.0rem !important; 
}
[data-testid="stVerticalBlockBorderWrapper"] [data-testid="stMetricLabel"] {
    font-size: 1.2rem !important;
    margin-bottom: -0.5rem !important;
}
[data-testid="stVerticalBlockBorderWrapper"] [data-testid="stMetricDelta"] {
    font-size: 1.2rem !important;
}
</style>
""", unsafe_allow_html=True)

# --- CACHING API CALLS ---
@st.cache_data(ttl=86400)
def fetch_details(ticker):
    try:
        return get_ticker_details(ticker)
    except Exception as e:
        st.sidebar.error(f"Details error: {e}")
        return None

@st.cache_data(ttl=60)
def fetch_snapshot(ticker):
    try:
        return get_snapshot_ticker(ticker)
    except Exception as e:
        st.sidebar.error(f"Snapshot error: {e}")
        return None

# --- SIDEBAR CONTROLS ---
st.sidebar.header("🔍 Monitor Settings")
ticker = st.sidebar.text_input("Stock Symbol", value="AAPL").upper()

timeframe = st.sidebar.selectbox(
    "Timeframe", 
    ["1 Day", "1 Week", "1 Month", "6 Months", "1 Year", "5 Years"],
    index=1
)

compare_ticker = st.sidebar.text_input("Compare Ticker (Optional)", value="").upper()

st.sidebar.markdown("---")

# --- SIDEBAR HEATMAP CONTROLS ---
st.sidebar.header("Heatmap Settings")
st.sidebar.caption(f"Primary Ticker ({ticker}) is automatically included.")

# Generate 5 distinct input boxes for the user to type individually
hm_tickers = []
defaults = ["NVDA", "JNJ", "XOM", "AMZN"]

for i in range(4):
    t = st.sidebar.text_input(f"Heatmap Ticker {i+1}", value=defaults[i]).upper()
    if t: # Only append if they didn't leave the box blank
        hm_tickers.append(t)

# ==========================================
# HEADER & LIVE SNAPSHOT (TOP RIGHT)
# ==========================================
details = fetch_details(ticker)
snapshot = fetch_snapshot(ticker)

# Create a master 2-column layout for the very top of the app
top_left, top_right = st.columns([1, 1.8])

with top_left:
    if details:
        col_logo, col_title = st.columns([1, 6])
        with col_logo:
            if details.get('icon_url'):
                api_key = st.secrets["POLYGON_API_KEY"] 
                auth_icon_url = f"{details['icon_url']}?apiKey={api_key}"
                st.image(auth_icon_url, width=45)
            else:
                st.write("")
                
        with col_title:
            st.title(f"{details.get('name', ticker)} ({ticker})")
        
        market_cap = details.get('market_cap', 0)
        cap_str = f"${market_cap / 1e9:,.2f}B" if market_cap else "N/A"
        
        st.caption(f"**Exchange:** {details.get('exchange', 'N/A')} | **Market Cap:** {cap_str} | [Website]({details.get('website', '#')})")
        
        if details.get('description'):
            with st.expander("About Company"):
                st.write(details['description'])
    else:
        st.title(f"{ticker} Market Data")

with top_right:
    if snapshot:
        st.write("") # Tiny spacer to push metrics down to align with title
        st.caption("**Live Market Snapshot (15m Delayed)**")
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Last Price", f"${snapshot['close']:.2f}", f"{snapshot['todays_change']:.2f} ({snapshot['todays_change_percent']:.2f}%)")
        col2.metric("Today's Vol", format_large_number(snapshot['volume']))
        col3.metric("Day High", f"${snapshot['high']:.2f}")
        col4.metric("Day Low", f"${snapshot['low']:.2f}")
        col5.metric("VWAP", f"${snapshot['vwap']:.2f}")
    else:
        st.warning("Live snapshot data is currently unavailable.")

st.markdown("---")

# ==========================================
# FETCH HISTORICAL DATA
# ==========================================
with st.spinner(f"Pulling {timeframe} data for {ticker}... (Auto-ingesting if needed)"):
    if timeframe in ["1 Day", "1 Week"]:
        df = get_minute_ticker_data(
            ticker, 
            one_day=(timeframe == "1 Day"), 
            one_week=(timeframe == "1 Week")
        )
        is_intraday = True
    else:
        df = get_eod_ticker_data(
            ticker,
            one_month=(timeframe == "1 Month"),
            half_yr=(timeframe == "6 Months"),
            one_yr=(timeframe == "1 Year"),
            five_yrs=(timeframe == "5 Years")
        )
        is_intraday = False

# ==========================================
# HISTORICAL CHARTING (NOW AT THE TOP)
# ==========================================
if df is None or df.empty:
    st.error(f"Failed to load historical database records for {ticker}.")
else:
    df = df.copy()

    # find the time column ('timestamp' for intraday, 'date' for EOD)
    time_col = 'timestamp' if 'timestamp' in df.columns else 'date' if 'date' in df.columns else None

    if time_col:
        if is_intraday:
            # Force lock as strict UTC
            df[time_col] = pd.to_datetime(df[time_col], utc=True)
        else:
            df[time_col] = pd.to_datetime(df[time_col])
            
        df.set_index(time_col, inplace=True)
        
    df.sort_index(inplace=True)

    # Apply the Wall Street Timezone Shift for to min data
    if is_intraday:
        # anchor it to UTC if handling goes wrong
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
            
        # Shift to US/Eastern 
        # force Plotly to read 09:30 as pure local time.
        df.index = df.index.tz_convert('US/Eastern').tz_localize(None)

    # --- RSI Toggle Logic ---
    valid_rsi_timeframes = ["6 Months", "1 Year", "5 Years"]
    show_rsi = False
    
    col_t1, col_t2, col_t3 = st.columns([1, 1, 1.5])
    with col_t1:
        show_ema = st.toggle("Show EMA (9 & 21)")
    with col_t2:
        show_vwap = st.toggle("Show VWAP")
    with col_t3:
        if not is_intraday and timeframe in valid_rsi_timeframes:
            show_rsi = st.toggle("Show RSI (14-Day Window)")
        elif not is_intraday:
            st.caption("*(RSI requires 6M+ timeframe)*")

    # Calculate EMAs if requested
    if show_ema:
        df['EMA_9'] = calculate_ema(df['close'], span=9)
        df['EMA_21'] = calculate_ema(df['close'], span=21)

    if show_rsi:
        with st.spinner("Calculating RSI..."):
            fetch_1_yr = (timeframe == "6 Months") 
            fetch_5_yr = (timeframe in ["1 Year", "5 Years"]) 
            
            padded_df = get_eod_ticker_data(
                ticker, one_month=False, half_yr=False, one_yr=fetch_1_yr, five_yrs=fetch_5_yr
            )
            
            if padded_df is not None and not padded_df.empty:
                if 'date' in padded_df.columns:
                    padded_df['date'] = pd.to_datetime(padded_df['date'])
                    padded_df.set_index('date', inplace=True)
                padded_df.sort_index(inplace=True)
                
                rsi_df = get_rsi(padded_df.copy(), window=14)
                df = df.join(rsi_df, how='left')

        fig = make_subplots(
            rows=3, cols=1, 
            shared_xaxes=True, 
            vertical_spacing=0.04, 
            row_heights=[0.6, 0.2, 0.2]
        )
    else:
        fig = make_subplots(
            rows=2, cols=1, 
            shared_xaxes=True, 
            vertical_spacing=0.03, 
            row_heights=[0.8, 0.2]
        )

    # --- Hover Templates ---
    if is_intraday:
        candle_hover = "<b>%{x|%b %d, %Y  %H:%M}</b><br><br>Open: $%{open:.2f}<br>High: $%{high:.2f}<br>Low: $%{low:.2f}<br>Close: $%{close:.2f}<extra></extra>"
        volume_hover = "<b>%{x|%b %d, %Y  %H:%M}</b><br>Volume: %{y:,.0f}<extra></extra>"
    else:
        candle_hover = "<b>%{x|%b %d, %Y}</b><br><br>Open: $%{open:.2f}<br>High: $%{high:.2f}<br>Low: $%{low:.2f}<br>Close: $%{close:.2f}<extra></extra>"
        volume_hover = "<b>%{x|%b %d, %Y}</b><br>Volume: %{y:,.0f}<extra></extra>"

    chart_df = df.copy()

    if is_intraday:
        # Strip out extended hours strictly for Plotly to prevent rendering glitches.
        # This leaves your original 'df' intact for your raw data tables.
        chart_df = chart_df.between_time('09:30', '15:59')

    # ==========================================
    # PLOTLY X-AXIS
    # ==========================================
    
    # 1. Force Y-Axis data into pure floats using the clean chart_df
    open_p = pd.to_numeric(chart_df['open'], errors='coerce').tolist()
    high_p = pd.to_numeric(chart_df['high'], errors='coerce').tolist()
    low_p = pd.to_numeric(chart_df['low'], errors='coerce').tolist()
    close_p = pd.to_numeric(chart_df['close'], errors='coerce').tolist()
    volumes = pd.to_numeric(chart_df['volume'], errors='coerce').tolist()

    # ROW 1: Candlesticks
    fig.add_trace(go.Candlestick(
        x=chart_df.index, open=open_p, high=high_p, low=low_p, close=close_p,
        name='Price', increasing_line_color='#26a69a', decreasing_line_color='#ef5350',
        hovertemplate=candle_hover
    ), row=1, col=1)

    # OVERLAYS
    if show_ema:
        ema9 = pd.to_numeric(chart_df['EMA_9'], errors='coerce').tolist()
        ema21 = pd.to_numeric(chart_df['EMA_21'], errors='coerce').tolist()
        fig.add_trace(go.Scatter(x=chart_df.index, y=ema9, name='EMA 9', line=dict(color='#29b6f6', width=1.5), hoverinfo='skip'), row=1, col=1)
        fig.add_trace(go.Scatter(x=chart_df.index, y=ema21, name='EMA 21', line=dict(color='#ab47bc', width=1.5), hoverinfo='skip'), row=1, col=1)

    if show_vwap and 'vwap' in chart_df.columns:
        vwap_v = pd.to_numeric(chart_df['vwap'], errors='coerce').tolist()
        fig.add_trace(go.Scatter(x=chart_df.index, y=vwap_v, name='VWAP', line=dict(color='#ffa726', width=2, dash='dot'), hoverinfo='skip'), row=1, col=1)

    # ROW 2: Volume
    colors = ['#26a69a' if c >= o else '#ef5350' for c, o in zip(close_p, open_p)]
    fig.add_trace(go.Bar(
        x=chart_df.index, y=volumes, name='Volume', marker_color=colors, opacity=0.8, hovertemplate=volume_hover
    ), row=2, col=1)

    # ROW 3: RSI
    if show_rsi and 'rsi' in chart_df.columns:
        rsi_vals = pd.to_numeric(chart_df['rsi'], errors='coerce').tolist()
        fig.add_trace(go.Scatter(x=chart_df.index, y=rsi_vals, name='RSI', line=dict(color='#ab47bc', width=1.5)), row=3, col=1)
        fig.add_trace(go.Scatter(x=chart_df.index, y=[70]*len(chart_df.index), line=dict(color='#ef5350', width=1, dash='dash'), hoverinfo='skip', showlegend=False), row=3, col=1)
        fig.add_trace(go.Scatter(x=chart_df.index, y=[30]*len(chart_df.index), line=dict(color='#26a69a', width=1, dash='dash'), hoverinfo='skip', showlegend=False), row=3, col=1)
        fig.update_yaxes(title_text="RSI", range=[0, 100], tickvals=[0, 30, 50, 70, 100], row=3, col=1)

    # --- Formatting Layout & Rangebreaks ---
    
    # Define rangebreaks (Plotly still needs this to pull the visual days/hours together)
    breaks = [dict(bounds=["sat", "mon"])] # Always skip weekends
    if is_intraday:
        breaks.append(dict(bounds=[16, 9.5], pattern="hour")) # Hide the visual gap overnight

    layout_update = dict(
        height=750 if show_rsi else 600, 
        template="plotly_dark", margin=dict(l=0, r=0, t=10, b=0), showlegend=False
    )
    
    fig.update_layout(**layout_update)
    
   
    # Calculate safe boundaries with a 1-day buffer to prevent rangebreak collision
    min_date = (chart_df.index.min() - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    max_date = (chart_df.index.max() + pd.Timedelta(days=1)).strftime('%Y-%m-%d')

    # Apply rangebreaks and strict zoom boundaries
    fig.update_xaxes(
        type="date", # Strictly force date handling
        rangebreaks=breaks,
        rangeslider_visible=False,
        minallowed=min_date,
        maxallowed=max_date
    )
    
    # ==========================================
    # STREAMLIT RENDER (WITH DYNAMIC KEY FIX)
    # ==========================================
    chart_key = f"main_chart_{ticker}_{timeframe}_{show_rsi}_{show_ema}"
    st.plotly_chart(fig, use_container_width=True, key=chart_key)
    
# ==========================================
# COMPACT RISK METRICS CONTAINER
# ==========================================
st.markdown("<br>", unsafe_allow_html=True) # Tiny bit of spacing after the chart

with st.container(border=True):
    st.subheader("Risk & Performance Metrics")
    st.caption("**1-Year Baseline vs SPY**")
    
    with st.spinner("Calculating quant metrics..."):
        
        quant_df = get_eod_ticker_data(ticker, one_yr=True)
        market_df = get_eod_ticker_data("SPY", one_yr=True)

        if not quant_df.empty and not market_df.empty:
            volatility = get_period_volatility(quant_df)
            mdd = get_max_drawdown(quant_df)
            sharpe = calculate_sharpe_ratio(quant_df)
            beta, alpha = calculate_beta_alpha(quant_df, market_df)

            qc1, qc2, qc3, qc4, qc5 = st.columns(5)
            qc1.metric("Beta", f"{beta:.2f}")
            qc2.metric("Alpha", f"{alpha * 100:.2f}%")
            qc3.metric("Sharpe Ratio", f"{sharpe:.2f}")
            qc4.metric("Volatility", f"{volatility * 100:.2f}%")
            qc5.metric("Max Drawdown", f"{mdd * 100:.2f}%", delta="Risk", delta_color="inverse")
        else:
            st.warning("Insufficient data to calculate quantitative metrics.")

# raw data expander below the metrics so it's completely out of the way
with st.expander("📂 View Raw Database Records"):
    if df is not None and not df.empty:
        st.dataframe(df.sort_index(ascending=False), use_container_width=True)

# ==========================================
# PAIRS TRADING & CORRELATION ANALYSIS
# ==========================================
if compare_ticker:
    valid_timeframes = ["6 Months", "1 Year", "5 Years"]

    if compare_ticker == ticker:
        st.markdown("---")
        st.warning(f"⚠️ **Invalid Comparison:** You are already viewing {ticker}. Please enter a different ticker symbol in the sidebar to run the statistical arbitrage analysis.")  
    elif timeframe not in valid_timeframes:
        st.markdown("---")
        st.info(f"📊 **Statistical Arbitrage requires a larger sample size.** \n\nA timeframe of 6 Months or greater is required to accurately calculate the 50-day rolling correlations for {ticker} vs {compare_ticker}.")
    else:
        st.markdown("---")
        st.subheader(f"Statistical Arbitrage: {ticker} vs {compare_ticker}")

        with st.spinner(f"Fetching {compare_ticker} and generating correlation channel..."):
            
            # FETCH EXTRA DATA
            fetch_1_yr = (timeframe == "6 Months") 
            fetch_5_yr = (timeframe in ["1 Year", "5 Years"]) 
            
            primary_padded_df = get_eod_ticker_data(
                ticker, one_month=False, half_yr=False, one_yr=fetch_1_yr, five_yrs=fetch_5_yr
            )
            compare_df = get_eod_ticker_data(
                compare_ticker, one_month=False, half_yr=False, one_yr=fetch_1_yr, five_yrs=fetch_5_yr
            )
            
            if not compare_df.empty and 'date' in compare_df.columns:
                compare_df['date'] = pd.to_datetime(compare_df['date'])
                compare_df.set_index('date', inplace=True)
                compare_df.sort_index(inplace=True)

            if compare_df is None or compare_df.empty or primary_padded_df is None or primary_padded_df.empty:
                st.warning(f"Could not load sufficient data to calculate correlation.")
            else:
                if 'date' in primary_padded_df.columns:
                    primary_padded_df['date'] = pd.to_datetime(primary_padded_df['date'])
                    primary_padded_df.set_index('date', inplace=True)
                    primary_padded_df.sort_index(inplace=True)

                corr_df = calculate_correlation_signal(primary_padded_df, compare_df, ticker, compare_ticker, std_multiplier=2.0)
                aligned_prices = combine_closed_stitcher(primary_padded_df, ticker, compare_df, compare_ticker)
                
                # SLICE 
                today = pd.Timestamp.now().normalize()
                if timeframe == "6 Months":
                    target_start = today - pd.DateOffset(months=6)
                elif timeframe == "1 Year":
                    target_start = today - pd.DateOffset(years=1)
                else:
                    target_start = today - pd.DateOffset(years=5)

                corr_df = corr_df[corr_df.index >= target_start]
                aligned_prices = aligned_prices[aligned_prices.index >= target_start]

                # ==========================================
                # FORCE NUMERIC CONVERSION FOR PLOTLY
                # ==========================================
                t_price = pd.to_numeric(aligned_prices[ticker], errors='coerce')
                c_price = pd.to_numeric(aligned_prices[compare_ticker], errors='coerce')

                # CALCULATE CUMULATIVE RETURN
                norm_ticker = ((t_price / t_price.iloc[0]) - 1) * 100
                norm_compare = ((c_price / c_price.iloc[0]) - 1) * 100

                # Extract pure float lists
                y_norm_ticker = norm_ticker.tolist()
                y_norm_compare = norm_compare.tolist()
                y_upper = pd.to_numeric(corr_df['upper_threshold'], errors='coerce').tolist()
                y_lower = pd.to_numeric(corr_df['lower_threshold'], errors='coerce').tolist()
                y_rolling = pd.to_numeric(corr_df['rolling_corr'], errors='coerce').tolist()
                y_avg = pd.to_numeric(corr_df['avg_corr'], errors='coerce').tolist()

                fig_corr = make_subplots(
                    rows=2, cols=1, 
                    shared_xaxes=True, 
                    vertical_spacing=0.08,
                    row_heights=[0.6, 0.4],
                    subplot_titles=(f"Cumulative Return (%)", f"Rolling Correlation (2.0σ Bands)")
                )

                # ROW 1: Normalized Prices
                fig_corr.add_trace(go.Scatter(x=aligned_prices.index, y=y_norm_ticker, name=ticker, line=dict(color='#26a69a', width=2)), row=1, col=1)
                fig_corr.add_trace(go.Scatter(x=aligned_prices.index, y=y_norm_compare, name=compare_ticker, line=dict(color='#ef5350', width=2)), row=1, col=1)

                # ROW 2: Correlation and Bands
                fig_corr.add_trace(go.Scatter(x=corr_df.index, y=y_upper, mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'), row=2, col=1)
                fig_corr.add_trace(go.Scatter(x=corr_df.index, y=y_lower, mode='lines', fill='tonexty', fillcolor='rgba(128, 128, 128, 0.2)', line=dict(width=0), name='Expected Range'), row=2, col=1)
                fig_corr.add_trace(go.Scatter(x=corr_df.index, y=y_rolling, name='Rolling Corr', line=dict(color='white', width=1.5)), row=2, col=1)
                fig_corr.add_trace(go.Scatter(x=corr_df.index, y=y_avg, name='Avg Corr', line=dict(color='gray', width=1, dash='dot')), row=2, col=1)

                # Signals
                buy_signals = corr_df[corr_df['signal'] == 1]
                sell_signals = corr_df[corr_df['signal'] == -1]

                if not buy_signals.empty:
                    y_buy = pd.to_numeric(buy_signals['rolling_corr'], errors='coerce').tolist()
                    fig_corr.add_trace(go.Scatter(x=buy_signals.index, y=y_buy, mode='markers', marker=dict(symbol='triangle-up', size=12, color='#26a69a'), name='Revert Up Signal'), row=2, col=1)

                if not sell_signals.empty:
                    y_sell = pd.to_numeric(sell_signals['rolling_corr'], errors='coerce').tolist()
                    fig_corr.add_trace(go.Scatter(x=sell_signals.index, y=y_sell, mode='markers', marker=dict(symbol='triangle-down', size=12, color='#ef5350'), name='Revert Down Signal'), row=2, col=1)

                # Format layout
                fig_corr.update_layout(
                    height=700, template="plotly_dark", margin=dict(l=0, r=0, t=40, b=0),
                    hovermode='x unified', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )

                fig_corr.update_yaxes(title_text=f"{ticker} Return (%)", secondary_y=False, row=1, col=1)
                fig_corr.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
                
                # Render with dynamic key to prevent ghost traces
                corr_key = f"corr_{ticker}_{compare_ticker}_{timeframe}"
                st.plotly_chart(fig_corr, use_container_width=True, key=corr_key)

# ==========================================
# MULTI-ASSET CORRELATION HEATMAP
# ==========================================
st.markdown("---")
st.subheader("Sector & Industry Correlation Heatmap")

basket_tickers = [ticker] + [t for t in hm_tickers if t != ticker]
basket_tickers = list(dict.fromkeys(basket_tickers))
        
if len(basket_tickers) > 1:
    with st.spinner("Compiling matrix data..."):
                
        fetch_1_yr = (timeframe == "6 Months") 
        fetch_5_yr = (timeframe in ["1 Year", "5 Years"]) 
                
        today = pd.Timestamp.now().normalize()
        if timeframe == "6 Months":
            target_start = today - pd.DateOffset(months=6)
        elif timeframe == "1 Year":
            target_start = today - pd.DateOffset(years=1)
        else:
            target_start = today - pd.DateOffset(years=5)

        close_prices = []
        failed_tickers = []
                
        for t in basket_tickers:
            t_df = get_eod_ticker_data(t, one_month=False, half_yr=False, one_yr=fetch_1_yr, five_yrs=fetch_5_yr)
            if t_df is not None and not t_df.empty and 'date' in t_df.columns:
                
                # Strip timezones and normalize to strictly dates to ensure perfect inner joins
                t_df['date'] = pd.to_datetime(t_df['date'], utc=True).dt.tz_localize(None).dt.normalize()
                t_df.set_index('date', inplace=True)
                t_df.sort_index(inplace=True)
                        
                t_df = t_df[t_df.index >= target_start]
                        
                if 'close' in t_df.columns:
                    close_series = t_df[['close']].copy()
                    close_series.columns = [t]
                    close_prices.append(close_series)
                else:
                    failed_tickers.append(t)
            else:
                failed_tickers.append(t)
                
        if failed_tickers:
            st.warning(f"Could not load data for: {', '.join(failed_tickers)}")

        # Stitch and Calculate
        if len(close_prices) > 1:
            basket_df = pd.concat(close_prices, axis=1, join='inner')
            
            if basket_df.empty:
                st.error("Data alignment failed. The fetched tickers have no overlapping dates.")
            else:
                # 💥 THE FIX: Ruthlessly force standard Python floats 
                basket_df = basket_df.astype('float64')
                
                corr_matrix = get_correlation_engine(basket_df)
                
                # 💥 THE FIX: Ensure Plotly gets raw python lists and strings, not numpy objects
                z_vals = np.round(corr_matrix.values, 2).astype('float64').tolist()
                x_vals = corr_matrix.columns.astype(str).tolist()
                y_vals = corr_matrix.index.astype(str).tolist()
                        
                fig_heat = go.Figure(data=go.Heatmap(
                    z=z_vals, x=x_vals, y=y_vals,
                    colorscale='RdBu', zmin=-1, zmax=1,   
                    text=z_vals, texttemplate="%{text}",
                    textfont={"size": 14, "color": "white"}, hoverinfo="x+y+z"
                ))

                fig_heat.update_layout(
                    template="plotly_dark", height=500 + (len(basket_tickers) * 20), 
                    margin=dict(l=0, r=0, t=30, b=0), yaxis_autorange='reversed' 
                )

                heat_key = f"heat_{'_'.join(basket_tickers)}_{timeframe}"
                st.plotly_chart(fig_heat, use_container_width=True, key=heat_key)
        else:
            st.info("Not enough valid tickers to generate a heatmap.")
else:
    st.info("Add comparison tickers in the sidebar to generate the correlation heatmap.")