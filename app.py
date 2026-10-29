import streamlit as st
import pandas as pd
import numpy as np
import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="IITL Stock Analytics Hub | NSE: IITL",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Modern CSS Styling for Financial Dashboard
# ---------------------------------------------------------
st.markdown("""
<style>
    /* Global Typography and Base Styling */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .stApp {
        background: radial-gradient(circle at top right, #111827 0%, #0a0d14 100%);
    }
    
    /* Top Header Banner */
    .hero-header {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px 28px;
        margin-bottom: 24px;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 16px;
    }
    
    .hero-title {
        margin: 0;
        font-size: 1.85rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        background: linear-gradient(90deg, #FFFFFF 0%, #94A3B8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .hero-subtitle {
        margin: 6px 0 0 0;
        color: #94A3B8;
        font-size: 0.95rem;
        font-weight: 400;
    }
    
    .badge-tag {
        display: inline-block;
        background: rgba(0, 210, 255, 0.12);
        color: #00D2FF;
        border: 1px solid rgba(0, 210, 255, 0.3);
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        margin-right: 6px;
    }
    
    .badge-nse {
        background: rgba(34, 197, 94, 0.12);
        color: #4ADE80;
        border: 1px solid rgba(34, 197, 94, 0.3);
    }
    
    /* Metric Cards Grid */
    .metric-card {
        background: linear-gradient(145deg, rgba(26, 34, 52, 0.6) 0%, rgba(17, 24, 39, 0.8) 100%);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 14px;
        padding: 16px 20px;
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(0, 210, 255, 0.3);
    }
    
    .metric-label {
        font-size: 0.78rem;
        color: #94A3B8;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 6px;
    }
    
    .metric-val {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.5rem;
        font-weight: 700;
        color: #F8FAFC;
        margin-bottom: 4px;
    }
    
    .metric-sub {
        font-size: 0.82rem;
        font-weight: 600;
    }
    
    .text-pos {
        color: #10B981;
    }
    
    .text-neg {
        color: #EF4444;
    }
    
    .text-neutral {
        color: #94A3B8;
    }
    
    /* Custom tab navigation styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(15, 23, 42, 0.6);
        padding: 6px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.06);
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 18px;
        font-weight: 600;
        font-size: 0.9rem;
        color: #94A3B8;
        border: none;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: rgba(0, 210, 255, 0.15) !important;
        color: #00D2FF !important;
    }
    
    /* Streamlit Metric Overrides */
    div[data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace !important;
        font-weight: 700 !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Data Loader and Processor
# ---------------------------------------------------------
@st.cache_data(show_spinner="Loading and processing market dataset...")
def load_data(filepath="IITL.NS_2014_2024.csv"):
    try:
        # Load skipping first 3 rows which contain multi-level metadata
        df = pd.read_csv(filepath, skiprows=3, header=None)
        df.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
        
        # Clean Datetime
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Numeric conversions
        for col in ['Close', 'High', 'Low', 'Open', 'Volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Drop invalid rows and sort chronologically
        df = df.dropna(subset=['Date', 'Close']).sort_values('Date').reset_index(drop=True)
        
        # Basic Price & Return Features
        df['Daily_Return'] = df['Close'].pct_change()
        df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
        df['Cumulative_Return'] = (1 + df['Daily_Return']).cumprod() - 1
        
        # Moving Averages
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
        
        # Bollinger Bands (20 days, 2 std)
        bb_mid = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Middle'] = bb_mid
        df['BB_Upper'] = bb_mid + (2 * bb_std)
        df['BB_Lower'] = bb_mid - (2 * bb_std)
        df['BB_Bandwidth'] = ((df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']) * 100
        df['BB_Percent'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # RSI (14 periods)
        delta = df['Close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        roll_gain = gain.rolling(window=14).mean()
        roll_loss = loss.rolling(window=14).mean()
        rs = roll_gain / (roll_loss + 1e-9)
        df['RSI_14'] = 100 - (100 / (1 + rs))
        
        # MACD (12, 26, 9)
        ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema_12 - ema_26
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        
        # ATR (Average True Range, 14 periods)
        high_low = df['High'] - df['Low']
        high_prev_close = (df['High'] - df['Close'].shift(1)).abs()
        low_prev_close = (df['Low'] - df['Close'].shift(1)).abs()
        tr = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)
        df['ATR_14'] = tr.rolling(window=14).mean()
        
        # Rolling Volatility (30 days annualized, 252 trading days)
        df['Vol_30D'] = df['Daily_Return'].rolling(window=30).std() * np.sqrt(252) * 100
        
        # Cumulative Peak and Drawdown
        df['Peak'] = df['Close'].cummax()
        df['Drawdown'] = (df['Close'] - df['Peak']) / df['Peak'] * 100
        
        return df
    except Exception as e:
        st.error(f"Error reading dataset: {e}")
        return pd.DataFrame()

raw_df = load_data()

if raw_df.empty:
    st.warning("Please verify that 'IITL.NS_2014_2024.csv' is present in the application directory.")
    st.stop()

# ---------------------------------------------------------
# Sidebar Controls & Analysis Settings
# ---------------------------------------------------------
st.sidebar.markdown("""
<div style="text-align: center; padding: 10px 0 16px 0;">
    <h3 style="margin: 0; color: #00D2FF; font-weight: 800;">NSE: IITL</h3>
    <p style="margin: 4px 0 0 0; color: #94A3B8; font-size: 0.8rem;">Industrial Investment Trust Ltd.</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("### 📅 Timeframe Filter")

min_date = raw_df['Date'].min().date()
max_date = raw_df['Date'].max().date()

preset_range = st.sidebar.selectbox(
    "Quick Preset",
    ["All Time (2012 - 2024)", "1 Month", "3 Months", "6 Months", "1 Year (2024)", "3 Years", "5 Years", "Custom Range"],
    index=0
)

if preset_range == "1 Month":
    start_date = max_date - datetime.timedelta(days=30)
    end_date = max_date
elif preset_range == "3 Months":
    start_date = max_date - datetime.timedelta(days=90)
    end_date = max_date
elif preset_range == "6 Months":
    start_date = max_date - datetime.timedelta(days=180)
    end_date = max_date
elif preset_range == "1 Year (2024)":
    start_date = max_date - datetime.timedelta(days=365)
    end_date = max_date
elif preset_range == "3 Years":
    start_date = max_date - datetime.timedelta(days=3*365)
    end_date = max_date
elif preset_range == "5 Years":
    start_date = max_date - datetime.timedelta(days=5*365)
    end_date = max_date
elif preset_range == "Custom Range":
    c1, c2 = st.sidebar.columns(2)
    start_date = c1.date_input("Start Date", min_date, min_value=min_date, max_value=max_date)
    end_date = c2.date_input("End Date", max_date, min_value=min_date, max_value=max_date)
else:
    start_date = min_date
    end_date = max_date

# Date Filtering
mask = (raw_df['Date'].dt.date >= start_date) & (raw_df['Date'].dt.date <= end_date)
df = raw_df.loc[mask].copy().reset_index(drop=True)

if df.empty:
    st.error("No trading sessions found within selected range. Please widen the date filter.")
    st.stop()

# Technical Indicators Toggles in Sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Chart Overlays")
chart_style = st.sidebar.radio("Price Chart Type", ["Candlestick", "Line Area"], index=0)
show_sma20 = st.sidebar.checkbox("SMA 20 (Short-term)", value=True)
show_sma50 = st.sidebar.checkbox("SMA 50 (Medium-term)", value=True)
show_sma200 = st.sidebar.checkbox("SMA 200 (Long-term)", value=False)
show_ema20 = st.sidebar.checkbox("EMA 20", value=False)
show_bb = st.sidebar.checkbox("Bollinger Bands (20, 2σ)", value=True)
log_scale = st.sidebar.checkbox("Logarithmic Price Scale", value=False)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Risk Configuration")
rf_rate = st.sidebar.slider("Risk-Free Rate (% p.a.)", min_value=3.0, max_value=10.0, value=6.5, step=0.1) / 100.0

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="font-size: 0.75rem; color: #64748B; line-height: 1.4;">
    <strong>IITL.NS Dataset Specs:</strong><br>
    • Total Sessions: 3,202<br>
    • Exchange: National Stock Exchange (NSE)<br>
    • Currency: INR (₹)
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Top Hero Header Banner
# ---------------------------------------------------------
latest_row = df.iloc[-1]
first_row = df.iloc[0]
prev_row = df.iloc[-2] if len(df) > 1 else latest_row

latest_price = latest_row['Close']
prev_price = prev_row['Close']
price_change = latest_price - prev_price
pct_change = (price_change / prev_price) * 100 if prev_price > 0 else 0

st.markdown(f"""
<div class="hero-header">
    <div>
        <div style="margin-bottom: 8px;">
            <span class="badge-tag badge-nse">NSE Listed</span>
            <span class="badge-tag">NBFC Sector</span>
            <span class="badge-tag">Ticker: IITL.NS</span>
        </div>
        <h1 class="hero-title">Industrial Investment Trust Limited</h1>
        <p class="hero-subtitle">Comprehensive Stock Analytics, Quantitative Risk Matrix & Algorithmic Insights (2012 – 2024)</p>
    </div>
    <div style="text-align: right;">
        <div style="font-size: 0.8rem; color: #94A3B8; text-transform: uppercase; font-weight: 600;">Latest Closing Price</div>
        <div style="font-family: 'JetBrains Mono', monospace; font-size: 2.2rem; font-weight: 800; color: #00D2FF;">
            ₹{latest_price:,.2f}
        </div>
        <div style="font-size: 0.95rem; font-weight: 600;" class="{ 'text-pos' if price_change >= 0 else 'text-neg' }">
            { '▲ +' if price_change >= 0 else '▼ ' }₹{abs(price_change):.2f} ({pct_change:+.2f}%)
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# KPI Metric Grid
# ---------------------------------------------------------
period_high = df['High'].max()
period_low = df['Low'].min()
cum_return = ((latest_price / first_row['Close']) - 1) * 100
total_volume = df['Volume'].sum()
avg_volume = df['Volume'].mean()

days_diff = (df['Date'].iloc[-1] - df['Date'].iloc[0]).days
cagr = (((latest_price / first_row['Close']) ** (365.25 / max(days_diff, 1))) - 1) * 100 if days_diff > 30 else 0

kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

with kpi1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Period Range</div>
        <div class="metric-val" style="font-size: 1.25rem;">₹{period_low:.1f} - ₹{period_high:.1f}</div>
        <div class="metric-sub text-neutral">Spread: ₹{(period_high - period_low):.1f}</div>
    </div>
    """, unsafe_allow_html=True)

with kpi2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Period Cumulative Return</div>
        <div class="metric-val { 'text-pos' if cum_return >= 0 else 'text-neg' }">{cum_return:+.1f}%</div>
        <div class="metric-sub text-neutral">Start: ₹{first_row['Close']:.2f}</div>
    </div>
    """, unsafe_allow_html=True)

with kpi3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Annualized Return (CAGR)</div>
        <div class="metric-val { 'text-pos' if cagr >= 0 else 'text-neg' }">{cagr:+.1f}%</div>
        <div class="metric-sub text-neutral">Over {days_diff/365.25:.1f} Years</div>
    </div>
    """, unsafe_allow_html=True)

with kpi4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">All-Time Peak (ATH)</div>
        <div class="metric-val">₹{raw_df['High'].max():.2f}</div>
        <div class="metric-sub text-neutral">ATL: ₹{raw_df['Low'].min():.2f}</div>
    </div>
    """, unsafe_allow_html=True)

with kpi5:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Avg Daily Traded Vol</div>
        <div class="metric-val" style="font-size: 1.25rem;">{avg_volume:,.0f}</div>
        <div class="metric-sub text-neutral">Total: {total_volume/1e6:.2f}M shares</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# ---------------------------------------------------------
# Interactive Analytics Tabs
# ---------------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Price Action & Volume",
    "🎯 Technical Indicators",
    "⚖️ Risk & Drawdown Matrix",
    "📅 Seasonality & Monthly Heatmap",
    "🔮 Trend Projection & Forecast",
    "🗃️ Raw Data & Export"
])

# ---------------------------------------------------------
# TAB 1: Price Action & Volume Analysis
# ---------------------------------------------------------
with tab1:
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.75, 0.25],
        subplot_titles=("IITL.NS Daily Price & Overlays (₹)", "Trading Volume (Shares)")
    )
    
    # Candlestick or Line
    if chart_style == "Candlestick":
        fig.add_trace(
            go.Candlestick(
                x=df['Date'],
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name="OHLC",
                increasing_line_color='#10B981',
                decreasing_line_color='#EF4444'
            ),
            row=1, col=1
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=df['Date'],
                y=df['Close'],
                mode='lines',
                name='Close Price',
                line=dict(color='#00D2FF', width=2),
                fill='tozeroy',
                fillcolor='rgba(0, 210, 255, 0.08)'
            ),
            row=1, col=1
        )
    
    # Overlays
    if show_sma20:
        fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA_20'], mode='lines', name='SMA 20', line=dict(color='#F59E0B', width=1.5)), row=1, col=1)
    if show_sma50:
        fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA_50'], mode='lines', name='SMA 50', line=dict(color='#3B82F6', width=1.5)), row=1, col=1)
    if show_sma200:
        fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA_200'], mode='lines', name='SMA 200', line=dict(color='#EC4899', width=1.8, dash='dash')), row=1, col=1)
    if show_ema20:
        fig.add_trace(go.Scatter(x=df['Date'], y=df['EMA_20'], mode='lines', name='EMA 20', line=dict(color='#A855F7', width=1.5)), row=1, col=1)
    if show_bb:
        fig.add_trace(go.Scatter(x=df['Date'], y=df['BB_Upper'], mode='lines', name='BB Upper (2σ)', line=dict(color='rgba(255, 255, 255, 0.25)', width=1, dash='dot')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['Date'], y=df['BB_Lower'], mode='lines', name='BB Lower (2σ)', line=dict(color='rgba(255, 255, 255, 0.25)', width=1, dash='dot'), fill='tonexty', fillcolor='rgba(255, 255, 255, 0.03)'), row=1, col=1)
    
    # Volume Bar Chart with Directional Coloring
    vol_colors = ['#10B981' if c >= o else '#EF4444' for c, o in zip(df['Close'], df['Open'])]
    fig.add_trace(
        go.Bar(
            x=df['Date'],
            y=df['Volume'],
            name="Volume",
            marker_color=vol_colors,
            opacity=0.8
        ),
        row=2, col=1
    )
    
    # Layout Tweaks
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15, 23, 42, 0.6)',
        height=680,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified"
    )
    
    if log_scale:
        fig.update_yaxes(type="log", row=1, col=1)
        
    st.plotly_chart(fig, use_container_width=True)
    
    # Quick Summary Table
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown(f"""
        **Session Extremes:**
        - Period Open: `₹{first_row['Open']:.2f}`
        - Period High: `₹{period_high:.2f}`
        - Period Low: `₹{period_low:.2f}`
        - Period Close: `₹{latest_price:.2f}`
        """)
    with col_b:
        up_days = (df['Daily_Return'] > 0).sum()
        down_days = (df['Daily_Return'] < 0).sum()
        flat_days = (df['Daily_Return'] == 0).sum()
        win_rate = (up_days / max(up_days + down_days, 1)) * 100
        st.markdown(f"""
        **Market Breadth:**
        - Bullish Sessions: `{up_days}` ({up_days/len(df)*100:.1f}%)
        - Bearish Sessions: `{down_days}` ({down_days/len(df)*100:.1f}%)
        - Flat Sessions: `{flat_days}`
        - Win Rate (Up/Down): `{win_rate:.1f}%`
        """)
    with col_c:
        st.markdown(f"""
        **Volume Statistics:**
        - Max Single-Day Vol: `{df['Volume'].max():,}`
        - Median Daily Vol: `{df['Volume'].median():,.0f}`
        - Zero-Volume Days: `{(df['Volume'] == 0).sum()}`
        - Days Analyzed: `{len(df)} sessions`
        """)

# ---------------------------------------------------------
# TAB 2: Technical Indicators
# ---------------------------------------------------------
with tab2:
    st.subheader("Technical Oscillators & Momentum")
    
    # RSI Chart
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=df['Date'], y=df['RSI_14'], mode='lines', name='RSI (14)', line=dict(color='#00D2FF', width=2)))
    fig_rsi.add_hline(y=70, line_dash="dash", line_color="#EF4444", annotation_text="Overbought (70)", annotation_position="top right")
    fig_rsi.add_hline(y=30, line_dash="dash", line_color="#10B981", annotation_text="Oversold (30)", annotation_position="bottom right")
    fig_rsi.add_hrect(y0=30, y1=70, fillcolor="rgba(255, 255, 255, 0.03)", line_width=0)
    fig_rsi.update_layout(
        title="Relative Strength Index (RSI 14)",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15, 23, 42, 0.6)',
        height=280,
        margin=dict(l=20, r=20, t=40, b=20),
        yaxis=dict(range=[0, 100])
    )
    st.plotly_chart(fig_rsi, use_container_width=True)
    
    # MACD Chart
    fig_macd = go.Figure()
    fig_macd.add_trace(go.Scatter(x=df['Date'], y=df['MACD'], mode='lines', name='MACD Line (12,26)', line=dict(color='#3B82F6', width=2)))
    fig_macd.add_trace(go.Scatter(x=df['Date'], y=df['MACD_Signal'], mode='lines', name='Signal Line (9)', line=dict(color='#F59E0B', width=1.5)))
    macd_bar_colors = ['#10B981' if h >= 0 else '#EF4444' for h in df['MACD_Hist']]
    fig_macd.add_trace(go.Bar(x=df['Date'], y=df['MACD_Hist'], name='MACD Histogram', marker_color=macd_bar_colors, opacity=0.8))
    fig_macd.update_layout(
        title="MACD (12, 26, 9)",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15, 23, 42, 0.6)',
        height=280,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig_macd, use_container_width=True)
    
    # Bollinger Bandwidth & 30-Day Volatility
    t_c1, t_c2 = st.columns(2)
    with t_c1:
        fig_bw = go.Figure()
        fig_bw.add_trace(go.Scatter(x=df['Date'], y=df['BB_Bandwidth'], mode='lines', line=dict(color='#A855F7', width=1.8), fill='tozeroy', fillcolor='rgba(168, 85, 247, 0.1)'))
        fig_bw.update_layout(
            title="Bollinger Bandwidth % (Volatility Squeeze Indicator)",
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(15, 23, 42, 0.6)',
            height=260,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_bw, use_container_width=True)
        
    with t_c2:
        fig_vol = go.Figure()
        fig_vol.add_trace(go.Scatter(x=df['Date'], y=df['Vol_30D'], mode='lines', line=dict(color='#EC4899', width=1.8), fill='tozeroy', fillcolor='rgba(236, 72, 153, 0.1)'))
        fig_vol.update_layout(
            title="Rolling 30-Day Annualized Volatility (%)",
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(15, 23, 42, 0.6)',
            height=260,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_vol, use_container_width=True)

# ---------------------------------------------------------
# TAB 3: Risk & Performance Matrix
# ---------------------------------------------------------
with tab3:
    st.subheader("Quantitative Risk & Drawdown Analytics")
    
    # Underwater Drawdown Chart
    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(
        x=df['Date'],
        y=df['Drawdown'],
        mode='lines',
        name='Drawdown %',
        line=dict(color='#EF4444', width=1.5),
        fill='tozeroy',
        fillcolor='rgba(239, 68, 68, 0.2)'
    ))
    fig_dd.update_layout(
        title="Historical Peak-to-Trough Drawdown (Underwater Chart)",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15, 23, 42, 0.6)',
        height=320,
        margin=dict(l=20, r=20, t=40, b=20),
        yaxis=dict(title="Drawdown (%)", ticksuffix="%")
    )
    st.plotly_chart(fig_dd, use_container_width=True)
    
    # Quantitative Risk Metrics Calculations
    daily_returns = df['Daily_Return'].dropna()
    mean_daily = daily_returns.mean()
    std_daily = daily_returns.std()
    annual_vol = std_daily * np.sqrt(252) * 100
    annual_ret = mean_daily * 252 * 100
    
    # Sharpe Ratio
    rf_daily = rf_rate / 252.0
    excess_returns = daily_returns - rf_daily
    sharpe_ratio = (excess_returns.mean() / (std_daily + 1e-9)) * np.sqrt(252) if std_daily > 0 else 0
    
    # Sortino Ratio (Downside deviation)
    downside_returns = daily_returns[daily_returns < 0]
    downside_std = downside_returns.std()
    sortino_ratio = (excess_returns.mean() / (downside_std + 1e-9)) * np.sqrt(252) if downside_std > 0 else 0
    
    # Max Drawdown
    max_dd = df['Drawdown'].min()
    
    # Value at Risk (Historical)
    var_95 = np.percentile(daily_returns, 5) * 100
    var_99 = np.percentile(daily_returns, 1) * 100
    
    # Distribution of Daily Returns
    rc1, rc2 = st.columns([1.2, 0.8])
    with rc1:
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=daily_returns * 100,
            nbinsx=60,
            name="Daily Return %",
            marker_color='#00D2FF',
            opacity=0.75
        ))
        fig_hist.update_layout(
            title="Distribution of Daily Returns (%)",
            xaxis_title="Daily Return (%)",
            yaxis_title="Frequency",
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(15, 23, 42, 0.6)',
            height=320,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_hist, use_container_width=True)
        
    with rc2:
        st.markdown(f"""
        ### 📐 Key Risk Ratios
        | Metric | Value | Reference / Meaning |
        | :--- | :--- | :--- |
        | **Annualized Volatility** | `{annual_vol:.2f}%` | Price fluctuation magnitude |
        | **Sharpe Ratio** | `{sharpe_ratio:.2f}` | Risk-adjusted return (Rf={rf_rate*100:.1f}%) |
        | **Sortino Ratio** | `{sortino_ratio:.2f}` | Downside risk-adjusted return |
        | **Maximum Drawdown (MDD)** | `{max_dd:.2f}%` | Worst loss from previous peak |
        | **Value at Risk (95% 1-Day)** | `{var_95:.2f}%` | 95% confidence max 1-day loss |
        | **Value at Risk (99% 1-Day)** | `{var_99:.2f}%` | 99% confidence max 1-day loss |
        | **Best Single Day** | `+{daily_returns.max()*100:.2f}%` | Strongest trading session |
        | **Worst Single Day** | `{daily_returns.min()*100:.2f}%` | Deepest trading contraction |
        """)

# ---------------------------------------------------------
# TAB 4: Seasonality & Monthly Heatmap
# ---------------------------------------------------------
with tab4:
    st.subheader("Calendar Returns & Seasonal Patterns")
    
    # Create Year-Month returns
    temp_df = raw_df.copy()
    temp_df['Year'] = temp_df['Date'].dt.year
    temp_df['Month'] = temp_df['Date'].dt.month
    
    monthly_prices = temp_df.groupby(['Year', 'Month'])['Close'].agg(['first', 'last']).reset_index()
    monthly_prices['Monthly_Return'] = ((monthly_prices['last'] / monthly_prices['first']) - 1) * 100
    
    pivot_table = monthly_prices.pivot(index='Year', columns='Month', values='Monthly_Return')
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    pivot_table.columns = [month_names[m - 1] for m in pivot_table.columns]
    
    fig_heat = px.imshow(
        pivot_table,
        labels=dict(x="Month", y="Year", color="Return (%)"),
        x=pivot_table.columns,
        y=pivot_table.index,
        color_continuous_scale=[[0, '#EF4444'], [0.5, '#1E293B'], [1, '#10B981']],
        color_continuous_midpoint=0,
        text_auto='.1f',
        aspect="auto"
    )
    fig_heat.update_layout(
        title="IITL.NS Monthly Returns Heatmap (%) (2012 - 2024)",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15, 23, 42, 0.6)',
        height=420,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig_heat, use_container_width=True)
    
    # Annual Return Bar Chart & Day of Week
    s_c1, s_c2 = st.columns(2)
    with s_c1:
        yearly_prices = temp_df.groupby('Year')['Close'].agg(['first', 'last']).reset_index()
        yearly_prices['Annual_Return'] = ((yearly_prices['last'] / yearly_prices['first']) - 1) * 100
        bar_colors = ['#10B981' if r >= 0 else '#EF4444' for r in yearly_prices['Annual_Return']]
        
        fig_year = go.Figure()
        fig_year.add_trace(go.Bar(
            x=yearly_prices['Year'],
            y=yearly_prices['Annual_Return'],
            marker_color=bar_colors,
            text=[f"{r:+.1f}%" for r in yearly_prices['Annual_Return']],
            textposition='outside'
        ))
        fig_year.update_layout(
            title="Annual Calendar Year Performance (%)",
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(15, 23, 42, 0.6)',
            height=320,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_year, use_container_width=True)
        
    with s_c2:
        temp_df['DayOfWeek'] = temp_df['Date'].dt.day_name()
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
        day_stats = temp_df.groupby('DayOfWeek')['Daily_Return'].mean() * 100
        day_stats = day_stats.reindex(day_order).dropna()
        
        fig_day = go.Figure()
        day_bar_colors = ['#10B981' if r >= 0 else '#EF4444' for r in day_stats]
        fig_day.add_trace(go.Bar(
            x=day_stats.index,
            y=day_stats.values,
            marker_color=day_bar_colors,
            text=[f"{r:+.2f}%" for r in day_stats.values],
            textposition='outside'
        ))
        fig_day.update_layout(
            title="Average Daily Return by Day of Week (%)",
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(15, 23, 42, 0.6)',
            height=320,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_day, use_container_width=True)

# ---------------------------------------------------------
# TAB 5: Trend Projection & Forecast
# ---------------------------------------------------------
with tab5:
    st.subheader("Statistical Trend Projection & Extrapolation")
    st.markdown("""
    This model computes linear and polynomial statistical trendlines based on historical price momentum and generates forward projection intervals for the upcoming trading sessions.
    """)
    
    horizon_days = st.slider("Forecast Horizon (Trading Days Ahead)", min_value=15, max_value=90, value=30, step=5)
    
    # Fit linear trend on selected period
    x_numeric = np.arange(len(df))
    y_close = df['Close'].values
    
    poly_fit = np.polyfit(x_numeric, y_close, deg=1)
    trend_line = np.polyval(poly_fit, x_numeric)
    
    # Future projection
    future_x = np.arange(len(df), len(df) + horizon_days)
    future_dates = [df['Date'].iloc[-1] + datetime.timedelta(days=i) for i in range(1, horizon_days + 1)]
    future_trend = np.polyval(poly_fit, future_x)
    
    # Confidence bounds based on residual standard deviation
    residuals = y_close - trend_line
    resid_std = np.std(residuals)
    upper_bound = future_trend + (1.96 * resid_std)
    lower_bound = np.maximum(future_trend - (1.96 * resid_std), 0)
    
    fig_proj = go.Figure()
    # Historical Close
    fig_proj.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Historical Close', line=dict(color='#00D2FF', width=2)))
    # Fitted Trend
    fig_proj.add_trace(go.Scatter(x=df['Date'], y=trend_line, mode='lines', name='Historical Linear Trend', line=dict(color='#F59E0B', dash='dash')))
    # Projected Trend
    fig_proj.add_trace(go.Scatter(x=future_dates, y=future_trend, mode='lines', name='Projected Trajectory', line=dict(color='#A855F7', width=2.5)))
    # Projection Bounds
    fig_proj.add_trace(go.Scatter(x=future_dates, y=upper_bound, mode='lines', name='Upper 95% Bound', line=dict(color='rgba(168, 85, 247, 0.4)', dash='dot')))
    fig_proj.add_trace(go.Scatter(x=future_dates, y=lower_bound, mode='lines', name='Lower 95% Bound', line=dict(color='rgba(168, 85, 247, 0.4)', dash='dot'), fill='tonexty', fillcolor='rgba(168, 85, 247, 0.1)'))
    
    fig_proj.update_layout(
        title=f"IITL.NS {horizon_days}-Day Trend Extrapolation & Confidence Corridor (₹)",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15, 23, 42, 0.6)',
        height=450,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig_proj, use_container_width=True)
    
    p_c1, p_c2, p_c3 = st.columns(3)
    p_c1.metric("Projected Price (+30 Days)", f"₹{future_trend[min(29, horizon_days-1)]:.2f}", f"{(future_trend[min(29, horizon_days-1)]/latest_price - 1)*100:+.2f}%")
    p_c2.metric(f"Projected Price (+{horizon_days} Days)", f"₹{future_trend[-1]:.2f}", f"{(future_trend[-1]/latest_price - 1)*100:+.2f}%")
    p_c3.metric("Trend Slope (₹ / Day)", f"{poly_fit[0]:+.4f}", "Daily Drift")

# ---------------------------------------------------------
# TAB 6: Raw Data & Export
# ---------------------------------------------------------
with tab6:
    st.subheader("Data Explorer & Export Center")
    
    st.dataframe(
        df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Daily_Return', 'SMA_20', 'RSI_14', 'Drawdown']],
        column_config={
            "Date": st.column_config.DateColumn("Date", format="YYYY-MM-DD"),
            "Open": st.column_config.NumberColumn("Open (₹)", format="₹%.2f"),
            "High": st.column_config.NumberColumn("High (₹)", format="₹%.2f"),
            "Low": st.column_config.NumberColumn("Low (₹)", format="₹%.2f"),
            "Close": st.column_config.NumberColumn("Close (₹)", format="₹%.2f"),
            "Volume": st.column_config.NumberColumn("Volume", format="%d"),
            "Daily_Return": st.column_config.NumberColumn("Daily Return", format="%.4f"),
            "SMA_20": st.column_config.NumberColumn("SMA 20", format="₹%.2f"),
            "RSI_14": st.column_config.NumberColumn("RSI (14)", format="%.2f"),
            "Drawdown": st.column_config.NumberColumn("Drawdown %", format="%.2f%%")
        },
        use_container_width=True,
        height=380
    )
    
    # Download Button for Processed Data
    csv_data = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Filtered Dataset as CSV",
        data=csv_data,
        file_name=f"IITL_NS_filtered_{start_date}_{end_date}.csv",
        mime="text/csv",
        use_container_width=True
    )
    
    st.markdown("### Descriptive Statistics")
    st.dataframe(df[['Open', 'High', 'Low', 'Close', 'Volume']].describe(), use_container_width=True)

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown("""
<div style="text-align: center; margin-top: 50px; padding: 20px; border-top: 1px solid rgba(255,255,255,0.08); color: #64748B; font-size: 0.8rem;">
    IITL Stock Analytics Dashboard • Built with Streamlit & Plotly • Deployed on Streamlit Cloud
</div>
""", unsafe_allow_html=True)
