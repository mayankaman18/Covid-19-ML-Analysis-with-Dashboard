import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
from covid_analysis import load_and_preprocess_data

# --- PAGE CONFIG ---
st.set_page_config(page_title="COVID-19 ML Dashboard", page_icon="🦠", layout="wide")

# --- CUSTOM CSS FOR PREMIUM DARK MODE & GLASSMORPHISM ---
st.markdown("""
<style>
    /* Global Background and Text */
    .stApp {
        background-color: #0e1117;
        color: #e0e6ed;
        font-family: 'Inter', sans-serif;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #ffffff !important;
        font-weight: 700;
    }
    
    /* Glassmorphism Metric Cards */
    div[data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        transition: transform 0.2s ease-in-out;
    }
    
    div[data-testid="metric-container"]:hover {
        transform: translateY(-5px);
        border-color: rgba(255, 255, 255, 0.2);
    }

    /* Metric Label */
    div[data-testid="stMetricLabel"] {
        color: #a3a8b8 !important;
        font-size: 1rem !important;
    }
    
    /* Metric Value */
    div[data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 2rem !important;
        font-weight: 800 !important;
    }
    
    /* Custom divider */
    hr {
        border-color: rgba(255, 255, 255, 0.1) !important;
    }
</style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.title("🦠 Global COVID-19 Insights & Forecasting")
st.markdown("<p style='color: #a3a8b8; font-size: 1.1rem;'>Advanced analytics and predictive modeling for global pandemic data.</p>", unsafe_allow_html=True)
st.markdown("---")

# --- SIDEBAR & DATA LOADING ---
@st.cache_data
def load_data():
    df = load_and_preprocess_data(country=None)
    # Ensure date is index and sorted
    if 'date' in df.columns:
        df.set_index('date', inplace=True)
    df.sort_index(inplace=True)
    return df

with st.spinner("Loading Global Dataset..."):
    full_df = load_data()

available_countries = sorted(full_df['location'].dropna().unique().tolist())

st.sidebar.header("⚙️ Dashboard Controls")

# Select Country
selected_country = st.sidebar.selectbox(
    "🌍 Select Region", 
    ["United States"] + [c for c in available_countries if c != "United States"]
)

# Filter by Date
min_date = full_df.index.min().date()
max_date = full_df.index.max().date()

st.sidebar.markdown("### 📅 Timeframe")
date_range = st.sidebar.slider(
    "Select Date Range",
    min_value=min_date,
    max_value=max_date,
    value=(min_date, max_date)
)

# Apply filters
start_date, end_date = date_range
mask = (full_df.index.date >= start_date) & (full_df.index.date <= end_date)
filtered_full_df = full_df.loc[mask]

df = filtered_full_df[filtered_full_df['location'] == selected_country].copy()

# Dataset Summary in Sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Data Summary")
st.sidebar.markdown(f"**Records for {selected_country}:** {len(df):,}")
if st.sidebar.button("⬇️ Download Data as CSV"):
    csv = df.to_csv()
    st.sidebar.download_button(
        label="Download",
        data=csv,
        file_name=f"covid_data_{selected_country}.csv",
        mime='text/csv'
    )

if len(df) == 0:
    st.error(f"No data available for {selected_country} in the selected date range.")
    st.stop()

# --- GLOBAL MAP VISUALIZATION ---
st.header("🌐 Global Overview")
st.markdown(f"Total Confirmed Cases as of **{end_date}**")

with st.spinner("Rendering Global Map..."):
    # Get the latest data point for each country within the date range
    latest_data = filtered_full_df.reset_index().groupby('location').last().reset_index()
    
    fig_map = px.choropleth(
        latest_data,
        locations="location",
        locationmode="country names",
        color="total_cases",
        hover_name="location",
        color_continuous_scale=px.colors.sequential.Plasma,
        template="plotly_dark"
    )
    fig_map.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        geo=dict(showframe=False, showcoastlines=True, projection_type='equirectangular', bgcolor='rgba(0,0,0,0)'),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig_map, width="stretch")

st.markdown("---")

# --- TOP LEVEL METRICS ---
st.header(f"📍 Deep Dive: {selected_country}")

total_cases = int(df['total_cases'].max())
total_deaths = int(df['total_deaths'].max())

# Calculate Deltas (Current week vs Previous week)
if len(df) >= 14:
    recent_7_days = df['new_cases'].iloc[-7:].sum()
    prev_7_days = df['new_cases'].iloc[-14:-7].sum()
    cases_delta = int(recent_7_days - prev_7_days)
    
    recent_7_deaths = df['total_deaths'].iloc[-1] - df['total_deaths'].iloc[-8]
    prev_7_deaths = df['total_deaths'].iloc[-8] - df['total_deaths'].iloc[-15]
    deaths_delta = int(recent_7_deaths - prev_7_deaths)
else:
    cases_delta = None
    deaths_delta = None

latest_new_cases = int(df['new_cases'].iloc[-1])
cfr = (total_deaths / total_cases * 100) if total_cases > 0 else 0

m1, m2, m3, m4 = st.columns(4)
m1.metric("Confirmed Cases", f"{total_cases:,}", delta=f"{cases_delta:,} (vs prev 7d)" if cases_delta else None, delta_color="inverse")
m2.metric("Total Deaths", f"{total_deaths:,}", delta=f"{deaths_delta:,} (vs prev 7d)" if deaths_delta else None, delta_color="inverse")
m3.metric("Latest New Cases", f"{latest_new_cases:,}")
m4.metric("Case Fatality Rate", f"{cfr:.2f}%")

st.markdown("<br>", unsafe_allow_html=True)

# --- EDA PLOTS ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Daily New Cases & 7-Day Rolling Average")
    df['new_cases_7d_avg'] = df['new_cases'].rolling(window=7).mean()
    
    fig_new_cases = go.Figure()
    fig_new_cases.add_trace(go.Bar(x=df.index, y=df['new_cases'], name="Daily New Cases", marker_color='rgba(100, 150, 255, 0.3)'))
    fig_new_cases.add_trace(go.Scatter(x=df.index, y=df['new_cases_7d_avg'], mode='lines', name="7-Day Rolling Avg", line=dict(color='#00ffcc', width=3)))
    
    fig_new_cases.update_layout(
        template="plotly_dark",
        height=400, 
        margin=dict(l=0, r=0, b=0, t=30), 
        hovermode="x unified",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    st.plotly_chart(fig_new_cases, width="stretch")

with col2:
    st.subheader("Cumulative Cases & Deaths (Log Scale)")
    fig_cum = go.Figure()
    fig_cum.add_trace(go.Scatter(x=df.index, y=df['total_cases'], mode='lines', name='Total Cases', fill='tozeroy', line=dict(color='#3b82f6'), fillcolor='rgba(59, 130, 246, 0.2)'))
    fig_cum.add_trace(go.Scatter(x=df.index, y=df['total_deaths'], mode='lines', name='Total Deaths', fill='tozeroy', line=dict(color='#ef4444'), fillcolor='rgba(239, 68, 68, 0.2)'))
    
    fig_cum.update_layout(
        template="plotly_dark",
        yaxis_type="log", 
        height=400, 
        margin=dict(l=0, r=0, b=0, t=30), 
        hovermode="x unified",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    st.plotly_chart(fig_cum, width="stretch")

st.markdown("---")

# --- ARIMA FORECASTING ---
st.header("📈 Predictive Analytics: Time Series Forecasting")
st.markdown("Forecasting **Weekly Average New Cases** to smooth out daily noise and predict upcoming trends.")

# Preprocess for ARIMA
weekly_cases = df['new_cases'].resample('W').mean()
# FIXED DEPRECATED PANDAS METHOD: replaced fillna(method='ffill') with ffill()
weekly_cases = weekly_cases.ffill()

if len(weekly_cases) < 10:
    st.warning("Not enough data points in the selected date range for accurate time-series forecasting. Please expand the date range.")
else:
    # Train test split
    train_size = int(len(weekly_cases) * 0.8)
    train, test = weekly_cases.iloc[:train_size], weekly_cases.iloc[train_size:]
    
    with st.spinner("Training ARIMA Model on selected timeframe..."):
        try:
            model = ARIMA(train, order=(5, 1, 0))
            model_fit = model.fit()
            predictions = model_fit.forecast(steps=len(test))
            
            mae = mean_absolute_error(test, predictions)
            rmse = np.sqrt(mean_squared_error(test, predictions))
            
            pcol1, pcol2 = st.columns([1, 2.5])
            with pcol1:
                st.success("✅ Model Trained Successfully!")
                st.markdown("#### Model Performance")
                st.metric("Mean Absolute Error", f"{mae:,.0f}")
                st.metric("Root Mean Sq Error", f"{rmse:,.0f}")
                
                st.markdown("#### Dataset Split")
                st.info(f"**Training:** {len(train)} weeks\n\n**Testing:** {len(test)} weeks")
                
            with pcol2:
                fig_arima = go.Figure()
                fig_arima.add_trace(go.Scatter(x=train.index, y=train, mode='lines', name='Train Actuals', line=dict(color='#8b5cf6', width=2)))
                fig_arima.add_trace(go.Scatter(x=test.index, y=test, mode='lines', name='Test Actuals', line=dict(color='#0ea5e9', width=2)))
                fig_arima.add_trace(go.Scatter(x=test.index, y=predictions, mode='lines', name='Predictions', line=dict(color='#f43f5e', dash='dot', width=3)))
                
                fig_arima.update_layout(
                    template="plotly_dark",
                    height=400, 
                    margin=dict(l=0, r=0, b=0, t=10), 
                    hovermode="x unified",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig_arima, width="stretch")
                
        except Exception as e:
            st.error(f"Failed to fit ARIMA model on this subset: {e}")
