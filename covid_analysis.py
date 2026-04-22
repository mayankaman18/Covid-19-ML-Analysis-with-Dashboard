import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os

# %% [markdown]
# # COVID-19 Data Analysis & ARIMA Forecasting Project
# This script is designed to run in the terminal or be converted to a Jupyter Notebook.
# It performs Data Loading, Preprocessing, EDA, and Time Series Forecasting.

# %%
def load_and_preprocess_data(file_path="data/full_grouped.csv", country="United States"):
    print(f"Loading OWID COVID-19 Dataset from local file: {file_path}...")
    
    try:
        # Load all columns first
        df = pd.read_csv(file_path)
        
        # Rename columns to match OWID format if it's the Kaggle dataset
        rename_map = {
            'Date': 'date',
            'Country/Region': 'location',
            'Confirmed': 'total_cases',
            'Deaths': 'total_deaths',
            'New cases': 'new_cases'
        }
        df.rename(columns=rename_map, inplace=True)
        
        # The Kaggle dataset uses 'US' instead of 'United States'
        if 'location' in df.columns:
            df['location'] = df['location'].replace({'US': 'United States'})
            
        cols = ['location', 'date', 'total_cases', 'new_cases', 'total_deaths']
        # Filter only existing columns from cols list
        existing_cols = [c for c in cols if c in df.columns]
        df = df[existing_cols]
        
        print("Data loaded successfully from local file!")
    except Exception as e:
        print(f"Failed to download data ({e}). Generating synthetic data for demonstration.")
        dates = pd.date_range(start="2020-03-01", end="2023-01-01", freq="D")
        np.random.seed(42)
        new_cases = np.abs(np.random.normal(loc=50000, scale=20000, size=len(dates)))
        total_cases = np.cumsum(new_cases)
        total_deaths = total_cases * 0.015  # roughly 1.5% mortality
        df = pd.DataFrame({
            'iso_code': ['USA'] * len(dates),
            'location': ['United States'] * len(dates),
            'date': dates,
            'total_cases': total_cases,
            'new_cases': new_cases,
            'total_deaths': total_deaths
        })
    
    print(f"Total dataset shape: {df.shape}")
    
    # Preprocessing
    df['date'] = pd.to_datetime(df['date'])
    
    # Focus on Specific Country
    if country:
        print(f"Filtering data for {country}...")
        res_df = df[df['location'] == country].copy()
    else:
        res_df = df.copy()
        
    res_df.sort_values('date', inplace=True)
    res_df.set_index('date', inplace=True)
    
    # Fill missing values with forward fill, then 0 for any at the start
    res_df.ffill(inplace=True)
    res_df.fillna(0, inplace=True)
    
    print(f"Dataset shape after filtering and cleaning: {res_df.shape}")
    print(f"\nFirst 5 rows of available Data:")
    print(res_df.head())
    
    return res_df

# %%
def perform_eda(df):
    print("\nStarting Exploratory Data Analysis (EDA)...")
    
    if not os.path.exists("plots"):
        os.makedirs("plots")
        
    # 1. Correlation Matrix
    numeric_df = df[['total_cases', 'new_cases', 'total_deaths']]
    corr = numeric_df.corr()
    print("\nCorrelation Matrix:")
    print(corr)
    
    plt.figure(figsize=(6, 4))
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Matrix of US COVID Metrics')
    plt.tight_layout()
    plt.savefig('plots/correlation_matrix.png')
    plt.close()
    print("Correlation Heatmap saved to 'plots/correlation_matrix.png'")
    
    # 2. Daily New Cases Trend
    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df['new_cases'], color='blue', alpha=0.7)
    plt.title('Daily New COVID-19 Cases in the US')
    plt.xlabel('Date')
    plt.ylabel('New Cases')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('plots/daily_new_cases.png')
    plt.close()
    print("Daily New Cases trend plot saved to 'plots/daily_new_cases.png'")
    
    # 3. 7-Day Rolling Average of New Cases
    plt.figure(figsize=(10, 5))
    df_rolling = df.copy()
    df_rolling['new_cases_7d_avg'] = df_rolling['new_cases'].rolling(window=7).mean()
    plt.plot(df_rolling.index, df_rolling['new_cases'], color='lightgray', alpha=0.5, label='Daily New Cases')
    plt.plot(df_rolling.index, df_rolling['new_cases_7d_avg'], color='red', linewidth=2, label='7-Day Rolling Avg')
    plt.title('Daily New COVID-19 Cases in the US (with 7-Day Rolling Average)')
    plt.xlabel('Date')
    plt.ylabel('New Cases')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('plots/rolling_avg_cases.png')
    plt.close()
    print("7-Day Rolling Average plot saved to 'plots/rolling_avg_cases.png'")

    # 4. Cumulative Cases and Deaths (Area Plot)
    plt.figure(figsize=(10, 5))
    plt.fill_between(df.index, df['total_cases'], color='skyblue', alpha=0.6, label='Total Cases')
    plt.fill_between(df.index, df['total_deaths'], color='salmon', alpha=0.9, label='Total Deaths')
    plt.title('Cumulative COVID-19 Cases and Deaths in the US Over Time')
    plt.xlabel('Date')
    plt.ylabel('Count')
    plt.yscale('log') # Log scale because cases outnumber deaths significantly
    plt.legend(loc='upper left')
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig('plots/cumulative_area_plot.png')
    plt.close()
    print("Cumulative Area Plot saved to 'plots/cumulative_area_plot.png'")# %%
def build_arima_model(df):
    print("\nBuilding ARIMA Forecasting Model...")
    
    # We will forecast 'new_cases'.
    # We resample to weekly freq to smooth noise a bit and speed up ARIMA.
    weekly_cases = df['new_cases'].resample('W').mean()
    
    # Train-test split (80-20)
    train_size = int(len(weekly_cases) * 0.8)
    train, test = weekly_cases.iloc[:train_size], weekly_cases.iloc[train_size:]
    
    print(f"Training on {len(train)} weeks, Testing on {len(test)} weeks.")
    
    # Fitting ARIMA model (simplified order for speed: (5,1,0))
    # Note: In a real scenario we'd use auto_arima to find optimal p,d,q
    print("Fitting ARIMA(5,1,0) model. This may take a few seconds...")
    model = ARIMA(train, order=(5, 1, 0))
    model_fit = model.fit()
    
    print(model_fit.summary().tables[1])
    
    # Predictions
    print("\nGenerating predictions on test set...")
    predictions = model_fit.forecast(steps=len(test))
    
    # Metrics
    mae = mean_absolute_error(test, predictions)
    rmse = np.sqrt(mean_squared_error(test, predictions))
    
    print("Model Evaluation Metrics:")
    print(f"  - Mean Absolute Error (MAE): {mae:.2f}")
    print(f"  - Root Mean Squared Error (RMSE): {rmse:.2f}")
    
    # Plotting Train, Test, and Predictions
    plt.figure(figsize=(10, 5))
    plt.plot(train.index, train, label='Train')
    plt.plot(test.index, test, label='Test Actuals')
    plt.plot(test.index, predictions, color='red', label='Predictions')
    plt.title('ARIMA Forecast vs Actuals (Weekly Average New Cases)')
    plt.xlabel('Date')
    plt.ylabel('Weekly Avg New Cases')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('plots/arima_forecast.png')
    plt.close()
    print("Forecasting plot saved to 'plots/arima_forecast.png'")

# %%
if __name__ == "__main__":
    import sys
    dataset_path = sys.argv[1] if len(sys.argv) > 1 else "data/full_grouped.csv"
    us_data = load_and_preprocess_data(dataset_path)
    perform_eda(us_data)
    build_arima_model(us_data)
    print("\nAll tasks completed successfully!")
