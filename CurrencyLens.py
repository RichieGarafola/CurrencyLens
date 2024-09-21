# Import necessary libraries
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go  # Use plotly for interactive charts
import streamlit as st
from statsmodels.tsa.stattools import adfuller
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from prophet import Prophet
from datetime import datetime, timedelta


# Streamlit interface
st.title("Forex Analysis Dashboard")

# Define a list of Forex pairs you want to track
forex_pairs = ["AUDNZD=X", "AUDUSD=X", "EURCAD=X", "EURCHF=X", "EURGBP=X", "EURUSD=X", 
               "GBPCAD=X", "GBPCHF=X", "GBPUSD=X", "NZDUSD=X", "USDCAD=X", "USDCHF=X", 
               "USDJPY=X", "AUDJPY=X", "EURJPY=X", "GBPJPY=X"]

# Function to fetch historical data for a given Forex pair, date range, and interval
def fetch_forex_data(pair_symbol, start_date, end_date, interval):
    pair_ticker = yf.Ticker(pair_symbol)
    # Fetch data for the specified date range and interval
    return pair_ticker.history(start=start_date, end=end_date, interval=interval)

# Sidebar selection for Forex pair
selected_pair = st.sidebar.selectbox("Select Forex Pair", forex_pairs)

# Allow the user to select the start date
start_date = st.sidebar.date_input("Select Start Date", datetime(2020, 1, 1))

# Allow the user to select the end date (default to today's date)
end_date = st.sidebar.date_input("Select End Date", datetime.today())

# Sidebar selection for time interval
timeframe = st.sidebar.selectbox("Select Timeframe", ["1m", "5m", "15m", "1h", "1d", "1wk", "1mo"])

# Limit the date range based on the selected timeframe
if timeframe in ["1m", "5m", "15m", "1h"]:
    max_days = 30  # Limit to 30 days for 5m, 15m, and 1h intervals
    if (end_date - start_date).days > max_days:
        st.sidebar.warning(f"{timeframe} data is only available for up to {max_days} days.")
        start_date = end_date - timedelta(days=max_days)

# Fetch historical data for the selected Forex pair, date range, and timeframe
selected_data = fetch_forex_data(selected_pair, start_date, end_date, timeframe)

# Check if data is empty and handle accordingly
if selected_data.empty:
    st.warning(f"No data available for {selected_pair.replace('=X', '')} in the selected timeframe ({timeframe}). Please try a different range or interval.")
else:
    # Display Forex Pair Price Trends with Plotly Candlestick Chart
    st.subheader(f"{selected_pair.replace('=X', '')} Price Trends ({timeframe})")

    # Prepare data for the candlestick chart
    fig = go.Figure(data=[go.Candlestick(
        x=selected_data.index,
        open=selected_data['Open'],
        high=selected_data['High'],
        low=selected_data['Low'],
        close=selected_data['Close'],
        increasing_line_color='green', decreasing_line_color='red'
    )])

    # Update layout for zooming
    fig.update_layout(
        title=f"{selected_pair.replace('=X', '')} Candlestick Chart ({timeframe} data)",
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,  # Hide the range slider
        hovermode="x unified",  # Show hover information for x-axis
        dragmode="zoom"  # Enable zooming
    )

    # Display the Plotly figure within Streamlit
    st.plotly_chart(fig)

# Sidebar options for analysis
analysis_option = st.sidebar.radio("Choose Analysis", ["Daily Returns", "Rolling Statistics", "Bollinger Bands", "Price Prediction", "Monte Carlo Simulation", "Prophet Forecast", "Augmented Dickey-Fuller Test", "Correlation Matrix"])


if analysis_option == "Daily Returns":
    # Calculate daily returns for the selected forexx pair
    selected_data['Daily Returns'] = selected_data['Close'].pct_change()
    st.subheader(f"{selected_pair.replace('=X', '')} Daily Returns")
    st.line_chart(selected_data['Daily Returns'])
    
elif analysis_option == "Rolling Statistics":
    # Calculate rolling statistics (e.g., moving average and standard deviation) for forex prices
    window = st.sidebar.slider("Select Rolling Window", 1, 500, 20)
    selected_data[f'{window}-Day MA'] = selected_data['Close'].rolling(window=window).mean()
    selected_data[f'{window}-Day Std'] = selected_data['Close'].rolling(window=window).std()
    st.subheader(f"{selected_pair.replace('=X', '')} Rolling Statistics")
    st.line_chart(selected_data[[f'{window}-Day MA', f'{window}-Day Std']])
    
elif analysis_option == "Bollinger Bands":
    # Calculate Bollinger Bands for the selected forex
    window = st.sidebar.slider("Select Bollinger Bands Window", 1, 100, 30)
    selected_data[f'{window}-Day MA'] = selected_data['Close'].rolling(window=window).mean()
    selected_data[f'{window}-Day Std'] = selected_data['Close'].rolling(window=window).std()
    selected_data['Upper Band'] = selected_data[f'{window}-Day MA'] + (2 * selected_data[f'{window}-Day Std'])
    selected_data['Lower Band'] = selected_data[f'{window}-Day MA'] - (2 * selected_data[f'{window}-Day Std'])
    st.subheader(f"{selected_pair.replace('=X', '')} Bollinger Bands")
    st.line_chart(selected_data[['Close', 'Upper Band', 'Lower Band']])
    
elif analysis_option == "Price Prediction":
    # Machine Learning: Linear Regression for Price Prediction
    st.subheader(f"{selected_pair.replace('=X', '')} Price Prediction")

    # Data preparation
    selected_data = selected_data.dropna()
    X = np.arange(len(selected_data)).reshape(-1, 1)
    y = selected_data['Close']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict prices
    y_pred = model.predict(X_test)

    # Calculate Mean Squared Error
    mse = mean_squared_error(y_test, y_pred)

    # Plot the actual vs. predicted prices
    # Create a Matplotlib figure and axis within Streamlit
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_title(f"{selected_pair.replace('=X', '')} Price Prediction")
    ax.plot(selected_data.index[-len(X_test):], y_test, label="Actual Price", color='blue', linestyle='--')
    ax.plot(selected_data.index[-len(X_test):], y_pred, label="Predicted Price", color='red')
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    ax.grid()

    # Display the Matplotlib figure within Streamlit
    st.pyplot(fig)

    st.write(f"Mean Squared Error (MSE): {mse:.2f}")
    
elif analysis_option == "Monte Carlo Simulation":
    # Monte Carlo Simulation for Price Forecasting
    st.subheader(f"{selected_pair.replace('=X', '')} Price Forecast (Monte Carlo Simulation)")

    # Data preparation for simulation
    last_price = selected_data['Close'].iloc[-1]

    # Check if 'Daily Returns' column exists, if not, calculate it
    if 'Daily Returns' not in selected_data.columns:
        selected_data['Daily Returns'] = selected_data['Close'].pct_change()

    # Continue with the rest of the code
    volatility = selected_data['Daily Returns'].std()
    
    # Number of simulations and days
    num_simulations = st.sidebar.slider("Number of Simulations", 1, 1000, 100)
    num_days = st.sidebar.slider("Number of Days to Forecast", 1, 365, 30)

    # Monte Carlo simulation
    simulation_df = pd.DataFrame()
    for i in range(num_simulations):
        daily_returns = np.random.normal(0, volatility, num_days) + 1
        price_series = [last_price]
        for j in range(num_days):
            price_series.append(price_series[-1] * daily_returns[j])
        simulation_df[f'Simulation {i+1}'] = price_series

    # Visualize Monte Carlo simulations using Matplotlib
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_title(f"{selected_pair.replace('=X', '')} Price Forecast (Monte Carlo Simulation)")
    for i in range(num_simulations):
        ax.plot(simulation_df.index, simulation_df[f'Simulation {i+1}'], lw=1)
    ax.set_xlabel("Days")
    ax.set_ylabel("Price (USD)")
    ax.grid()

    # Display the Matplotlib figure within Streamlit
    st.pyplot(fig)
    
elif analysis_option == "Prophet Forecast":
    # Prophet Forecasting
    st.subheader(f"{selected_pair.replace('=X', '')} Price Forecast (Prophet Forecast)")

    # Ensure that 'Date' is not the index but a column in the DataFrame
    df_prophet = selected_data[['Close']].reset_index()  # Reset index to move the 'Date' index to a column

    # Identify the actual column name for the date (likely it will be the name of the index)
    date_column_name = df_prophet.columns[0]  # This should be the first column after reset_index

    # Rename the 'date_column_name' and 'Close' columns to 'ds' and 'y' as required by Prophet
    df_prophet.rename(columns={date_column_name: 'ds', 'Close': 'y'}, inplace=True)

    # Check the renamed columns
    st.write("Data for Prophet model:", df_prophet.head())

    # Remove timezone information from the 'ds' column (if any)
    df_prophet['ds'] = pd.to_datetime(df_prophet['ds']).dt.tz_localize(None)

    # Create a Prophet model
    model_prophet = Prophet(yearly_seasonality=True)

    # Fit the model with the historical data
    model_prophet.fit(df_prophet)

    # Create a DataFrame for future dates and make predictions
    future_prophet = model_prophet.make_future_dataframe(periods=52, freq="W")
    forecast_prophet = model_prophet.predict(future_prophet)

    # Plot predictions
    st.subheader(f"{selected_pair.replace('=X', '')} Prophet Forecast Predictions")
    fig, ax = plt.subplots()
    ax.plot(forecast_prophet['ds'], forecast_prophet['yhat'], label='yhat', color='blue')
    ax.fill_between(
        forecast_prophet['ds'],
        forecast_prophet['yhat_lower'],
        forecast_prophet['yhat_upper'],
        color='gray',
        alpha=0.2,
        label='yhat_lower and yhat_upper'
    )
    ax.legend()

    # Display the Prophet forecast plot
    st.pyplot(fig)

    # Display Prophet forecast components
    st.subheader(f"{selected_pair.replace('=X', '')} Prophet Forecast Components")
    fig_components = model_prophet.plot_components(forecast_prophet)
    st.pyplot(fig_components)

    # Display 21-day forecast
    st.subheader("21 Day Forecast")
    forecast_21_days = forecast_prophet[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(21)
    forecast_21_days.rename(columns={'ds': 'Date', 'yhat': 'Most Likely Case', 'yhat_lower': 'Worst Case', 'yhat_upper': 'Best Case'}, inplace=True)
    st.write(forecast_21_days)
    

# Augmented Dickey-Fuller Test
elif analysis_option == "Augmented Dickey-Fuller Test":
    result = adfuller(selected_data['Close'], autolag='AIC')
    st.subheader("Augmented Dickey-Fuller Test for Stationarity")
    st.write(f"ADF Statistic: {result[0]:.4f}")
    st.write(f"P-value: {result[1]:.4f}")
    st.write("Critical Values:")
    for key, value in result[4].items():
        st.write(f"  {key}: {value:.4f}")
    if result[1] <= 0.05:
        st.write("Stationary (Reject null hypothesis)")
    else:
        st.write("Non-Stationary (Fail to reject null hypothesis)")

# Correlation Matrix        
elif analysis_option == "Correlation Matrix":
    # Create a DataFrame to store closing prices of Forex pairs
    price_data = pd.DataFrame()

    # Fetch closing prices for all Forex pairs and store in the DataFrame
    for pair in forex_pairs:
        price_data[pair] = fetch_forex_data(pair, start_date, end_date, '1d')['Close']

    # Calculate the correlation matrix of the closing prices
    correlation_matrix = price_data.corr()

    # Display the correlation matrix as a heatmap using Seaborn
    st.subheader("Forex Closing Price Correlation Matrix")
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, cmap='coolwarm', annot=True, fmt=".2f", linewidths=.5, square=True, cbar_kws={"shrink": 0.75})
    st.pyplot(plt.gcf())