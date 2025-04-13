from fastapi.responses import JSONResponse
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
from LSTMLoader import LOADED_MODELS


# RSI Compute function
def compute_RSI(data, window=14):
    """
    Calculate the Relative Strength Index (RSI) for a given DataFrame.
    """
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    # Calculate the rolling mean of gains and losses
    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_bollinger_bands(data, window=20, num_std=2):
    """
    Calculate Bollinger Bands: middle band is the SMA,
    while upper and lower bands are set num_std standard deviations away.
    """
    sma = data['Close'].rolling(window=window).mean()
    std = data['Close'].rolling(window=window).std()
    upper_band = sma + num_std * std
    lower_band = sma - num_std * std
    return sma, upper_band, lower_band

def compute_ATR(data, window=14):
    """
    Calculate the Average True Range (ATR) using a rolling window.
    """
    high_low = data['High'] - data['Low']
    high_close = (data['High'] - data['Close'].shift()).abs()
    low_close = (data['Low'] - data['Close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=window).mean()
    return atr

def compute_sharpe_ratio(data, window=14, risk_free_rate=0):
    """
    Calculate a rolling Sharpe Ratio using daily returns.
    The ratio is annualized by multiplying by sqrt(252) (typical number of trading days).
    """
    daily_return = data['Close'].pct_change()
    rolling_mean = daily_return.rolling(window=window).mean()
    rolling_std = daily_return.rolling(window=window).std()
    sharpe = (rolling_mean - risk_free_rate) / rolling_std * np.sqrt(252)
    return sharpe


def get_predicted_price(ticker):
    # --- 1. Find the Loaded Model ---
    model_entry = next((m for m in LOADED_MODELS if m["name"].lower() == ticker.lower()), None)
    if model_entry is None:
        return JSONResponse(status_code=404, content={"detail": f"No loaded model found for ticker {ticker}"})
    model = model_entry["model"]


    # --- 2. Download Recent Data ---
    # Download approximately the last 2 years of data
    data = yf.download(ticker, start="2000-01-01", end="2025-04-11")
    if data.empty:
        return JSONResponse(status_code=404, content={"detail": f"No market data found for ticker {ticker}"})

    # --- 3. Compute Technical Indicators ---
    data['RSI'] = compute_RSI(data, window=14)
    data['BB_Middle'], data['BB_Upper'], data['BB_Lower'] = compute_bollinger_bands(data, window=20, num_std=2)

    window_ma = 20
    data['SMA'] = data['Close'].rolling(window=window_ma).mean()
    data['EMA'] = data['Close'].ewm(span=window_ma, adjust=False).mean()

    ema_12 = data['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = ema_12 - ema_26
    data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()

    data['ATR'] = compute_ATR(data, window=14)
    data['Sharpe_Ratio'] = compute_sharpe_ratio(data, window=14, risk_free_rate=0)

    stock_info = yf.Ticker(ticker).info
    pe_ratio = stock_info.get('trailingPE', np.nan)
    data['PE_Ratio'] = pe_ratio

    # Drop rows with missing values from technical indicator computations
    #data = data.dropna()
    #if data.empty:
    #    return JSONResponse(status_code=400, content={"detail": "Not enough processed data for prediction after computing features."})

    data['Close_1year'] = data['Close'].shift(-252)

    temp_data = data.copy()
    # Drop rows with NaN values coming from indicator computations and from the target shift
    final_data = data.dropna().copy()

    # Inspect the last few rows of the data to check the new target and features
    #print(final_data[['Close', 'Close_1year', 'RSI', 'BB_Middle', 'BB_Upper', 'BB_Lower',
                    #'SMA', 'EMA', 'MACD', 'MACD_Signal', 'ATR', 'Sharpe_Ratio', 'PE_Ratio']].tail())

    final_data.columns = ['Close', 'High', 'Low', 'Open', 'Volume',
                        'RSI', 'BB_Middle', 'BB_Upper', 'BB_Lower',
                        'SMA', 'EMA', 'MACD', 'MACD_Signal',
                        'ATR', 'Sharpe_Ratio', 'PE_Ratio', 'Close_1year']

    data = final_data.copy()

    # Set today's date dynamically.
    today = datetime.now()

    # Fixed training start date.
    train_start = datetime(2017, 1, 1)

    # The usual split date: one year before today.
    split_date = today - timedelta(days=365)
    # To include one extra day for future trial (as context), we set its start to be one day earlier.

    # Assign training data: from train_start up to the split date.
    train_data = data[(data.index >= train_start) & (data.index < split_date)].copy()

    # Assign future trial data: from the split date up to today.
    future_trial_data = temp_data[(temp_data.index >= split_date) & (temp_data.index < today)].copy()

    features = ['RSI', 'BB_Middle', 'BB_Upper', 'BB_Lower',
                'SMA', 'EMA', 'MACD', 'MACD_Signal',
                'ATR', 'Sharpe_Ratio', 'PE_Ratio',
                'Open', 'High', 'Low', 'Close']
    target = 'Close_1year'

    X_train = train_data[features]
    y_train = train_data[target]
    future_trial_data_test = future_trial_data[features]
    future_trial_data_test_target = future_trial_data[target]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    future_trial_data_scaled = scaler.transform(future_trial_data_test)

    lookback = 1  # Change this to >1 if you want sequences over multiple days

    def reshape_for_lstm(X, lookback):
        # If lookback==1, we simply add a time dimension.
        return np.reshape(X, (X.shape[0] - lookback + 1, lookback, X.shape[1]))

    # For a basic model we use lookback = 1. If you decide to use a larger lookback, you must adjust the target accordingly.
    X_train_seq = reshape_for_lstm(X_train_scaled, lookback)
    future_trial_data_seq = reshape_for_lstm(future_trial_data_scaled, lookback)

    y_train_seq = y_train.values[(lookback - 1):]

    train_predictions = model.predict(X_train_seq).flatten()
    future_trial_predictions = model.predict(future_trial_data_seq).flatten()

    '''plt.figure(figsize=(12, 6))
    # Plot future trial actual and predictions
    plt.plot(future_trial_data.index[(lookback - 1):], future_trial_predictions, label='Future Trial Predicted Close_1year', linewidth=2)
    plt.plot(train_data.index[(lookback - 1):], y_train_seq, label='Training Actual Close_1year', linewidth=2, linestyle='--', color='gray')
    #plt.plot(train_data.index[(lookback - 1):], train_predictions, label='Training Predicted Close_1year', linestyle='--', linewidth=2)

    plt.xlabel('Date')
    plt.ylabel('Close_1year')
    plt.title('Actual vs Predicted Close_1year Over Time')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()'''

    train_data = temp_data[(temp_data.index >= train_start) & (temp_data.index <= today)].copy()
    print(train_data.tail())
    prev_model_pred = future_trial_predictions[0]
    prev_actual_price = train_data['Close'].iloc[-1]
    scaled_future_preds = []
    for pred in future_trial_predictions[1:]:
        new_pred = (pred / prev_model_pred) * prev_actual_price
        scaled_future_preds.append(new_pred)
        prev_actual_price = new_pred
        prev_model_pred = pred

    # Convert to a numpy array if needed
    scaled_future_preds = np.array(scaled_future_preds)

    #Remove first element of future_trial_data
    temp_future_data = future_trial_data.iloc[1:]
    temp_future_data['Scaled_Predicted_Close_1year'] = scaled_future_preds
    temp_future_data.index = temp_future_data.index + pd.DateOffset(years=1)
    # Create DataFrames as before:
    df_train = pd.DataFrame({
        'Date': train_data.index,
        'Close': train_data['Close']
    })
    df_future = pd.DataFrame({
        'Date': temp_future_data.index,
        'Close': scaled_future_preds
    })
    combined_df = pd.concat([df_train, df_future]).reset_index(drop=True)
    combined_df = combined_df.sort_values(by='Date').reset_index(drop=True)

    plt.figure(figsize=(12, 6))
    plt.plot(combined_df['Date'], combined_df['Close'], marker='o', linestyle='-', label='Close')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.title('Combined Training and Future Predicted Close Prices')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    '''plt.figure(figsize=(12, 6))
    plt.plot(train_data.index, train_data['Close'], label='Training Actual Close', linewidth=2)
    plt.plot(temp_future_data.index, scaled_future_preds, label='Future Trial Scaled Prediction', linestyle='--', linewidth=2)
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.title('Actual vs Scaled Predicted Stock Price Over Time')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Save the plot to a temporary file
    os.makedirs("figures", exist_ok=True)
    figure_path = f"figures/prediction_{ticker}.png"
    plt.savefig(figure_path, dpi=300)
    plt.close()'''

    # --- 8. Return the Graph Location and Prediction ---
    return JSONResponse(
        status_code=200,
        content={
            "message": f"Prediction graph generated for ticker {ticker}.",
            "predicted_close_1year": future_trial_predictions[-1],
            "figure_path": figure_path
        }
    )