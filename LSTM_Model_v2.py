import os
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from datetime import datetime, timedelta
import ast


# Technical Indicator functions (unchanged)
def compute_RSI(data, window=14):
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_bollinger_bands(data, window=20, num_std=2):
    sma = data['Close'].rolling(window=window).mean()
    std = data['Close'].rolling(window=window).std()
    upper_band = sma + num_std * std
    lower_band = sma - num_std * std
    return sma, upper_band, lower_band

def compute_ATR(data, window=14):
    high_low = data['High'] - data['Low']
    high_close = (data['High'] - data['Close'].shift()).abs()
    low_close = (data['Low'] - data['Close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=window).mean()
    return atr

def compute_sharpe_ratio(data, window=14, risk_free_rate=0):
    daily_return = data['Close'].pct_change()
    rolling_mean = daily_return.rolling(window=window).mean()
    rolling_std = daily_return.rolling(window=window).std()
    sharpe = (rolling_mean - risk_free_rate) / rolling_std * np.sqrt(252)
    return sharpe

# Define a function that encapsulates all steps for one ticker symbol.
def train_and_save_model(ticker_symbol):
    print(f"\nProcessing ticker: {ticker_symbol}")

    # Download data
    data = yf.download(ticker_symbol, start="2000-01-01", end="2025-04-11")
    data.dropna(inplace=True)

    # Compute technical indicators
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

    # Example of adding a fundamental indicator (here we use trailing P/E ratio)
    stock_info = yf.Ticker(ticker_symbol).info
    pe_ratio = stock_info.get('trailingPE', np.nan)
    data['PE_Ratio'] = pe_ratio

    # Generate the target variable: Close Price after 1 Year (~252 trading days ahead)
    data['Close_1year'] = data['Close'].shift(-252)

    temp_data = data.copy()
    temp_data.columns = ['Close', 'High', 'Low', 'Open', 'Volume',
                         'RSI', 'BB_Middle', 'BB_Upper', 'BB_Lower',
                         'SMA', 'EMA', 'MACD', 'MACD_Signal',
                         'ATR', 'Sharpe_Ratio', 'PE_Ratio', 'Close_1year']
    # Prepare the final dataset by dropping rows with NaN values
    final_data = data.dropna().copy()
    final_data.columns = ['Close', 'High', 'Low', 'Open', 'Volume',
                           'RSI', 'BB_Middle', 'BB_Upper', 'BB_Lower',
                           'SMA', 'EMA', 'MACD', 'MACD_Signal',
                           'ATR', 'Sharpe_Ratio', 'PE_Ratio', 'Close_1year']

    # Split the data (example: use data from 2018 to 2022 for training, and later periods for testing/future trial)
    train_data = final_data[(final_data.index >= '2017-01-01')  & (final_data.index < '2024-04-11')].copy()
    future_trial_data = temp_data[(temp_data.index >= '2024-04-11') & (temp_data.index < '2025-04-12')].copy()

    features = ['RSI', 'BB_Middle', 'BB_Upper', 'BB_Lower', 'SMA', 'EMA',
                'MACD', 'MACD_Signal', 'ATR', 'Sharpe_Ratio', 'PE_Ratio',
                'Open', 'High', 'Low', 'Close']
    target = 'Close_1year'

    X_train = train_data[features]
    y_train = train_data[target]
    future_trial_data_test = future_trial_data[features]
    future_trial_data_test_target = future_trial_data[target]

    # Optional Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    future_trial_data_scaled = scaler.transform(future_trial_data_test)

    # Prepare Data for the LSTM model
    lookback = 1  # If you wish to change this, you must also adjust the target arrays accordingly.

    def reshape_for_lstm(X, lookback):
        return np.reshape(X, (X.shape[0] - lookback + 1, lookback, X.shape[1]))

    X_train_seq = reshape_for_lstm(X_train_scaled, lookback)
    future_trial_data_seq = reshape_for_lstm(future_trial_data_scaled, lookback)

    y_train_seq = y_train.values[(lookback - 1):]
    future_trial_target_seq = future_trial_data_test_target.values[(lookback - 1):]

    # ----------------------------
    # Build the LSTM Model (MODEL LOGIC - DO NOT CHANGE)
    # ----------------------------
    model = Sequential()
    model.add(LSTM(units=50, activation='relu', input_shape=(lookback, X_train_scaled.shape[1])))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mae')

    early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
    history = model.fit(X_train_seq, y_train_seq,
                        epochs=90,
                        batch_size=32,
                        verbose=1,
                        callbacks=[early_stopping])

    # Make Predictions (optional, here for diagnostic purposes)
    train_predictions = model.predict(X_train_seq).flatten()
    future_trial_predictions = model.predict(future_trial_data_seq).flatten()

    print("Training R²:", r2_score(y_train_seq, train_predictions))
    train_mse = mean_squared_error(y_train_seq, train_predictions)
    train_r2 = r2_score(y_train_seq, train_predictions)
    print(f"Training MSE: {train_mse:.4f}")
    print(f"Training R²: {train_r2:.4f}")

    # Save model with a unique name for the ticker symbol
    model_name = ticker_symbol
    model.save('models/' + model_name + '.h5')
    print(f"Model for {ticker_symbol} saved as models/{model_name}.h5")

    # Optionally, plot predictions vs. actual values
    plt.figure(figsize=(12, 6))
    plt.plot(train_data.index[(lookback - 1):], y_train_seq, label='Training Actual Close_1year', linewidth=2)
    plt.plot(future_trial_data.index[(lookback - 1):], future_trial_predictions, label='Future Trial Predicted Close_1year', linewidth=2)
    #plt.plot(train_data.index[(lookback - 1):], train_predictions, label='Training Predicted Close_1year', linestyle='--', linewidth=2)
    plt.xlabel('Date')
    plt.ylabel('Close_1year')
    plt.title(f'Actual vs Predicted Close_1year for {ticker_symbol}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # Check if the directory exists, if not, create it
    os.makedirs('figures', exist_ok=True)
    plt.savefig(f"figures/future_trial_predictions_{ticker_symbol}.png", dpi=300)

# ----------------------------
# Loop over different ticker symbols and run the full process
# ----------------------------


'''ticker_symbols = []

with open('all_tickers.txt', 'r', encoding='utf-8') as f:
        input = f.read()

full_dict = ast.literal_eval(input)
ticker_symbols = list(full_dict.keys())

tickers = ticker_symbols'''

url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'

# Read all the tables in the Wikipedia page; this returns a list of DataFrames
tables = pd.read_html(url)

# The first table in the list usually contains the S&P 500 companies data
sp500_table = tables[0]

# Display the first few rows to inspect the table structure
#print(sp500_table.head())

# Get the tickers from the 'Symbol' column as a list
tickers = sp500_table['Symbol'].tolist()

# Print the list of tickers
print(tickers)
print(len(tickers))
#print(tickers[0:5])
#tickers = ['AAPL', 'MSFT', 'GOOGL', 'SPY', 'KO', 'TSLA']  # Replace or extend the list with your desired tickers

for ticker in tickers:
    # Check if the ticker throws an error when downloading data
    try:
        train_and_save_model(ticker)
    except Exception as e:
        print(f"Error downloading data for {ticker}: {e}")
        continue
=======
import os
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from datetime import datetime, timedelta
import ast


# Technical Indicator functions (unchanged)
def compute_RSI(data, window=14):
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_bollinger_bands(data, window=20, num_std=2):
    sma = data['Close'].rolling(window=window).mean()
    std = data['Close'].rolling(window=window).std()
    upper_band = sma + num_std * std
    lower_band = sma - num_std * std
    return sma, upper_band, lower_band

def compute_ATR(data, window=14):
    high_low = data['High'] - data['Low']
    high_close = (data['High'] - data['Close'].shift()).abs()
    low_close = (data['Low'] - data['Close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=window).mean()
    return atr

def compute_sharpe_ratio(data, window=14, risk_free_rate=0):
    daily_return = data['Close'].pct_change()
    rolling_mean = daily_return.rolling(window=window).mean()
    rolling_std = daily_return.rolling(window=window).std()
    sharpe = (rolling_mean - risk_free_rate) / rolling_std * np.sqrt(252)
    return sharpe

# Define a function that encapsulates all steps for one ticker symbol.
def train_and_save_model(ticker_symbol):
    print(f"\nProcessing ticker: {ticker_symbol}")

    # Download data
    data = yf.download(ticker_symbol, start="2000-01-01", end="2025-04-11")
    data.dropna(inplace=True)

    # Compute technical indicators
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

    # Example of adding a fundamental indicator (here we use trailing P/E ratio)
    stock_info = yf.Ticker(ticker_symbol).info
    pe_ratio = stock_info.get('trailingPE', np.nan)
    data['PE_Ratio'] = pe_ratio

    # Generate the target variable: Close Price after 1 Year (~252 trading days ahead)
    data['Close_1year'] = data['Close'].shift(-252)

    temp_data = data.copy()
    temp_data.columns = ['Close', 'High', 'Low', 'Open', 'Volume',
                         'RSI', 'BB_Middle', 'BB_Upper', 'BB_Lower',
                         'SMA', 'EMA', 'MACD', 'MACD_Signal',
                         'ATR', 'Sharpe_Ratio', 'PE_Ratio', 'Close_1year']
    # Prepare the final dataset by dropping rows with NaN values
    final_data = data.dropna().copy()
    final_data.columns = ['Close', 'High', 'Low', 'Open', 'Volume',
                           'RSI', 'BB_Middle', 'BB_Upper', 'BB_Lower',
                           'SMA', 'EMA', 'MACD', 'MACD_Signal',
                           'ATR', 'Sharpe_Ratio', 'PE_Ratio', 'Close_1year']

    # Split the data (example: use data from 2018 to 2022 for training, and later periods for testing/future trial)
    train_data = final_data[(final_data.index >= '2017-01-01')  & (final_data.index < '2024-04-11')].copy()
    future_trial_data = temp_data[(temp_data.index >= '2024-04-11') & (temp_data.index < '2025-04-12')].copy()

    features = ['RSI', 'BB_Middle', 'BB_Upper', 'BB_Lower', 'SMA', 'EMA',
                'MACD', 'MACD_Signal', 'ATR', 'Sharpe_Ratio', 'PE_Ratio',
                'Open', 'High', 'Low', 'Close']
    target = 'Close_1year'

    X_train = train_data[features]
    y_train = train_data[target]
    future_trial_data_test = future_trial_data[features]
    future_trial_data_test_target = future_trial_data[target]

    # Optional Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    future_trial_data_scaled = scaler.transform(future_trial_data_test)

    # Prepare Data for the LSTM model
    lookback = 1  # If you wish to change this, you must also adjust the target arrays accordingly.

    def reshape_for_lstm(X, lookback):
        return np.reshape(X, (X.shape[0] - lookback + 1, lookback, X.shape[1]))

    X_train_seq = reshape_for_lstm(X_train_scaled, lookback)
    future_trial_data_seq = reshape_for_lstm(future_trial_data_scaled, lookback)

    y_train_seq = y_train.values[(lookback - 1):]
    future_trial_target_seq = future_trial_data_test_target.values[(lookback - 1):]

    # ----------------------------
    # Build the LSTM Model (MODEL LOGIC - DO NOT CHANGE)
    # ----------------------------
    model = Sequential()
    model.add(LSTM(units=50, activation='relu', input_shape=(lookback, X_train_scaled.shape[1])))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mae')

    early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
    history = model.fit(X_train_seq, y_train_seq,
                        epochs=90,
                        batch_size=32,
                        verbose=1,
                        callbacks=[early_stopping])

    # Make Predictions (optional, here for diagnostic purposes)
    train_predictions = model.predict(X_train_seq).flatten()
    future_trial_predictions = model.predict(future_trial_data_seq).flatten()

    print("Training R²:", r2_score(y_train_seq, train_predictions))
    train_mse = mean_squared_error(y_train_seq, train_predictions)
    train_r2 = r2_score(y_train_seq, train_predictions)
    print(f"Training MSE: {train_mse:.4f}")
    print(f"Training R²: {train_r2:.4f}")

    # Save model with a unique name for the ticker symbol
    model_name = ticker_symbol
    model.save('models/' + model_name + '.h5')
    print(f"Model for {ticker_symbol} saved as models/{model_name}.h5")

    # Optionally, plot predictions vs. actual values
    plt.figure(figsize=(12, 6))
    plt.plot(train_data.index[(lookback - 1):], y_train_seq, label='Training Actual Close_1year', linewidth=2)
    plt.plot(future_trial_data.index[(lookback - 1):], future_trial_predictions, label='Future Trial Predicted Close_1year', linewidth=2)
    #plt.plot(train_data.index[(lookback - 1):], train_predictions, label='Training Predicted Close_1year', linestyle='--', linewidth=2)
    plt.xlabel('Date')
    plt.ylabel('Close_1year')
    plt.title(f'Actual vs Predicted Close_1year for {ticker_symbol}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # Check if the directory exists, if not, create it
    os.makedirs('figures', exist_ok=True)
    plt.savefig(f"figures/future_trial_predictions_{ticker_symbol}.png", dpi=300)

# ----------------------------
# Loop over different ticker symbols and run the full process
# ----------------------------


'''ticker_symbols = []

with open('all_tickers.txt', 'r', encoding='utf-8') as f:
        input = f.read()

full_dict = ast.literal_eval(input)
ticker_symbols = list(full_dict.keys())

tickers = ticker_symbols'''

url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'

# Read all the tables in the Wikipedia page; this returns a list of DataFrames
tables = pd.read_html(url)

# The first table in the list usually contains the S&P 500 companies data
sp500_table = tables[0]

# Display the first few rows to inspect the table structure
#print(sp500_table.head())

# Get the tickers from the 'Symbol' column as a list
tickers = sp500_table['Symbol'].tolist()

# Print the list of tickers
print(tickers)
print(len(tickers))
#print(tickers[0:5])
#tickers = ['AAPL', 'MSFT', 'GOOGL', 'SPY', 'KO', 'TSLA']  # Replace or extend the list with your desired tickers

for ticker in tickers:
    # Check if the ticker throws an error when downloading data
    try:
        train_and_save_model(ticker)
    except Exception as e:
        print(f"Error downloading data for {ticker}: {e}")
        continue
