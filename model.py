# Import necessary libraries
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense




# ===============================
# 1. Download Stock Data Using yfinance
# ===============================
ticker = ['GOOGL']  # Google ticker symbol


# Download historical data from January 1, 2010 to January 1, 2023 for training
train_data = yf.download(ticker, start='2010-01-01', end='2018-01-01')
print("Training data head:")
print(train_data.head())
# Focus on the 'Close' price
data_Close = train_data[['Close']]


# ===============================
# 2. Preprocess the Data for Training
# ===============================
# Scale the closing prices to the range [0, 1]
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data_Close)


# Create training sequences using a sliding window
window_size = 21


X_train = []
y_train = []


for i in range(window_size, len(scaled_data)):
    X_train.append(scaled_data[i-window_size:i, 0])  # 60-day window
    y_train.append(scaled_data[i, 0])                # next day's closing price


# Convert to numpy arrays
X_train, y_train = np.array(X_train), np.array(y_train)


# Reshape training data to 3 dimensions: (samples, timesteps, features)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


# ===============================
# 3. Build the LSTM Model
# ===============================
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(1))


# Compile the model using the Adam optimizer and mean squared error loss
model.compile(optimizer='adam', loss='mean_squared_error')


# ===============================
# 4. Train the Model
# ===============================
# Train for 20 epochs with a batch size of 32
model.fit(X_train, y_train, epochs=20, batch_size=32)


# ===============================
# 5. Download and Prepare 2024 Test Data
# ===============================
# Download new test data for 2024 (first 3 months)
test_data = yf.download(ticker, start='2021-01-01', end='2024-12-31')
print("Test data head (2024):")
print(test_data.head())


# Focus on the 'Close' price for test data
test_Close = test_data[['Close']]


# To create a sliding window for test predictions, combine the last window_size days from training with the test data.
last_train_days = data_Close.tail(window_size)
combined = pd.concat([last_train_days, test_Close], ignore_index=False)


# Scale the combined data using the same scaler used for training data
scaled_combined = scaler.transform(combined)


# Create test sequences
X_test = []
y_test = []


# Build sliding window on the combined scaled data
# Note: The first window_size points are from the tail of training data; predictions will be for the test period.
for i in range(window_size, len(scaled_combined)):
    X_test.append(scaled_combined[i-window_size:i, 0])
    # To evaluate, we also store the true values (which will only be available for the test period)
    y_test.append(scaled_combined[i, 0])


X_test = np.array(X_test)
y_test = np.array(y_test)


# Reshape X_test to match LSTM input shape
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))


# ===============================
# 6. Make Predictions on the 2024 Data
# ===============================
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)


# Get the actual closing prices for the test period from the combined data,
# discarding the initial training tail
# We'll use the index from the test_data as our x-axis for plotting.
actual_prices = combined.iloc[window_size:]['Close']


# ===============================
# 7. Visualize the Results
# ===============================
plt.figure(figsize=(10, 6))
plt.plot(actual_prices.index, actual_prices.values, color='blue', label='Real Google Stock Price (2024)')
plt.plot(actual_prices.index, predictions, color='red', label='Predicted Google Stock Price (2024)')
plt.title('Google Stock Price Prediction using LSTM (Testing on 2024 Data)')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.legend()
plt.show()