from LSTM_helper import get_predicted_price
from LSTMLoader import download_extract_and_load_lstm_models

def main():
    #download_extract_and_load_lstm_models()
    # Choose a ticker to test—make sure there is a corresponding model loaded.
    ticker = "AAPL"
    print(f"Testing prediction for ticker: {ticker}")

    # Get the prediction result using the helper function
    result = get_predicted_price(ticker)
    print("Result:")
    print(result)

if __name__ == "__main__":
    main()
