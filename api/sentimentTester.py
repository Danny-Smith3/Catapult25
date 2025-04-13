from sentimentAI import generate_sentiment_summary, StockSentimentRequest

if __name__ == "__main__":
    # Example test ticker
    test_ticker = "TSLA"
    request = StockSentimentRequest(ticker=test_ticker)
    
    # Call the function and print result
    result = generate_sentiment_summary(request)
    print("\nSentiment Analysis Result:\n")
    print(result)
