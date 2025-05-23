from sentimentAI import generate_sentiment_summary, StockSentimentRequest

if __name__ == "__main__":
    # Example test ticker
    test_ticker = "AAPL"
    request = StockSentimentRequest(ticker=test_ticker)
    
    # Call the function and print result
    result = generate_sentiment_summary(request)
    print("\n\n\nSentiment Analysis Result:\n")
    print(result)

    test_ticker = "TSLA"
    request = StockSentimentRequest(ticker=test_ticker)
    
    # Call the function and print result
    result = generate_sentiment_summary(request)
    print("\n\n\nSentiment Analysis Result:\n")
    print(result)

    test_ticker = "XOM"
    request = StockSentimentRequest(ticker=test_ticker)
    
    # Call the function and print result
    result = generate_sentiment_summary(request)
    print("\n\n\nSentiment Analysis Result:\n")
    print(result)

    test_ticker = "JPM"
    request = StockSentimentRequest(ticker=test_ticker)
    
    # Call the function and print result
    result = generate_sentiment_summary(request)
    print("\n\n\nSentiment Analysis Result:\n")
    print(result)

    test_ticker = "CAT"
    request = StockSentimentRequest(ticker=test_ticker)
    
    # Call the function and print result
    result = generate_sentiment_summary(request)
    print("\n\n\nSentiment Analysis Result:\n")
    print(result)

    test_ticker = "PFE"
    request = StockSentimentRequest(ticker=test_ticker)
    
    # Call the function and print result
    result = generate_sentiment_summary(request)
    print("\n\n\nSentiment Analysis Result:\n")
    print(result)
