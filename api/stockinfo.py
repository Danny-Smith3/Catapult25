import yfinance as yf

def get_stock_data(ticker: str) -> dict:
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        return {
            "open": info.get('open', None),
            "high": info.get('dayHigh', None),
            "low": info.get('dayLow', None),
            "market_cap": round(info.get('marketCap', 0) / 1e12, 2) if info.get('marketCap') else None,
            "pe_ratio": info.get('trailingPE', None),
            "div_yield": round(info.get('dividendYield', 0) * 100, 2) if info.get('dividendYield') else None,
            "52_week_high": info.get('fiftyTwoWeekHigh', None),
            "52_week_low": info.get('fiftyTwoWeekLow', None),
            "name:": info.get('displayName', None)
        }

    except Exception as e:
        return { "error": str(e) }
