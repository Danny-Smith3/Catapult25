import ast
import pandas as pd

def get_tickers():
    with open('all_tickers.txt', 'r', encoding='utf-8') as f:
        input = f.read()

    full_dict = ast.literal_eval(input)
    ticker_symbols = list(full_dict.keys())

    return ticker_symbols

def get_sp500(all_tickers):
    df = pd.read_csv('constituents.csv')
    first_col = df.iloc[:, 0].to_list()
    sp500_matches = [ticker for ticker in all_tickers if ticker.upper() in first_col]
    return sp500_matches