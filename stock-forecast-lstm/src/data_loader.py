# src/data_loader.py
import yfinance as yf
import pandas as pd
import os
from datetime import datetime, timedelta

def download_price(ticker: str, period='5y', interval='1d', save_csv=None):
    """
    Download OHLCV from yfinance.
    """
    t = yf.Ticker(ticker)
    df = t.history(period=period, interval=interval, auto_adjust=False)
    df = df.reset_index().rename(columns={'Date':'date'})
    df['ticker'] = ticker
    if save_csv:
        os.makedirs(os.path.dirname(save_csv), exist_ok=True)
        df.to_csv(save_csv, index=False)
    return df

def get_option_chain(ticker: str, date=None):
    """
    Pull options chain for nearest expiry if date is None.
    Returns a dict {'calls': DataFrame, 'puts': DataFrame}
    """
    t = yf.Ticker(ticker)
    expiries = t.options
    if not expiries:
        return {'calls': pd.DataFrame(), 'puts': pd.DataFrame()}
    if date and date in expiries:
        ex = date
    else:
        ex = expiries[0]
    chain = t.option_chain(ex)
    calls = chain.calls.copy()
    puts = chain.puts.copy()
    calls['expiry'] = ex
    puts['expiry'] = ex
    return {'calls': calls, 'puts': puts}

def get_fundamentals(ticker: str):
    """
    Get some fundamental metrics from yfinance info.
    """
    t = yf.Ticker(ticker)
    info = t.info
    # extract a safe subset; fields may be missing
    keys = ['trailingPE','priceToBook','trailingEps','dividendYield','returnOnEquity','marketCap','beta']
    data = {k: info.get(k, None) for k in keys}
    data['ticker'] = ticker
    return data
