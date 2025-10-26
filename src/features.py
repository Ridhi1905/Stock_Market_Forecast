import pandas as pd
import numpy as np
from ta.trend import SMAIndicator, EMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange, BollingerBands

def add_technical_indicators(df):
    df = df.copy()
    df = df.sort_values('date').reset_index(drop=True)
    close = df['Close']
    high = df['High']
    low = df['Low']
    volume = df['Volume']

    # SMA, EMA
    df['sma_14'] = SMAIndicator(close, window=14).sma_indicator()
    df['ema_14'] = EMAIndicator(close, window=14).ema_indicator()

    # RSI
    df['rsi_14'] = RSIIndicator(close, window=14).rsi()

    # ATR
    atr = AverageTrueRange(high, low, close, window=14)
    df['atr_14'] = atr.average_true_range()

    # ADX
    adx = ADXIndicator(high, low, close, window=14)
    df['adx_14'] = adx.adx()

    # MACD
    macd = MACD(close)
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()

    # Bollinger Bands
    bb = BollingerBands(close, window=20, window_dev=2)
    df['bb_h'] = bb.bollinger_hband()
    df['bb_l'] = bb.bollinger_lband()
    df['bb_m'] = bb.bollinger_mavg()
    df['bb_w'] = (df['bb_h'] - df['bb_l']) / df['bb_m']

    # Price returns
    df['return_1'] = close.pct_change(1)
    df['return_5'] = close.pct_change(5)
    df['volatility_14'] = df['return_1'].rolling(14).std()

    # Drop rows with NaNs introduced by indicators
    df = df.dropna().reset_index(drop=True)
    return df

def add_fundamentals_to_df(df, fundamentals: dict):
    """
    Add scalar fundamental values (PE, PB, EPS, dividend yield...) as constant columns.
    If fundamentals missing, fillna with median later.
    """
    df = df.copy()
    for k,v in fundamentals.items():
        if k == 'ticker':
            continue
        df[k] = v
    return df
