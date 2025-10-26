import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from src.data_loader import download_price, get_fundamentals, get_option_chain
from src.features import add_technical_indicators, add_fundamentals_to_df
from src.dataset import prepare_data
from src.model import build_lstm
import joblib
import argparse
from datetime import datetime

def train(ticker='AAPL', seq_len=60, epochs=30, batch_size=32, period='5y', output_dir='outputs'):
    os.makedirs(output_dir, exist_ok=True)

    print(f"[{datetime.now()}] Downloading data for {ticker} ...")
    df = download_price(ticker, period=period)
    if df.empty:
        raise RuntimeError("No price data downloaded; check ticker or yfinance.")

    print("Adding technical indicators ...")
    df = add_technical_indicators(df)

    print("Fetching fundamentals ...")
    fundamentals = get_fundamentals(ticker)
    df = add_fundamentals_to_df(df, fundamentals)

    # choose features
    feature_cols = [
        'Open','High','Low','Close','Volume',
        'sma_14','ema_14','rsi_14','atr_14','adx_14',
        'macd','macd_signal','bb_h','bb_l','bb_w',
        'return_1','return_5','volatility_14',
        # fundamentals
        'trailingPE','priceToBook','trailingEps','dividendYield','returnOnEquity','marketCap','beta'
    ]
    # keep only columns present
    feature_cols = [c for c in feature_cols if c in df.columns]

    print("Preparing dataset ...")
    X_train, y_train, X_test, y_test, _, _, scaler = prepare_data(df, feature_cols, seq_len=seq_len, scaler_save_path=os.path.join(output_dir,'scaler.gz'))

    print("Building model ...")
    model = build_lstm(seq_len, len(feature_cols))
    ckpt_path = os.path.join(output_dir, 'best_model.h5')
    cp = ModelCheckpoint(ckpt_path, monitor='val_loss', save_best_only=True, mode='min', verbose=1)
    es = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=1)

    print("Training ...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs, batch_size=batch_size,
        callbacks=[cp, es],
        verbose=2
    )

    model.save(os.path.join(output_dir,'final_model.h5'))
    joblib.dump(history.history, os.path.join(output_dir,'history.pkl'))

    # save metadata
    meta = {
        'ticker': ticker,
        'seq_len': seq_len,
        'feature_cols': feature_cols,
        'epochs': epochs,
        'batch_size': batch_size,
        'period': period
    }
    with open(os.path.join(output_dir,'meta.json'),'w') as f:
        json.dump(meta, f, indent=2)

    print("Training complete. Artifacts saved to", output_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ticker', default='AAPL')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--seq_len', type=int, default=60)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--output_dir', default='outputs')
    parser.add_argument('--period', default='5y')
    args = parser.parse_args()
    train(args.ticker, seq_len=args.seq_len, epochs=args.epochs, batch_size=args.batch_size, period=args.period, output_dir=args.output_dir)
