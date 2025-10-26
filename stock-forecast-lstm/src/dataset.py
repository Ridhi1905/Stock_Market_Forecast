import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib

def create_sequences(df, features, target_col='Close', seq_len=60):
    """
    df: DataFrame sorted by date
    features: list of feature column names
    returns X, y, dates
    """
    data = df[features].values
    targets = df[target_col].values
    X, y, dates = [], [], []
    for i in range(seq_len, len(df)):
        X.append(data[i-seq_len:i])
        y.append(targets[i])
        dates.append(df.loc[i,'date'])
    X = np.array(X)
    y = np.array(y)
    return X, y, dates

def prepare_data(df, feature_cols, target_col='Close', seq_len=60, train_frac=0.8, scaler_save_path=None):
    df = df.sort_values('date').reset_index(drop=True)
    # fill missing fundamentals/features with median
    df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())
    scaler = MinMaxScaler()
    # scale features column-wise (fit on full dataset to avoid leakage for example scripts; you may fit only on train)
    scaled = scaler.fit_transform(df[feature_cols])
    df_scaled = df.copy()
    df_scaled[feature_cols] = scaled

    X, y, dates = create_sequences(df_scaled, feature_cols, target_col=target_col, seq_len=seq_len)

    split_idx = int(len(X) * train_frac)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    dates_train, dates_test = dates[:split_idx], dates[split_idx:]

    if scaler_save_path:
        joblib.dump(scaler, scaler_save_path)
    return X_train, y_train, X_test, y_test, dates_train, dates_test, scaler
