import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization

def build_lstm(seq_len, n_features, units1=64, units2=32, dropout=0.2):
    model = Sequential()
    model.add(LSTM(units1, return_sequences=True, input_shape=(seq_len, n_features)))
    model.add(Dropout(dropout))
    model.add(BatchNormalization())
    model.add(LSTM(units2, return_sequences=False))
    model.add(Dropout(dropout))
    model.add(BatchNormalization())
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='linear'))  # predicting price
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model
