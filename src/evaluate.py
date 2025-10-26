import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, precision_score, recall_score, f1_score

def evaluate_model(model, X_test, y_test, scaler=None, plot=True):
    preds = model.predict(X_test).flatten()
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    # Direction classification metrics: up/down
    y_dir = (y_test[1:] > y_test[:-1]).astype(int)
    p_dir = (preds[1:] > preds[:-1]).astype(int)
    # ensure same length
    minlen = min(len(y_dir), len(p_dir))
    y_dir, p_dir = y_dir[:minlen], p_dir[:minlen]
    prec = precision_score(y_dir, p_dir, zero_division=0)
    rec = recall_score(y_dir, p_dir, zero_division=0)
    f1 = f1_score(y_dir, p_dir, zero_division=0)

    results = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'Precision_dir': prec,
        'Recall_dir': rec,
        'F1_dir': f1
    }

    if plot:
        plt.figure(figsize=(12,6))
        plt.plot(y_test, label='True')
        plt.plot(preds, label='Predicted')
        plt.legend()
        plt.title('True vs Predicted')
        plt.show()

    return results
