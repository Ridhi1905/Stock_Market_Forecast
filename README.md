
# Stock Forecast LSTM:
# "A Comprehensive Approach to Financial Market Forecasting: Integrating Fundamental and Derivative Analysis and Prediction" 
This repository trains a stacked LSTM to forecast stock prices using:
- Technical indicators (SMA, EMA, RSI, ATR, ADX, MACD, Bollinger Bands)
- Fundamental metrics from Yahoo Finance
- Option chain extraction (basic)

## Quickstart (local)
1. Clone:
   ```bash
   git clone <https://github.com/Ridhi1905/Stock_Market_Forecast>
   cd stock-forecast-lstm

2. Create virtual env and install:
   ```bash
   python -m venv venv
   source venv/bin/activate     # or `venv\Scripts\activate` on Windows
   pip install -r requirements.txt
3. Train
   ```bash
   bash run_example.sh AAPL
4. Evaluate:
   ```
   Load outputs/AAPL/final_model.h5 in a notebook and run src.evaluate.evaluate_model.
