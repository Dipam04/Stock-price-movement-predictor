# Stock-price-movement-predictor
A machine learning pipeline for predicting stock price movement using technical indicators and engineered features. This project demonstrates end-to-end ML engineering practices, including data generation, feature engineering, validation, and testing.

Project Overview

This project builds a complete pipeline to:

Generate synthetic stock market data (OHLCV)

Engineer technical indicators (RSI, MACD, Bollinger Bands, etc.)

Create a binary classification target (price up/down)

Split data chronologically (train/validation/test)

Validate the entire pipeline using unit tests

🧠 Key Features

✔️ Synthetic stock data generation

✔️ 25+ engineered technical features

✔️ No lookahead bias (time-series safe)

✔️ Chronological data splitting

✔️ Strong unit testing coverage

✔️ Modular and production-style code structure

🏗️ Project Structure
├── src/
│   ├── pipeline.py        # Data generation, feature engineering, splitting
│   ├── features.py        # Technical indicators
│
├── tests/
│   ├── test_pipeline.py   # Unit tests for full pipeline
│
├── README.md
⚙️ Installation


Install dependencies:

pip install -r requirements.txt
 Usage

Run the pipeline inside Python:

from pipeline import generate_synthetic_stock_data, engineer_features, split_data

df = generate_synthetic_stock_data(n_days=500)
features = engineer_features(df)
splits = split_data(features)

train = splits["train"]
val = splits["val"]
test = splits["test"]
 Running Tests

Run unit tests using:

python tests/test_pipeline.py

Or (recommended):

pytest

# Features Implemented Technical Indicators:

RSI (Relative Strength Index)

SMA / EMA

MACD

Bollinger Bands

Average True Range (ATR)

Historical Volatility

On-Balance Volume (OBV)

Candle Features:

Candle body

Upper wick

Lower wick

Bullish/Bearish signal

# Important ML Practices Followed

 No data leakage (chronological split)

 No lookahead bias

 Feature validation

 Reproducible data generation

 Robust unit testing

# Future Improvements

Add real stock market data (e.g., Yahoo Finance API)

Train ML/DL models (Logistic Regression, XGBoost, LSTM)

Model evaluation metrics (Accuracy, F1-score, ROC-AUC)

Deploy model using FastAPI

Build a dashboard for predictions

Handling time-series data correctly

Building production-ready ML systems
