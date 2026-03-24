# src/pipeline.py

import numpy as np
import pandas as pd
from features import *

def generate_synthetic_stock_data(n_days=500, seed=42):
    rng = np.random.default_rng(seed)

    close = 100 * np.cumprod(1 + rng.normal(0, 0.01, n_days))
    high = close * (1 + rng.uniform(0, 0.02, n_days))
    low = close * (1 - rng.uniform(0, 0.02, n_days))
    open_ = close * (1 + rng.normal(0, 0.005, n_days))
    volume = rng.integers(100_000, 1_000_000, n_days)

    df = pd.DataFrame({
        "Open": open_,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": volume,
        "Ticker": "SYNTH"
    })

    return df


def engineer_features(df):
    df = df.copy()

    df["rsi_14"] = rsi(df["Close"])
    df["sma_20"] = sma(df["Close"], 20)
    df["ema_12"] = ema(df["Close"], 12)

    macd_df = macd(df["Close"])
    df = pd.concat([df, macd_df], axis=1)

    bb = bollinger_bands(df["Close"])
    df = pd.concat([df, bb], axis=1)

    df["atr"] = average_true_range(df["High"], df["Low"], df["Close"])
    df["volatility"] = historical_volatility(df["Close"])
    df["obv"] = on_balance_volume(df["Close"], df["Volume"])

    candle = candle_features(df["Open"], df["High"], df["Low"], df["Close"])
    df = pd.concat([df, candle], axis=1)

    # Target (next day movement)
    df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

    return df


def split_data(df, train_size=0.7, val_size=0.15):
    n = len(df)

    train_end = int(n * train_size)
    val_end = int(n * (train_size + val_size))

    return {
        "train": df.iloc[:train_end],
        "val": df.iloc[train_end:val_end],
        "test": df.iloc[val_end:]
    }