"""Data download, feature engineering and windowing."""
from pathlib import Path

import numpy as np
import pandas as pd


def download(ticker: str, start: str, end: str | None, cache_dir: str = "data/raw") -> pd.DataFrame:
    path = Path(cache_dir) / f"{ticker}.csv"
    if path.exists():
        return pd.read_csv(path, index_col=0, parse_dates=True)
    import yfinance as yf

    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
    return df


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Stationary features only (returns/ratios), no raw prices."""
    f = pd.DataFrame(index=df.index)
    f["ret"] = np.log(df["Close"]).diff()
    f["hl"] = (df["High"] - df["Low"]) / df["Close"]
    f["oc"] = (df["Close"] - df["Open"]) / df["Open"]
    f["vol_chg"] = np.log(df["Volume"].replace(0, np.nan)).diff()
    f["vol20"] = f["ret"].rolling(20).std()
    f["ma_ratio"] = df["Close"] / df["Close"].rolling(20).mean() - 1
    return f


def make_windows(features: pd.DataFrame, close: pd.Series, window: int, horizon: int):
    """Return X (N, window, F) and binary y (N,): 1 if close is higher `horizon` days ahead."""
    future_ret = np.log(close).shift(-horizon) - np.log(close)
    data = features.join(future_ret.rename("target")).dropna()
    x = data.drop(columns="target").to_numpy(dtype=np.float32)
    y = (data["target"].to_numpy() > 0).astype(np.float32)
    idx = np.arange(window, len(data) + 1)
    X = np.stack([x[i - window : i] for i in idx])
    return X, y[window - 1 :]


def chrono_split(X, y, train_frac: float, val_frac: float):
    n = len(X)
    a, b = int(n * train_frac), int(n * (train_frac + val_frac))
    return (X[:a], y[:a]), (X[a:b], y[a:b]), (X[b:], y[b:])


def standardize(train, *others):
    """Scale using train-set statistics only (no lookahead)."""
    mu = train.mean(axis=(0, 1), keepdims=True)
    sd = train.std(axis=(0, 1), keepdims=True) + 1e-8
    return [(a - mu) / sd for a in (train, *others)]
