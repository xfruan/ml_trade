import numpy as np
import pandas as pd
import torch

from ml_trade.data import chrono_split, make_features, make_windows, standardize
from ml_trade.models import build_model


def fake_prices(n=300):
    rng = np.random.default_rng(0)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))),
                      index=pd.date_range("2020-01-01", periods=n))
    return pd.DataFrame({"Open": close * (1 + rng.normal(0, 0.003, n)), "High": close * (1 + rng.uniform(0.005, 0.02, n)), "Low": close * (1 - rng.uniform(0.005, 0.02, n)),
                         "Close": close, "Volume": rng.integers(1_000, 2_000, n)})


def test_windows_and_split():
    df = fake_prices()
    X, y = make_windows(make_features(df), df["Close"], window=30, horizon=1)
    assert X.shape[0] == len(y) and X.shape[1] == 30
    tr, va, te = chrono_split(X, y, 0.7, 0.15)
    Xtr, Xva, Xte = standardize(tr[0], va[0], te[0])
    assert abs(Xtr.mean()) < 1e-3


def test_models_forward():
    x = torch.randn(4, 30, 6)
    for kind in ("lstm", "cnn"):
        assert build_model(kind, 6)(x).shape == (4,)
