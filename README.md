# ml_trade

CNN and LSTM models for financial market direction prediction (PyTorch).

## Setup
```
python -m venv .venv && source .venv/bin/activate
pip install -e . pytest
```

## Train
```
python -m ml_trade.train --config configs/default.yaml --model lstm
python -m ml_trade.train --config configs/default.yaml --model cnn
```
Edit `configs/default.yaml` for ticker, window, horizon and hyperparameters.

## Layout
- `src/ml_trade/data.py` – download (yfinance, cached in `data/raw`), features, windowing, chronological split
- `src/ml_trade/models.py` – `LSTMClassifier`, `CNNClassifier`
- `src/ml_trade/train.py` – training/eval entry point
- `tests/` – `pytest`

Splits are chronological and scaling uses train statistics only to avoid lookahead.
Compare test accuracy against the printed up-day baseline; results are not trading advice.
