import torch
from torch import nn


class LSTMClassifier(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64, layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x):  # x: (B, T, F)
        out, _ = self.lstm(x)
        return self.head(out[:, -1]).squeeze(-1)


class CNNClassifier(nn.Module):
    """1D temporal convolutions over the lookback window."""

    def __init__(self, n_features: int, hidden: int = 64, layers: int = 2, dropout: float = 0.2):
        super().__init__()
        blocks, c = [], n_features
        for _ in range(layers):
            blocks += [nn.Conv1d(c, hidden, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm1d(hidden)]
            c = hidden
        self.conv = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x):  # x: (B, T, F)
        h = self.conv(x.transpose(1, 2)).mean(dim=2)
        return self.head(h).squeeze(-1)


def build_model(kind: str, n_features: int, **kw) -> nn.Module:
    models = {"lstm": LSTMClassifier, "cnn": CNNClassifier}
    return models[kind](n_features, **kw)
