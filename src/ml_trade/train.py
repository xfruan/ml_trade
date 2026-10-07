"""Usage: python -m ml_trade.train --config configs/default.yaml [--model cnn]"""
import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .data import chrono_split, download, make_features, make_windows, standardize
from .models import build_model


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in loader:
            pred = (model(xb.to(device)) > 0).float().cpu()
            correct += (pred == yb).sum().item()
            total += len(yb)
    return correct / total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--model", choices=["lstm", "cnn"])
    args = p.parse_args()
    cfg = yaml.safe_load(open(args.config))
    if args.model:
        cfg["model"]["type"] = args.model
    d, m, t = cfg["data"], cfg["model"], cfg["train"]

    torch.manual_seed(t["seed"]); np.random.seed(t["seed"])
    device = "cuda" if t["device"] == "auto" and torch.cuda.is_available() else \
        ("cpu" if t["device"] == "auto" else t["device"])

    df = download(d["ticker"], d["start"], d["end"])
    X, y = make_windows(make_features(df), df["Close"], d["window"], d["horizon"])
    (Xtr, ytr), (Xva, yva), (Xte, yte) = chrono_split(X, y, d["train_frac"], d["val_frac"])
    Xtr, Xva, Xte = standardize(Xtr, Xva, Xte)

    def loader(X, y, shuffle):
        ds = TensorDataset(torch.tensor(X), torch.tensor(y))
        return DataLoader(ds, batch_size=t["batch_size"], shuffle=shuffle)

    tr, va, te = loader(Xtr, ytr, True), loader(Xva, yva, False), loader(Xte, yte, False)
    model = build_model(m["type"], X.shape[2], hidden=m["hidden"], layers=m["layers"],
                        dropout=m["dropout"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=t["lr"])
    loss_fn = nn.BCEWithLogitsLoss()

    best, ckpt = 0.0, Path("checkpoints") / f"{d['ticker']}_{m['type']}.pt"
    for epoch in range(t["epochs"]):
        model.train()
        for xb, yb in tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
        acc = evaluate(model, va, device)
        print(f"epoch {epoch + 1:3d}  val_acc {acc:.4f}")
        if acc > best:
            best = acc
            torch.save(model.state_dict(), ckpt)

    model.load_state_dict(torch.load(ckpt))
    print(f"test_acc {evaluate(model, te, device):.4f}  (baseline up-rate {yte.mean():.4f})")


if __name__ == "__main__":
    main()
