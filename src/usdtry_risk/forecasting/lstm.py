"""Small LSTM that forecasts the next day's log return from recent returns."""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class LSTMConfig:
    lookback: int = 20
    hidden_size: int = 16
    max_epochs: int = 100
    patience: int = 10
    batch_size: int = 64
    learning_rate: float = 1e-3


class ReturnLSTM(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)  # x: (batch, lookback, 1)
        return self.head(out[:, -1]).squeeze(-1)


def make_windows(z: np.ndarray, lookback: int) -> tuple[np.ndarray, np.ndarray]:
    """Inputs ``z[i-lookback:i]`` and target ``z[i]`` for every ``i >= lookback``."""
    X = np.lib.stride_tricks.sliding_window_view(z[:-1], lookback)
    y = z[lookback:]
    return X[..., None].astype(np.float32), y.astype(np.float32)


def train_model(
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: LSTMConfig,
    seed: int,
) -> tuple[ReturnLSTM, dict]:
    """Fit on MSE with Adam and keep the weights with the lowest validation loss."""
    torch.manual_seed(seed)
    model = ReturnLSTM(config.hidden_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_fn = nn.MSELoss()
    Xf, yf = torch.from_numpy(X_fit), torch.from_numpy(y_fit)
    Xv, yv = torch.from_numpy(X_val), torch.from_numpy(y_val)
    generator = torch.Generator().manual_seed(seed)

    best_loss, best_state, best_epoch, stale = math.inf, None, 0, 0
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        for idx in torch.randperm(len(Xf), generator=generator).split(config.batch_size):
            optimizer.zero_grad()
            loss_fn(model(Xf[idx]), yf[idx]).backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xv), yv).item()
        if val_loss < best_loss:
            best_loss, best_epoch, stale = val_loss, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= config.patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    return model, {"best_epoch": best_epoch, "epochs_run": epoch, "val_loss": best_loss}


def predict(model: ReturnLSTM, X: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return model(torch.from_numpy(X)).numpy()
