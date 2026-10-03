"""Expanding-window walk-forward evaluation of the LSTM, one test fold per year."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from usdtry_risk.forecasting.lstm import LSTMConfig, make_windows, predict, train_model
from usdtry_risk.forecasting.scaling import StandardScaler
from usdtry_risk.seed import SEED


@dataclass(frozen=True)
class Fold:
    test_start: pd.Timestamp
    test_end: pd.Timestamp  # inclusive


def yearly_folds(index: pd.DatetimeIndex, first_test_year: int) -> list[Fold]:
    """One fold per calendar year from ``first_test_year``; training is all earlier dates."""
    years = sorted(set(index.year[index.year >= first_test_year]))
    return [Fold(index[index.year == y][0], index[index.year == y][-1]) for y in years]


def walk_forward_lstm(
    returns: pd.Series,
    first_test_year: int = 2010,
    config: LSTMConfig = LSTMConfig(),
    val_size: int = 252,
    seed: int = SEED,
    scaler_factory=StandardScaler,
) -> tuple[pd.Series, pd.DataFrame]:
    """LSTM log-return forecasts for every test day, plus a per-fold training log.

    Each fold uses only returns dated before its test year. The last
    ``val_size`` of those form the early-stopping slice; the scaler is fit on
    the rest, so it never sees validation or test returns.
    """
    dates = returns.index
    forecasts, log = [], []
    for fold in yearly_folds(dates, first_test_year):
        train = returns[dates < fold.test_start]
        if len(train) <= val_size + config.lookback:
            raise ValueError(f"too little history before {fold.test_start:%Y-%m-%d}")
        val_start = train.index[-val_size]
        scaler = scaler_factory().fit(train[train.index < val_start])

        z = scaler.transform(returns.loc[: fold.test_end]).to_numpy()
        X, y = make_windows(z, config.lookback)
        target_dates = dates[config.lookback : len(z)]
        fit_mask = target_dates < val_start
        val_mask = (target_dates >= val_start) & (target_dates < fold.test_start)
        test_mask = target_dates >= fold.test_start

        model, info = train_model(
            X[fit_mask], y[fit_mask], X[val_mask], y[val_mask], config, seed
        )
        pred = scaler.inverse_transform(predict(model, X[test_mask]))
        forecasts.append(pd.Series(pred, index=target_dates[test_mask]))
        log.append(
            {
                "test_year": fold.test_start.year,
                "n_fit": int(fit_mask.sum()),
                "n_val": int(val_mask.sum()),
                "n_test": int(test_mask.sum()),
                **info,
            }
        )
    return pd.concat(forecasts).rename("lstm"), pd.DataFrame(log).set_index("test_year")
