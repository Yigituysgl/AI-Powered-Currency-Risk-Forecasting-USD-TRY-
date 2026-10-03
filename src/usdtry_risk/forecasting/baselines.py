"""Benchmark one-day-ahead forecasts every model must beat.

Convention: the value at date t is the forecast *for* t, built only from
prices up to and including t-1.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def naive_forecast(prices: pd.Series) -> pd.Series:
    """Random walk: tomorrow = today."""
    return prices.shift(1).rename("naive")


def drift_forecast(
    prices: pd.Series, window: int | None = None, min_periods: int = 20
) -> pd.Series:
    """Random walk with drift: P_{t-1} * exp(mean log return up to t-1).

    ``window=None`` averages all past returns; an integer uses a rolling window
    of that many returns.
    """
    r = np.log(prices).diff()
    if window is None:
        mu = r.expanding(min_periods=min_periods).mean()
        name = "drift_expanding"
    else:
        mu = r.rolling(window, min_periods=window).mean()
        name = f"drift_{window}d"
    return (prices * np.exp(mu)).shift(1).rename(name)
