"""Value at Risk for an exporter that will receive a fixed USD amount.

The exporter converts USD into TRY, so it loses when USD/TRY falls. Over a
horizon with log return R the TRY value changes by exp(R) - 1, so the loss
as a fraction of today's TRY value is 1 - exp(R). VaR is that loss at the
(1 - confidence) quantile of R, quoted as a positive fraction.
"""
from __future__ import annotations

from statistics import NormalDist

import numpy as np
import pandas as pd

EXPOSURE_USD = 1_000_000


def loss_fraction(log_return):
    """Exporter's loss as a fraction of its TRY value; negative when USD/TRY rises."""
    return 1 - np.exp(log_return)


def parametric_var(variance, confidence: float):
    """VaR fraction when the horizon log return is normal with zero mean and ``variance``."""
    z = NormalDist().inv_cdf(confidence)
    return loss_fraction(-z * np.sqrt(variance))


def historical_var(returns: pd.Series, confidence: float, window: int = 500, horizon: int = 1) -> pd.Series:
    """Rolling historical VaR fraction, indexed by the date it is made.

    The value at date t uses the ``window`` daily returns up to and including
    t. For ``horizon`` > 1 it uses the overlapping ``horizon``-day log returns
    inside that window, which are not independent of each other.
    """
    horizon_returns = returns.rolling(horizon).sum()
    q = horizon_returns.rolling(window - horizon + 1).quantile(1 - confidence)
    return loss_fraction(q).dropna().rename("historical")


def var_try(fraction, usd_try: float, exposure_usd: float = EXPOSURE_USD):
    """VaR in TRY for ``exposure_usd`` valued at the ``usd_try`` rate."""
    return exposure_usd * usd_try * fraction
