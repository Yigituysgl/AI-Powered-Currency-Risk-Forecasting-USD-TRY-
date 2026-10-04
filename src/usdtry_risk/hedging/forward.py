"""USD/TRY forward rates from covered interest parity with simple interest."""
from __future__ import annotations

import pandas as pd


def cip_forward(spot, r_try_pct, r_usd_pct, days):
    """F = S * (1 + r_TRY * t) / (1 + r_USD * t), rates in percent per year, t = days / 365."""
    t = days / 365
    return spot * (1 + r_try_pct / 100 * t) / (1 + r_usd_pct / 100 * t)


def monthly_available_from(rates: pd.Series, lag_months: int = 3) -> pd.Series:
    """Re-index a monthly series to the first day it can be used.

    A monthly value is published weeks after its month ends, so the value for
    month m is used only from the first day of month m + ``lag_months``.
    """
    return pd.Series(rates.to_numpy(), index=rates.index + pd.DateOffset(months=lag_months), name=rates.name)


def rate_known_before(rates: pd.Series, date: pd.Timestamp) -> float:
    """Latest rate dated strictly before ``date``.

    Daily rates for a date are published after the ECB fixing on that date,
    so only earlier dates count as known.
    """
    known = rates[rates.index < date]
    if known.empty:
        raise ValueError(f"no {rates.name} rate known before {date:%Y-%m-%d}")
    return float(known.iloc[-1])
