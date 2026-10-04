"""One-day VaR backtest with Kupiec's proportion-of-failures test."""
from __future__ import annotations

import math

import pandas as pd

from usdtry_risk.risk.var import loss_fraction


def kupiec(breaches: int, n: int, expected_rate: float) -> tuple[float, float]:
    """Likelihood-ratio statistic and p-value (chi-squared, 1 df) for a breach rate."""
    if not 0 <= breaches <= n or n == 0:
        raise ValueError("need 0 <= breaches <= n and n > 0")
    p, x = expected_rate, breaches
    observed = x / n

    def loglik(rate):
        # 0 * log(0) is taken as 0, so zero breaches or all breaches stay finite.
        hits = x * math.log(rate) if x else 0.0
        misses = (n - x) * math.log(1 - rate) if x < n else 0.0
        return hits + misses

    lr = max(-2 * (loglik(p) - loglik(observed)), 0.0)
    return lr, math.erfc(math.sqrt(lr / 2))


def backtest_var(
    returns: pd.Series,
    var_forecasts: dict[str, pd.Series],
    confidence: float,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Breach counts and Kupiec tests on the days where every forecast exists.

    Each forecast is indexed by the date it is made and is checked against the
    next day's return; a breach is a loss larger than the VaR.
    """
    forecast_for_day = {
        name: f.reindex(returns.index).shift(1) for name, f in var_forecasts.items()
    }
    frame = pd.concat({"loss": loss_fraction(returns), **forecast_for_day}, axis=1).loc[start:end].dropna()
    if frame.empty:
        raise ValueError("no dates where all forecasts are available")
    expected = 1 - confidence
    rows = {}
    for name in var_forecasts:
        breaches = int((frame["loss"] > frame[name]).sum())
        lr, p = kupiec(breaches, len(frame), expected)
        rows[name] = {
            "breaches": breaches,
            "expected": expected * len(frame),
            "breach_rate": breaches / len(frame),
            "kupiec_lr": lr,
            "kupiec_p": p,
        }
    result = pd.DataFrame.from_dict(rows, orient="index")
    result.attrs.update(n=len(frame), start=frame.index[0], end=frame.index[-1])
    return result
