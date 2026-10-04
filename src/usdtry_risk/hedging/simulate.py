"""Monthly USD receivables hedged with forwards, and their TRY outcomes.

On the last ECB date of each month the exporter books a USD receivable due
in three months and chooses a hedge ratio h. At maturity it receives
amount * (h * F + (1 - h) * S_T) in TRY, where F is the forward fixed on the
decision date and S_T is the spot rate on the maturity date.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from usdtry_risk.hedging.forward import cip_forward, rate_known_before
from usdtry_risk.risk.volatility import ewma_variance

AMOUNT_USD = 1_000_000


def decision_dates(prices: pd.Series, start: str) -> pd.DatetimeIndex:
    """Last price date of each month from the month of ``start`` on."""
    dates = prices.index[prices.index >= pd.Timestamp(start).to_period("M").start_time]
    return pd.DatetimeIndex(dates.to_series().groupby(dates.to_period("M")).max().to_numpy())


def build_receivables(
    prices: pd.Series,
    usd_rates: pd.Series,
    try_rates: pd.Series,
    dates: pd.DatetimeIndex,
    months: int = 3,
) -> pd.DataFrame:
    """Spot, rates, forward and maturity spot for each decision date that has matured.

    Maturity is the first price date on or after the decision date plus
    ``months``; rates are the latest known before the decision date.
    """
    rows = {}
    for d in dates:
        pos = prices.index.searchsorted(d + pd.DateOffset(months=months))
        if pos == len(prices):
            continue  # not yet matured in this snapshot
        maturity = prices.index[pos]
        days = (maturity - d).days
        r_try, r_usd = rate_known_before(try_rates, d), rate_known_before(usd_rates, d)
        rows[d] = {
            "maturity": maturity,
            "days": days,
            "spot": prices[d],
            "r_try": r_try,
            "r_usd": r_usd,
            "forward": cip_forward(prices[d], r_try, r_usd, days),
            "spot_maturity": prices.iloc[pos],
        }
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("decision")


def dynamic_hedge_ratio(returns: pd.Series, dates: pd.DatetimeIndex) -> pd.Series:
    """1.0 when EWMA volatility is above its median to date, else 0.5.

    Both the EWMA value and the median use only returns up to the decision date.
    """
    ewma = ewma_variance(returns)
    median_to_date = ewma.expanding().median()
    ratio = np.where(ewma > median_to_date, 1.0, 0.5)
    return pd.Series(ratio, index=ewma.index, name="dynamic").reindex(dates)


def hedge_outcomes(
    receivables: pd.DataFrame, hedge_ratios: dict[str, float | pd.Series], amount_usd: float = AMOUNT_USD
) -> pd.DataFrame:
    """TRY received at maturity for each strategy and decision date."""
    out = {}
    for name, h in hedge_ratios.items():
        if isinstance(h, pd.Series):
            h = h.reindex(receivables.index)
            if h.isna().any():
                raise ValueError(f"{name} has no hedge ratio on some decision dates")
        out[name] = amount_usd * (h * receivables["forward"] + (1 - h) * receivables["spot_maturity"])
    return pd.DataFrame(out)


def summarize(
    outcomes: pd.DataFrame,
    receivables: pd.DataFrame,
    baseline: str = "no_hedge",
    amount_usd: float = AMOUNT_USD,
) -> pd.DataFrame:
    """Average, standard deviation and 5th percentile of each strategy, in TRY and in %.

    The % measure is TRY received relative to the amount at the decision-date
    spot, which is comparable across years. ``beat_baseline`` is the share of
    receivables where the strategy received strictly more TRY than ``baseline``.
    """
    pct = 100 * (outcomes.div(amount_usd * receivables["spot"], axis=0) - 1)
    rows = {}
    for name in outcomes:
        rows[name] = {
            "mean_try": outcomes[name].mean(),
            "std_try": outcomes[name].std(),
            "p5_try": outcomes[name].quantile(0.05),
            "mean_pct": pct[name].mean(),
            "std_pct": pct[name].std(),
            "p5_pct": pct[name].quantile(0.05),
            "beat_baseline": float((outcomes[name] > outcomes[baseline]).mean())
            if name != baseline
            else np.nan,
        }
    result = pd.DataFrame.from_dict(rows, orient="index")
    result.attrs.update(n=len(outcomes), start=outcomes.index[0], end=outcomes.index[-1])
    return result
