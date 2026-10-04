import numpy as np
import pandas as pd
import pytest

from usdtry_risk.data.prepare import log_returns
from usdtry_risk.hedging.forward import cip_forward
from usdtry_risk.hedging.simulate import (
    build_receivables,
    decision_dates,
    dynamic_hedge_ratio,
    hedge_outcomes,
    summarize,
)


def random_prices(n: int = 1500, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=n)
    return pd.Series(5 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, n))), index=dates)


def test_decision_dates_are_month_ends_from_start():
    prices = random_prices(100)
    dates = decision_dates(prices, "2015-02")
    assert dates[0] == pd.Timestamp("2015-02-27")  # last business day of February 2015
    assert all(d in prices.index for d in dates)
    assert dates.to_period("M").is_unique


def test_receivable_rates_maturity_and_forward():
    prices = random_prices(200)
    usd = pd.Series([1.0, 2.0], index=pd.to_datetime(["2015-01-01", "2015-03-31"]))
    trl = pd.Series([10.0, 20.0], index=pd.to_datetime(["2015-01-01", "2015-03-31"]))
    rec = build_receivables(prices, usd, trl, decision_dates(prices, "2015-03"))

    first = rec.iloc[0]
    assert rec.index[0] == pd.Timestamp("2015-03-31")
    assert first["maturity"] == pd.Timestamp("2015-06-30")
    # A rate dated on the decision date is not yet known.
    assert first["r_try"] == 10.0 and first["r_usd"] == 1.0
    assert first["forward"] == pytest.approx(cip_forward(first["spot"], 10.0, 1.0, first["days"]))
    assert rec.iloc[1]["r_try"] == 20.0
    # Receivables that mature after the last price date are left out.
    assert (rec["maturity"] <= prices.index[-1]).all()
    assert rec.index[-1] + pd.DateOffset(months=3) <= prices.index[-1]


def test_hedge_outcomes_by_ratio():
    rec = pd.DataFrame({"forward": [40.0], "spot_maturity": [36.0], "spot": [38.0]})
    out = hedge_outcomes(rec, {"no_hedge": 0.0, "half": 0.5, "full": 1.0}, amount_usd=1_000_000)
    assert out.loc[0].tolist() == pytest.approx([36_000_000, 38_000_000, 40_000_000])


def test_dynamic_ratio_has_no_lookahead():
    prices = random_prices()
    returns = log_returns(prices)
    dates = decision_dates(prices, "2016-06")
    changed = returns.copy()
    changed.iloc[-200:] *= 4
    a, b = dynamic_hedge_ratio(returns, dates), dynamic_hedge_ratio(changed, dates)
    cutoff = returns.index[-201]
    pd.testing.assert_series_equal(a[a.index <= cutoff], b[b.index <= cutoff])
    assert set(a.unique()) <= {0.5, 1.0}


def test_dynamic_ratio_follows_the_median_rule():
    rng = np.random.default_rng(1)
    dates = pd.bdate_range("2015-01-01", periods=600)
    calm, wild = rng.normal(0, 0.002, 400), rng.normal(0, 0.03, 200)
    returns = pd.Series(np.concatenate([calm, wild]), index=dates)
    ratio = dynamic_hedge_ratio(returns, pd.DatetimeIndex([dates[350], dates[599]]))
    assert ratio.tolist() == [0.5, 1.0]


def test_summarize_measures():
    rec = pd.DataFrame({"spot": [10.0, 10.0, 20.0, 20.0]})
    outcomes = pd.DataFrame(
        {"no_hedge": [9e6, 11e6, 18e6, 22e6], "full": [10.5e6, 10.5e6, 21e6, 21e6]}
    )
    result = summarize(outcomes, rec, amount_usd=1_000_000)
    assert result.loc["full", "mean_pct"] == pytest.approx(5.0)
    assert result.loc["no_hedge", "mean_pct"] == pytest.approx(0.0)
    assert result.loc["full", "beat_baseline"] == pytest.approx(0.5)
    assert np.isnan(result.loc["no_hedge", "beat_baseline"])
    assert result.loc["no_hedge", "p5_try"] == pytest.approx(np.quantile([9e6, 11e6, 18e6, 22e6], 0.05))
