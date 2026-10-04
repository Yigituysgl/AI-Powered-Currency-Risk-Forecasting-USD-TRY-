import math

import numpy as np
import pandas as pd
import pytest

from usdtry_risk.risk.backtest import backtest_var, kupiec
from usdtry_risk.risk.var import parametric_var


def test_kupiec_is_zero_at_the_expected_rate():
    lr, p = kupiec(50, 1000, 0.05)
    assert lr == pytest.approx(0.0)
    assert p == pytest.approx(1.0)


def test_kupiec_hand_computed():
    lr, p = kupiec(20, 1000, 0.01)
    expected = -2 * (
        980 * math.log(0.99) + 20 * math.log(0.01) - 980 * math.log(0.98) - 20 * math.log(0.02)
    )
    assert lr == pytest.approx(expected)
    assert p == pytest.approx(math.erfc(math.sqrt(expected / 2)))
    assert p < 0.01


def test_kupiec_handles_zero_breaches():
    lr, p = kupiec(0, 1000, 0.01)
    assert lr == pytest.approx(-2 * 1000 * math.log(0.99))
    assert p < 0.001


def test_backtest_flags_a_breach_on_the_next_day():
    dates = pd.bdate_range("2020-01-01", periods=4)
    returns = pd.Series([0.0, -0.05, -0.01, 0.02], index=dates)
    # VaR made on each of the first three days, checked against the following return.
    var = pd.Series([0.02, 0.02, 0.02], index=dates[:3])
    result = backtest_var(returns, {"flat": var}, confidence=0.99)
    assert result.attrs["n"] == 3 and result.attrs["start"] == dates[1]
    assert result.loc["flat", "breaches"] == 1


def test_backtest_uses_common_dates():
    dates = pd.bdate_range("2020-01-01", periods=10)
    returns = pd.Series(0.0, index=dates)
    short = pd.Series(0.01, index=dates[5:])
    long = pd.Series(0.01, index=dates)
    result = backtest_var(returns, {"short": short, "long": long}, confidence=0.95)
    assert result.attrs["n"] == 4 and result.attrs["start"] == dates[6]


def test_correct_model_passes_kupiec():
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2000-01-01", periods=5000)
    sigma = pd.Series(rng.uniform(0.005, 0.02, len(dates)), index=dates)
    returns = sigma.shift(1).fillna(0.01) * rng.standard_normal(len(dates))
    for confidence in (0.95, 0.99):
        right = backtest_var(returns, {"m": parametric_var(sigma**2, confidence)}, confidence)
        wrong = backtest_var(returns, {"m": parametric_var((sigma / 2) ** 2, confidence)}, confidence)
        # A correct model is still rejected at 5% one time in twenty.
        assert right.loc["m", "kupiec_p"] > 0.01
        assert wrong.loc["m", "kupiec_p"] < 0.001
