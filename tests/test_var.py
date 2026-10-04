import math

import numpy as np
import pandas as pd
import pytest

from usdtry_risk.risk.var import historical_var, loss_fraction, parametric_var, var_try


def test_loss_is_positive_when_usd_try_falls():
    assert loss_fraction(math.log(0.9)) == pytest.approx(0.1)
    assert loss_fraction(math.log(1.1)) == pytest.approx(-0.1)


def test_parametric_var_hand_computed():
    sigma = 0.01
    assert parametric_var(sigma**2, 0.95) == pytest.approx(1 - math.exp(-1.6448536 * sigma))
    assert parametric_var(sigma**2, 0.99) == pytest.approx(1 - math.exp(-2.3263479 * sigma))


def test_square_root_of_time_scaling():
    one_day_sigma = 0.01
    var_21 = parametric_var(21 * one_day_sigma**2, 0.99)
    assert var_21 == pytest.approx(1 - math.exp(-2.3263479 * one_day_sigma * math.sqrt(21)))


def test_var_try_scales_exposure():
    assert var_try(0.02, usd_try=40.0) == pytest.approx(800_000)


def test_historical_var_is_the_lower_return_quantile():
    returns = pd.Series(np.arange(1, 101) / 1000.0 - 0.05, index=pd.bdate_range("2020-01-01", periods=100))
    var = historical_var(returns, confidence=0.95, window=100)
    assert len(var) == 1 and var.index[0] == returns.index[-1]
    assert var.iloc[0] == pytest.approx(loss_fraction(np.quantile(returns, 0.05)))


def test_historical_var_uses_only_past_returns():
    rng = np.random.default_rng(0)
    returns = pd.Series(rng.normal(0, 0.01, 800), index=pd.bdate_range("2020-01-01", periods=800))
    changed = returns.copy()
    changed.iloc[600:] = -0.2
    for horizon in (1, 21):
        a = historical_var(returns, 0.99, horizon=horizon)
        b = historical_var(changed, 0.99, horizon=horizon)
        cutoff = returns.index[599]
        pd.testing.assert_series_equal(a[:cutoff], b[:cutoff])


def test_historical_multi_day_var_uses_horizon_returns():
    rng = np.random.default_rng(1)
    returns = pd.Series(rng.normal(0, 0.01, 520), index=pd.bdate_range("2020-01-01", periods=520))
    var = historical_var(returns, 0.95, window=500, horizon=21)
    sums = returns.rolling(21).sum().iloc[-480:]
    assert var.iloc[-1] == pytest.approx(loss_fraction(np.quantile(sums, 0.05)))
