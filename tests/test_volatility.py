import numpy as np
import pandas as pd
import pytest

from usdtry_risk.risk.volatility import (
    ewma_variance,
    fit_garch,
    garch_horizon_variance,
    garch_variance,
    walk_forward_garch,
)


def simulated_garch(n: int = 1500, seed: int = 0) -> pd.Series:
    """Daily returns from a GARCH(1,1) with omega 1e-6, alpha 0.08, beta 0.9."""
    rng = np.random.default_rng(seed)
    r, h = np.empty(n), 1e-6 / 0.02
    for t in range(n):
        r[t] = np.sqrt(h) * rng.standard_normal()
        h = 1e-6 + 0.08 * r[t] ** 2 + 0.9 * h
    return pd.Series(r, index=pd.bdate_range("2015-01-01", periods=n))


def test_ewma_hand_computed():
    returns = pd.Series([0.01, -0.02, 0.03], index=pd.bdate_range("2020-01-01", periods=3))
    var = ewma_variance(returns, lam=0.94, init=2)
    assert list(var.index) == list(returns.index[1:])
    assert var.iloc[0] == pytest.approx((0.01**2 + 0.02**2) / 2)
    assert var.iloc[1] == pytest.approx(0.94 * var.iloc[0] + 0.06 * 0.03**2)


def test_ewma_has_no_lookahead():
    returns = simulated_garch(600)
    changed = returns.copy()
    changed.iloc[400:] *= 5
    a, b = ewma_variance(returns), ewma_variance(changed)
    cutoff = returns.index[399]
    pd.testing.assert_series_equal(a[:cutoff], b[:cutoff])
    assert not np.allclose(a[returns.index[400] :], b[returns.index[400] :])


def test_garch_next_day_matches_arch_forecast():
    returns = simulated_garch()
    params = fit_garch(returns)
    one_day, total = garch_horizon_variance(returns, params, horizon=1)
    assert garch_variance(returns, params).iloc[-1] == pytest.approx(one_day)
    assert total == pytest.approx(one_day)


def test_garch_horizon_variance_follows_mean_reversion():
    returns = simulated_garch()
    params = fit_garch(returns)
    one_day, total = garch_horizon_variance(returns, params, horizon=21)
    persistence = params["alpha[1]"] + params["beta[1]"]
    long_run = params["omega"] / (1 - persistence) / 100**2
    expected = sum(long_run + persistence**k * (one_day - long_run) for k in range(21))
    assert total == pytest.approx(expected)


def test_walk_forward_garch_has_no_lookahead():
    returns = simulated_garch(1500)
    changed = returns.copy()
    last_year = returns.index.year.max()
    changed[changed.index.year == last_year] *= 5
    a, params_a = walk_forward_garch(returns, first_year=2018)
    b, params_b = walk_forward_garch(changed, first_year=2018)
    earlier = a.index.year < last_year
    pd.testing.assert_series_equal(a[earlier], b[earlier])
    pd.testing.assert_frame_equal(params_a, params_b)
    assert a.index[0] == returns.index[returns.index.year == 2018][0]
    assert a.index.equals(returns.index[returns.index.year >= 2018])
