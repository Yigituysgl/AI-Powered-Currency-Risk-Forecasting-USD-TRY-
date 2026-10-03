import numpy as np
import pandas as pd
import pytest

from usdtry_risk.forecasting.baselines import drift_forecast, naive_forecast
from usdtry_risk.forecasting.metrics import evaluate_forecasts
from usdtry_risk.seed import set_seed

DATES = pd.bdate_range("2020-01-01", periods=300)


def random_walk(seed=0):
    rng = np.random.default_rng(seed)
    return pd.Series(10 * np.exp(np.cumsum(rng.normal(0.001, 0.01, len(DATES)))), index=DATES)


def test_naive_is_previous_price():
    prices = random_walk()
    f = naive_forecast(prices)
    assert np.isnan(f.iloc[0])
    assert (f.iloc[1:].to_numpy() == prices.iloc[:-1].to_numpy()).all()


def test_drift_is_exact_on_constant_growth():
    prices = pd.Series(100 * np.exp(0.002 * np.arange(len(DATES))), index=DATES)
    for f in (drift_forecast(prices), drift_forecast(prices, window=50)):
        valid = f.dropna()
        assert len(valid) > 200
        assert valid.to_numpy() == pytest.approx(prices[valid.index].to_numpy())


def test_rolling_drift_needs_full_window():
    f = drift_forecast(random_walk(), window=50)
    # 50 returns need 51 prices, and the forecast applies to the following day.
    assert f.first_valid_index() == DATES[51]


@pytest.mark.parametrize(
    "make", [naive_forecast, drift_forecast, lambda p: drift_forecast(p, window=50)]
)
def test_no_lookahead(make):
    prices = random_walk()
    k = 200
    shocked = prices.copy()
    shocked.iloc[k:] *= 3.0  # change everything from day k on
    before, after = make(prices), make(shocked)
    # Forecasts up to and including day k use only prices before day k.
    pd.testing.assert_series_equal(before.iloc[: k + 1], after.iloc[: k + 1])
    assert not np.allclose(before.iloc[k + 1 :], after.iloc[k + 1 :])


def test_evaluate_forecasts_known_values():
    idx = pd.bdate_range("2024-01-01", periods=4)
    prices = pd.Series([10.0, 11.0, 12.0, 12.0], index=idx)
    result = evaluate_forecasts(prices, {"naive": naive_forecast(prices)})
    # Errors on the last three days: 1, 1, 0.
    assert result.loc["naive", "mae_try"] == pytest.approx(2 / 3)
    assert result.loc["naive", "rmse_try"] == pytest.approx(np.sqrt(2 / 3))
    expected_log = 100 * np.array([np.log(1.1), np.log(12 / 11), 0.0])
    assert result.loc["naive", "mae_ret_pct"] == pytest.approx(np.abs(expected_log).mean())
    assert np.isnan(result.loc["naive", "dir_acc"])  # a zero forecast has no direction
    assert result.attrs["n"] == 3
    assert result.attrs["up_share"] == pytest.approx(2 / 3)


def test_evaluate_uses_common_sample_and_date_range():
    prices = random_walk()
    forecasts = {"naive": naive_forecast(prices), "drift": drift_forecast(prices, window=50)}
    result = evaluate_forecasts(prices, forecasts, start="2020-06-01")
    assert result.attrs["start"] >= pd.Timestamp("2020-06-01")
    assert result.attrs["n"] == len(prices.loc["2020-06-01":])


def test_set_seed_is_deterministic():
    set_seed()
    a = np.random.rand(5)
    set_seed()
    assert (np.random.rand(5) == a).all()
