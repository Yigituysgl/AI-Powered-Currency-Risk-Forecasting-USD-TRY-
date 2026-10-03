import math

import numpy as np
import pandas as pd
import pytest

from usdtry_risk.forecasting.baselines import drift_forecast, naive_forecast
from usdtry_risk.forecasting.metrics import diebold_mariano, evaluate_forecasts, return_errors

IDX = pd.RangeIndex(2000)


def test_diebold_mariano_hand_computed():
    a = pd.Series([1.0, 1.0, 1.0, 1.0, 1.0])
    b = pd.Series([0.0, 2.0, 0.0, 2.0, 0.0])
    # Absolute-loss differences 1, -1, 1, -1, 1: mean 0.2, variance 0.96.
    stat, p = diebold_mariano(a, b, loss="absolute")
    expected = 0.2 / math.sqrt(0.96 / 5) * math.sqrt(4 / 5)
    assert stat == pytest.approx(expected)
    assert p == pytest.approx(math.erfc(expected / math.sqrt(2)))


def test_diebold_mariano_detects_a_better_forecast():
    rng = np.random.default_rng(0)
    good = pd.Series(rng.normal(0, 1, len(IDX)), index=IDX)
    bad = good + pd.Series(rng.normal(0, 1, len(IDX)), index=IDX)
    for loss in ("squared", "absolute"):
        stat, p = diebold_mariano(good, bad, loss)
        assert stat < 0 and p < 0.01


def test_diebold_mariano_equal_accuracy_is_not_rejected():
    rng = np.random.default_rng(1)
    a = pd.Series(rng.normal(0, 1, len(IDX)), index=IDX)
    b = pd.Series(rng.normal(0, 1, len(IDX)), index=IDX)
    _, p = diebold_mariano(a, b)
    assert p > 0.05


def test_diebold_mariano_is_symmetric():
    rng = np.random.default_rng(2)
    a = pd.Series(rng.normal(0, 1, len(IDX)), index=IDX)
    b = pd.Series(rng.normal(0, 1.2, len(IDX)), index=IDX)
    stat_ab, p_ab = diebold_mariano(a, b)
    stat_ba, p_ba = diebold_mariano(b, a)
    assert stat_ab == pytest.approx(-stat_ba)
    assert p_ab == pytest.approx(p_ba)


def test_diebold_mariano_rejects_bad_input():
    a = pd.Series([0.1, -0.2, 0.3])
    with pytest.raises(ValueError, match="same dates"):
        diebold_mariano(a, pd.Series([0.1, -0.2, 0.3], index=[5, 6, 7]))
    with pytest.raises(ValueError, match="constant"):
        diebold_mariano(a, a)


def test_return_errors_use_the_evaluation_sample():
    dates = pd.bdate_range("2020-01-01", periods=300)
    rng = np.random.default_rng(3)
    prices = pd.Series(10 * np.exp(np.cumsum(rng.normal(0.001, 0.01, len(dates)))), index=dates)
    forecasts = {"naive": naive_forecast(prices), "drift": drift_forecast(prices, window=50)}

    errors = return_errors(prices, forecasts)
    scores = evaluate_forecasts(prices, forecasts)
    assert len(errors) == scores.attrs["n"]
    assert errors.index[0] == scores.attrs["start"]
    assert errors["naive"].abs().mean() == pytest.approx(scores.loc["naive", "mae_ret_pct"])
