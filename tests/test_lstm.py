import numpy as np
import pandas as pd
import pytest

from usdtry_risk.forecasting.lstm import LSTMConfig, make_windows
from usdtry_risk.forecasting.metrics import directional_accuracy, to_price_forecast
from usdtry_risk.forecasting.scaling import StandardScaler
from usdtry_risk.forecasting.walk_forward import walk_forward_lstm, yearly_folds

TINY = LSTMConfig(lookback=5, hidden_size=4, max_epochs=3, patience=2, batch_size=32)
VAL_SIZE = 60


def synthetic_returns(seed=0):
    dates = pd.bdate_range("2018-01-01", "2021-06-30")
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0.0005, 0.01, len(dates)), index=dates)


def run(returns, **kwargs):
    return walk_forward_lstm(returns, 2020, TINY, val_size=VAL_SIZE, **kwargs)


def test_make_windows_alignment():
    X, y = make_windows(np.arange(10.0), lookback=3)
    assert X.shape == (7, 3, 1)
    assert X[0, :, 0].tolist() == [0, 1, 2] and y[0] == 3
    assert X[-1, :, 0].tolist() == [6, 7, 8] and y[-1] == 9


def test_yearly_folds_are_contiguous_and_expanding():
    dates = synthetic_returns().index
    folds = yearly_folds(dates, 2019)
    assert [f.test_start.year for f in folds] == [2019, 2020, 2021]
    assert folds[0].test_start == dates[dates.year == 2019][0]
    assert folds[-1].test_end == dates[-1]
    for a, b in zip(folds, folds[1:]):
        assert dates[dates.get_loc(a.test_end) + 1] == b.test_start


def test_scaler_never_sees_validation_or_test_dates():
    fitted = []

    class SpyScaler(StandardScaler):
        def fit(self, x):
            fitted.append(x.index)
            return super().fit(x)

    returns = synthetic_returns()
    run(returns, scaler_factory=SpyScaler)

    folds = yearly_folds(returns.index, 2020)
    assert len(fitted) == len(folds)
    for fold, fit_dates in zip(folds, fitted):
        train_dates = returns.index[returns.index < fold.test_start]
        val_start = train_dates[-VAL_SIZE]
        assert fit_dates.max() < val_start < fold.test_start
        assert fit_dates.equals(train_dates[train_dates < val_start])


def test_forecasts_cover_exactly_the_test_days():
    returns = synthetic_returns()
    forecast, log = run(returns)
    assert forecast.index.equals(returns.index[returns.index.year >= 2020])
    assert np.isfinite(forecast).all()
    assert (log["n_val"] == VAL_SIZE).all()
    assert log["n_test"].sum() == len(forecast)


def test_no_lookahead():
    returns = synthetic_returns()
    cut = returns.index[returns.index.year == 2021][40]
    shocked = returns.copy()
    shocked[shocked.index >= cut] *= 10.0

    before, _ = run(returns)
    after, _ = run(shocked)
    # The forecast for `cut` uses returns up to the day before, so it must not change.
    pd.testing.assert_series_equal(before[:cut], after[:cut])
    assert not np.allclose(before[cut:].iloc[1:], after[cut:].iloc[1:])


def test_directional_accuracy():
    actual = pd.Series([0.01, -0.02, 0.03, -0.01])
    assert directional_accuracy(actual, pd.Series([0.5, 0.5, 0.5, -0.5])) == pytest.approx(0.75)
    assert directional_accuracy(actual, pd.Series([0.0, 0.5, 0.0, 0.0])) == 0.0
    assert np.isnan(directional_accuracy(actual, pd.Series([0.0] * 4)))


def test_to_price_forecast():
    idx = pd.bdate_range("2024-01-01", periods=3)
    prices = pd.Series([10.0, 11.0, 12.0], index=idx)
    f = to_price_forecast(prices, pd.Series([np.log(1.5), 0.0], index=idx[1:], name="m"))
    assert f.tolist() == pytest.approx([15.0, 11.0])
    assert f.name == "m"
