"""Forecast error metrics on a common evaluation sample."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


def rmse(errors: pd.Series) -> float:
    return float(np.sqrt(np.mean(np.square(errors))))


def mae(errors: pd.Series) -> float:
    return float(np.mean(np.abs(errors)))


def directional_accuracy(actual: pd.Series, forecast: pd.Series) -> float:
    """Share of days where forecast and realised return have the same sign.

    A zero forecast has no direction, so those days are excluded; the result
    is NaN when every forecast is zero.
    """
    has_direction = forecast != 0
    if not has_direction.any():
        return float("nan")
    return float((np.sign(actual[has_direction]) == np.sign(forecast[has_direction])).mean())


def to_price_forecast(prices: pd.Series, return_forecast: pd.Series) -> pd.Series:
    """Price forecast for day t from a log-return forecast: P_{t-1} * exp(r_hat_t)."""
    prev = prices.shift(1).reindex(return_forecast.index)
    return (prev * np.exp(return_forecast)).rename(return_forecast.name)


def diebold_mariano(
    errors_a: pd.Series, errors_b: pd.Series, loss: str = "squared"
) -> tuple[float, float]:
    """Test equal accuracy of two one-day-ahead forecasts.

    Uses the Harvey-Leybourne-Newbold small-sample correction and returns the
    statistic and a two-sided p-value from the normal approximation. A
    negative statistic means forecast A has the lower average loss.
    """
    if not errors_a.index.equals(errors_b.index):
        raise ValueError("both forecasts must be scored on the same dates")
    loss_fn = {"squared": np.square, "absolute": np.abs}[loss]
    d = (loss_fn(errors_a) - loss_fn(errors_b)).to_numpy()
    n = len(d)
    variance = np.mean((d - d.mean()) ** 2)
    if not variance > 0:
        raise ValueError("loss differences are constant; the test is undefined")
    stat = d.mean() / math.sqrt(variance / n) * math.sqrt((n - 1) / n)
    return float(stat), math.erfc(abs(stat) / math.sqrt(2))


def _common_sample(prices, forecasts, start, end) -> pd.DataFrame:
    """Actual, previous and forecast prices on dates where every forecast exists."""
    frame = (
        pd.concat({"actual": prices, "prev": prices.shift(1), **forecasts}, axis=1)
        .loc[start:end]
        .dropna()
    )
    if frame.empty:
        raise ValueError("no dates where all forecasts are available")
    return frame


def return_errors(
    prices: pd.Series,
    forecasts: dict[str, pd.Series],
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Daily log-return errors in percent, ln(actual / forecast), on the common sample."""
    frame = _common_sample(prices, forecasts, start, end)
    return pd.DataFrame({name: 100 * np.log(frame["actual"] / frame[name]) for name in forecasts})


def evaluate_forecasts(
    prices: pd.Series,
    forecasts: dict[str, pd.Series],
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Score price forecasts on the dates where every forecast exists.

    ``*_try`` errors are in TRY. ``*_ret_pct`` errors are on daily log returns
    in percent, ln(actual / forecast), which is comparable across price levels.
    ``dir_acc`` is the directional accuracy of the implied return forecast.
    """
    frame = _common_sample(prices, forecasts, start, end)
    actual_ret = np.log(frame["actual"] / frame["prev"])
    rows = {}
    for name in forecasts:
        price_err = frame["actual"] - frame[name]
        ret_err = 100 * np.log(frame["actual"] / frame[name])
        rows[name] = {
            "rmse_try": rmse(price_err),
            "mae_try": mae(price_err),
            "rmse_ret_pct": rmse(ret_err),
            "mae_ret_pct": mae(ret_err),
            "dir_acc": directional_accuracy(actual_ret, np.log(frame[name] / frame["prev"])),
        }
    result = pd.DataFrame.from_dict(rows, orient="index")
    result.attrs.update(
        n=len(frame),
        start=frame.index[0],
        end=frame.index[-1],
        up_share=float((actual_ret > 0).mean()),
    )
    return result
