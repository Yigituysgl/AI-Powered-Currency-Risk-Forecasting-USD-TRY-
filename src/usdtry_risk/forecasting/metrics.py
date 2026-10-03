"""Forecast error metrics on a common evaluation sample."""
from __future__ import annotations

import numpy as np
import pandas as pd


def rmse(errors: pd.Series) -> float:
    return float(np.sqrt(np.mean(np.square(errors))))


def mae(errors: pd.Series) -> float:
    return float(np.mean(np.abs(errors)))


def evaluate_forecasts(
    prices: pd.Series,
    forecasts: dict[str, pd.Series],
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Score forecasts on the dates where every forecast exists.

    Price errors are in TRY. Log errors are ln(actual / forecast) in percent,
    which equals the realised log return minus the forecast log return and is
    comparable across price levels.
    """
    frame = pd.concat({"actual": prices, **forecasts}, axis=1).loc[start:end].dropna()
    if frame.empty:
        raise ValueError("no dates where all forecasts are available")
    rows = {}
    for name in forecasts:
        price_err = frame["actual"] - frame[name]
        log_err = 100 * np.log(frame["actual"] / frame[name])
        rows[name] = {
            "rmse_try": rmse(price_err),
            "mae_try": mae(price_err),
            "rmse_log_pct": rmse(log_err),
            "mae_log_pct": mae(log_err),
        }
    result = pd.DataFrame.from_dict(rows, orient="index")
    result.attrs.update(n=len(frame), start=frame.index[0], end=frame.index[-1])
    return result
