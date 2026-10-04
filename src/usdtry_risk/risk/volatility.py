"""EWMA and GARCH(1,1) forecasts of the variance of daily log returns.

Every forecast series is indexed by the date it is made: the value at date t
uses returns up to and including t and forecasts the variance of the return
on the next business day.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from arch import arch_model

RISKMETRICS_LAMBDA = 0.94

# arch's optimiser is better conditioned on returns in percent.
_SCALE = 100.0


def ewma_variance(returns: pd.Series, lam: float = RISKMETRICS_LAMBDA, init: int = 252) -> pd.Series:
    """RiskMetrics variance: s2_{t+1} = lam * s2_t + (1 - lam) * r_t^2.

    The recursion starts from the mean squared return of the first ``init``
    returns, so the first forecast is made at date ``init - 1``.
    """
    r2 = np.square(returns.to_numpy())
    if len(r2) < init:
        raise ValueError(f"need at least {init} returns")
    out = np.empty(len(r2) - init + 1)
    out[0] = r2[:init].mean()
    for i, t in enumerate(range(init, len(r2)), start=1):
        out[i] = lam * out[i - 1] + (1 - lam) * r2[t]
    return pd.Series(out, index=returns.index[init - 1 :], name="ewma")


def _garch_model(returns: pd.Series):
    return arch_model(_SCALE * returns, mean="Zero", vol="GARCH", p=1, q=1, dist="normal")


def fit_garch(returns: pd.Series) -> pd.Series:
    """GARCH(1,1) parameters (omega, alpha[1], beta[1]) for returns in percent."""
    res = _garch_model(returns).fit(disp="off")
    if res.convergence_flag != 0:
        raise RuntimeError("GARCH estimation did not converge")
    return res.params


def garch_variance(returns: pd.Series, params: pd.Series) -> pd.Series:
    """Next-day variance at every date of ``returns`` with fixed GARCH parameters."""
    res = _garch_model(returns).fix(params)
    h = np.square(res.conditional_volatility.to_numpy())
    r = _SCALE * returns.to_numpy()
    nxt = params["omega"] + params["alpha[1]"] * r**2 + params["beta[1]"] * h
    return pd.Series(nxt / _SCALE**2, index=returns.index, name="garch")


def walk_forward_garch(returns: pd.Series, first_year: int = 2010) -> tuple[pd.Series, pd.DataFrame]:
    """Next-day GARCH variance made at each date from ``first_year`` on, plus fitted parameters.

    Forecasts made during a year use parameters fit on all returns before that
    year; the variance itself updates daily with each new return.
    """
    dates = returns.index
    forecasts, params_log = [], {}
    for year in sorted(set(dates.year[dates.year >= first_year])):
        params = fit_garch(returns[dates.year < year])
        params_log[year] = params
        var = garch_variance(returns[dates.year <= year], params)
        forecasts.append(var[var.index.year == year])
    return pd.concat(forecasts), pd.DataFrame(params_log).T


def garch_horizon_variance(returns: pd.Series, params: pd.Series, horizon: int) -> tuple[float, float]:
    """Next-day and total ``horizon``-day variance after the last return."""
    res = _garch_model(returns).fix(params)
    path = res.forecast(horizon=horizon, reindex=False).variance.iloc[-1].to_numpy() / _SCALE**2
    return float(path[0]), float(path.sum())
