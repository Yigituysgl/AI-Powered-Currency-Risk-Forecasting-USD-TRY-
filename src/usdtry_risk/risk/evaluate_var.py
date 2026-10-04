"""Current VaR for a USD 1,000,000 receivable and a walk-forward 1-day VaR backtest.

Usage:
    python -m usdtry_risk.risk.evaluate_var [--first-year 2010]
"""
from __future__ import annotations

import argparse

import pandas as pd

from usdtry_risk.data.prepare import load_snapshot, log_returns
from usdtry_risk.risk.backtest import backtest_var
from usdtry_risk.risk.var import EXPOSURE_USD, historical_var, parametric_var, var_try
from usdtry_risk.risk.volatility import (
    ewma_variance,
    fit_garch,
    garch_horizon_variance,
    walk_forward_garch,
)

CONFIDENCES = (0.95, 0.99)
HORIZONS = (1, 21)


def current_var(prices: pd.Series, returns: pd.Series, garch_params: pd.Series) -> pd.DataFrame:
    """VaR made at the last snapshot date, in percent of exposure and in TRY."""
    ewma_1d = ewma_variance(returns).iloc[-1]
    garch_1d, garch_21d = garch_horizon_variance(returns, garch_params, horizon=21)
    variances = {"ewma": {1: ewma_1d, 21: 21 * ewma_1d}, "garch": {1: garch_1d, 21: garch_21d}}

    rows = []
    for horizon in HORIZONS:
        for confidence in CONFIDENCES:
            fractions = {name: parametric_var(v[horizon], confidence) for name, v in variances.items()}
            fractions["historical"] = historical_var(returns, confidence, horizon=horizon).iloc[-1]
            for model, fraction in fractions.items():
                rows.append(
                    {
                        "horizon_days": horizon,
                        "confidence": confidence,
                        "model": model,
                        "var_pct": 100 * fraction,
                        "var_try": var_try(fraction, prices.iloc[-1]),
                    }
                )
    return pd.DataFrame(rows).set_index(["horizon_days", "confidence", "model"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--first-year", type=int, default=2010)
    args = parser.parse_args()

    prices = load_snapshot()
    returns = log_returns(prices)
    full_params = fit_garch(returns)
    current = current_var(prices, returns, full_params)
    # Historical VaR follows the recent trend, which zero-mean parametric VaR ignores.
    recent_21d = returns.iloc[-500:].rolling(21).sum().dropna()

    ewma = ewma_variance(returns)
    garch, garch_params = walk_forward_garch(returns, args.first_year)
    backtests = {}
    for confidence in CONFIDENCES:
        forecasts = {
            "ewma": parametric_var(ewma, confidence),
            "garch": parametric_var(garch, confidence),
            "historical": historical_var(returns, confidence),
        }
        backtests[confidence] = backtest_var(returns, forecasts, confidence, start=str(args.first_year))
    a = backtests[CONFIDENCES[0]].attrs

    with pd.option_context(
        "display.float_format", "{:,.4f}".format, "display.width", 160, "display.max_columns", None
    ):
        print(
            f"Current VaR at {prices.index[-1]:%Y-%m-%d}, USD/TRY {prices.iloc[-1]:.4f}, "
            f"exposure USD {EXPOSURE_USD:,}:"
        )
        print(current)
        print(
            f"Rising 21-day windows in the last 500 days: "
            f"{(recent_21d > 0).sum()} of {len(recent_21d)}"
        )
        print("\nGARCH(1,1) parameters fit on all returns (returns in percent):")
        print(full_params.to_frame().T)
        print("\nGARCH(1,1) parameters by year, fit on earlier returns:")
        print(garch_params)
        print(f"\n1-day VaR backtest: {a['n']} days, {a['start']:%Y-%m-%d} to {a['end']:%Y-%m-%d}")
        for confidence, table in backtests.items():
            print(f"\n{confidence:.0%} VaR (mean_var is a fraction of exposure):")
            print(table)


if __name__ == "__main__":
    main()
