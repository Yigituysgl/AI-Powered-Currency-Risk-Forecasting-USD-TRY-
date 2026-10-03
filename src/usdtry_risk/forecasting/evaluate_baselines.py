"""Print baseline forecast errors for the latest snapshot.

Usage:
    python -m usdtry_risk.forecasting.evaluate_baselines [--start YYYY-MM-DD] [--end YYYY-MM-DD]
"""
from __future__ import annotations

import argparse

import pandas as pd

from usdtry_risk.data.prepare import load_snapshot
from usdtry_risk.forecasting.baselines import drift_forecast, naive_forecast
from usdtry_risk.forecasting.metrics import evaluate_forecasts
from usdtry_risk.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    args = parser.parse_args()

    set_seed()
    prices = load_snapshot()
    forecasts = {
        "naive": naive_forecast(prices),
        "drift_expanding": drift_forecast(prices),
        "drift_252d": drift_forecast(prices, window=252),
    }
    result = evaluate_forecasts(prices, forecasts, args.start, args.end)
    a = result.attrs
    print(f"{a['n']} forecast days, {a['start']:%Y-%m-%d} to {a['end']:%Y-%m-%d}")
    with pd.option_context("display.float_format", "{:.4f}".format):
        print(result)


if __name__ == "__main__":
    main()
