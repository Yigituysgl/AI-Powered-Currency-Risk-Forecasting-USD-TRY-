"""Compare walk-forward LSTM forecasts with naive and drift on the same test days.

Usage:
    python -m usdtry_risk.forecasting.compare_lstm [--first-test-year 2010]
"""
from __future__ import annotations

import argparse

import pandas as pd

from usdtry_risk.data.prepare import load_snapshot, log_returns
from usdtry_risk.forecasting.baselines import drift_forecast, naive_forecast
from usdtry_risk.forecasting.metrics import evaluate_forecasts, to_price_forecast
from usdtry_risk.forecasting.walk_forward import walk_forward_lstm
from usdtry_risk.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--first-test-year", type=int, default=2010)
    args = parser.parse_args()

    set_seed()
    prices = load_snapshot()
    lstm_returns, fold_log = walk_forward_lstm(log_returns(prices), args.first_test_year)
    forecasts = {
        "lstm": to_price_forecast(prices, lstm_returns),
        "naive": naive_forecast(prices),
        "drift_expanding": drift_forecast(prices),
        "drift_252d": drift_forecast(prices, window=252),
    }

    # Per-fold scores show whether one regime drives the overall result.
    per_fold = {}
    for year in fold_log.index:
        scores = evaluate_forecasts(prices, forecasts, str(year), str(year))
        per_fold[year] = {
            "lstm_mae": scores.loc["lstm", "mae_ret_pct"],
            "naive_mae": scores.loc["naive", "mae_ret_pct"],
            "lstm_dir_acc": scores.loc["lstm", "dir_acc"],
            "up_share": scores.attrs["up_share"],
        }
    fold_table = fold_log.join(pd.DataFrame.from_dict(per_fold, orient="index"))

    result = evaluate_forecasts(prices, forecasts)
    a = result.attrs
    with pd.option_context(
        "display.float_format", "{:.4f}".format, "display.width", 160, "display.max_columns", None
    ):
        print("Per fold (MAE on daily log returns, %):")
        print(fold_table)
        print(f"\nAll folds: {a['n']} test days, {a['start']:%Y-%m-%d} to {a['end']:%Y-%m-%d}")
        print(result[["mae_ret_pct", "rmse_ret_pct", "dir_acc"]])
        print(f"Share of up days: {a['up_share']:.4f}")


if __name__ == "__main__":
    main()
