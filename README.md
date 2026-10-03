# USD/TRY Currency Risk & Hedging Tool

Forecasting, volatility, Value at Risk and hedging analysis for USD/TRY
exposure, built on a reproducible ECB data snapshot.

Planned components:

1. Reproducible data and one-day-ahead forecasts compared against naive and
   drift baselines on the same dates
2. Volatility forecasting and Value at Risk with backtesting
3. Hedging simulator for a USD exposure
4. Streamlit dashboard

## Data

Daily ECB euro reference rates, with USD/TRY derived as EUR/TRY ÷ EUR/USD.
**Source: ECB statistics.** See [data/README.md](data/README.md) for
provenance, terms of use and column definitions.

## Setup

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python -m pip install -e .
.venv/Scripts/python -m pytest
```

On Linux or macOS use `.venv/bin/python`.

## Usage

```bash
python -m usdtry_risk.data.fetch_ecb
python -m usdtry_risk.forecasting.compare_lstm
python -m usdtry_risk.forecasting.evaluate_baselines --start 2023-06-19 --end 2025-05-07
```

`fetch_ecb` saves a new dated snapshot. `compare_lstm` runs the walk-forward
LSTM and scores it against the baselines on the same test days.
`evaluate_baselines` scores only the baselines, optionally on a date range.

## Results

One-day-ahead forecasts of the daily log return on snapshot
`usdtry_ecb_2026-10-03`. The LSTM (one layer, 16 hidden units, 20-day
lookback) is evaluated walk-forward with one fold per calendar year: each fold
trains on all earlier returns, stops early on the last 252 of them, and fits
its scaler on the rest. All forecasts are scored on the same 4,290 test days,
2010-01-04 to 2026-10-02. Errors are in percent of daily log return.

| Forecast                 | MAE (%) | RMSE (%) | Directional accuracy |
|--------------------------|--------:|---------:|---------------------:|
| LSTM                     |  0.4927 |   0.9781 |               0.5814 |
| Naive (tomorrow = today) |  0.4946 |   0.9844 |                  n/a |
| Drift, expanding mean    |  0.4916 |   0.9813 |               0.5848 |
| Drift, 252-day mean      |  0.4974 |   0.9834 |               0.5804 |

The share of up days in the test period is 0.5848. Naive forecasts no change,
so it has no direction. 96.4% of the LSTM forecasts are positive and their
within-year standard deviation stays below 0.31% against 0.98% for realised
returns: the LSTM behaves like a drift forecast. It does not beat the
expanding-mean drift on MAE or directional accuracy; its RMSE is the lowest
(0.9781 against 0.9813).

Diebold-Mariano tests on the same 4,290 days (Harvey-Leybourne-Newbold
correction, two-sided p-values; a negative statistic means the LSTM has the
lower loss). The absolute-error test matches MAE, the squared-error test
matches RMSE.

| Comparison                   | DM (absolute) | p (absolute) | DM (squared) | p (squared) |
|------------------------------|--------------:|-------------:|-------------:|------------:|
| LSTM vs naive                |       -1.4042 |       0.1603 |      -2.3378 |      0.0194 |
| LSTM vs drift, expanding mean |       1.2050 |       0.2282 |      -1.0616 |      0.2884 |

In plain terms: the LSTM beats tomorrow = today only on squared error
(p = 0.0194) and is not significantly different from a simple drift forecast
on either measure, so there is no evidence it adds anything beyond the lira's
average trend.

## Repository layout

- `src/usdtry_risk/data/`: ECB download, snapshot loading, log returns
- `src/usdtry_risk/forecasting/`: baselines, LSTM, walk-forward evaluation, metrics
- `tests/`: data, baseline and LSTM tests, including lookahead and scaler-leakage checks
- `notebooks/00_legacy_lstm.ipynb`: exploratory LSTM notebook; it reads data
  from outside the repository, so its outputs are not part of the results above
