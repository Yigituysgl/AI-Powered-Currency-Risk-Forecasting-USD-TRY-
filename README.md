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
pip install -r requirements.txt
pip install -e .
python -m pytest
```

## Usage

```bash
python -m usdtry_risk.data.fetch_ecb
python -m usdtry_risk.forecasting.evaluate_baselines --start 2023-06-19 --end 2025-05-07
```

`fetch_ecb` saves a new dated snapshot; `evaluate_baselines` scores the
baselines on the latest snapshot, optionally restricted to a date range.

## Results

One-day-ahead forecasts on snapshot `usdtry_ecb_2026-10-03`, 6,853 forecast
days from 1999-12-23 to 2026-10-02, all scored on the same dates. Log errors
are ln(actual / forecast) in percent.

| Forecast              | RMSE (TRY) | MAE (TRY) | RMSE (log %) | MAE (log %) |
|-----------------------|-----------:|----------:|-------------:|------------:|
| Naive (tomorrow = today) | 0.0892  |   0.0257  |       1.2322 |      0.5890 |
| Drift, expanding mean |     0.0884 |    0.0251 |       1.2313 |      0.5909 |
| Drift, 252-day mean   |     0.0888 |    0.0257 |       1.2328 |      0.5918 |

## Repository layout

- `src/usdtry_risk/data/`: ECB download, snapshot loading, log returns
- `src/usdtry_risk/forecasting/`: baselines and error metrics
- `tests/`: data and baseline tests, including a lookahead check
- `notebooks/00_legacy_lstm.ipynb`: exploratory LSTM notebook; it reads data
  from outside the repository, so its outputs are not part of the results above
