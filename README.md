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
python -m usdtry_risk.risk.evaluate_var
```

`fetch_ecb` saves a new dated snapshot. `compare_lstm` runs the walk-forward
LSTM and scores it against the baselines on the same test days.
`evaluate_baselines` scores only the baselines, optionally on a date range.
`evaluate_var` prints the current VaR and the 1-day VaR backtest.

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

## Volatility and Value at Risk

The exposure is an exporter that will receive USD 1,000,000 and convert it to
TRY. It loses when USD/TRY falls (the lira strengthens). The loss over a
horizon is 1 − exp(R), where R is the USD/TRY log return, as a share of the
exposure's TRY value. VaR is the loss that should be exceeded only 5% (95%
VaR) or 1% (99% VaR) of the time.

Three models, all on snapshot `usdtry_ecb_2026-10-03`:

- **EWMA** (RiskMetrics, λ = 0.94): tomorrow's variance is 94% of today's
  estimate plus 6% of today's squared return.
- **GARCH(1,1)** (zero mean, normal errors, `arch` package): variance is a
  constant plus weights on yesterday's squared return and yesterday's variance.
- **Historical**: the empirical loss quantile of the last 500 daily returns,
  with no distribution assumed. It is the simple baseline the two models
  should beat.

EWMA and GARCH VaR assume a normal, zero-mean return. The 21-day EWMA
variance is 21 times the 1-day variance (square-root-of-time). The 21-day
GARCH variance is the sum of its own 21 daily forecasts. The 21-day historical
VaR uses the overlapping 21-day returns inside the 500-day window.

### Current VaR

Made at 2026-10-02 with USD/TRY at 49.1448, so the exposure is worth
TRY 49,144,766. GARCH here is fit on all returns.

| Horizon | Confidence | Model      | VaR (% of exposure) |    VaR (TRY) |
|---------|-----------:|------------|--------------------:|-------------:|
| 1 day   |        95% | EWMA       |              0.1825 |       89,712 |
| 1 day   |        95% | GARCH      |              0.6657 |      327,157 |
| 1 day   |        95% | Historical |              0.1165 |       57,237 |
| 1 day   |        99% | EWMA       |              0.2581 |      126,833 |
| 1 day   |        99% | GARCH      |              0.9402 |      462,066 |
| 1 day   |        99% | Historical |              0.2611 |      128,301 |
| 21 days |        95% | EWMA       |              0.8338 |      409,768 |
| 21 days |        95% | GARCH      |              6.6891 |    3,287,367 |
| 21 days |        95% | Historical |             −0.8785 |     −431,727 |
| 21 days |        99% | EWMA       |              1.1772 |      578,540 |
| 21 days |        99% | GARCH      |              9.3277 |    4,584,094 |
| 21 days |        99% | Historical |             −0.7147 |     −351,261 |

In plain terms:

- **EWMA:** on 99 days out of 100, the exporter should lose less than
  TRY 126,833 by tomorrow, and less than TRY 578,540 over the next month.
- **Historical:** the 1-day figures are close to EWMA at 99%. The 21-day
  figures are negative: in the last 500 days USD/TRY rose over all 480
  overlapping 21-day windows, so even the worst month in that window was a
  gain for the exporter. This is a property of that calm, steadily weakening
  lira period, not a guarantee. The window contains no month in which the
  lira strengthened.
- **GARCH:** far above EWMA at every horizon and confidence. Its fitted
  parameters are omega 0.0679, alpha 0.4887 and beta 0.5113 (returns in
  percent): alpha + beta is 1, the edge of what the model allows, likely
  because a few very large one-day moves dominate the fit under normal
  errors. As a result the variance never falls below a
  floor set by omega, and it grows with the horizon instead of returning to
  an average. That is why the 21-day GARCH VaR is so large.

### Backtest of 1-day VaR

Walk-forward with no future data. Each day's VaR uses only returns up to the
day before. EWMA and historical update daily. GARCH parameters are refit once
a year on all earlier returns and the variance updates daily. The fitted
alpha + beta is 1.0000 (four decimals) in every year from 2010 to 2026. All
three models are checked on the same 4,289 days, 2010-01-05 to 2026-10-02.

A breach is a day when the actual loss was larger than the VaR. The Kupiec
test asks whether the breach rate is consistent with the target rate; a
p-value below 0.05 means it is not. Mean VaR is the average VaR level as a
share of the exposure, which shows how much a model pays for few breaches.

**95% VaR** (expected breaches: 214.45, rate 0.05)

| Model      | Breaches | Breach rate | Kupiec LR | Kupiec p | Mean VaR |
|------------|---------:|------------:|----------:|---------:|---------:|
| EWMA       |      121 |      0.0282 |   50.5339 |   0.0000 |   0.0114 |
| GARCH      |       88 |      0.0205 |  100.0141 |   0.0000 |   0.0132 |
| Historical |      194 |      0.0452 |    2.1177 |   0.1456 |   0.0094 |

**99% VaR** (expected breaches: 42.89, rate 0.01)

| Model      | Breaches | Breach rate | Kupiec LR | Kupiec p | Mean VaR |
|------------|---------:|------------:|----------:|---------:|---------:|
| EWMA       |       39 |      0.0091 |    0.3676 |   0.5443 |   0.0161 |
| GARCH      |       27 |      0.0063 |    6.8481 |   0.0089 |   0.0185 |
| Historical |       50 |      0.0117 |    1.1303 |   0.2877 |   0.0190 |

In plain terms:

- **Historical** passes at both levels: 194 breaches against 214.45 expected
  at 95%, and 50 against 42.89 at 99%.
- **EWMA** passes at 99% (39 against 42.89) but fails at 95%: it breached
  only 121 times against 214.45 expected, so it is too cautious there. Being
  too cautious at 95% but about right at 99% is the usual pattern when returns
  have fatter tails than a normal distribution: more quiet days and more
  extreme days. Assuming a zero mean also ignores the
  lira's long-run weakening, which makes the exporter's losses rarer than
  the model expects.
- **GARCH** fails at both levels with too few breaches (88 and 27) and the
  highest mean VaR at both levels. That is consistent with the boundary fit
  described above.
- Overall, the simplest model, historical VaR, is the only one whose breach
  rate matches the target at both levels. At 99%, EWMA matches it with a
  lower average VaR (0.0161 against 0.0190).

## Repository layout

- `src/usdtry_risk/data/`: ECB download, snapshot loading, log returns
- `src/usdtry_risk/forecasting/`: baselines, LSTM, walk-forward evaluation, metrics
- `src/usdtry_risk/risk/`: EWMA and GARCH volatility, VaR, backtest and Kupiec test
- `tests/`: data, baseline, LSTM, volatility, VaR and backtest tests, including
  lookahead and scaler-leakage checks
- `notebooks/00_legacy_lstm.ipynb`: exploratory LSTM notebook; it reads data
  from outside the repository, so its outputs are not part of the results above
