# Data

**Source: ECB statistics.**

Exchange rates come from the European Central Bank's euro foreign exchange
reference rates (dataset `EXR`), downloaded from the ECB Data Portal API:

```
https://data-api.ecb.europa.eu/service/data/EXR/D.TRY+USD.EUR.SP00.A
```

## Files

`raw/usdtry_ecb_<download-date>.csv` is a snapshot of one download.
`raw/usdtry_ecb_<download-date>.manifest.json` records the source URL, the
download time (UTC), the row count, the date range and the SHA-256 hash of the
CSV. Code loads snapshots through `usdtry_risk.data.prepare.load_snapshot`,
which refuses a file that does not match its manifest. Snapshots are never
edited. To refresh, download a new one:

```bash
python -m usdtry_risk.data.fetch_ecb
```

| Column    | Meaning                                                              |
|-----------|----------------------------------------------------------------------|
| `date`    | Reference rate date (TARGET business days only)                      |
| `eur_try` | Turkish lira per 1 euro, **as published by the ECB, unmodified**     |
| `eur_usd` | US dollars per 1 euro, **as published by the ECB, unmodified**       |
| `usd_try` | Turkish lira per 1 US dollar, **derived by this project**            |

## Derived series

The ECB does not publish a USD/TRY rate. This project computes
`usd_try = eur_try / eur_usd`, rounded to 6 decimals. **`usd_try` is this
project's own calculation, not an ECB statistic.** The ECB values it is built
from are kept unmodified in the same file.

## Terms of use

The ECB allows free reuse of its public statistics provided the source is
quoted ("Source: ECB statistics.") and the statistics are not modified
([ECB statistics usage policy](https://www.ecb.europa.eu/stats/ecb_statistics/governance_and_quality_framework/html/usage_policy.en.html)).

The ECB states that the euro reference rates "are published for information
purposes only" and that using them for transaction purposes is strongly
discouraged ([euro reference rates](https://www.ecb.europa.eu/stats/policy_and_exchange_rates/euro_reference_exchange_rates/html/index.en.html)).
The ECB accepts no liability for decisions made in reliance on the rates
([ECB disclaimer](https://www.ecb.europa.eu/home/disclaimer/html/index.en.html)).
This project is for research and education, not for trading or pricing.

## Notes on the data

- Rates are a fixing taken around 14:10–14:15 CET, not a market close. On
  days with large intraday moves the fixing can differ markedly from the
  closing price.
- The ECB `TRY` series starts in 1999. Rates before the 2005 redenomination
  are expressed in new lira (1 TRY = 1,000,000 old TRL).
- Tail events such as the February 2001 float and the 2018 and 2021 crises
  are real and are kept.

## Interest rates

Downloaded with `python -m usdtry_risk.data.fetch_rates`, one snapshot and
manifest per series, loaded through `usdtry_risk.data.rates.rate_series`,
which refuses a file that does not match its manifest. Values are stored as
published.

| File prefix            | Series                                                    | Source |
|------------------------|-----------------------------------------------------------|--------|
| `usd_dtb3_fred_`       | 3-month US Treasury bill, secondary market, discount basis, daily | Board of Governors of the Federal Reserve System (US), H.15, via [FRED DTB3](https://fred.stlouisfed.org/series/DTB3) |
| `try_policy_tcmb_`     | TCMB one-week repo lending rate (policy rate), one row per change | [CBRT, 1 Week Repo](https://www.tcmb.gov.tr/wps/wcm/connect/EN/TCMB+EN/Main+Menu/Core+Functions/Monetary+Policy/Central+Bank+Interest+Rates/1+Week+Repo) |
| `try_callmoney_fred_`  | Turkish call money / interbank overnight rate, monthly    | OECD Main Economic Indicators, via [FRED IRSTCI01TRM156N](https://fred.stlouisfed.org/series/IRSTCI01TRM156N) |

- **FRED DTB3:** public domain, citation requested. Empty values are US
  non-business days. A few values are slightly negative (2015, 2020).
- **TCMB:** the CBRT allows its published information to be quoted with a
  specific reference; commercial use needs its prior written permission
  ([CBRT disclaimer](https://www.tcmb.gov.tr/wps/wcm/connect/EN/TCMB+EN/Bottom+Menu/Other/Disclaimer)).
  The one-week repo rate starts on 2010-05-20. The page lists dates in
  `DD.MM.YYYY` form and at least one rate with a comma decimal (`5,00`), so
  the file keeps the published strings plus parsed `date` and `rate_pct`
  columns derived by this project.
- **OECD call money rate:** copyrighted by the OECD, citation required. Each
  value is dated the first day of the month it refers to, but it is
  published later.
