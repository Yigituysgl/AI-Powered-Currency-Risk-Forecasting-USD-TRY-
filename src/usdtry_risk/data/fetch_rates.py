"""Download USD and TRY interest rates and save dated snapshots.

Three series, each saved as its own CSV with a manifest:

- ``usd_dtb3_fred_*``: 3-month US Treasury bill rate (FRED series DTB3)
- ``try_policy_tcmb_*``: TCMB one-week repo lending rate (policy rate)
- ``try_callmoney_fred_*``: OECD call money rate for Turkey (FRED series
  IRSTCI01TRM156N)

Values are stored as published, as strings; parsed columns are added next to
them where the published format is not ISO dates and plain numbers.

Usage:
    python -m usdtry_risk.data.fetch_rates [--out-dir data/raw] [--force]
"""
from __future__ import annotations

import argparse
import io
import json
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

import pandas as pd
import requests

from usdtry_risk.data.fetch_ecb import DEFAULT_RAW_DIR, _sha256

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
TCMB_REPO_URL = (
    "https://www.tcmb.gov.tr/wps/wcm/connect/EN/TCMB+EN/Main+Menu/Core+Functions/"
    "Monetary+Policy/Central+Bank+Interest+Rates/1+Week+Repo"
)

SOURCES = {
    "usd_dtb3_fred_": {
        "source": "Board of Governors of the Federal Reserve System (US), H.15 Selected "
        "Interest Rates, 3-Month Treasury Bill Secondary Market Rate, Discount Basis "
        "[DTB3], retrieved from FRED, Federal Reserve Bank of St. Louis",
        "source_url": FRED_CSV_URL.format(series="DTB3"),
        "attribution": "Source: Board of Governors of the Federal Reserve System (US) via FRED.",
        "columns": {
            "date": "observation date",
            "value": "percent per year, discount basis, as published; empty on non-business days",
        },
    },
    "try_policy_tcmb_": {
        "source": "Central Bank of the Republic of Turkiye (CBRT), Central Bank Interest "
        "Rates, 1 Week Repo",
        "source_url": TCMB_REPO_URL,
        "attribution": "Source: Central Bank of the Republic of Turkiye (CBRT).",
        "columns": {
            "date_published": "date of the rate change, DD.MM.YYYY, as published",
            "borrowing": "borrowing rate, as published",
            "lending": "one-week repo lending rate in percent, as published",
            "date": "date_published as an ISO date, derived by this project",
            "rate_pct": "lending as a number (comma read as decimal point), derived by this project",
        },
    },
    "try_callmoney_fred_": {
        "source": "OECD, Main Economic Indicators, Interest Rates: Immediate Rates (< 24 Hours): "
        "Call Money/Interbank Rate: Total for Turkey [IRSTCI01TRM156N], retrieved from FRED, "
        "Federal Reserve Bank of St. Louis",
        "source_url": FRED_CSV_URL.format(series="IRSTCI01TRM156N"),
        "attribution": "Source: OECD Main Economic Indicators via FRED.",
        "columns": {
            "date": "first day of the month the value refers to",
            "value": "percent per year, monthly, as published",
        },
    },
}


def _get(url: str, timeout: float = 60) -> str:
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.text


def parse_fred_csv(text: str, series: str) -> pd.DataFrame:
    """FRED CSV as published, with columns ``date`` and ``value`` (strings)."""
    df = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)
    if list(df.columns) != ["observation_date", series]:
        raise ValueError(f"unexpected FRED columns {list(df.columns)} for {series}")
    df = df.rename(columns={"observation_date": "date", series: "value"})
    pd.to_datetime(df["date"], format="%Y-%m-%d")  # rejects malformed dates
    pd.to_numeric(df["value"].replace("", None))  # rejects non-numeric values
    return df


class _TableParser(HTMLParser):
    """Collect the text of every table cell, row by row."""

    def __init__(self):
        super().__init__()
        self.rows, self._row, self._cell = [], None, None

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self._row = []
        elif tag in ("td", "th") and self._row is not None:
            self._cell = []

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self._cell is not None:
            self._row.append(" ".join("".join(self._cell).split()))
            self._cell = None
        elif tag == "tr" and self._row is not None:
            self.rows.append(self._row)
            self._row = None

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)


def parse_tcmb_repo(html: str) -> pd.DataFrame:
    """Rows of the TCMB one-week repo table, as published plus parsed date and rate."""
    parser = _TableParser()
    parser.feed(html)
    header = ["DATE", "Borrowing", "Lending"]
    if header not in parser.rows:
        raise ValueError("TCMB page has no DATE / Borrowing / Lending table")
    body = parser.rows[parser.rows.index(header) + 1 :]
    rows = [r for r in body if len(r) == 3 and r[0]]
    df = pd.DataFrame(rows, columns=["date_published", "borrowing", "lending"])
    df["date"] = pd.to_datetime(df["date_published"], format="%d.%m.%Y")
    df["rate_pct"] = pd.to_numeric(df["lending"].str.replace(",", ".", regex=False))
    if not df["date"].is_monotonic_increasing or df["date"].duplicated().any():
        raise ValueError("TCMB rate dates must be unique and sorted")
    return df


def save_rate_snapshot(
    df: pd.DataFrame,
    prefix: str,
    out_dir: Path = DEFAULT_RAW_DIR,
    downloaded_at: datetime | None = None,
    force: bool = False,
) -> tuple[Path, Path]:
    """Write the dated CSV and its manifest; return both paths."""
    downloaded_at = downloaded_at or datetime.now(timezone.utc)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{prefix}{downloaded_at:%Y-%m-%d}"
    csv_path = out_dir / f"{stem}.csv"
    manifest_path = out_dir / f"{stem}.manifest.json"
    if csv_path.exists() and not force:
        raise FileExistsError(f"{csv_path} already exists; pass force=True to replace it")

    # LF line endings on every OS keep the hash identical on Windows and Linux.
    df.to_csv(csv_path, index=False, date_format="%Y-%m-%d", lineterminator="\n")
    dates = pd.to_datetime(df["date"])
    manifest = {
        **{k: SOURCES[prefix][k] for k in ("source", "source_url", "attribution")},
        "download_date": downloaded_at.isoformat(timespec="seconds"),
        "row_count": len(df),
        "first_date": f"{dates.min():%Y-%m-%d}",
        "last_date": f"{dates.max():%Y-%m-%d}",
        "columns": SOURCES[prefix]["columns"],
        "sha256": _sha256(csv_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return csv_path, manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--force", action="store_true", help="overwrite today's snapshots")
    args = parser.parse_args()

    frames = {
        "usd_dtb3_fred_": parse_fred_csv(_get(SOURCES["usd_dtb3_fred_"]["source_url"]), "DTB3"),
        "try_policy_tcmb_": parse_tcmb_repo(_get(TCMB_REPO_URL)),
        "try_callmoney_fred_": parse_fred_csv(
            _get(SOURCES["try_callmoney_fred_"]["source_url"]), "IRSTCI01TRM156N"
        ),
    }
    for prefix, df in frames.items():
        csv_path, _ = save_rate_snapshot(df, prefix, args.out_dir, force=args.force)
        print(f"{csv_path.name}: {len(df)} rows, {df['date'].min()} to {df['date'].max()}")


if __name__ == "__main__":
    main()
