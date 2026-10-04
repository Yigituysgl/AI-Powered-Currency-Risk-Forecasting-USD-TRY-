import json
from datetime import datetime, timezone

import pandas as pd
import pytest

from usdtry_risk.data.fetch_rates import parse_fred_csv, parse_tcmb_repo, save_rate_snapshot
from usdtry_risk.data.rates import (
    TRY_CALL_MONEY,
    TRY_POLICY,
    USD_TBILL,
    load_rate_snapshot,
    rate_series,
)

FRED_CSV = "observation_date,DTB3\n2024-01-01,\n2024-01-02,5.25\n2024-01-03,5.20\n"

# Shaped like the TCMB page: header row, an empty spacer row, a comma decimal.
TCMB_HTML = """
<html><body><table>
<tr><th>DATE</th><th>Borrowing</th><th>Lending</th></tr>
<tr><td>&nbsp;</td><td>&nbsp;</td><td>&nbsp;</td></tr>
<tr><td>20.05.2010</td><td>-</td><td>7.00</td></tr>
<tr><td>17.04.2013</td><td>-</td><td>5,00</td></tr>
<tr><td><p>23.01.2026</p></td><td>-</td><td>37.00</td></tr>
</table></body></html>
"""


def test_parse_fred_keeps_published_strings():
    df = parse_fred_csv(FRED_CSV, "DTB3")
    assert list(df.columns) == ["date", "value"]
    assert df["value"].tolist() == ["", "5.25", "5.20"]


def test_parse_fred_rejects_other_series():
    with pytest.raises(ValueError, match="unexpected FRED columns"):
        parse_fred_csv(FRED_CSV, "DTB6")


def test_parse_tcmb_reads_dates_and_comma_decimals():
    df = parse_tcmb_repo(TCMB_HTML)
    assert df["date_published"].tolist() == ["20.05.2010", "17.04.2013", "23.01.2026"]
    assert df["lending"].tolist() == ["7.00", "5,00", "37.00"]
    assert df["rate_pct"].tolist() == [7.0, 5.0, 37.0]
    assert df["date"].iloc[1] == pd.Timestamp("2013-04-17")


def test_parse_tcmb_rejects_a_page_without_the_table():
    with pytest.raises(ValueError, match="no DATE"):
        parse_tcmb_repo("<html><table><tr><td>x</td></tr></table></html>")


def test_rate_snapshot_roundtrip(tmp_path):
    when = datetime(2024, 1, 5, tzinfo=timezone.utc)
    csv_path, manifest_path = save_rate_snapshot(
        parse_tcmb_repo(TCMB_HTML), TRY_POLICY, tmp_path, downloaded_at=when
    )
    manifest = json.loads(manifest_path.read_text())
    assert csv_path.name == "try_policy_tcmb_2024-01-05.csv"
    assert manifest["row_count"] == 3 and manifest["last_date"] == "2026-01-23"
    assert manifest["attribution"].startswith("Source: Central Bank")
    rates = rate_series(TRY_POLICY, tmp_path)
    assert rates.tolist() == [7.0, 5.0, 37.0]

    save_rate_snapshot(parse_fred_csv(FRED_CSV, "DTB3"), USD_TBILL, tmp_path, downloaded_at=when)
    assert rate_series(USD_TBILL, tmp_path).tolist() == [5.25, 5.20]


def test_rate_snapshot_detects_tampering(tmp_path):
    csv_path, _ = save_rate_snapshot(parse_fred_csv(FRED_CSV, "DTB3"), USD_TBILL, tmp_path)
    csv_path.write_text(csv_path.read_text().replace("5.25", "1.00"))
    with pytest.raises(ValueError, match="hash"):
        load_rate_snapshot(USD_TBILL, tmp_path)


@pytest.mark.parametrize("prefix", [USD_TBILL, TRY_POLICY, TRY_CALL_MONEY])
def test_committed_rate_snapshots_are_valid(prefix):
    try:
        rates = rate_series(prefix)
    except FileNotFoundError:
        pytest.skip("no rate snapshot downloaded")
    assert rates.index.is_monotonic_increasing and not rates.index.has_duplicates
    # T-bill yields dipped slightly below zero in 2015 and 2020.
    assert (rates > -1).all()
