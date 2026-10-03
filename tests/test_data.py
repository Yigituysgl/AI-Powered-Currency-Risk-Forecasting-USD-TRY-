import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from usdtry_risk.data import fetch_ecb
from usdtry_risk.data.fetch_ecb import build_usdtry, download_ecb_csv, save_snapshot
from usdtry_risk.data.prepare import latest_snapshot, load_snapshot, log_returns

HEADER = "KEY,FREQ,CURRENCY,CURRENCY_DENOM,EXR_TYPE,EXR_SUFFIX,TIME_PERIOD,OBS_VALUE,OBS_STATUS"
FAKE_ECB_CSV = "\n".join(
    [
        HEADER,
        "EXR.D.TRY.EUR.SP00.A,D,TRY,EUR,SP00,A,2024-01-03,32.5000,A",
        "EXR.D.TRY.EUR.SP00.A,D,TRY,EUR,SP00,A,2024-01-02,32.0000,A",
        "EXR.D.TRY.EUR.SP00.A,D,TRY,EUR,SP00,A,2024-01-04,33.0000,A",
        "EXR.D.USD.EUR.SP00.A,D,USD,EUR,SP00,A,2024-01-02,1.1000,A",
        "EXR.D.USD.EUR.SP00.A,D,USD,EUR,SP00,A,2024-01-03,1.0000,A",
        # No USD rate on 2024-01-04, so that date must be dropped.
    ]
)


def test_build_usdtry_derives_cross_rate_and_keeps_ecb_values():
    df = build_usdtry(FAKE_ECB_CSV)
    assert list(df.columns) == ["date", "eur_try", "eur_usd", "usd_try"]
    assert list(df["date"].dt.strftime("%Y-%m-%d")) == ["2024-01-02", "2024-01-03"]
    assert list(df["eur_usd"]) == ["1.1000", "1.0000"]  # published strings, unmodified
    assert df["usd_try"].tolist() == pytest.approx([32.0 / 1.1, 32.5])


def test_build_usdtry_rejects_duplicates():
    dup = FAKE_ECB_CSV + "\nEXR.D.USD.EUR.SP00.A,D,USD,EUR,SP00,A,2024-01-03,1.0000,A"
    with pytest.raises(ValueError, match="duplicate"):
        build_usdtry(dup)


def test_download_rejects_non_csv(monkeypatch):
    class FakeResponse:
        text = "<html>maintenance</html>"

        def raise_for_status(self):
            pass

    monkeypatch.setattr(fetch_ecb.requests, "get", lambda *a, **k: FakeResponse())
    with pytest.raises(ValueError, match="did not return CSV"):
        download_ecb_csv()


def test_snapshot_roundtrip_and_manifest(tmp_path):
    df = build_usdtry(FAKE_ECB_CSV)
    when = datetime(2024, 1, 5, tzinfo=timezone.utc)
    csv_path, manifest_path = save_snapshot(df, tmp_path, downloaded_at=when)

    assert csv_path.name == "usdtry_ecb_2024-01-05.csv"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["row_count"] == 2
    assert manifest["source_url"] == fetch_ecb.ECB_URL
    assert manifest["attribution"] == "Source: ECB statistics."
    assert manifest["download_date"].startswith("2024-01-05")

    prices = load_snapshot(csv_path)
    assert prices.tolist() == pytest.approx(df["usd_try"].tolist())
    assert latest_snapshot(tmp_path) == csv_path

    with pytest.raises(FileExistsError):
        save_snapshot(df, tmp_path, downloaded_at=when)


def test_load_snapshot_detects_tampering(tmp_path):
    csv_path, _ = save_snapshot(build_usdtry(FAKE_ECB_CSV), tmp_path)
    csv_path.write_text(csv_path.read_text().replace("32.5", "99.9"))
    with pytest.raises(ValueError, match="hash"):
        load_snapshot(csv_path)


def test_log_returns():
    prices = pd.Series([10.0, 11.0, 9.9], index=pd.date_range("2024-01-01", periods=3))
    r = log_returns(prices)
    assert len(r) == 2
    assert r.tolist() == pytest.approx([np.log(1.1), np.log(0.9)])


def test_committed_snapshot_is_valid():
    try:
        path = latest_snapshot()
    except FileNotFoundError:
        pytest.skip("no snapshot downloaded")
    prices = load_snapshot(path)  # verifies hash and row count
    assert prices.notna().all() and (prices > 0).all()
    assert prices.index.is_monotonic_increasing and not prices.index.has_duplicates
    assert prices.index[0] <= pd.Timestamp("2005-01-03")
    # ECB publishes on TARGET business days only; no gap should exceed a long holiday.
    assert prices.index.to_series().diff().dt.days.max() <= 7
