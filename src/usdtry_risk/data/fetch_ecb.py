"""Download ECB euro reference rates and save a dated USD/TRY snapshot.

The ECB quotes every currency per 1 EUR, so USD/TRY is derived here as
EUR/TRY / EUR/USD. The ECB values are stored exactly as published (as
strings) next to the derived rate, because the ECB reuse policy requires
that its statistics are not modified.

Usage:
    python -m usdtry_risk.data.fetch_ecb [--out-dir data/raw] [--force]
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

ECB_URL = "https://data-api.ecb.europa.eu/service/data/EXR/D.TRY+USD.EUR.SP00.A"
ATTRIBUTION = "Source: ECB statistics."
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RAW_DIR = REPO_ROOT / "data" / "raw"
SNAPSHOT_PREFIX = "usdtry_ecb_"


def download_ecb_csv(url: str = ECB_URL, timeout: float = 60) -> str:
    """Return the raw CSV text from the ECB data API."""
    # Without an explicit Accept header the API may wrap the CSV in HTML.
    resp = requests.get(url, headers={"Accept": "text/csv"}, timeout=timeout)
    resp.raise_for_status()
    if not resp.text.startswith("KEY,"):
        raise ValueError(f"ECB API did not return CSV, got: {resp.text[:80]!r}")
    return resp.text


def build_usdtry(ecb_csv: str) -> pd.DataFrame:
    """Turn the long ECB CSV into one row per date with the derived USD/TRY.

    Only dates on which the ECB published both TRY and USD are kept.
    """
    raw = pd.read_csv(
        io.StringIO(ecb_csv),
        usecols=["CURRENCY", "CURRENCY_DENOM", "TIME_PERIOD", "OBS_VALUE"],
        dtype=str,
    ).dropna(subset=["OBS_VALUE"])

    if set(raw["CURRENCY_DENOM"]) != {"EUR"}:
        raise ValueError("expected all ECB rates to be quoted per EUR")
    if set(raw["CURRENCY"]) != {"TRY", "USD"}:
        raise ValueError(f"expected TRY and USD series, got {sorted(set(raw['CURRENCY']))}")
    if raw.duplicated(["CURRENCY", "TIME_PERIOD"]).any():
        raise ValueError("duplicate (currency, date) observations in ECB data")

    wide = (
        raw.pivot(index="TIME_PERIOD", columns="CURRENCY", values="OBS_VALUE")
        .dropna()
        .sort_index()
    )
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(wide.index, format="%Y-%m-%d"),
            "eur_try": wide["TRY"].to_numpy(),
            "eur_usd": wide["USD"].to_numpy(),
        }
    )
    usd_try = out["eur_try"].astype(float) / out["eur_usd"].astype(float)
    if not (usd_try > 0).all():
        raise ValueError("non-positive exchange rate in ECB data")
    out["usd_try"] = usd_try.round(6)
    return out.reset_index(drop=True)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_snapshot(
    df: pd.DataFrame,
    out_dir: Path = DEFAULT_RAW_DIR,
    source_url: str = ECB_URL,
    downloaded_at: datetime | None = None,
    force: bool = False,
) -> tuple[Path, Path]:
    """Write the dated CSV and its manifest; return both paths."""
    downloaded_at = downloaded_at or datetime.now(timezone.utc)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{SNAPSHOT_PREFIX}{downloaded_at:%Y-%m-%d}"
    csv_path = out_dir / f"{stem}.csv"
    manifest_path = out_dir / f"{stem}.manifest.json"
    if csv_path.exists() and not force:
        raise FileExistsError(f"{csv_path} already exists; pass force=True to replace it")

    # LF line endings on every OS keep the hash identical on Windows and Linux.
    df.to_csv(csv_path, index=False, date_format="%Y-%m-%d", lineterminator="\n")
    manifest = {
        "source": "European Central Bank, euro foreign exchange reference rates (EXR)",
        "source_url": source_url,
        "attribution": ATTRIBUTION,
        "download_date": downloaded_at.isoformat(timespec="seconds"),
        "row_count": len(df),
        "first_date": df["date"].min().strftime("%Y-%m-%d"),
        "last_date": df["date"].max().strftime("%Y-%m-%d"),
        "columns": {
            "date": "ECB reference rate date (TARGET business days)",
            "eur_try": "TRY per 1 EUR, as published by the ECB",
            "eur_usd": "USD per 1 EUR, as published by the ECB",
            "usd_try": "TRY per 1 USD, derived by this project as eur_try / eur_usd (not an ECB statistic)",
        },
        "sha256": _sha256(csv_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return csv_path, manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--force", action="store_true", help="overwrite today's snapshot")
    args = parser.parse_args()

    df = build_usdtry(download_ecb_csv())
    csv_path, manifest_path = save_snapshot(df, args.out_dir, force=args.force)
    print(f"Saved {len(df)} rows ({df['date'].min():%Y-%m-%d} to {df['date'].max():%Y-%m-%d})")
    print(f"  {csv_path}\n  {manifest_path}")


if __name__ == "__main__":
    main()
