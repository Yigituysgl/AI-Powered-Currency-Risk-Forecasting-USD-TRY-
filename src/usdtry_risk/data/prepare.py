"""Load a verified USD/TRY snapshot and compute daily log returns."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from usdtry_risk.data.fetch_ecb import DEFAULT_RAW_DIR, SNAPSHOT_PREFIX


def latest_snapshot(raw_dir: Path = DEFAULT_RAW_DIR) -> Path:
    """Return the most recent snapshot CSV (file names sort by date)."""
    snapshots = sorted(Path(raw_dir).glob(f"{SNAPSHOT_PREFIX}*.csv"))
    if not snapshots:
        raise FileNotFoundError(
            f"no snapshot in {raw_dir}; run `python -m usdtry_risk.data.fetch_ecb` first"
        )
    return snapshots[-1]


def load_snapshot(path: Path | None = None, verify: bool = True) -> pd.Series:
    """Load USD/TRY prices indexed by date.

    With ``verify`` the file must match the SHA-256 hash and row count in its
    manifest, so every result can be traced to an unchanged snapshot.
    """
    path = Path(path) if path is not None else latest_snapshot()
    if verify:
        manifest = json.loads(path.with_suffix(".manifest.json").read_text(encoding="utf-8"))
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["sha256"]:
            raise ValueError(f"{path.name} does not match the hash in its manifest")

    df = pd.read_csv(path, parse_dates=["date"])
    if verify and len(df) != manifest["row_count"]:
        raise ValueError(f"{path.name} has {len(df)} rows, manifest says {manifest['row_count']}")
    prices = df.set_index("date")["usd_try"].rename("usd_try")
    if not prices.index.is_monotonic_increasing or prices.index.has_duplicates:
        raise ValueError("snapshot dates must be unique and sorted")
    return prices


def log_returns(prices: pd.Series) -> pd.Series:
    """Daily log returns ln(P_t / P_{t-1}); the first date is dropped.

    Returns run between consecutive ECB business days, so a Monday return
    covers the weekend.
    """
    return np.log(prices).diff().dropna().rename("log_return")


def largest_moves(prices: pd.Series, n: int = 10) -> pd.DataFrame:
    """The ``n`` largest one-day moves by size, in date order, as simple % changes."""
    returns = log_returns(prices)
    dates = returns.abs().nlargest(n).index.sort_values()
    return pd.DataFrame(
        {"usd_try": prices[dates], "move_pct": 100 * np.expm1(returns[dates])},
        index=dates.rename("date"),
    )
