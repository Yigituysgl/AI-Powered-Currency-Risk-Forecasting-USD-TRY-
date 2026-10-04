"""Load verified interest-rate snapshots as annual rates in percent."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from usdtry_risk.data.fetch_ecb import DEFAULT_RAW_DIR

USD_TBILL = "usd_dtb3_fred_"
TRY_POLICY = "try_policy_tcmb_"
TRY_CALL_MONEY = "try_callmoney_fred_"


def load_rate_snapshot(prefix: str, raw_dir: Path = DEFAULT_RAW_DIR) -> pd.DataFrame:
    """Latest snapshot for ``prefix``, refused unless it matches its manifest."""
    snapshots = sorted(Path(raw_dir).glob(f"{prefix}*.csv"))
    if not snapshots:
        raise FileNotFoundError(
            f"no {prefix}* snapshot in {raw_dir}; run `python -m usdtry_risk.data.fetch_rates` first"
        )
    path = snapshots[-1]
    manifest = json.loads(path.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["sha256"]:
        raise ValueError(f"{path.name} does not match the hash in its manifest")
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    if len(df) != manifest["row_count"]:
        raise ValueError(f"{path.name} has {len(df)} rows, manifest says {manifest['row_count']}")
    return df


def rate_series(prefix: str, raw_dir: Path = DEFAULT_RAW_DIR) -> pd.Series:
    """Rate in percent per year, indexed by the date it applies from; missing days dropped."""
    df = load_rate_snapshot(prefix, raw_dir)
    column = "rate_pct" if prefix == TRY_POLICY else "value"
    values = pd.to_numeric(df[column].replace("", None))
    series = pd.Series(values.to_numpy(), index=pd.to_datetime(df["date"]), name=prefix.rstrip("_"))
    return series.dropna()
