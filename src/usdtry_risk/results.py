"""Save and load result tables so the dashboard and figures never recompute them.

Each table is a CSV with a ``.meta.json`` beside it holding the table's
``attrs`` (sample size, date range) and the snapshot it was computed from.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from usdtry_risk.data.prepare import latest_snapshot

RESULTS_DIR = Path(__file__).resolve().parents[2] / "reports" / "results"


def _jsonable(value):
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if hasattr(value, "item"):  # numpy scalars
        return value.item()
    return value


def save_table(df: pd.DataFrame, name: str, results_dir: Path = RESULTS_DIR) -> Path:
    """Write ``df`` to ``<name>.csv`` and its attrs to ``<name>.meta.json``."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / f"{name}.csv"
    df.to_csv(csv_path, lineterminator="\n", date_format="%Y-%m-%d")
    meta = {k: _jsonable(v) for k, v in df.attrs.items()}
    meta["snapshot"] = latest_snapshot().name
    (results_dir / f"{name}.meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return csv_path


def load_table(name: str, results_dir: Path = RESULTS_DIR, index_cols: int = 1) -> pd.DataFrame:
    """Read a saved table; its metadata is restored to ``attrs``."""
    results_dir = Path(results_dir)
    df = pd.read_csv(results_dir / f"{name}.csv", index_col=list(range(index_cols)))
    df.attrs.update(json.loads((results_dir / f"{name}.meta.json").read_text(encoding="utf-8")))
    return df
