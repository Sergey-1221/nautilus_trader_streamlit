# csv_data.py
# -*- coding: utf-8 -*-
"""
csv_data — load OHLC(V) data from CSV into a pandas.DataFrame.

• Accepts any CSV file with candles as long as it has columns
  timestamp / time / date and open, high, low, close (volume is optional).
• Automatically determines the time units (s / ms / µs / ns).
• Returns a sorted DataFrame with a DatetimeIndex (UTC).

Test run
--------
Simply execute::

    python csv_data.py

By default the file "BINANCE_BTCUSD, 15.csv" will be loaded.
If you need another one, change the `CSV_DEFAULT_PATH` constant below.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import pandas as pd

__all__: List[str] = ["load_ohlcv_csv"]

_logger = logging.getLogger(__name__)

# ────────────────────────────── default CSV path
CSV_DEFAULT_PATH = "BINANCE_BTCUSD, 15.csv"
# ─────────────────────────────────────────────────────


def load_ohlcv_csv(csv_path: str) -> pd.DataFrame:
    """
    Read CSV and return a DataFrame with a DatetimeIndex (UTC).

    Parameters
    ----------
    csv_path : str
        Path to the CSV file.

    Returns
    -------
    pandas.DataFrame
        Columns: open, high, low, close, (volume).
        Index: DatetimeIndex (UTC).
    """
    csv_file = Path(csv_path)
    if not csv_file.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_file, decimal=".")

    # ── case-insensitive matching of column names
    col_map = {c.lower(): c for c in df.columns}

    ts_col = next((col_map.get(k) for k in ("timestamp", "time", "date") if k in col_map), None)
    if ts_col is None:
        raise ValueError("CSV must contain one of: timestamp, time, or date columns")

    required = ["open", "high", "low", "close"]
    missing = [c for c in required if c not in col_map]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    target_cols = [ts_col] + [col_map[c] for c in required]
    has_vol = "volume" in col_map
    if has_vol:
        target_cols.append(col_map["volume"])

    # ── normalized names
    df = df[target_cols]
    df.columns = ["timestamp"] + required + (["volume"] if has_vol else [])

    # ── data types
    df[required] = df[required].astype("float64")
    if has_vol:
        df["volume"] = df["volume"].astype("float64")

    # ── timestamp → datetime UTC
    ts = df["timestamp"]
    if pd.api.types.is_numeric_dtype(ts):
        max_ts = ts.max()
        unit = (
            "ns" if max_ts > 2e18 else
            "us" if max_ts > 2e15 else
            "ms" if max_ts > 2e12 else
            "s"
        )
        df["timestamp"] = pd.to_datetime(ts, unit=unit, utc=True)
    else:
        df["timestamp"] = pd.to_datetime(ts, utc=True, errors="coerce")

    df.dropna(subset=["timestamp"], inplace=True)
    df.set_index("timestamp", inplace=True)
    df.sort_index(inplace=True)

    _logger.debug("Loaded %d rows from %s", len(df), csv_file.name)
    return df


# ──────────────────────────────────────────────────────────────────────
# Mini test when running "python csv_data.py"
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    path = CSV_DEFAULT_PATH
    _logger.info("Loading CSV: %s", path)

    try:
        df_test = load_ohlcv_csv(path)
    except Exception as exc:
        _logger.error("❌ %s", exc)
        raise SystemExit(2) from exc

    _logger.info("✅ Loaded %d rows", len(df_test))
    _logger.info("First 5 rows:\n%s", df_test.head())
