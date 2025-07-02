# data_connector.py
# -*- coding: utf-8 -*-
"""Unified interface for loading OHLCV data from CSV files or ClickHouse."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import re

import pandas as pd

from .csv_data import load_ohlcv_csv
from .clickhouse import ClickHouseConnector

__all__ = ["DataConnector"]


CSV_NAME_RE = re.compile(r"(?P<ex>[A-Z]+)_(?P<sym>[A-Z]+),\s*(?P<tf>\d+)\.csv")


class DataConnector:
    """Load OHLCV data and expose available sources."""

    def __init__(
        self,
        csv_dir: str | Path = ".",
        clickhouse_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._csv_dir = Path(csv_dir)
        self._csv_info: Optional[List[Dict[str, str]]] = None
        self._ch_params = clickhouse_params or {}
        self._ch: Optional[ClickHouseConnector] = None

    # ------------------------------------------------------------------
    def _scan_csv(self) -> List[Dict[str, str]]:
        """Return cached info about CSV files."""
        if self._csv_info is None:
            info: List[Dict[str, str]] = []
            for p in self._csv_dir.glob("*.csv"):
                m = CSV_NAME_RE.match(p.name)
                if m:
                    tf = f"{m.group('tf')}min"
                    info.append(
                        {
                            "path": str(p),
                            "exchange": m.group("ex"),
                            "symbol": m.group("sym"),
                            "timeframe": tf,
                        }
                    )
            self._csv_info = info
        return self._csv_info

    def _get_ch(self) -> ClickHouseConnector:
        if self._ch is None:
            self._ch = ClickHouseConnector(**self._ch_params)
        return self._ch

    # ------------------------------------------------------------------
    def get_exchanges(self, source: str) -> List[str]:
        source_u = source.upper()
        if source_u == "CSV":
            return sorted({i["exchange"] for i in self._scan_csv()})
        if source_u == "CLICKHOUSE":
            from .clickhouse import EXCHANGE_NAME_TO_ID

            return sorted(EXCHANGE_NAME_TO_ID.keys())
        raise ValueError(f"Unknown source: {source}")

    def get_symbols(self, source: str, exchange: str | None = None) -> List[str]:
        source_u = source.upper()
        if source_u == "CSV":
            return sorted(
                {
                    i["symbol"]
                    for i in self._scan_csv()
                    if exchange is None or i["exchange"] == exchange
                }
            )
        if source_u == "CLICKHOUSE":
            return []  # symbol list not implemented
        raise ValueError(f"Unknown source: {source}")

    def get_timeframes(self, source: str) -> List[str]:
        source_u = source.upper()
        if source_u == "CSV":
            return sorted({i["timeframe"] for i in self._scan_csv()})
        if source_u == "CLICKHOUSE":
            from .clickhouse import INTERVAL_STR_TO_CODE

            def _conv(tf: str) -> str:
                if tf.endswith("m"):
                    return tf[:-1] + "min"
                return tf

            return sorted(_conv(tf) for tf in INTERVAL_STR_TO_CODE)
        raise ValueError(f"Unknown source: {source}")

    def get_csv_path(self, exchange: str, symbol: str, timeframe: str) -> str:
        tf_norm = timeframe.lower().replace("min", "")
        for i in self._scan_csv():
            if (
                i["exchange"].upper() == exchange.upper()
                and i["symbol"].upper() == symbol.upper()
                and i["timeframe"].lower().replace("min", "") == tf_norm
            ):
                return i["path"]
        raise FileNotFoundError(f"CSV not found for {exchange} {symbol} {timeframe}")

    def load(
        self,
        source: str,
        spec: Any,
        *,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Return a DataFrame with OHLCV data for the given source."""
        source_u = source.upper()
        if source_u == "CSV":
            df = load_ohlcv_csv(spec)
            if start or end:
                start_dt = start or df.index[0]
                end_dt = end or df.index[-1]
                df = df.loc[start_dt:end_dt]
            return df

        if source_u == "CLICKHOUSE":
            ch = self._get_ch()
            if not isinstance(spec, dict):
                raise TypeError("spec must be dict for ClickHouse")
            return ch.candles(**spec, auto_clip=True)

        raise ValueError(f"Unknown data source: {source}")
