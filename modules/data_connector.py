# data_connector.py
# -*- coding: utf-8 -*-
"""Unified interface for loading OHLCV data from CSV files or ClickHouse."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd

from .csv_data import load_ohlcv_csv
from .clickhouse import ClickHouseConnector

__all__ = ["DataConnector"]


class DataConnector:
    """Load OHLCV data from CSV or ClickHouse using a single API."""

    def __init__(self, clickhouse_params: Optional[Dict[str, Any]] = None) -> None:
        self._ch_params = clickhouse_params or {}
        self._ch: Optional[ClickHouseConnector] = None

    def _get_ch(self) -> ClickHouseConnector:
        if self._ch is None:
            self._ch = ClickHouseConnector(**self._ch_params)
        return self._ch

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
