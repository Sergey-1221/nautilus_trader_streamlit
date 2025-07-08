#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
clickhouse_instrument_provider_no_bus.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Instrument provider for Nautilus Trader that creates Instrument objects
from ClickHouse metadata. **Version without MessageBus**. Supports:
  • a generic ClickHouse connector (with extended diagnostics);
  • the currency_pair_from_db() factory;
  • the ClickHouseInstrumentProvider class (adapter port);
  • usage examples – 4 cases in ``__main__``.

Dependencies
------------
pip install clickhouse-driver pandas nautilus-trader python-dotenv
"""

from __future__ import annotations

import os
import re
import sys

from dotenv import load_dotenv
from datetime import datetime, timezone, timedelta
from types import ModuleType
from typing import Dict, Optional, Tuple

import pandas as pd
from clickhouse_driver import Client

load_dotenv()

# ── Nautilus Trader ─────────────────────────────────────────────── #
from nautilus_trader.model import InstrumentId, Symbol, Venue
from nautilus_trader.model.currencies import Currency
from nautilus_trader.model.instruments import CurrencyPair
# NB: the path to the InstrumentProvider base class changed after v1.200
from nautilus_trader.common.providers import InstrumentProvider
from nautilus_trader.cache.cache import Cache

# ────────────────── ClickHouse connection settings ─────────────── #
CH_HOST = os.getenv("CH_HOST")
CH_USER = os.getenv("CH_USER")
CH_PASSWORD = os.getenv("CH_PASSWORD")
CH_DATABASE = os.getenv("CH_DATABASE")

EXCHANGE_NAME_TO_ID: Dict[str, int] = {
    "BINANCE": 1,
    # add other exchanges if needed …
}

INTERVAL_STR_TO_CODE: Dict[str, int] = {
    "1s": 1, "1m": 2, "3m": 3, "5m": 4, "15m": 5, "30m": 6,
    "1h": 7, "2h": 8, "4h": 9, "6h": 10, "8h": 11, "12h": 12,
    "1d": 13, "3d": 14, "1w": 15, "1mo": 16,
}
CODE_TO_INTERVAL_STR = {v: k for k, v in INTERVAL_STR_TO_CODE.items()}

MKT_ENUM: Dict[str, int] = {"spot": 1, "usdm": 2, "coinm": 3}

# ─────────────────────────── Utilities ──────────────────────────── #
_SYMBOL_RE = re.compile(
    r"^(.*?)(USDT|BUSD|FDUSD|USDC|BTC|ETH|BNB|SOL|TRX|TRY|EUR|GBP|AUD|RUB|USD)$"
)


def parse_symbol(sym: str) -> Tuple[str, str]:
    """Split a Binance ticker into base and quote currencies.
    If the suffix is not found, split the string in half."""
    m = _SYMBOL_RE.match(sym)
    if m:
        return m.group(1), m.group(2)
    mid = len(sym) // 2
    return sym[:mid], sym[mid:]


def _get_currency(code: str, module: ModuleType) -> Currency:
    """Try to get an existing Currency from nautilus_trader.model.currencies.
    If the currency is missing, create it on-the-fly (name=iso=code, numeric=0)."""
    try:
        return getattr(module, code.upper())
    except AttributeError:
        return Currency(code.upper(), code.upper(), 0)


# ─────────────────────── ClickHouse Connector ───────────────────── #
class ClickHouseConnector:
    """Lightweight wrapper around ``clickhouse-driver`` with extended diagnostics."""

    def __init__(
        self,
        host: str = CH_HOST,
        user: str = CH_USER,
        password: str = CH_PASSWORD,
        database: str = CH_DATABASE,
    ):
        try:
            self.cli = Client(
                host=host,
                user=user,
                password=password,
                database=database,
            )
            self.cli.execute("SELECT 1")  # verify connection
        except Exception as exc:
            raise ConnectionError(
                f"Failed to connect to ClickHouse "
                f"(host={host}, db={database}): {exc}"
            ) from exc

# ────────────────── Public method: candles ─────────────────── #
    def candles(
        self,
        *,
        exchange: str,
        symbol: str,
        timeframe: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        mkt: str = "spot",
        debug: bool = False,
        auto_clip: bool = False,
    ) -> pd.DataFrame:
        """Return a DataFrame with candles; if the result set is empty,
        diagnose the range and optionally clip to the available range."""
        exchange = exchange.upper()
        if exchange not in EXCHANGE_NAME_TO_ID:
            raise ValueError(f"Unknown exchange: {exchange}")
        if timeframe not in INTERVAL_STR_TO_CODE:
            raise ValueError(f"Unsupported timeframe: {timeframe}")

        ex_id = EXCHANGE_NAME_TO_ID[exchange]
        interval_code = INTERVAL_STR_TO_CODE[timeframe]
        base, quote = parse_symbol(symbol)

        params = {
            "ex": ex_id,
            "b": base,
            "q": quote,
            "m": MKT_ENUM[mkt],
            "iv": interval_code,
        }
        conds = [
            "i.ex_id   = %(ex)s",
            "b.code    = %(b)s",
            "q.code    = %(q)s",
            "i.mkt     = %(m)s",
            "c.interval = %(iv)s",
        ]
        if start is not None:
            params["ts0"] = start
            conds.append("c.open_time >= %(ts0)s")
        if end is not None:
            params["ts1"] = end
            conds.append("c.open_time <= %(ts1)s")

        sql = f"""
        SELECT
            c.open_time,
            c.open,  c.high,  c.low,  c.close,
            c.volume, c.quote_vol, c.trades,
            c.taker_base, c.taker_quote
        FROM   {CH_DATABASE}.candles   AS c
               JOIN   {CH_DATABASE}.instrument AS i ON c.inst_id = i.id
               JOIN   {CH_DATABASE}.currency  AS b ON i.base   = b.id
               JOIN   {CH_DATABASE}.currency  AS q ON i.quote  = q.id
        WHERE  {' AND '.join(conds)}
        ORDER BY c.open_time
        """

        if debug:
            print("SQL  :", sql)
            print("PARAM:", params)

        rows = self.cli.execute(sql, params)
        if rows:
            df = pd.DataFrame(
                rows,
                columns=[
                    "timestamp", "open", "high", "low", "close",
                    "volume", "quote_vol", "trades",
                    "taker_base", "taker_quote",
                ],
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df.set_index("timestamp", inplace=True)
            num_cols = [
                "open",
                "high",
                "low",
                "close",
                "volume",
                "quote_vol",
                "trades",
                "taker_base",
                "taker_quote",
            ]
            df[num_cols] = df[num_cols].astype("float64")
            return df

        # ──────── no data → diagnose min/max ─────────── #
        diag_sql = f"""
        SELECT
            min(c.open_time) AS min_time,
            max(c.open_time) AS max_time
        FROM   {CH_DATABASE}.candles AS c
               JOIN   {CH_DATABASE}.instrument AS i ON c.inst_id = i.id
               JOIN   {CH_DATABASE}.currency  AS b ON i.base   = b.id
               JOIN   {CH_DATABASE}.currency  AS q ON i.quote  = q.id
        WHERE  i.ex_id = %(ex)s
          AND  b.code  = %(b)s
          AND  q.code  = %(q)s
          AND  i.mkt   = %(m)s
          AND  c.interval = %(iv)s
        """
        min_time, max_time = self.cli.execute(diag_sql, params)[0]

        if min_time is None:
            raise RuntimeError(
                f"No data for {symbol} on {exchange} ({mkt}) for interval "
                f"'{timeframe}' (code {interval_code})."
            )

        # Data exists, but the range does not match
        if auto_clip:
            clipped_start = max(min_time, start) if start else min_time
            clipped_end = min(max_time, end) if end else max_time
            if debug:
                print(f"⤵️  Auto-clipping: {clipped_start} … {clipped_end}")
            return self.candles(
                exchange=exchange,
                symbol=symbol,
                timeframe=timeframe,
                start=clipped_start,
                end=clipped_end,
                mkt=mkt,
                debug=debug,
                auto_clip=False,
            )

        raise RuntimeError(
            f"No rows for {symbol} {timeframe} in range "
            f"{start} … {end}.\n"
            f"In the DB this interval covers:\n"
            f"    {min_time} … {max_time}\n"
            f"Check the date or use `auto_clip=True`."
        )


# ─────────────── Create CurrencyPair from DB ───────────────────── #

def currency_pair_from_db(
    ch: ClickHouseConnector,
    *,
    exchange: str,
    symbol: str,
    mkt: str = "spot",
) -> CurrencyPair:
    """Read pair specifications from ClickHouse and return a ``CurrencyPair``."""
    exchange_u = exchange.upper()
    base, quote = parse_symbol(symbol)
    row = ch.cli.execute(
        """
        SELECT
            i.price_digits,
            i.qty_digits,
            b.code AS base_code,
            q.code AS quote_code
        FROM   crypto.instrument AS i
               JOIN crypto.currency AS b ON i.base  = b.id
               JOIN crypto.currency AS q ON i.quote = q.id
        WHERE  i.ex_id = %(ex)s
          AND  b.code  = %(b)s
          AND  q.code  = %(q)s
          AND  i.mkt   = %(m)s
        LIMIT  1
        """,
        {
            "ex": EXCHANGE_NAME_TO_ID[exchange_u],
            "b": base,
            "q": quote,
            "m": MKT_ENUM[mkt],
        },
    )

    if not row:
        raise RuntimeError(f"Instrument {symbol} {mkt} on {exchange_u} not found.")

    price_digits, qty_digits, base_code, quote_code = row[0]

    currencies_mod: ModuleType = sys.modules["nautilus_trader.model.currencies"]
    base_cur = _get_currency(base_code, currencies_mod)
    quote_cur = _get_currency(quote_code, currencies_mod)

    now_ns = int(datetime.now(timezone.utc).timestamp() * 1e9)
    return CurrencyPair(
        instrument_id=InstrumentId(symbol.upper(), Venue(exchange_u)),
        symbol=Symbol(symbol.upper()),
        base_currency=base_cur,
        quote_currency=quote_cur,
        price_precision=price_digits,          # int, see releases >= 1.200
        size_precision=qty_digits,             # int
        ts_init=now_ns,
        ts_event=now_ns,
    )


# ──────────────── ClickHouse InstrumentProvider ─────────────────── #
class ClickHouseInstrumentProvider(InstrumentProvider):
    """Nautilus adapter provider for loading instruments from ClickHouse
    (without MessageBus support)."""

    def __init__(
        self,
        connector: ClickHouseConnector,
        cache: Cache | None = None,
    ):
        # super().__init__ expects bus and cache; we omit bus (None)
        super().__init__(cache=cache)
        self._ch = connector

    # Simple synchronous implementation; an async wrapper could be added
    def load_all(self) -> None:
        raise NotImplementedError("load_all() is not implemented in this example.")

    # mini-API for single requests
    def currency_pair_from_db(
        self,
        *,
        exchange: str,
        symbol: str,
        mkt: str = "spot",
    ) -> CurrencyPair:
        pair = currency_pair_from_db(
            self._ch, exchange=exchange, symbol=symbol, mkt=mkt
        )
        # publish only to Cache if it exists; MessageBus is omitted
        if self._cache is not None:
            self._cache.add_instrument(pair)
        return pair


# ─────────────────────────── Examples ──────────────────────────── #
if __name__ == "__main__":
    ch = ClickHouseConnector()
    provider = ClickHouseInstrumentProvider(ch)  # demo without Bus/Cache

    # 1) BNB/USDT: hourly candles
    print("\n— BNB/USDT 1h —")
    start = datetime.now(timezone.utc) - timedelta(days=1)
    end = datetime.now(timezone.utc)
    df = ch.candles(
        exchange="BINANCE",
        symbol="BNBUSDT",
        timeframe="1h",
        start=start,
        end=end,
        auto_clip=True,
    )
    print(df.tail())
    print(f"⏱  rows received: {len(df)}")

    # 2) ETH/USDT: one-minute candles
    print("\n— ETH/USDT 1m —")
    df_eth = ch.candles(
        exchange="BINANCE",
        symbol="ETHUSDT",
        timeframe="1m",
        start=start,
        end=end,
        auto_clip=True,
    )
    print(df_eth.tail())
    print(f"⏱  rows received: {len(df_eth)}")

    # 3) Range with no data (auto_clip=False)
    print("\n— Empty range (expecting error) —")
    try:
        ch.candles(
            exchange="BINANCE",
            symbol="BTCUSDT",
            timeframe="1m",
            start=datetime(2015, 1, 1, tzinfo=timezone.utc),
            end=datetime(2015, 1, 2, tzinfo=timezone.utc),
            auto_clip=False,
        )
    except RuntimeError as err:
        print("‼", err)

    # 4) CurrencyPair via provider
    print("\n— CurrencyPair from provider —")
    pair = provider.currency_pair_from_db(
        exchange="BINANCE",
        symbol="BNBUSDT",
        mkt="spot",
    )
    print(
        f"ID: {pair.instrument_id}, "
        f"price_precision={pair.price_precision}, "
        f"size_precision={pair.size_precision}"
    )
