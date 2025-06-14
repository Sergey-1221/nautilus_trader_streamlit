#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
clickhouse_instrument_provider_no_bus.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Инструмент‑провайдер Nautilus Trader, который создаёт объекты Instrument
на основе метаданных из ClickHouse. **Версия без MessageBus**. Поддерживает:
  • общий коннектор ClickHouse (+ расширенная диагностика);
  • фабрику currency_pair_from_db();
  • класс ClickHouseInstrumentProvider (порт адаптера);
  • примеры использования – 4 кейса в __main__.

Зависимости
-----------
pip install clickhouse-driver pandas nautilus-trader python-dotenv
"""

from __future__ import annotations

import os
import re
import sys
from datetime import datetime, timezone, timedelta
from types import ModuleType
from typing import Dict, Optional, Tuple

import pandas as pd
from clickhouse_driver import Client

# ── Nautilus Trader ─────────────────────────────────────────────── #
from nautilus_trader.model import InstrumentId, Symbol, Venue
from nautilus_trader.model.currencies import Currency
from nautilus_trader.model.instruments import CurrencyPair
# NB: путь к базовому классу InstrumentProvider поменялся после v1.200
from nautilus_trader.common.providers import InstrumentProvider
from nautilus_trader.cache.cache import Cache

# ────────────────── ClickHouse connection settings ─────────────── #
CH_HOST = os.getenv("CH_HOST")
CH_USER = os.getenv("CH_USER")
CH_PASSWORD = os.getenv("CH_PASSWORD")
CH_DATABASE = os.getenv("CH_DATABASE")

EXCHANGE_NAME_TO_ID: Dict[str, int] = {
    "BINANCE": 1,
    # добавьте другие биржи при необходимости …
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
    """Разделяет тикер Binance на базовую и котируемую валюты.
    Если суффикс не найден – делит посередине."""
    m = _SYMBOL_RE.match(sym)
    if m:
        return m.group(1), m.group(2)
    mid = len(sym) // 2
    return sym[:mid], sym[mid:]


def _get_currency(code: str, module: ModuleType) -> Currency:
    """Пытаемся взять готовый Currency из nautilus_trader.model.currencies.
    Если валюты нет — создаём on‑the‑fly (name=iso=code, numeric=0)."""
    try:
        return getattr(module, code.upper())
    except AttributeError:
        return Currency(code.upper(), code.upper(), 0)


# ─────────────────────── ClickHouse Connector ───────────────────── #
class ClickHouseConnector:
    """Лёгкая обёртка вокруг clickhouse-driver с расширенной диагностикой."""

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
            self.cli.execute("SELECT 1")  # проверяем соединение
        except Exception as exc:
            raise ConnectionError(
                f"Не удалось подключиться к ClickHouse "
                f"(host={host}, db={database}): {exc}"
            ) from exc

    # ────────────────── Публичный метод: свечи ─────────────────── #
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
        """Возвращает DataFrame со свечами; если выборка пуста —
        диагностирует диапазон и (опц.) подрезает к available range."""
        exchange = exchange.upper()
        if exchange not in EXCHANGE_NAME_TO_ID:
            raise ValueError(f"Неизвестная биржа: {exchange}")
        if timeframe not in INTERVAL_STR_TO_CODE:
            raise ValueError(f"Неподдерживаемый таймфрейм: {timeframe}")

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
            return df

        # ──────── данных нет → диагностируем min/max ─────────── #
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
                f"Для {symbol} на {exchange} ({mkt}) нет данных интервала "
                f"'{timeframe}' вовсе (код {interval_code})."
            )

        # Данные есть, но диапазон не совпал
        if auto_clip:
            clipped_start = max(min_time, start) if start else min_time
            clipped_end = min(max_time, end) if end else max_time
            if debug:
                print(f"⤵️  Авто-обрезка: {clipped_start} … {clipped_end}")
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
            f"Нет строк для {symbol} {timeframe} в диапазоне "
            f"{start} … {end}.\n"
            f"В БД этот интервал закрывает:\n"
            f"    {min_time} … {max_time}\n"
            f"Проверьте дату либо используйте `auto_clip=True`."
        )


# ─────────────── Создание CurrencyPair из БД ───────────────────── #

def currency_pair_from_db(
    ch: ClickHouseConnector,
    *,
    exchange: str,
    symbol: str,
    mkt: str = "spot",
) -> CurrencyPair:
    """Читает спецификацию пары из ClickHouse и возвращает CurrencyPair."""
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
        raise RuntimeError(f"Инструмент {symbol} {mkt} на {exchange_u} не найден.")

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
        price_precision=price_digits,          # int, см. релизы >= 1.200
        size_precision=qty_digits,             # int
        ts_init=now_ns,
        ts_event=now_ns,
    )


# ──────────────── ClickHouse InstrumentProvider ─────────────────── #
class ClickHouseInstrumentProvider(InstrumentProvider):
    """Адаптер‑провайдер Nautilus для загрузки инструментов из ClickHouse
    (без поддержки MessageBus)."""

    def __init__(
        self,
        connector: ClickHouseConnector,
        cache: Cache | None = None,
    ):
        # super().__init__ принимает bus и cache; bus опускаем = None
        super().__init__(cache=cache)
        self._ch = connector

    # Простейшая синхронная реализация; можно добавить async‑wrapper
    def load_all(self) -> None:
        raise NotImplementedError("load_all() не реализован в примере.")

    # мини‑API для единичных запросов
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
        # публикуем лишь в Cache, если он существует; MessageBus убран
        if self._cache is not None:
            self._cache.add_instrument(pair)
        return pair


# ─────────────────────────── Примеры ──────────────────────────── #
if __name__ == "__main__":
    ch = ClickHouseConnector()
    provider = ClickHouseInstrumentProvider(ch)  # демо без Bus/Cache

    # 1) BNB/USDT: часовые свечи
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
    print(f"⏱  получено строк: {len(df)}")

    # 2) ETH/USDT: минутки
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
    print(f"⏱  получено строк: {len(df_eth)}")

    # 3) Диапазон, где данных нет (auto_clip=False)
    print("\n— Пустой диапазон (ожидаем ошибку) —")
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

    # 4) CurrencyPair через провайдер
    print("\n— CurrencyPair из провайдера —")
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
