from __future__ import annotations

"""
Backtest Runner Module (patched)
--------------------------------
Utilities for
* loading historical bar data from CSV,
* configuring and running a Nautilus‑Trader `BacktestEngine`,
* post‑processing fills -> trades -> equity curve, and
* packaging results and metrics.

**Patch 2025‑05‑09**
• `generate_order_fills_report()` now returns a `DataFrame` in modern Nautilus‑Trader.  
  The old implementation here was turning that into a plain list, losing column names
  and causing `KeyError: 'No timestamp field …'`.  
  The helper now:
  1. Accepts the DataFrame directly (fast‑path).
  2. Falls back to converting a legacy `list[OrderFill]`.
  3. Throws a clear `TypeError` for unknown formats.
• Timestamp handling simplified – once the DataFrame path is taken we just normalise
  a single `timestamp` column.

Everything else is functionally identical.
"""

import logging
import re
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Optional, Type

import pandas as pd
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.config import StrategyConfig
from nautilus_trader.model import BarSpecification, BarType
from nautilus_trader.model.currencies import BTC, USDT
from nautilus_trader.model.enums import (
    AggregationSource,
    AccountType,
    BarAggregation,
    OmsType,
    PriceType,
)
from nautilus_trader.model.identifiers import Venue
from nautilus_trader.model.objects import Money
from nautilus_trader.persistence.wranglers import BarDataWrangler
from nautilus_trader.test_kit.providers import TestInstrumentProvider
from nautilus_trader.trading.strategy import Strategy

_logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────
# 1. Load CSV and convert to bars
# ────────────────────────────────────────────────────────────────

def load_bars(csv_path: str):
    """Load OHLC(V) data from CSV and convert to `Bar` objects."""
    csv = Path(csv_path)
    if not csv.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv, decimal=".")

    # Map lowercase column names ➜ original names
    col_map = {col.lower(): col for col in df.columns}

    ts_col = next(
        (col_map.get(key) for key in ("timestamp", "time", "date") if key in col_map),
        None,
    )
    if ts_col is None:
        raise ValueError("CSV must contain one of: timestamp, time, or date columns")

    # Required OHLC
    ohlc = ["open", "high", "low", "close"]
    missing = set(ohlc) - set(col_map)
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")

    selected = [ts_col] + [col_map[name] for name in ohlc]
    if "volume" in col_map:
        selected.append(col_map["volume"])
    df = df[selected]
    df.columns = ["timestamp"] + ohlc + (["volume"] if "volume" in df.columns else [])

    df[ohlc] = df[ohlc].astype("float64")
    if "volume" in df.columns:
        df["volume"] = df["volume"].astype("float64")

    ts = df["timestamp"]
    if pd.api.types.is_numeric_dtype(ts):
        unit = "ms" if ts.max() > 1e12 else "s"
        df["timestamp"] = pd.to_datetime(ts, unit=unit, utc=True)
    else:
        df["timestamp"] = pd.to_datetime(ts, utc=True, errors="coerce")
    df.dropna(subset=["timestamp"], inplace=True)
    df.set_index("timestamp", inplace=True)
    df.sort_index(inplace=True)

    # Infer bar interval (minutes)
    try:
        delta = df.index.to_series().diff().mode()[0]
        interval = int(pd.to_timedelta(delta).seconds / 60) or 1
    except Exception:
        m = re.search(r"(\d+)", csv.stem.split(",")[-1])
        interval = int(m.group(1)) if m else 1

    instr = TestInstrumentProvider.btcusdt_binance()
    bar_type = BarType(
        instr.id,
        BarSpecification(interval, BarAggregation.MINUTE, PriceType.LAST),
        AggregationSource.EXTERNAL,
    )

    bars = BarDataWrangler(bar_type, instr).process(df)
    if not bars:
        raise RuntimeError("No bars produced—verify CSV structure and column types.")

    return instr, bar_type, bars, df


# ────────────────────────────────────────────────────────────────
# 2. Initialise Backtest Engine
# ────────────────────────────────────────────────────────────────

def _init_engine(instr, bars, balance: float = 10_000.0) -> BacktestEngine:
    engine = BacktestEngine()
    engine.add_venue(
        Venue("BINANCE"),
        oms_type=OmsType.NETTING,
        account_type=AccountType.CASH,
        starting_balances=[Money(balance, USDT), Money(1.0, BTC)],
        base_currency=None,
    )
    engine.add_instrument(instr)
    engine.add_data(bars)
    return engine


# ────────────────────────────────────────────────────────────────
# 3. Build engine, attach strategy & actor, run
# ────────────────────────────────────────────────────────────────

def build_engine_with_actor(
    strat_cls: Type[Strategy],
    cfg_cls: Type[StrategyConfig],
    params: Dict[str, Any],
    csv: str,
    actor_cls: Type,
) -> BacktestEngine:
    instr, bar_type, bars, _ = load_bars(csv)
    engine = _init_engine(instr, bars)

    cfg_args = {
        key: (Decimal(str(val)) if cfg_cls.__annotations__.get(key) is Decimal else val)
        for key, val in params.items()
        if key not in ("instrument_id", "bar_type")
    }
    cfg_args.update(instrument_id=instr.id, bar_type=bar_type)

    engine.add_strategy(strat_cls(cfg_cls(**cfg_args)))
    engine.add_actor(actor_cls())
    engine.run()
    return engine


# ────────────────────────────────────────────────────────────────
# 4. Run backtest and post‑process
# ────────────────────────────────────────────────────────────────

def _order_fills_to_dataframe(fills) -> pd.DataFrame:
    """Unified path converting whatever `generate_order_fills_report()` gives into a clean DF."""
    # Modern Nautilus‑Trader ➜ DataFrame straight away
    if isinstance(fills, pd.DataFrame):
        return fills.copy()

    # Legacy API ➜ iterable of OrderFill‑like objects / mapping / dataclass
    if isinstance(fills, (list, tuple)):
        records = []
        for f in fills:
            if isinstance(f, dict):
                records.append(f)
            elif hasattr(f, "to_dict"):
                records.append(f.to_dict())
            elif hasattr(f, "_asdict"):
                records.append(f._asdict())
            else:
                records.append(vars(f))
        return pd.DataFrame(records)

    raise TypeError(
        "Unexpected return type from generate_order_fills_report(): " + str(type(fills))
    )


def run_backtest(
    strat_cls: Type[Strategy],
    cfg_cls: Type[StrategyConfig],
    params: Dict[str, Any],
    data_file: str,
    reuse_engine: Optional[BacktestEngine] = None,
) -> Dict[str, Any]:

    instr, bar_type, bars, price_df = load_bars(data_file)

    # 1) Engine + strategy
    if reuse_engine is None:
        engine = _init_engine(instr, bars)
        cfg_args = {
            key: (Decimal(str(val)) if cfg_cls.__annotations__.get(key) is Decimal else val)
            for key, val in params.items()
            if key not in ("instrument_id", "bar_type")
        }
        cfg_args.update(instrument_id=instr.id, bar_type=bar_type)
        engine.add_strategy(strat_cls(cfg_cls(**cfg_args)))
        engine.run()
    else:
        engine = reuse_engine

    # 2) Fills ➜ DataFrame (patched section)
    fills_raw = engine.trader.generate_order_fills_report()
    fills_df = _order_fills_to_dataframe(fills_raw)

    if fills_df.empty:
        fills_df = pd.DataFrame(columns=["timestamp", "order_side", "price", "quantity"])

    # Timestamp normalisation
    rename_map = {c: "timestamp" for c in ("ts_event", "ts_fill", "ts_init", "ts") if c in fills_df.columns}
    fills_df.rename(columns=rename_map, inplace=True)
    if "timestamp" not in fills_df.columns:
        raise KeyError(f"No timestamp field in fills columns: {fills_df.columns.tolist()}")
    fills_df["timestamp"] = pd.to_datetime(fills_df["timestamp"], utc=True, errors="coerce")

    # Ensure fields side / price / quantity exist
    if "order_side" not in fills_df.columns:
        side_key = next((c for c in ("side",) if c in fills_df.columns), None)
        if side_key:
            fills_df["order_side"] = fills_df[side_key]
    price_key = next((c for c in ("price", "avg_px", "fill_px", "px") if c in fills_df.columns), None)
    if price_key is None:
        raise KeyError(f"No price field in fills columns: {fills_df.columns.tolist()}")
    fills_df["price"] = pd.to_numeric(fills_df[price_key], errors="coerce")
    qty_key = next((c for c in ("quantity", "qty", "filled_qty") if c in fills_df.columns), None)
    if qty_key is None:
        raise KeyError(f"No quantity field in fills columns: {fills_df.columns.tolist()}")
    fills_df["quantity"] = pd.to_numeric(fills_df[qty_key], errors="coerce")

    fills_df.fillna({"order_side": "", "price": 0.0, "quantity": 0.0}, inplace=True)

    # 3) Reconstruct trades
    trades: list[Dict[str, Any]] = []
    pos_qty = 0.0
    entry_px: Optional[float] = None
    entry_ts: Optional[pd.Timestamp] = None
    entry_side: Optional[str] = None

    def record_trade(exit_ts: pd.Timestamp, exit_px: float) -> None:
        profit = ((exit_px - entry_px) if pos_qty > 0 else (entry_px - exit_px)) * abs(pos_qty)
        trades.append({
            "entry_time": entry_ts,
            "exit_time": exit_ts,
            "entry_side": entry_side,
            "entry_price": entry_px,
            "exit_price": exit_px,
            "profit": round(profit, 2),
        })

    for _, row in fills_df.sort_values("timestamp").iterrows():
        side = str(row["order_side"]).upper()
        px, qty, ts = float(row["price"]), float(row["quantity"]), row["timestamp"]

        if side == "BUY":
            if pos_qty < 0:
                cover = min(qty, abs(pos_qty))
                pos_qty += cover
                if pos_qty == 0:
                    record_trade(ts, px)
                qty -= cover
            if qty > 0:
                if pos_qty == 0:
                    entry_px, entry_ts, entry_side = px, ts, "LONG"
                entry_px = ((entry_px * pos_qty + px * qty) / (pos_qty + qty)) if pos_qty else px
                pos_qty += qty

        elif side == "SELL":
            if pos_qty > 0:
                close_qty = min(qty, pos_qty)
                pos_qty -= close_qty
                if pos_qty == 0:
                    record_trade(ts, px)
                qty -= close_qty
            if qty > 0:
                if pos_qty == 0:
                    entry_px, entry_ts, entry_side = px, ts, "SHORT"
                entry_px = ((entry_px * abs(pos_qty) + px * qty) / (abs(pos_qty) + qty)) if pos_qty else px
                pos_qty -= qty

    if pos_qty != 0:
        last_ts = price_df.index[-1]
        last_price = price_df["close"].iloc[-1]
        record_trade(last_ts, last_price)

    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        trades_df["entry_time"] = trades_df["entry_time"].dt.tz_localize(None)
        trades_df["exit_time"] = trades_df["exit_time"].dt.tz_localize(None)

    # 4) Build equity curve
    cash = 10_000.0
    pos_qty = 0.0
    events = sorted([
        (row["timestamp"], str(row["order_side"]).upper(), float(row["price"]), float(row["quantity"]))
        for _, row in fills_df.iterrows()
    ], key=lambda x: x[0])

    e_idx = 0
    equity_vals: list[float] = []
    equity_ts: list[pd.Timestamp] = []
    max_eq = cash
    max_dd = 0.0

    for ts, price_row in price_df.iterrows():
        while e_idx < len(events) and events[e_idx][0] == ts:
            _, side, p, q = events[e_idx]
            if side == "BUY":
                cash -= p * q
                pos_qty += q
            else:
                cash += p * q
                pos_qty -= q
            e_idx += 1
        equity = cash + pos_qty * price_row["close"]
        equity_vals.append(equity)
        equity_ts.append(ts)
        max_eq = max(max_eq, equity)
        max_dd = max(max_dd, max_eq - equity)

    equity_df = pd.DataFrame({"equity": equity_vals}, index=equity_ts)

    ret_stats: dict = {}
    pnl_stats: dict = {}
    gen_stats: dict = {}

    portfolio = getattr(getattr(engine, 'trader'), 'portfolio', None) or getattr(getattr(engine, 'trader'), '_portfolio', None)
    if portfolio is not None and hasattr(portfolio, "analyzer"):
        try:
            analyzer = portfolio.analyzer
            analyzer.reset()

            account_obj = None
            try:
                account_obj = engine.trader.get_account(Venue("BINANCE"))
            except Exception:
                accounts = getattr(engine.trader, "accounts", None)
                if isinstance(accounts, dict):
                    account_obj = next(iter(accounts.values()), None)
                elif isinstance(accounts, list) and accounts:
                    account_obj = accounts[0]

            analyzer.calculate_statistics(
                account_obj,
                list(engine.trader.get_positions()) if hasattr(engine.trader, "get_positions") else [],
            )
            equity_df = analyzer.equity_curve().to_frame(name="equity")
            ret_stats = analyzer.get_performance_stats_returns()
            pnl_stats = analyzer.get_performance_stats_pnls()
            gen_stats = analyzer.get_performance_stats_general()

            roll_max = equity_df.equity.cummax()
            max_dd = (roll_max - equity_df.equity).max()
        except Exception as exc:
            _logger.warning("PortfolioAnalyzer failed: %s", exc, exc_info=True)

    total_profit = float(trades_df['profit'].sum()) if not trades_df.empty else 0.0
    num_trades = len(trades_df)
    win_rate = round((trades_df['profit'] > 0).sum() / num_trades * 100, 2) if num_trades else 0.0

    metrics = {
        'total_profit': round(total_profit, 2),
        'max_drawdown': round(float(max_dd), 2),
        'num_trades': num_trades,
        'win_rate': win_rate,
    }

    try:
        account_obj = engine.trader.get_account(Venue("BINANCE"))
    except Exception:
        account_obj = None
    commissions: Dict[str, float] = {}
    if account_obj is not None and hasattr(account_obj, 'commissions'):
        comms = account_obj.commissions()
        commissions = {str(k): float(v.amount) for k, v in comms.items()}

    orders_count = fills_df['order_id'].nunique() if 'order_id' in fills_df.columns else len(fills_df)
    positions_count = len(engine.trader.get_positions()) if hasattr(engine.trader, 'get_positions') else 0

    return {
        'price_df': price_df,
        'trades_df': trades_df,
        'fills_df': fills_df,
        'equity_df': equity_df,
        'metrics': metrics,
        'stats': {'returns': ret_stats, 'pnl': pnl_stats, 'general': gen_stats},
        'fills_count': len(fills_df),
        'orders_count': orders_count,
        'positions_count': positions_count,
        'commissions': commissions,
    }
