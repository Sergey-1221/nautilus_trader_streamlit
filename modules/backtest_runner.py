from __future__ import annotations

import logging
import re
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Optional, Type

import numpy as np
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
from nautilus_trader.model.objects import (
    Money,
)  # Money is imported but used as_decimal()
from nautilus_trader.persistence.wranglers import BarDataWrangler
from nautilus_trader.test_kit.providers import TestInstrumentProvider
from nautilus_trader.trading.strategy import Strategy

_logger = logging.getLogger(__name__)
from .csv_data import load_ohlcv_csv


# ──────────────────────────────────────────────────────────────────────
# 2. DataFrame → Nautilus Bars
# ──────────────────────────────────────────────────────────────────────
def dataframe_to_bars(
    df: pd.DataFrame,
    instrument_factory=TestInstrumentProvider.btcusdt_binance,
):
    """
    Accept a DataFrame (as from ``load_ohlcv_csv``) and return
    ``(instrument, bar_type, bars, df)``.

    *Interval* is inferred automatically from index differences.
    """
    if df.empty:
        raise ValueError("DataFrame is empty – nothing to convert.")

    # ── determine interval and Aggregation Unit ───────────────────────
    diffs = df.index.to_series().diff().dropna()
    if diffs.empty:
        interval, agg = 1, BarAggregation.MINUTE
    else:
        delta = diffs.mode()[0]
        secs = int(delta.total_seconds())
        if secs % 86_400 == 0:  # days
            interval, agg = secs // 86_400, BarAggregation.DAY
        elif secs % 3_600 == 0:  # hours
            interval, agg = secs // 3_600, BarAggregation.HOUR
        elif secs % 60 == 0:  # minutes
            interval, agg = secs // 60, BarAggregation.MINUTE
        else:  # non-standard → fallback to minutes
            interval, agg = max(1, secs // 60), BarAggregation.MINUTE

    # ── Instrument & BarType ─────────────────────────────────────────
    instr = instrument_factory()
    bar_type = BarType(
        instr.id,
        BarSpecification(interval, agg, PriceType.LAST),
        AggregationSource.EXTERNAL,
    )

    # ── DataFrame → Bars ─────────────────────────────────────────────
    wrangler = BarDataWrangler(bar_type=bar_type, instrument=instr)
    bars = wrangler.process(df)

    if not bars:
        raise RuntimeError("No bars produced – verify DataFrame structure.")

    return instr, bar_type, bars, df


# ────────────────────────────────────────────────────────────────
# 1. Load CSV and convert to bars
# ────────────────────────────────────────────────────────────────


def load_bars(csv_path: str):
    """Load OHLC(V) data from CSV and convert to `Bar` objects."""
    csv_file = Path(csv_path)
    if not csv_file.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_file, decimal=".")

    # Map lowercase column names ➜ original names
    col_map = {col.lower(): col for col in df.columns}

    ts_col_original = next(
        (col_map.get(key) for key in ("timestamp", "time", "date") if key in col_map),
        None,
    )
    if ts_col_original is None:
        raise ValueError("CSV must contain one of: timestamp, time, or date columns")

    # Required OHLC columns (target names)
    ohlc_target_names = ["open", "high", "low", "close"]
    ohlc_original_names = []
    missing = set(ohlc_target_names) - set(col_map)
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")

    for name in ohlc_target_names:
        ohlc_original_names.append(col_map[name])

    # Build the list of original column names to select
    selected_original_names = [ts_col_original] + ohlc_original_names
    has_volume = "volume" in col_map  # Check if volume exists in original columns map
    if has_volume:
        selected_original_names.append(col_map["volume"])

    # Select the columns using original names
    df = df[selected_original_names]

    # Build the list of target column names for the selected DataFrame
    target_columns = ["timestamp"] + ohlc_target_names
    if has_volume:
        target_columns.append("volume")

    df.columns = target_columns

    # Convert dtypes using target names
    df[ohlc_target_names] = df[ohlc_target_names].astype("float64")
    if has_volume:
        df["volume"] = df["volume"].astype("float64")

    ts = df["timestamp"]
    if pd.api.types.is_numeric_dtype(ts):
        max_ts = ts.max()
        # Adjusted logic based on common timestamp scales
        if max_ts > 2e18:  # Likely nanoseconds (max int64 is ~9e18)
            unit = "ns"
        elif max_ts > 2e15:  # Likely microseconds
            unit = "us"
        elif max_ts > 2e12:  # Likely milliseconds
            unit = "ms"
        else:  # Likely seconds
            unit = "s"
        df["timestamp"] = pd.to_datetime(ts, unit=unit, utc=True)
    else:
        df["timestamp"] = pd.to_datetime(ts, utc=True, errors="coerce")

    df.dropna(subset=["timestamp"], inplace=True)
    df.set_index("timestamp", inplace=True)
    df.sort_index(inplace=True)

    interval = 1
    try:
        if len(df) > 1:
            diffs = df.index.to_series().diff().dropna()
            if not diffs.empty:
                # Use the most frequent difference as the interval
                delta = diffs.mode()[0]
                interval_seconds = delta.total_seconds()
                if interval_seconds > 0:
                    # Determine the appropriate BarAggregation unit
                    if interval_seconds % (60 * 60 * 24) == 0:  # Days
                        interval = int(interval_seconds / (60 * 60 * 24))
                        agg = BarAggregation.DAY
                    elif interval_seconds % (60 * 60) == 0:  # Hours
                        interval = int(interval_seconds / (60 * 60))
                        agg = BarAggregation.HOUR
                    elif interval_seconds % 60 == 0:  # Minutes
                        interval = int(interval_seconds / 60)
                        agg = BarAggregation.MINUTE
                    else:  # Seconds (NautilusTrader might not support seconds directly for BarType spec)
                        # Fallback to minute aggregation if seconds are not standard
                        interval = max(
                            1, int(interval_seconds / 60)
                        )  # At least 1 minute
                        agg = BarAggregation.MINUTE
                else:
                    interval = 1
                    agg = (
                        BarAggregation.MINUTE
                    )  # Default if interval_seconds is 0 or less
            else:  # Only one row or no difference
                interval = 1
                agg = BarAggregation.MINUTE
        else:  # len(df) <= 1
            interval = 1
            agg = BarAggregation.MINUTE
    except Exception:
        # Fallback if datetime diff calculation fails
        _logger.warning(
            f"Could not determine bar interval from timestamps for {csv_file.name}. Attempting from filename."
        )
        m = re.search(r"(\d+)", csv_file.stem.split(",")[-1])
        interval = int(m.group(1)) if m else 1
        agg = BarAggregation.MINUTE  # Assume minute if filename parsing is the fallback

    interval = max(1, interval)  # Ensure interval is at least 1

    instr = TestInstrumentProvider.btcusdt_binance()
    # BarType requires InstrumentId, BarSpecification, and AggregationSource
    bar_type = BarType(
        instr.id,
        BarSpecification(interval, agg, PriceType.LAST),  # Use determined aggregation
        AggregationSource.EXTERNAL,
    )

    # Use BarDataWrangler
    wrangler = BarDataWrangler(bar_type=bar_type, instrument=instr)
    bars = wrangler.process(df)

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
        # base_currency=None → multi-currency account since version 1.171
        starting_balances=[
            Money(Decimal(str(balance)), USDT),
            # second balance in BTC (0 is enough) enables shorting
            Money(Decimal("0"), BTC),
        ],
        base_currency=None,
    )
    engine.add_instrument(instr)
    engine.add_data(bars)
    return engine


# ────────────────────────────────────────────────────────────────
# 3. Build engine, attach strategy & actor, run
# ────────────────────────────────────────────────────────────────


# This build_engine_with_actor function already accepted actor_cls
def build_engine_with_actor(
    strat_cls: Type[Strategy],
    cfg_cls: Type[StrategyConfig],
    params: Dict[str, Any],
    csv: str,
    actor_cls: Type,
) -> BacktestEngine:
    csv = load_ohlcv_csv(csv)
    instr, bar_type, bars, _ = dataframe_to_bars(csv)
    engine = _init_engine(instr, bars)

    cfg_args = {
        key: (Decimal(str(val)) if cfg_cls.__annotations__.get(key) is Decimal else val)
        for key, val in params.items()
        if key not in ("instrument_id", "bar_type")
    }
    cfg_args.update(instrument_id=instr.id, bar_type=bar_type)

    engine.add_strategy(strat_cls(cfg_cls(**cfg_args)))
    engine.add_actor(actor_cls())  # Add actor
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
                # Attempt to get attributes via vars() or direct access
                try:
                    record = vars(f)
                except TypeError:  # vars() might not work on all objects
                    record = {
                        attr: getattr(f, attr)
                        for attr in dir(f)
                        if not attr.startswith("_") and not callable(getattr(f, attr))
                    }
                    # Filter out non-serializable or unwanted attributes if necessary
                records.append(record)

        if not records:
            return pd.DataFrame()  # Return empty DF if no records could be processed

        return pd.DataFrame(records)

    # Handle None or other unexpected types gracefully
    if fills is None:
        return pd.DataFrame()

    raise TypeError(
        "Unexpected return type from generate_order_fills_report(): " + str(type(fills))
    )


def rebuild_equity_curve(
    price_index: pd.DatetimeIndex,
    trades_df: pd.DataFrame,
    start_balance: float = 10_000.0,
) -> pd.Series:
    """Construct a simple equity curve from trade profits."""
    price_index = pd.to_datetime(price_index)
    if getattr(price_index, "tz", None) is not None:
        price_index = price_index.tz_convert(None)

    if trades_df.empty:
        return pd.Series(start_balance, index=price_index)

    pnl = (
        trades_df.sort_values("exit_time")
        .set_index("exit_time")["profit"]
        .cumsum()
    )
    if getattr(pnl.index, "tz", None) is not None:
        pnl.index = pnl.index.tz_convert(None)
    pnl = pnl.reindex(price_index, method="ffill").fillna(0.0)
    equity = start_balance + pnl
    equity = equity.reindex(price_index, method="ffill")
    equity.iloc[0] = start_balance
    return equity


def _dd_periods(series: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Return start/end indices for all drawdown periods."""
    if series.empty:
        return []
    peak = series.iloc[0]
    start = None
    periods: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for ts, val in series.items():
        if val < peak:
            if start is None:
                start = ts
        else:
            if start is not None:
                periods.append((start, ts))
                start = None
            peak = val
    if start is not None:
        periods.append((start, series.index[-1]))
    return periods


def _longest_dd_days(series: pd.Series) -> float:
    if series.empty:
        return np.nan
    peak_idx = series.index[0]
    peak_val = series.iloc[0]
    longest = 0
    for idx, val in series.items():
        if val >= peak_val:
            longest = max(longest, (idx - peak_idx).days)
            peak_idx = idx
            peak_val = val
    longest = max(longest, (series.index[-1] - peak_idx).days)
    return float(longest)


def _avg_dd_days(series: pd.Series) -> float:
    durs = [(e - s).days for s, e in _dd_periods(series)]
    if not durs:
        return 0.0
    return float(np.mean(durs))


def _total_dd_days(series: pd.Series) -> float:
    durs = [(e - s).days for s, e in _dd_periods(series)]
    if not durs:
        return 0.0
    return float(np.sum(durs))


def run_backtest(
    strat_cls: Type[Strategy],
    cfg_cls: Type[StrategyConfig],
    params: Dict[str, Any],
    data: Any,
    actor_cls: Type,
    reuse_engine: Optional[BacktestEngine] = None,
) -> Dict[str, Any]:
    """Run a back-test using either a CSV path or a ready DataFrame."""

    if isinstance(data, str):
        csv = load_ohlcv_csv(data)
    elif isinstance(data, pd.DataFrame):
        csv = data
    else:
        raise TypeError("data must be a path to CSV or pandas.DataFrame")
    instr, bar_type, bars, price_df = dataframe_to_bars(csv)

    # 1) Engine + strategy + actor
    if reuse_engine is None:
        engine = _init_engine(instr, bars)
        cfg_args = {
            key: (
                Decimal(str(val))
                if cfg_cls.__annotations__.get(key) is Decimal
                else val
            )
            for key, val in params.items()
            if key not in ("instrument_id", "bar_type")
        }
        cfg_args.update(instrument_id=instr.id, bar_type=bar_type)
        engine.add_strategy(strat_cls(cfg_cls(**cfg_args)))
        engine.add_actor(actor_cls())
        engine.run()
    else:
        engine = reuse_engine
        # If the engine is reused it's assumed the strategy and actor are already added.
        # Otherwise you may need to adjust logic depending on the reuse scenario.
        # For simplicity we don't add the actor/strategy again here.

    # 2) Fills ➜ DataFrame (patched section)
    # Use generate_order_fills_report as in the original code
    fills_raw = engine.trader.generate_order_fills_report()

    fills_df = _order_fills_to_dataframe(fills_raw)

    if fills_df.empty:
        # Ensure the empty DataFrame has expected columns for further processing
        fills_df = pd.DataFrame(
            columns=["timestamp", "order_side", "price", "quantity", "order_id"]
        )

    # Timestamp normalisation
    # Look for any field resembling a timestamp and rename it to 'timestamp'
    timestamp_cols = [
        c for c in fills_df.columns if "ts_" in c or c in ("timestamp", "time", "date")
    ]
        if not timestamp_cols:
            # If no explicit time fields, try using the index if it's datetime
        if isinstance(fills_df.index, pd.DatetimeIndex):
            fills_df["timestamp"] = fills_df.index.to_series()
        else:
            raise KeyError(
                f"No timestamp field or DatetimeIndex found in fills columns: {fills_df.columns.tolist()}"
            )
    else:
        # Use the first found time field and rename it
        ts_col_to_use = timestamp_cols[0]
        if ts_col_to_use != "timestamp":
            fills_df.rename(columns={ts_col_to_use: "timestamp"}, inplace=True)

    # Ensure the 'timestamp' column is datetime with UTC
    fills_df["timestamp"] = pd.to_datetime(
        fills_df["timestamp"], utc=True, errors="coerce"
    )
    fills_df.dropna(
        subset=["timestamp"], inplace=True
    )  # Remove rows with invalid time
    fills_df.sort_values("timestamp", inplace=True)  # Sort by time

    # Ensure fields side / price / quantity exist and are correct types
    # Use .str.upper() for reliability if side is not a string
    if "order_side" not in fills_df.columns:
        side_key = next((c for c in ("side",) if c in fills_df.columns), None)
        if side_key:
            fills_df["order_side"] = fills_df[side_key].astype(str).str.upper()
        else:
            # If neither 'order_side' nor 'side' exists, you might skip this step or raise an error
            _logger.warning("Could not find 'order_side' or 'side' column in fills_df.")
            fills_df["order_side"] = (
                ""  # Add empty column to avoid KeyError
            )

    # Search and normalize price column
    price_keys = (
        "price",
        "avg_px",
        "fill_px",
        "px",
        "last_px",
    )  # last_px added from context
    price_key = next((c for c in price_keys if c in fills_df.columns), None)
    if price_key is None:
        raise KeyError(
            f"No price field found in fills columns. Tried: {price_keys}. Available: {fills_df.columns.tolist()}"
        )
    fills_df["price"] = pd.to_numeric(fills_df[price_key], errors="coerce")

    # Search and normalize quantity column
    qty_keys = (
        "quantity",
        "qty",
        "filled_qty",
        "last_qty",
    )  # last_qty added from context
    qty_key = next((c for c in qty_keys if c in fills_df.columns), None)
    if qty_key is None:
        raise KeyError(
            f"No quantity field found in fills columns. Tried: {qty_keys}. Available: {fills_df.columns.tolist()}"
        )
    fills_df["quantity"] = pd.to_numeric(fills_df[qty_key], errors="coerce")

    # Fill missing values in critical columns
    fills_df.fillna({"order_side": "", "price": 0.0, "quantity": 0.0}, inplace=True)

    # 3) Reconstruct trades
    trades: list[Dict[str, Any]] = []
    pos_qty = 0.0
    entry_px: Optional[float] = None
    entry_ts: Optional[pd.Timestamp] = None
    entry_side: Optional[str] = None

    def record_trade(exit_ts: pd.Timestamp, exit_px: float) -> None:
        # Ensure entry_px is not None before calculating
        if entry_px is None:
            _logger.warning("Attempted to record trade with None entry_price.")
            return

        profit = ((exit_px - entry_px) if pos_qty > 0 else (entry_px - exit_px)) * abs(
            pos_qty
        )
        trades.append(
            {
                "entry_time": entry_ts,
                "exit_time": exit_ts,
                "entry_side": entry_side,
                "entry_price": entry_px,
                "exit_price": exit_px,
                "profit": round(profit, 2),
            }
        )

    # Use fills_df which is already sorted by timestamp
    for _, row in fills_df.iterrows():
        side = str(row["order_side"]).upper()
        px, qty, ts = float(row["price"]), float(row["quantity"]), row["timestamp"]

        # Skip rows with zero quantity or price
        if qty == 0.0 or px == 0.0:
            continue

        if side == "BUY":
            if pos_qty < 0:
                # Close part or all of a short position
                cover = min(qty, abs(pos_qty))
                pos_qty += cover
                if pos_qty == 0:
                    record_trade(ts, px)  # Complete closing of short position
                qty -= cover
            if qty > 0:
                # Open or increase a long position
                if pos_qty == 0:
                    entry_px, entry_ts, entry_side = px, ts, "LONG"
                else:
                    # Average entry price for a long position
                    entry_px = (
                        ((entry_px * pos_qty + px * qty) / (pos_qty + qty))
                        if entry_px is not None
                        else px
                    )  # Ensure entry_px is not None
                pos_qty += qty

        elif side == "SELL":
            if pos_qty > 0:
                # Close part or all of a long position
                close_qty = min(qty, pos_qty)
                pos_qty -= close_qty
                if pos_qty == 0:
                    record_trade(ts, px)  # Complete closing of long position
                qty -= close_qty
            if qty > 0:
                # Open or increase a short position
                if pos_qty == 0:
                    entry_px, entry_ts, entry_side = px, ts, "SHORT"
                else:
                    # Average entry price for a short position
                    entry_px = (
                        ((entry_px * abs(pos_qty) + px * qty) / (abs(pos_qty) + qty))
                        if entry_px is not None
                        else px
                    )  # Ensure entry_px is not None
                pos_qty -= qty
        else:
            _logger.warning(
                f"Unknown order side encountered in fills: {row['order_side']}"
            )

    # Close remaining position at the end of the backtest
    if pos_qty != 0 and not price_df.empty:
        last_ts = price_df.index[-1]
        last_price = price_df["close"].iloc[-1]
        record_trade(last_ts, last_price)  # Close at the last bar price

    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        # Ensure time columns have correct type and remove timezone for compatibility
        trades_df["entry_time"] = pd.to_datetime(
            trades_df["entry_time"]
        ).dt.tz_localize(None)
        trades_df["exit_time"] = pd.to_datetime(trades_df["exit_time"]).dt.tz_localize(
            None
        )

    # 4) Build equity curve
    # Use PortfolioAnalyzer for more accurate calculations if available
    equity_df = pd.DataFrame({"equity": []})  # Initialize empty DF
    ret_stats: dict = {}
    pnl_stats: dict = {}
    gen_stats: dict = {}
    max_dd = 0.0  # Initialize max_dd

    # Get Trader object from engine
    trader = getattr(engine, "trader", None) or getattr(engine, "_trader", None)

    if trader is not None:
        portfolio = getattr(trader, "portfolio", None) or getattr(
            trader, "_portfolio", None
        )
        if portfolio is not None and hasattr(portfolio, "analyzer"):
            try:
                analyzer = portfolio.analyzer
                analyzer.reset()  # Reset analyzer before use

                # Try to obtain the account object
                account_obj = None
                try:
                    # Assume the trader has a get_account method
                    account_obj = trader.get_account(Venue("BINANCE"))
                except Exception:
                    # Fallback: look for accounts in trader attributes
                    accounts = getattr(trader, "accounts", None)
                    if isinstance(accounts, dict):
                        account_obj = next(iter(accounts.values()), None)
                    elif isinstance(accounts, list) and accounts:
                        account_obj = accounts[0]

                # Obtain positions
                positions = []
                if hasattr(trader, "get_positions"):
                    positions = list(trader.get_positions())
                elif hasattr(
                    portfolio, "positions"
                ):  # Fallback: Fallback: look for positions in the portfolio
                    positions = list(
                        portfolio.positions.values()
                    )  # Assume dict or list

                # Calculate statistics
                analyzer.calculate_statistics(
                    account_obj,
                    positions,
                )
                # Get equity curve and stats from the analyzer
                equity_df = analyzer.equity_curve().to_frame(name="equity")
                if not equity_df.empty:
                    equity_df.index = pd.to_datetime(equity_df.index)
                    if getattr(equity_df.index, "tz", None) is not None:
                        equity_df.index = equity_df.index.tz_convert(None)
                    equity_df.sort_index(inplace=True)
                ret_stats = analyzer.get_performance_stats_returns()
                pnl_stats = analyzer.get_performance_stats_pnls()
                gen_stats = analyzer.get_performance_stats_general()

                # Recalculate max drawdown based on analyzer equity curve
                if not equity_df.empty:
                    roll_max = equity_df.equity.cummax()
                    max_dd = (roll_max - equity_df.equity).max()
                else:
                    max_dd = 0.0  # If equity curve is empty

            except Exception as exc:
                _logger.warning("PortfolioAnalyzer failed: %s", exc, exc_info=True)
                _logger.warning("Falling back to manual equity calculation.")
                start_balance = 10_000.0
                try:
                    account_obj = trader.get_account(Venue("BINANCE"))
                    bal = getattr(account_obj, "cash_balance", None)
                    if callable(bal):
                        start_balance = float(bal(USDT).as_double())
                except Exception:
                    pass
                equity_series = rebuild_equity_curve(price_df.index, trades_df, start_balance)
                equity_df = equity_series.to_frame(name="equity")
                if not equity_df.empty:
                    equity_df.index = pd.to_datetime(equity_df.index)
                    if getattr(equity_df.index, "tz", None) is not None:
                        equity_df.index = equity_df.index.tz_convert(None)
                    equity_df.sort_index(inplace=True)
                if not equity_df.empty:
                    roll_max = equity_df.equity.cummax()
                    max_dd = (roll_max - equity_df.equity).max()
                else:
                    max_dd = 0.0

    # If PortfolioAnalyzer was not used or didn't return an equity curve,
    # you can add manual calculations here as a fallback if needed.
    # In the current version, if the analyzer failed, equity_df will be empty.

    # Calculate metrics from trades_df and max_dd
    total_profit = float(trades_df["profit"].sum()) if not trades_df.empty else 0.0
    num_trades = len(trades_df)
    # Avoid division by zero
    win_rate = (
        round((trades_df["profit"] > 0).sum() / num_trades * 100, 2)
        if num_trades > 0
        else 0.0
    )

    profit_factor = np.nan
    if not trades_df.empty:
        gains = trades_df.loc[trades_df["profit"] > 0, "profit"].sum()
        losses = trades_df.loc[trades_df["profit"] < 0, "profit"].sum()
        if losses == 0:
            profit_factor = np.inf if gains > 0 else np.nan
        else:
            profit_factor = gains / abs(losses)

    # ── Additional risk metrics --------------------------------------------
    sortino = np.nan
    pnl_dd_ratio = np.nan
    calmar = np.nan
    romad = np.nan
    longest_dd_len = np.nan
    avg_dd_len = np.nan
    total_dd_len = np.nan
    if not equity_df.empty:
        returns = equity_df["equity"].pct_change().dropna()
        neg_returns = returns[returns < 0]
        if not neg_returns.empty and neg_returns.std(ddof=0) > 0:
            sortino = (returns.mean() / neg_returns.std(ddof=0)) * np.sqrt(252)
        max_dd_pct = ((equity_df.equity.cummax() - equity_df.equity) / equity_df.equity.cummax()).max()
        total_return = (equity_df.equity.iloc[-1] - equity_df.equity.iloc[0]) / equity_df.equity.iloc[0]
        period_seconds = (equity_df.index[-1] - equity_df.index[0]).total_seconds()
        annual_return = (1 + total_return) ** (365 * 24 * 3600 / period_seconds) - 1 if period_seconds > 0 else np.nan
        if not np.isnan(total_return):
            if max_dd_pct == 0:
                pnl_dd_ratio = np.inf if total_return > 0 else np.nan
                romad = np.inf if total_return > 0 else np.nan
            elif not np.isnan(max_dd_pct):
                pnl_dd_ratio = (total_return * 100) / (abs(max_dd_pct) * 100)
                romad = total_return / abs(max_dd_pct)
        if not np.isnan(annual_return):
            if max_dd_pct == 0:
                calmar = np.inf if annual_return > 0 else np.nan
            elif not np.isnan(max_dd_pct):
                calmar = annual_return / abs(max_dd_pct)
        longest_dd_len = _longest_dd_days(equity_df["equity"])
        avg_dd_len = _avg_dd_days(equity_df["equity"])
        total_dd_len = _total_dd_days(equity_df["equity"])

    metrics = {
        "total_profit": round(total_profit, 2),
          "max_drawdown": round(float(max_dd), 2),  # max_dd already float or 0.0
        "num_trades": num_trades,
        "win_rate": win_rate,
        "profit_factor": round(float(profit_factor), 2)
        if not np.isnan(profit_factor)
        else np.nan,
        "profit_dd": round(float(pnl_dd_ratio), 2) if not np.isnan(pnl_dd_ratio) else np.nan,
        "calmar": round(float(calmar), 2) if not np.isnan(calmar) else np.nan,
        "romad": round(float(romad), 2) if not np.isnan(romad) else np.nan,
        "sortino": round(float(sortino), 2) if not np.isnan(sortino) else np.nan,
        "longest_dd_days": round(float(longest_dd_len), 2) if not np.isnan(longest_dd_len) else np.nan,
        "avg_dd_days": round(float(avg_dd_len), 2) if not np.isnan(avg_dd_len) else np.nan,
        "total_dd_days": round(float(total_dd_len), 2) if not np.isnan(total_dd_len) else np.nan,
    }

    # Retrieve commissions
    commissions: Dict[str, float] = {}
    if trader is not None:
        try:
            account_obj = trader.get_account(Venue("BINANCE"))
            if account_obj is not None and hasattr(account_obj, "commissions"):
                # commissions() returns dict[Currency, Money]
                comms = account_obj.commissions()
                # Convert Money to float for the report
                commissions = {
                    str(k): float(v.as_double()) for k, v in comms.items()
                  }  # Use as_double() or as_decimal()
        except Exception as exc:
            _logger.warning("Could not retrieve commissions: %s", exc)

    # Counting orders and positions
    # fills_df['order_id'] may not exist; use client_order_id from context
    orders_count = (
        fills_df["client_order_id"].nunique()
        if "client_order_id" in fills_df.columns
        else len(fills_df)
    )
    positions_count = (
        len(trader.get_positions())
        if trader is not None and hasattr(trader, "get_positions")
        else 0
    )

    return {
        "price_df": price_df,
        "trades_df": trades_df,
        "fills_df": fills_df,
        "equity_df": equity_df,
        "metrics": metrics,
        "stats": {"returns": ret_stats, "pnl": pnl_stats, "general": gen_stats},
        "fills_count": len(fills_df),
        "orders_count": orders_count,
        "positions_count": positions_count,
        "commissions": commissions,
    }
