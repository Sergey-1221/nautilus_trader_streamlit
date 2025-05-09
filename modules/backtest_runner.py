from __future__ import annotations

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


"""
Backtest Runner Module

Provides utilities for loading historical bar data from CSV,
initializing a backtest engine, running strategies, and
post-processing results into reports and metrics.
"""

# ────────────────────────────────────────────────────────────────
# 1. Load CSV and convert to bars
# ────────────────────────────────────────────────────────────────
def load_bars(csv_path: str):
    """
    Load OHLC(V) data from a CSV file and convert to Bar objects.

    Returns:
        instr: Instrument object for BTC/USDT pair
        bar_type: BarType describing interval and aggregation
        bars: list of Bar objects ready for backtesting
        df: original DataFrame with parsed timestamps
    """
    csv = Path(csv_path)
    if not csv.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    # Read CSV, interpreting decimals with dot
    df = pd.read_csv(csv, decimal=".")

    # Map lowercase column names to original names
    col_map = {col.lower(): col for col in df.columns}

    # Identify the timestamp column
    ts_col = next(
        (col_map.get(key) for key in ("timestamp", "time", "date") if key in col_map),
        None,
    )
    if ts_col is None:
        raise ValueError("CSV must contain one of: timestamp, time, or date columns")

    # Ensure required OHLC columns are present
    ohlc = ["open", "high", "low", "close"]
    missing = set(ohlc) - set(col_map)
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")

    # Select and rename columns
    selected = [ts_col] + [col_map[name] for name in ohlc]
    if "volume" in col_map:
        selected.append(col_map["volume"])
    df = df[selected]
    df.columns = ["timestamp", *ohlc, *("volume",) if "volume" in df.columns else ()]

    # Convert types
    df[ohlc] = df[ohlc].astype("float64")
    if "volume" in df.columns:
        df["volume"] = df["volume"].astype("float64", copy=False)

    # Parse timestamp column to datetime
    ts = df["timestamp"]
    if pd.api.types.is_numeric_dtype(ts):
        # Numeric epoch: choose ms vs s based on magnitude
        unit = "ms" if ts.max() > 1e12 else "s"
        df["timestamp"] = pd.to_datetime(ts, unit=unit, utc=True)
    else:
        df["timestamp"] = pd.to_datetime(ts, utc=True, errors="coerce")
    df.dropna(subset=["timestamp"], inplace=True)
    df.set_index("timestamp", inplace=True)
    df.sort_index(inplace=True)

    # Infer bar interval in minutes by mode of time diff
    try:
        delta = df.index.to_series().diff().mode()[0]
        interval = int(pd.to_timedelta(delta).seconds / 60) or 1
    except Exception:
        # Fallback: parse interval from filename
        m = re.search(r"(\d+)", csv.stem.split(",")[-1])
        interval = int(m.group(1)) if m else 1

    # Prepare BarType for BTC/USDT on Binance test provider
    instr = TestInstrumentProvider.btcusdt_binance()
    bar_type = BarType(
        instr.id,
        BarSpecification(interval, BarAggregation.MINUTE, PriceType.LAST),
        AggregationSource.EXTERNAL,
    )

    # Process DataFrame into Bar objects
    bars = BarDataWrangler(bar_type, instr).process(df)
    if not bars:
        raise RuntimeError("No bars produced—verify CSV structure and column types.")

    return instr, bar_type, bars, df


# ────────────────────────────────────────────────────────────────
# 2. Initialize Backtest Engine
# ────────────────────────────────────────────────────────────────
def _init_engine(instr, bars, balance: float = 10_000.0) -> BacktestEngine:
    """
    Create and configure a BacktestEngine with initial balances and market data.

    Args:
        instr: Instrument object to add
        bars: Preprocessed Bar objects
        balance: Starting USDT balance (default 10k)

    Returns:
        Configured BacktestEngine instance
    """
    engine = BacktestEngine()
    # Add Binance venue with netting OMS and cash account
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
# 3. Build engine, attach strategy & actor, and run
# ────────────────────────────────────────────────────────────────
def build_engine_with_actor(
    strat_cls: Type[Strategy],
    cfg_cls: Type[StrategyConfig],
    params: Dict[str, Any],
    csv: str,
    actor_cls: Type,
) -> BacktestEngine:
    """
    Helper to run a strategy with a custom Actor attached.

    1. Load bars from CSV
    2. Initialize engine
    3. Instantiate strategy with parameters
    4. Add strategy and actor
    5. Execute backtest

    Returns:
        The BacktestEngine after run (for inspection)
    """
    instr, bar_type, bars, _ = load_bars(csv)
    engine = _init_engine(instr, bars)

    # Prepare strategy configuration, casting Decimal fields properly
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
# 4. Run backtest and post-process results
# ────────────────────────────────────────────────────────────────
def run_backtest(
    strat_cls: Type[Strategy],
    cfg_cls: Type[StrategyConfig],
    params: Dict[str, Any],
    data_file: str,
    reuse_engine: Optional[BacktestEngine] = None,
) -> Dict[str, Any]:
    """
    Execute a backtest and return detailed results and metrics.

    Args:
        strat_cls: Strategy class to test
        cfg_cls: StrategyConfig class for configuration
        params: Dict of strategy parameters
        data_file: Path to CSV data
        reuse_engine: Optional existing engine to reuse

    Returns:
        Dictionary with:
          - price_df: DataFrame of input price bars
          - fills_df: DataFrame of order fills
          - trades_df: DataFrame of reconstructed trades
          - equity_df: DataFrame of equity curve
          - metrics: summary metrics (profit, drawdown, win rate)
          - stats: detailed analyzer stats if available
          - counts and commissions data
    """
    instr, bar_type, bars, price_df = load_bars(data_file)

    # 1) Setup engine and strategy
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

    # 2) Build order fills report DataFrame
    fills = engine.trader.generate_order_fills_report()
    fills_df = pd.DataFrame(fills if isinstance(fills, list) else list(fills))
    if fills_df.empty:
        fills_df = pd.DataFrame(columns=["timestamp", "order_side", "price", "quantity"])

    # Normalize timestamp column
    ts_key = next(
        (c for c in ("timestamp", "ts_event", "ts_fill", "ts_init", "ts") if c in fills_df.columns),
        None,
    )
    if ts_key is None:
        raise KeyError(f"No timestamp field in fills columns: {fills_df.columns.tolist()}")
    fills_df["timestamp"] = pd.to_datetime(fills_df[ts_key], utc=True, errors="coerce")

    # Ensure order_side, price, and quantity columns
    if "order_side" not in fills_df and "side" in fills_df:
        fills_df["order_side"] = fills_df["side"]
    price_key = next((c for c in ("price", "avg_px", "fill_px", "px") if c in fills_df.columns), None)
    if price_key is None:
        raise KeyError(f"No price field in fills columns: {fills_df.columns.tolist()}")
    fills_df["price"] = pd.to_numeric(fills_df[price_key], errors="coerce")
    qty_key = next((c for c in ("quantity", "qty", "filled_qty") if c in fills_df.columns), None)
    if qty_key is None:
        raise KeyError(f"No quantity field in fills columns: {fills_df.columns.tolist()}")
    fills_df["quantity"] = pd.to_numeric(fills_df[qty_key], errors="coerce")
    fills_df.fillna({"order_side": "", "price": 0.0, "quantity": 0.0}, inplace=True)

    # 3) Reconstruct trades from fills
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

    # Process each fill in chronological order
    for _, row in fills_df.sort_values("timestamp").iterrows():
        side = str(row["order_side"]).upper()
        px, qty, ts = float(row["price"]), float(row["quantity"]), row["timestamp"]

        if side == "BUY":
            # Close any short position first
            if pos_qty < 0:
                cover = min(qty, abs(pos_qty))
                pos_qty += cover
                if pos_qty == 0:
                    record_trade(ts, px)
                qty -= cover
            # Open or add to long
            if qty > 0:
                if pos_qty == 0:
                    entry_px, entry_ts, entry_side = px, ts, "LONG"
                entry_px = ((entry_px * pos_qty + px * qty) / (pos_qty + qty)) if pos_qty else px
                pos_qty += qty

        elif side == "SELL":
            # Close any long first
            if pos_qty > 0:
                close_qty = min(qty, pos_qty)
                pos_qty -= close_qty
                if pos_qty == 0:
                    record_trade(ts, px)
                qty -= close_qty
            # Open or add to short
            if qty > 0:
                if pos_qty == 0:
                    entry_px, entry_ts, entry_side = px, ts, "SHORT"
                entry_px = ((entry_px * abs(pos_qty) + px * qty) / (abs(pos_qty) + qty)) if pos_qty else px
                pos_qty -= qty

    # Force-close any open position at last bar close
    if pos_qty != 0:
        last_ts = price_df.index[-1]
        last_price = price_df["close"].iloc[-1]
        record_trade(last_ts, last_price)

    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        # Drop timezone for CSV export compatibility
        trades_df["entry_time"] = trades_df["entry_time"].dt.tz_localize(None)
        trades_df["exit_time"] = trades_df["exit_time"].dt.tz_localize(None)

    # 4) Build baseline equity curve
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
        # Apply fills at this timestamp
        while e_idx < len(events) and events[e_idx][0] == ts:
            _, side, p, q = events[e_idx]
            if side == "BUY":
                cash -= p * q
                pos_qty += q
            else:
                cash += p * q
                pos_qty -= q
            e_idx += 1
        # Compute equity and track max drawdown
        equity = cash + pos_qty * price_row["close"]
        equity_vals.append(equity)
        equity_ts.append(ts)
        max_eq = max(max_eq, equity)
        max_dd = max(max_dd, max_eq - equity)

    equity_df = pd.DataFrame({"equity": equity_vals}, index=equity_ts)

    ret_stats: dict = {}
    pnl_stats: dict = {}
    gen_stats: dict = {}

    # 5) Use PortfolioAnalyzer if available for more detailed stats
    portfolio = getattr(getattr(engine, 'trader'), 'portfolio', None) or getattr(getattr(engine, 'trader'), '_portfolio', None)
    if portfolio is not None and hasattr(portfolio, "analyzer"):
        try:
            analyzer = portfolio.analyzer
            analyzer.reset()

            # Attempt to retrieve the account object
            account_obj = None
            try:
                account_obj = engine.trader.get_account(Venue("BINANCE"))
            except Exception:
                # Fallback to first account in list or dict
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

            # Recompute max drawdown from analyzer curve
            roll_max = equity_df.equity.cummax()
            max_dd = (roll_max - equity_df.equity).max()
        except Exception as exc:
            _logger.warning("PortfolioAnalyzer failed: %s", exc, exc_info=True)

    # 6) Headline metrics summary
    total_profit = float(trades_df['profit'].sum()) if not trades_df.empty else 0.0
    num_trades = len(trades_df)
    win_rate = round((trades_df['profit'] > 0).sum() / num_trades * 100, 2) if num_trades else 0.0

    metrics = {
        'total_profit': round(total_profit, 2),
        'max_drawdown': round(float(max_dd), 2),
        'num_trades': num_trades,
        'win_rate': win_rate,
    }

    # 7) Additional technical info: commissions, order/position counts
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

    # 8) Return comprehensive results
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