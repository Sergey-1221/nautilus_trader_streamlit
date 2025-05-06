# Streamlit dashboard for Nautilus Trader
import sys, pathlib, numbers, dataclasses, inspect
from decimal import Decimal
from typing import Any, Dict, get_origin

import numpy as np
import plotly.graph_objects as go
import streamlit as st

# ── allow  import modules.*
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from modules.strategy_loader import discover_strategies
from modules.backtest_runner import run_backtest, build_engine_with_actor
from modules.data_collector import DataCollector
from modules.storage import save_backtest_result, load_backtest_result
from modules.dashboard_actor import DashboardPublisher


# ────────────────── page
st.set_page_config(page_title="NautilusTrader Dashboard", layout="wide")
st.title("NautilusTrader — universal backtest / live dashboard")


# ────────────────── helpers
def is_simple(v: Any) -> bool:
    return isinstance(v, (numbers.Number, str, bool, Decimal)) or v is None


def _try_default_factory(obj):
    fac = getattr(obj, "default_factory", None)
    if fac not in (None, dataclasses.MISSING, ..., Ellipsis):
        try:
            return fac()
        except Exception:
            pass
    return None


def get_field_default(cfg_cls, field: str):
    """
    Universal default extractor for StrategyConfig descendants.

    Supports:
    • dataclass fields
    • Pydantic v1 (__fields__)
    • Pydantic v2 (model_fields)
    • plain class attributes
    • constructor signature fallback
    """
    # --- dataclass ----------------------------------------------------------
    dfields = getattr(cfg_cls, "__dataclass_fields__", {})
    if field in dfields:
        f = dfields[field]
        if f.default is not dataclasses.MISSING:
            return f.default
        val = _try_default_factory(f)
        if val is not None:
            return val

    # --- Pydantic v2 --------------------------------------------------------
    mf = getattr(cfg_cls, "model_fields", None)
    if mf and field in mf:
        default = mf[field].default
        if default not in (dataclasses.MISSING, Ellipsis, None):
            return default

    # --- Pydantic v1 --------------------------------------------------------
    pf = getattr(cfg_cls, "__fields__", None)
    if pf and field in pf:
        default = pf[field].default
        if default not in (dataclasses.MISSING, Ellipsis, None):
            return default

    # --- regular attribute --------------------------------------------------
    attr = getattr(cfg_cls, field, dataclasses.MISSING)
    if is_simple(attr):
        return attr
    if hasattr(attr, "default"):
        default_attr = getattr(attr, "default")
        if default_attr not in (dataclasses.MISSING, Ellipsis, None):
            return default_attr
    val = _try_default_factory(attr)
    if val is not None:
        return val

    # --- constructor signature ---------------------------------------------
    try:
        sig_param = inspect.signature(cfg_cls).parameters.get(field)
        if sig_param and sig_param.default is not inspect._empty:
            return sig_param.default
    except (TypeError, ValueError):
        pass

    return None



def issub(tp, cls_or_tuple) -> bool:
    """Безопасный issubclass (работает с typing-типами)."""
    try:
        return issubclass(tp, cls_or_tuple)
    except TypeError:
        origin = get_origin(tp)
        return issub(origin, cls_or_tuple) if origin else False


# ────────────────── discover strategies
strategies = discover_strategies("strategies")
if not strategies:
    st.error("No strategies found in /strategies")
    st.stop()


# ────────────────── sidebar UI
with st.sidebar:
    st.header("Set-up")

    strat_name = st.selectbox("Strategy", list(strategies))
    info       = strategies[strat_name]
    if info.doc:
        st.caption(info.doc)

    timeframe = st.selectbox("Timeframe", ["1min", "15min"])
    data_file = "BINANCE_BTCUSD, 1.csv" if timeframe == "1min" else "BINANCE_BTCUSD, 15.csv"
    st.write(f"Data file: **{data_file}**")

    st.subheader("Parameters")
    params: Dict[str, Any] = {}

    for field, annotation in info.cfg_cls.__annotations__.items():
        if field in ("instrument_id", "bar_type"):
            continue

        default = get_field_default(info.cfg_cls, field)
        label   = field.replace("_", " ").title()

        # bool
        if isinstance(default, bool) or issub(annotation, bool):
            params[field] = st.checkbox(label, value=bool(default) if default is not None else False)

        # int
        elif isinstance(default, int) or issub(annotation, int):
            params[field] = st.number_input(label,
                                            value=int(default) if default is not None else 0,
                                            step=1, format="%d")

        # float / Decimal
        elif isinstance(default, (float, Decimal)) or issub(annotation, (float, Decimal)):
            params[field] = st.number_input(label,
                                            value=float(default) if default is not None else 0.0,
                                            format="%.6f")

        # fallback-строка
        else:
            params[field] = st.text_input(label, value=str(default or ""))

    run_bt = st.button("Run back-test", type="primary")


# ────────────────── run back-test
if run_bt:
    with st.spinner("Running …"):
        engine = build_engine_with_actor(
            info.strategy_cls, info.cfg_cls, params, data_file, DashboardPublisher
        )

        bus = getattr(engine, "msgbus", None)
        if bus is None:  # новые версии NT
            trader = getattr(engine, "trader", None) or getattr(engine, "_trader", None)
            bus = getattr(trader, "msgbus", None) if trader else None
        if bus is not None:
            DataCollector(bus, "DASHBOARD").drain()

        result = run_backtest(
            info.strategy_cls, info.cfg_cls, params, data_file, reuse_engine=engine
        )

    # metrics
    m = result["metrics"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Profit",  f"{m['total_profit']:.2f} USDT")
    c2.metric("Max DD",  f"{m['max_drawdown']:.2f} USDT")
    c3.metric("Trades",   m["num_trades"])
    c4.metric("Win %",   f"{m['win_rate']} %")

    # price chart
    price_df, trades_df = result["price_df"], result["trades_df"]
    ohlc = price_df[["open", "high", "low", "close"]]
    if len(ohlc) > 4_000:
        ohlc = ohlc.iloc[:: int(np.ceil(len(ohlc) / 4_000))]

    fig = go.Figure(go.Candlestick(
        x=ohlc.index, open=ohlc.open, high=ohlc.high, low=ohlc.low, close=ohlc.close))
    if not trades_df.empty:
        fig.add_trace(go.Scattergl(
            x=trades_df.entry_time, y=trades_df.entry_price,
            mode="markers", name="Entry",
            marker=dict(symbol="triangle-up", color="green", size=8)))
        fig.add_trace(go.Scattergl(
            x=trades_df.exit_time,  y=trades_df.exit_price,
            mode="markers", name="Exit",
            marker=dict(symbol="triangle-down", color="red", size=8)))
    st.plotly_chart(fig, use_container_width=True)

    # equity curve
    eq = result["equity_df"]
    st.plotly_chart(go.Figure(go.Scattergl(x=eq.index, y=eq.equity, mode="lines")),
                    use_container_width=True)

    # trades table
    if not trades_df.empty:
        st.subheader("Trades")
        st.dataframe(trades_df)

    # save
    fname = st.text_input("Filename", f"{strat_name.lower()}_{timeframe}")
    if st.button("Save result"):
        save_backtest_result(result, f"{fname}.pkl")
        st.success("Saved!")

# ────────────────── load pickle
st.markdown("---")
st.header("Load saved result")
upl = st.file_uploader("Pickle file", type="pkl")
if upl:
    res = load_backtest_result(upl)
    st.success("Loaded!")
    st.dataframe(res["trades_df"].head())
