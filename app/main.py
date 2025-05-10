# -*- coding: utf‑8 -*-
"""
Combined NautilusTrader back‑test dashboard + “Universal Back‑test Dashboard”
visual components — **single file, ready to run**.

Run from project root::

    streamlit run main_dashboard.py

Key enhancements (v3)
--------------------
1. **Run metadata** block restyled with a soft‑grey background and the unique run
   ID injected directly into the expander title for instant traceability.
2. **Unified “💹 Account & Performance” section**: the former independent blocks
   (💰 Balances & Fees, 📈 PnL statistics, 📊 Return / Risk, ⚖ General stats,
   Back‑test KPI grid) are now presented as four easy‑to‑navigate **tabs** inside
   a single bordered container — keeping the screen tidy while letting users
   focus on what matters.
3. Each tab leverages `st.metric` grids for compact, colour‑coded visualisation
   of key numbers, plus sparklines or data frames where helpful.
4. All comments remain **English‑only** and the file stays self‑contained.
"""

from __future__ import annotations

# ───────────────────────────── standard libs ────────────────────────────────
import sys
import pathlib
import numbers
import dataclasses
import inspect
import io
import uuid
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, get_origin

# ────────────────────────────── 3rd‑party ────────────────────────────────────
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ────────────────────────────── local code ───────────────────────────────────
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from modules.strategy_loader import discover_strategies
from modules.backtest_runner import run_backtest
from modules.dashboard_actor import DashboardPublisher  # optional, only if supported

# ───────────────────────────── Streamlit page ────────────────────────────────
st.set_page_config(page_title="NautilusTrader Dashboard", layout="wide")

# --- CSS tweaks --------------------------------------------------------------
st.markdown(
    """
    <style>
    /* grey background for the first expander header */
    div[data-testid="stExpander"] > details > summary {
        background-color:#f3f4f6 !important;  /* tailwind gray‑100 */
        border:1px solid #e5e7eb !important;
        border-radius:6px;
    }
    /* reduce padding on tab labels to save vertical space */
    .stTabs [data-baseweb="tab"] { padding-top:4px; padding-bottom:4px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("NautilusTrader — live / back‑test dashboard")

# ╭──────────────────────── helper utilities ─────────────────────────────────╮
def is_simple(v: Any) -> bool:
    """Return True for “simple” JSON‑like scalars."""
    return isinstance(v, (numbers.Number, str, bool, Decimal)) or v is None


def _try_default_factory(obj):
    """Attempt to call dataclass default_factory without raising."""
    factory = getattr(obj, "default_factory", None)
    if factory not in (None, dataclasses.MISSING, ..., Ellipsis):
        try:
            return factory()
        except Exception:
            pass
    return None


def get_field_default(cfg_cls, field: str):
    """
    Best‑effort extraction of default value for a dataclass / Pydantic model
    field, supporting dataclass, Pydantic v1 & v2.
    """
    # dataclass
    dfields = getattr(cfg_cls, "__dataclass_fields__", {})
    if field in dfields:
        f = dfields[field]
        if f.default is not dataclasses.MISSING:
            return f.default
        val = _try_default_factory(f)
        if val is not None:
            return val

    # pydantic v2
    pf_v2 = getattr(cfg_cls, "model_fields", None)
    if pf_v2 and field in pf_v2:
        default = pf_v2[field].default
        if default not in (dataclasses.MISSING, Ellipsis, None):
            return default

    # pydantic v1
    pf_v1 = getattr(cfg_cls, "__fields__", None)
    if pf_v1 and field in pf_v1:
        default = pf_v1[field].default
        if default not in (dataclasses.MISSING, Ellipsis, None):
            return default

    # plain class attribute
    attr = getattr(cfg_cls, field, dataclasses.MISSING)
    if is_simple(attr):
        return attr
    if hasattr(attr, "default"):
        if attr.default not in (dataclasses.MISSING, Ellipsis, None):
            return attr.default
    val = _try_default_factory(attr)
    if val is not None:
        return val

    # __init__ signature default
    try:
        sig_param = inspect.signature(cfg_cls).parameters.get(field)
        if sig_param and sig_param.default is not inspect._empty:
            return sig_param.default
    except (TypeError, ValueError):
        pass
    return None


def issub(tp, cls_or_tuple) -> bool:
    """Safe issubclass that also handles typing constructs."""
    try:
        return issubclass(tp, cls_or_tuple)
    except TypeError:
        origin = get_origin(tp)
        return issub(origin, cls_or_tuple) if origin else False


def parse_extra_stats(log_text: str) -> Dict[str, float]:
    """Extract “BacktestEngine: Key: value” lines into a dict of floats."""
    stats: Dict[str, float] = {}
    for line in log_text.splitlines():
        if "BacktestEngine:" not in line:
            continue
        tail = line.split("BacktestEngine:")[1].strip()
        if tail.startswith(("-", "_")) or ":" not in tail:
            continue
        key, val = tail.split(":", 1)
        key, val = key.strip(), val.strip().rstrip("%")
        try:
            stats[key] = float(val)
        except ValueError:
            pass
    return stats


# ────────── pretty format helpers (ISO 8601, BTC, USDT, etc.) ────────────────
def _fmt_dt(dt: datetime | str) -> str:
    """Return ISO‑8601 string with trailing “Z” (UTC) and no microseconds."""
    if isinstance(dt, str):
        return dt
    return dt.replace(microsecond=0).isoformat() + "Z"


def _fmt_usd(num: float | Decimal | None) -> str:
    return "—" if num is None else f"{num:,.8f} USDT"


def _fmt_btc(num: float | Decimal | None) -> str:
    return "—" if num is None else f"{num:,.8f} BTC"


# ╭──────────────────────── dashboard renderer ───────────────────────────────╮
def draw_dashboard(result: dict, log_text: str, TPL: str, ACCENT: str, NEG: str) -> None:
    """
    Build the entire Streamlit dashboard.
    Five high‑level blocks are always present (with fallbacks if data missing).
    """

    # ── 0. basic run metadata (needed multiple times) ------------------------
    run_meta = {
        "Run ID":        getattr(result, "run_id", uuid.uuid4()),
        "Run started":   result.get("run_started", datetime.utcnow()),
        "Run finished":  result.get("run_finished", datetime.utcnow()),
        "Elapsed time":  result.get("elapsed", "—"),
        "Backtest start": result["price_df"].index[0],
        "Backtest end":   result["price_df"].index[-1],
        "Backtest range": str(result["price_df"].index[-1] - result["price_df"].index[0]),
        "Iterations":     result.get("iterations", "—"),
        "Total events":   result.get("total_events", "—"),
        "Total orders":   result.get("orders_count", "—"),
        "Total positions": result.get("positions_count", "—"),
    }

    # ── 1. extract core DataFrames ------------------------------------------
    price_df   = result["price_df"].copy()
    equity_df  = result["equity_df"].copy()
    trades_df  = result["trades_df"].copy()

    for df in (price_df, equity_df, trades_df):
        if not df.empty and not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

    price_series = price_df["close"] if "close" in price_df else price_df.iloc[:, 0]

    # ── 2. fast KPI calc -----------------------------------------------------
    strategy_returns = (
        equity_df["equity"].pct_change().dropna()
        if not equity_df.empty
        else pd.Series(dtype=float)
    )
    benchmark_returns = price_series.pct_change().dropna()

    def sharpe(series: pd.Series) -> float:
        return (
            (series.mean() / series.std(ddof=0)) * np.sqrt(252)
            if not series.empty and series.std()
            else np.nan
        )

    def sortino(series: pd.Series) -> float:
        neg = series[series < 0]
        return (
            (series.mean() / neg.std(ddof=0)) * np.sqrt(252)
            if not neg.empty and neg.std()
            else np.nan
        )

    def max_dd(series: pd.Series) -> float:
        return (
            ((series.cummax() - series) / series.cummax()).max()
            if not series.empty
            else np.nan
        )

    comm_total = sum(result.get("commissions", {}).values())

    kpi = {
        "PnL ($)":       result.get("metrics", {}).get("total_profit", np.nan),
        "PnL (%)":       (
            (equity_df["equity"].iloc[-1] - equity_df["equity"].iloc[0])
            / equity_df["equity"].iloc[0]
            if not equity_df.empty
            else np.nan
        ),
        "Win Rate":      result.get("metrics", {}).get("win_rate", np.nan),
        "Sharpe":        sharpe(strategy_returns),
        "Sortino":       sortino(strategy_returns),
        "Max DD (%)":    max_dd(equity_df["equity"]) * 100 if not equity_df.empty else np.nan,
        "Profit Factor": result.get("metrics", {}).get("profit_factor", np.nan),
        "Volatility (252d)": strategy_returns.std(ddof=0) * np.sqrt(252)
        if not strategy_returns.empty
        else np.nan,
    }

    extra_stats = parse_extra_stats(log_text)

    # ╭──────────────────── 📄  RUN METADATA (collapsed) ──────────────────────╮
    with st.expander(
        f"📄 Run metadata — ID: {run_meta['Run ID']}",
        expanded=False,
    ):
        meta_cols = st.columns(3)
        with meta_cols[0]:
            st.write(f"**Venue:** {result.get('venue', '—')}")
            st.write(f"**Iterations:** {run_meta['Iterations']}")
            st.write(f"**Total events:** {run_meta['Total events']}")
        with meta_cols[1]:
            st.write(f"**Run started:** {_fmt_dt(run_meta['Run started'])}")
            st.write(f"**Run finished:** {_fmt_dt(run_meta['Run finished'])}")
            st.write(f"**Elapsed:** {run_meta['Elapsed time']}")
        with meta_cols[2]:
            st.write("**Back‑test period:**")
            st.markdown(
                f"""<div style='line-height:1.35'>
                Start: {_fmt_dt(run_meta['Backtest start'])}<br>
                End:&nbsp;&nbsp; {_fmt_dt(run_meta['Backtest end'])}<br>
                Duration: {run_meta['Backtest range']}
                </div>""",
                unsafe_allow_html=True,
            )

    # ╭──────────────── SUMMARY KPI GRID ─────────────────────────────────────╮
    st.markdown("## ⚡ Back‑test summary")
    hdr = st.columns(4)
    hdr[0].metric("Started",   _fmt_dt(run_meta["Run started"]))
    hdr[1].metric("Finished",  _fmt_dt(run_meta["Run finished"]))
    hdr[2].metric("Elapsed",   run_meta["Elapsed time"])
    hdr[3].metric("Orders",    run_meta["Total orders"])

    kcols = st.columns(len(kpi))
    for (label, value), col in zip(kpi.items(), kcols):
        text = (
            "—"
            if np.isnan(value)
            else (
                f"{value*100:.2f}%"
                if "%" in label and ("PnL" in label or "DD" in label)
                else f"{value:,.2f}"
            )
        )
        color = NEG if (("PnL" in label and value < 0) or ("DD" in label and value > 0)) else ACCENT
        col.markdown(
            f"<span style='font-size:1.1rem;color:{color}'><b>{text}</b><br>{label}</span>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ╭──────────────────── 💹 ACCOUNT & PERFORMANCE (TABS) ───────────────────╮
    with st.container(border=True):
        st.subheader("💹 Account & Performance")
        perf_tabs = st.tabs([
            "Balances & Fees",
            "PnL",
            "Return & Risk",
            "General",
        ])

        # ── Tab 0: Balances & Fees ───────────────────────────────────────────
        with perf_tabs[0]:
            bal_cols = st.columns(4)
            initial_bal  = result.get("initial_balances", {"USDT": 10_000, "BTC": 1})
            final_bal    = result.get("final_balances",   {"USDT": 9_872.05788320, "BTC": 1})
            unrealised   = result.get("unrealised_pnl", None)

            bal_cols[0].metric("Initial USDT", _fmt_usd(initial_bal.get("USDT")))
            bal_cols[1].metric("Initial BTC",  _fmt_btc(initial_bal.get("BTC")))
            bal_cols[2].metric("Final USDT",   _fmt_usd(final_bal.get("USDT")))
            bal_cols[3].metric("Final BTC",    _fmt_btc(final_bal.get("BTC")))

            fee_cols = st.columns(2)
            fee_cols[0].metric("Total fees", _fmt_usd(-comm_total if comm_total else None))
            fee_cols[1].metric("Unrealised PnL", "None" if unrealised is None else f"{unrealised:+,.2f}")

        # ── Tab 1: PnL statistics ────────────────────────────────────────────
        with perf_tabs[1]:
            pnl_metrics = result.get("metrics", {})
            btc = pnl_metrics.get("btc", {})
            usd = pnl_metrics.get("usdt", {})
            pnl_cols = st.columns(4)
            pnl_cols[0].metric("BTC PnL",   btc.get("total", "—"))
            pnl_cols[1].metric("BTC PnL %", btc.get("pct", "—"))
            pnl_cols[2].metric("USDT PnL",  usd.get("total", kpi["PnL ($)"]))
            pnl_cols[3].metric("USDT PnL %", usd.get("pct", kpi["PnL (%)"]))

            st.dataframe(
                pd.DataFrame({
                    "Metric": [
                        "Max winner", "Avg winner", "Min winner",
                        "Max loser",  "Avg loser",  "Min loser",
                        "Expectancy", "Win rate",
                    ],
                    "BTC": [
                        btc.get("max_win"), btc.get("avg_win"), btc.get("min_win"),
                        btc.get("max_loss"), btc.get("avg_loss"), btc.get("min_loss"),
                        btc.get("expectancy"), btc.get("win_rate"),
                    ],
                    "USDT": [
                        usd.get("max_win"), usd.get("avg_win"), usd.get("min_win"),
                        usd.get("max_loss"), usd.get("avg_loss"), usd.get("min_loss"),
                        usd.get("expectancy"), usd.get("win_rate", kpi["Win Rate"]),
                    ]
                }).set_index("Metric"),
                height=250,
            )

        # ── Tab 2: Return & Risk statistics ──────────────────────────────────
        with perf_tabs[2]:
            if strategy_returns.empty:
                st.info("Not enough data to compute return stats.")
            else:
                ret_stats = {
                    "Volatility (252d)":        kpi["Volatility (252d)"],
                    "Avg daily return":         strategy_returns.mean(),
                    "Avg loss (daily)":         strategy_returns[strategy_returns < 0].mean(),
                    "Avg win (daily)":          strategy_returns[strategy_returns > 0].mean(),
                    "Sharpe (252d)":            kpi["Sharpe"],
                    "Sortino (252d)":           kpi["Sortino"],
                    "Profit factor":            kpi["Profit Factor"],
                    "Risk / Return": (
                        abs(kpi["Max DD (%)"] / 100) / kpi["PnL (%)"]
                        if not np.isnan(kpi["PnL (%)"]) and kpi["PnL (%)"] != 0
                        else np.nan
                    ),
                }
                rc1, rc2, rc3, rc4 = st.columns(4)
                for (lbl, val), col in zip(ret_stats.items(), (rc1, rc2, rc3, rc4)*3):
                    col.metric(lbl, "—" if np.isnan(val) else f"{val:,.4f}")

                st.dataframe(pd.Series(ret_stats, name="Value"))

        # ── Tab 3: General statistics ────────────────────────────────────────
        with perf_tabs[3]:
            long_ratio = result.get("long_ratio")
            if long_ratio is None and not trades_df.empty and "side" in trades_df:
                long_ratio = (trades_df["side"].str.upper() == "BUY").mean()

            st.metric("Long ratio", "—" if long_ratio is None else f"{long_ratio:.0%}")
            st.metric("Positions", run_meta["Total positions"])
            st.metric("Trades", len(trades_df) if not trades_df.empty else 0)

    st.markdown("---")

    # ① Price & Trades --------------------------------------------------------
    st.subheader("📉 Price & Trades")
    fig_pt = go.Figure()
    fig_pt = go.Figure()
    fig_pt.add_trace(go.Scatter(x=price_series.index, y=price_series, mode="lines", name="Price"))

    if not trades_df.empty:
        print("entry_side", trades_df.get("entry_side", "").str.upper())
        buys = trades_df[trades_df.get("entry_side", "").str.upper() == "LONG"]
        sells = trades_df[trades_df.get("entry_side", "").str.upper() == "SELL"]

        if not buys.empty:
            fig_pt.add_trace(
                go.Scatter(
                    x=buys["entry_time"],
                    y=buys["entry_price"],
                    mode="markers",
                    marker_symbol="triangle-up",
                    marker_color=ACCENT,
                    name="Buy (entry)",
                )
            )
        if not sells.empty:
            fig_pt.add_trace(
                go.Scatter(
                    x=sells["entry_time"],
                    y=sells["entry_price"],
                    mode="markers",
                    marker_symbol="triangle-down",
                    marker_color=NEG,
                    name="Sell short (entry)",
                )
            )
        if {"exit_time", "exit_price"}.issubset(trades_df.columns):
            fig_pt.add_trace(
                go.Scatter(
                    x=trades_df["exit_time"],
                    y=trades_df["exit_price"],
                    mode="markers",
                    marker_symbol="triangle-down",
                    marker_size=9,
                    marker_color="#EF4444",
                    name="Exit",
                )
            )

    fig_pt.update_layout(template=TPL, height=420, margin=dict(l=0, r=0, b=0, t=25))
    st.plotly_chart(fig_pt, use_container_width=True)
    st.markdown("---")

    # ② Equity | Drawdown | Fees ---------------------------------------------
    st.subheader("📈 Equity | Drawdown | Fees")
    tabs_eq = st.tabs(["Equity", "Drawdown", "Fees"])

    start_balance_series = pd.Series(equity_df["equity"].iloc[0], index=equity_df.index)
    eq_plot_df = pd.DataFrame(
        {"Equity": equity_df["equity"], "Start Balance": start_balance_series}
    ).dropna()

    tabs_eq[0].plotly_chart(
        px.line(
            eq_plot_df,
            x=eq_plot_df.index,
            y=["Equity", "Start Balance"],
            template=TPL,
            labels={"value": "Series value", "variable": "Series"},
        ),
        use_container_width=True,
    )

    if not equity_df.empty:
        dd = (equity_df["equity"].cummax() - equity_df["equity"]) / equity_df["equity"].cummax()
        tabs_eq[1].plotly_chart(
            px.area(x=dd.index, y=dd.values, template=TPL),
            use_container_width=True,
        )
    else:
        tabs_eq[1].warning("Equity data unavailable.")

    if "commission" in trades_df.columns:
        trades_df["comm_cum"] = trades_df["commission"].cumsum()
        fee_series = trades_df.set_index("exit_time")["comm_cum"]
    elif comm_total:
        fee_series = pd.Series(
            [0, -comm_total],
            index=[price_series.index[0], price_series.index[-1]],
        )
    else:
        fee_series = pd.Series(dtype=float)

    if not fee_series.empty:
        tabs_eq[2].plotly_chart(
            px.line(x=fee_series.index, y=fee_series.values, template=TPL),
            use_container_width=True,
        )
    else:
        tabs_eq[2].info("No commissions recorded.")
    st.markdown("---")

    # ③ Risk & Seasonality ----------------------------------------------------
    st.subheader("📊 Risk & Seasonality")
    roll = st.slider("Rolling window (days)", 5, 60, 20, key="roll")

    if strategy_returns.empty:
        st.info("Not enough data for risk calculations.")
    else:
        rvol   = strategy_returns.rolling(roll).std(ddof=0).mul(np.sqrt(252)).dropna()
        cov    = strategy_returns.rolling(roll).cov(benchmark_returns)
        rbeta  = (cov / benchmark_returns.rolling(roll).var(ddof=0)).dropna()
        rsharp = strategy_returns.rolling(roll).apply(lambda s: sharpe(s)).dropna()

        risk_tabs = st.tabs(["Distribution & VaR", "Rolling metrics", "Seasonality"])

        var5 = np.percentile(strategy_returns, 5)
        hist = px.histogram(strategy_returns, nbins=60, template=TPL, title="Return distribution")
        hist.add_vline(x=var5, line_color=NEG, annotation_text="VaR 5%")
        risk_tabs[0].plotly_chart(hist, use_container_width=True)

        fig_roll = go.Figure()
        fig_roll.add_trace(go.Scatter(x=rsharp.index, y=rsharp, name="Rolling Sharpe"))
        fig_roll.add_trace(go.Scatter(x=rvol.index,   y=rvol,   name="Rolling Volatility"))
        fig_roll.add_trace(go.Scatter(x=rbeta.index,  y=rbeta,  name="Rolling Beta"))
        fig_roll.update_layout(template=TPL, height=350, legend_orientation="h")
        risk_tabs[1].plotly_chart(fig_roll, use_container_width=True)

        wd_ret = strategy_returns.groupby(strategy_returns.index.weekday).mean() * 100
        week_bar = px.bar(
            x=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
            y=wd_ret.reindex(range(7)).fillna(0),
            template=TPL,
            title="Average return by weekday",
        )
        monthly_heat = (
            strategy_returns.resample("M").sum().to_frame("ret").assign(
                Year=lambda d: d.index.year,
                Month=lambda d: d.index.month_name().str[:3],
            )
        )
        pivot = monthly_heat.pivot(index="Year", columns="Month", values="ret").fillna(0)
        heatmap = px.imshow(
            pivot,
            color_continuous_scale="RdYlGn",
            template=TPL,
            title="Monthly return heatmap",
        )
        risk_tabs[2].plotly_chart(week_bar, use_container_width=True)
        risk_tabs[2].plotly_chart(heatmap,   use_container_width=True)
    st.markdown("---")

    # ④ Trades stats & allocation --------------------------------------------
    left_bot, right_bot = st.columns(2)

    with left_bot:
        st.subheader("Trades statistics")
        if not trades_df.empty:
            if "duration_h" not in trades_df.columns:
                trades_df["duration_h"] = (
                    trades_df["exit_time"] - trades_df["entry_time"]
                ).dt.total_seconds() / 3600.0
            st.plotly_chart(
                px.histogram(
                    trades_df, x="duration_h", nbins=40, template=TPL, title="Trade duration (h)"
                ),
                use_container_width=True,
            )
            wins = (trades_df["profit"] > 0).sum()
            losses = (trades_df["profit"] <= 0).sum()
            st.plotly_chart(
                px.pie(
                    values=[wins, losses],
                    names=["Wins", "Losses"],
                    template=TPL,
                    title="Win / Loss",
                ),
                use_container_width=True,
            )
        else:
            st.info("No trades to display.")

    with right_bot:
        st.subheader("Portfolio allocation")
        pos_df = result.get("positions_df", pd.DataFrame())
        if not pos_df.empty and {"symbol", "size"}.issubset(pos_df.columns):
            alloc = pos_df.groupby("symbol")["size"].sum().abs()
            alloc = alloc / alloc.sum()
        else:
            alloc = pd.Series({"BTC": 1.0})
        st.plotly_chart(
            px.pie(values=alloc.values, names=alloc.index, template=TPL),
            use_container_width=True,
        )
        st.table(alloc.rename("Weight"))
    st.markdown("---")

    # ⑤ Metrics, raw log & engine stats --------------------------------------
    tab_metrics, tab_log, tab_extra = st.tabs(["Key metrics", "Trades log", "Engine stats"])
    tab_metrics.dataframe(
        pd.DataFrame.from_dict({**kpi, **extra_stats}, orient="index", columns=["Value"])
    )

    if not trades_df.empty:
        cols_show = [
            "entry_time",
            "exit_time",
            "entry_price",
            "exit_price",
            "profit",
            "commission" if "commission" in trades_df.columns else None,
        ]
        cols_show = [c for c in cols_show if c and c in trades_df.columns]
        tab_log.dataframe(trades_df[cols_show], use_container_width=True, height=350)
    else:
        tab_log.info("No trades recorded.")

    tab_extra.code(
        "\n".join(line for line in log_text.splitlines() if "BacktestEngine:" in line),
        language="text",
    )
    if log_text:
        with st.expander("Full back‑test log"):
            st.code(log_text, language="text")


# ╭──────────────────────── sidebar (user inputs) ─────────────────────────────╮
strategies = discover_strategies("strategies")
if not strategies:
    st.error("No strategies found under /strategies — add at least one and reload.")
    st.stop()

with st.sidebar:
    st.header("Configuration")
    strat_name = st.selectbox("Strategy", list(strategies))
    info = strategies[strat_name]
    if info.doc:
        st.caption(info.doc)

    timeframe = st.selectbox("Timeframe", ["1min", "15min"])
    data_file = "BINANCE_BTCUSD, 1.csv" if timeframe == "1min" else "BINANCE_BTCUSD, 15.csv"
    st.write(f"Data file: **{data_file}**")

    st.subheader("Parameters")
    params: Dict[str, Any] = {}
    for field, ann in info.cfg_cls.__annotations__.items():
        if field in ("instrument_id", "bar_type"):
            continue
        default = get_field_default(info.cfg_cls, field)
        label = field.replace("_", " ").title()

        if isinstance(default, bool) or issub(ann, bool):
            params[field] = st.checkbox(label, value=bool(default) if default is not None else False)
        elif isinstance(default, int) or issub(ann, int):
            params[field] = st.number_input(label, value=int(default) if default is not None else 0,
                                            step=1, format="%d")
        elif isinstance(default, (float, Decimal)) or issub(ann, (float, Decimal)):
            params[field] = st.number_input(label, value=float(default) if default is not None else 0.0,
                                            format="%.6f")
        else:
            params[field] = st.text_input(label, value=str(default or ""))

    theme = "Light"
    TPL = "plotly_dark" if theme == "Dark" else "plotly_white"
    ACCENT, NEG = "#10B981", "#EF4444"

    st.markdown("---")
    run_bt = st.button("Run back‑test", type="primary")

# ╭──────────────────────── run back‑test on click ────────────────────────────╮
if run_bt:
    with st.spinner("Running back‑test… please wait"):
        log_stream = io.StringIO()
        with redirect_stdout(log_stream), redirect_stderr(log_stream):
            try:
                result = run_backtest(
                    info.strategy_cls,
                    info.cfg_cls,
                    params,
                    data_file,
                    actor_cls=DashboardPublisher,  # only if supported
                )
            except TypeError:  # actor_cls not accepted
                result = run_backtest(
                    info.strategy_cls,
                    info.cfg_cls,
                    params,
                    data_file,
                )
        log_text = log_stream.getvalue()

    draw_dashboard(result, log_text, TPL, ACCENT, NEG)
