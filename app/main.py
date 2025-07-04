# -*- coding: utf‑8 -*-

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
from datetime import datetime, timezone
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
from modules.data_connector import DataConnector
from modules.csv_data import load_ohlcv_csv
from datetime import timedelta

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
    /* compact dataframe cells */
    .stDataFrame tbody tr td { padding-top:2px; padding-bottom:2px; }
    /* tighter spacing for sidebar data source section */
    [data-testid="stSidebar"] .data-source-header {
        margin-bottom:-2.2rem;
    }
    [data-testid="stSidebar"] [data-testid="stTabs"] {
        margin-top:0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("NautilusTrader — dashboard")


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


def style_trades(df: pd.DataFrame) -> pd.DataFrame | pd.Styler:
    """Apply styling to the trades log table."""
    if "profit" in df.columns:
        styler = df.style.applymap(
            lambda v: f"color: {'#10B981' if v > 0 else '#EF4444'}",
            subset=["profit"],
        )
        return styler
    return df


# ╭──────────────────────── dashboard renderer ───────────────────────────────╮
def draw_dashboard(
    result: dict, log_text: str, TPL: str, ACCENT: str, NEG: str
) -> None:
    """
    Build the entire Streamlit dashboard.
    Five high‑level blocks are always present (with fallbacks if data missing).
    """

    # ── 0. basic run metadata (needed multiple times) ------------------------
    run_meta = {
        "Run ID": getattr(result, "run_id", uuid.uuid4()),
        "Run started": result.get("run_started", datetime.now(timezone.utc)),
        "Run finished": result.get("run_finished", datetime.now(timezone.utc)),
        "Elapsed time": result.get("elapsed", "—"),
        "Backtest start": result["price_df"].index[0],
        "Backtest end": result["price_df"].index[-1],
        "Backtest range": str(
            result["price_df"].index[-1] - result["price_df"].index[0]
        ),
        "Iterations": result.get("iterations", "—"),
        "Total events": result.get("total_events", "—"),
        "Total orders": result.get("orders_count", "—"),
        "Total positions": result.get("positions_count", "—"),
    }

    # ── 1. extract core DataFrames ------------------------------------------
    price_df = result["price_df"].copy()
    equity_df = result["equity_df"].copy()
    trades_df = result["trades_df"].copy()

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

    def max_dd_days(series: pd.Series) -> float:
        if series.empty:
            return np.nan
        running_max = series.cummax()
        dd = running_max - series
        end = dd.idxmax()
        start = series.loc[:end].idxmax()
        return (end - start).days

    comm_total = sum(result.get("commissions", {}).values())

    # Additional performance metrics
    period_seconds = (
        run_meta["Backtest end"] - run_meta["Backtest start"]
    ).total_seconds()
    total_return = (
        (equity_df["equity"].iloc[-1] - equity_df["equity"].iloc[0])
        / equity_df["equity"].iloc[0]
        if not equity_df.empty
        else np.nan
    )
    annual_return = (
        (1 + total_return) ** (365 * 24 * 3600 / period_seconds) - 1
        if period_seconds > 0 and not np.isnan(total_return)
        else np.nan
    )
    tim = (
        (
            trades_df["exit_time"] - trades_df["entry_time"]
        ).dt.total_seconds().sum()
        / period_seconds * 100
        if period_seconds > 0 and not trades_df.empty
        else np.nan
    )
    max_dd_pct = max_dd(equity_df["equity"]) * 100 if not equity_df.empty else np.nan
    pnl_dd_ratio = (
        (total_return * 100) / abs(max_dd_pct)
        if not np.isnan(max_dd_pct) and not np.isnan(total_return)
        else np.nan
    )
    max_dd_len = (
        max_dd_days(equity_df["equity"]) if not equity_df.empty else np.nan
    )

    kpi = {
        "PnL ($)": result.get("metrics", {}).get("total_profit", np.nan),
        "PnL (%)": (
            (equity_df["equity"].iloc[-1] - equity_df["equity"].iloc[0])
            / equity_df["equity"].iloc[0]
            if not equity_df.empty
            else np.nan
        ),
        "Win Rate": result.get("metrics", {}).get("win_rate", np.nan),
        "Sharpe": sharpe(strategy_returns),
        "Sortino": sortino(strategy_returns),
        "Max DD (%)": max_dd_pct,
        "Max DD (days)": max_dd_len,
        "Profit Factor": result.get("metrics", {}).get("profit_factor", np.nan),
        "Volatility (252d)": (
            strategy_returns.std(ddof=0) * np.sqrt(252)
            if not strategy_returns.empty
            else np.nan
        ),
        "Annual Return": annual_return,
        "Profit/DD": pnl_dd_ratio,
        "Time in Market": tim,
    }
    KPI_ICONS = {
        "PnL ($)": "💰",
        "PnL (%)": "📈",
        "Win Rate": "🏆",
        "Sharpe": "⚖️",
        "Sortino": "📐",
        "Max DD (%)": "📉",
        "Max DD (days)": "🕳️",
        "Profit Factor": "🚀",
        "Volatility (252d)": "📊",
        "Annual Return": "📅",
        "Profit/DD": "⚡",
        "Time in Market": "⏱️",
    }

    KPI_TOOLTIPS = {
        "PnL ($)": "Net profit in base currency",
        "PnL (%)": "Total return over the test period",
        "Annual Return": "Compound annual growth rate",
        "Profit/DD": "Profit to drawdown ratio",
        "Time in Market": "Percentage of time with open positions",
        "Max DD (%)": "Maximum equity drawdown in percent",
        "Max DD (days)": "Duration of the largest drawdown",
        "Sharpe": "Risk-adjusted return ratio",
        "Sortino": "Downside risk-adjusted return",
        "Profit Factor": "Gross profit divided by gross loss",
        "Win Rate": "Share of profitable trades",
    }

    extra_stats = parse_extra_stats(log_text)

    # ╭──────────────────── 📄  RUN METADATA (collapsed) ──────────────────────╮
    with st.expander(
        f"📄 Мetadata — ID: {run_meta['Run ID']}",
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

    # ╭──────────────────── 📊 OVERVIEW ─────────────────────────────────────╮
    with st.container(border=True):
        st.subheader("📊 Overview")
        overview_keys = [
            "PnL ($)",
            "PnL (%)",
            "Annual Return",
            "Max DD (%)",
            "Max DD (days)",
            "Win Rate",
            "Sharpe",
            "Sortino",
            "Profit Factor",
        ]
        ocols = st.columns(len(overview_keys))
        pct_labels = {
            "PnL (%)",
            "Win Rate",
            "Max DD (%)",
            "Annual Return",
            "Time in Market",
        }
        for key, col in zip(overview_keys, ocols):
            val = kpi.get(key)
            icon = KPI_ICONS.get(key, "")
            tip = KPI_TOOLTIPS.get(key, "")
            is_pct = key in pct_labels
            precision = 0 if key == "Max DD (days)" else 2
            text = _fmt_pct(val) if is_pct else _fmt_num(val, precision)
            col.metric(f"{icon} {key}", text, help=tip)

    # ╭──────────────────── 💹 ACCOUNT & Performance ────────────────────────────╮

    _fmt_pct = lambda v: (
        "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:+.2%}"
    )
    _fmt_num = lambda v, p=2: (
        "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:,.{p}f}"
    )

    with st.container(border=True):
        st.subheader("💹 Account & Performance")

        # ------------------------------------------------------------------
        # We analyse the entire back-test by default.
        # If strategy_returns is empty, returns_view will also be empty.
        # ------------------------------------------------------------------
        returns_view = strategy_returns

        # ── Tabs -----------------------------------------------------------
        perf_tabs = st.tabs(
            [
                "Summary",
                "Balances & Fees",
                "PnL",
                "Return & Risk",
                "General",
            ]
        )

        # ── Tab 0: Summary -------------------------------------------------
        with perf_tabs[0]:
            st.subheader("⚡ Back-test summary")
            hdr = st.columns(4)
            hdr[0].metric("Started", _fmt_dt(run_meta["Run started"]))
            hdr[1].metric("Finished", _fmt_dt(run_meta["Run finished"]))
            hdr[2].metric("Elapsed", run_meta["Elapsed time"])
            hdr[3].metric("Orders", run_meta["Total orders"])

            # KPI grid
            kcols = st.columns(len(kpi))
            summary_pct_labels = {
                "PnL (%)",
                "Win Rate",
                "Max DD (%)",
                "Annual Return",
                "Time in Market",
            }
            for (label, value), col in zip(kpi.items(), kcols):
                icon = KPI_ICONS.get(label, "")
                is_pct = label in summary_pct_labels
                text = _fmt_pct(value) if is_pct else _fmt_num(value)
                col.metric(f"{icon} {label}", text)

        # === Tab 1: Balances & Fees ==============================================
        with perf_tabs[1]:
            bal_cols = st.columns(4)
            initial_bal = result.get("initial_balances", {"USDT": 10_000, "BTC": 1})
            final_bal = result.get("final_balances", {"USDT": 9_872.06, "BTC": 1})
            unrealised = result.get("unrealised_pnl")

            bal_cols[0].metric("Initial USDT", _fmt_usd(initial_bal.get("USDT")))
            bal_cols[1].metric("Initial BTC", _fmt_btc(initial_bal.get("BTC")))
            bal_cols[2].metric("Final USDT", _fmt_usd(final_bal.get("USDT")))
            bal_cols[3].metric("Final BTC", _fmt_btc(final_bal.get("BTC")))

            fee_cols = st.columns(2)
            fee_cols[0].metric(
                "Total fees", _fmt_usd(-comm_total) if comm_total else "—"
            )
            fee_cols[1].metric(
                "Unrealised PnL",
                _fmt_num(unrealised) if unrealised is not None else "—",
            )

        # === Tab 2: PnL ==========================================================
        with perf_tabs[2]:
            pnl_metrics = result.get("metrics", {})
            btc, usd = pnl_metrics.get("btc", {}), pnl_metrics.get("usdt", {})

            mcols = st.columns(4)
            mcols[0].metric("BTC PnL", btc.get("total", "—"))
            mcols[1].metric("BTC PnL %", btc.get("pct", "—"))
            mcols[2].metric("USDT PnL", usd.get("total", kpi.get("PnL ($)", "—")))
            mcols[3].metric("USDT PnL %", usd.get("pct", kpi.get("PnL (%)", "—")))

            st.markdown("#### Raw PnL metrics")
            raw = [
                ("Max win BTC", btc.get("max_win")),
                ("Avg win BTC", btc.get("avg_win")),
                ("Min win BTC", btc.get("min_win")),
                ("Max loss BTC", btc.get("max_loss")),
                ("Avg loss BTC", btc.get("avg_loss")),
                ("Min loss BTC", btc.get("min_loss")),
                ("Expectancy BTC", btc.get("expectancy")),
                ("Win rate BTC", btc.get("win_rate")),
                ("Max win USDT", usd.get("max_win")),
                ("Avg win USDT", usd.get("avg_win")),
                ("Min win USDT", usd.get("min_win")),
                ("Max loss USDT", usd.get("max_loss")),
                ("Avg loss USDT", usd.get("avg_loss")),
                ("Min loss USDT", usd.get("min_loss")),
                ("Expectancy USDT", usd.get("expectancy")),
                ("Win rate USDT", usd.get("win_rate", kpi.get("Win Rate"))),
            ]
            grid = st.columns(4)
            for (lbl, val), col in zip(raw, grid * ((len(raw) // 4) + 1)):
                pct = "rate" in lbl.lower()
                col.metric(lbl, _fmt_pct(val) if pct else _fmt_num(val))

        # === Tab 3: Return & Risk ==============================================
        with perf_tabs[3]:
            if returns_view.empty:
                st.info("Not enough data to compute return stats.")
            else:
                stats = {
                    "Volatility (252d)": kpi.get("Volatility (252d)"),
                    "Avg daily return": returns_view.mean(),
                    "Avg loss (daily)": returns_view[returns_view < 0].mean(),
                    "Avg win (daily)": returns_view[returns_view > 0].mean(),
                    "Sharpe (252d)": kpi.get("Sharpe"),
                    "Sortino (252d)": kpi.get("Sortino"),
                    "Profit factor": kpi.get("Profit Factor"),
                    "Risk / Return": (
                        abs(kpi.get("Max DD (%)", np.nan)) / 100 / kpi.get("PnL (%)")
                        if kpi.get("PnL (%)") not in (None, 0, np.nan)
                        else np.nan
                    ),
                }
                rcols = st.columns(4)
                for (lbl, val), col in zip(stats.items(), rcols * 2):
                    pct = "%" in lbl or lbl.endswith("(%)")
                    col.metric(lbl, _fmt_pct(val) if pct else _fmt_num(val, 4))

        # === Tab 4: General =====================================================
        with perf_tabs[4]:
            long_ratio = result.get("long_ratio")
            if long_ratio is None and not trades_df.empty and "side" in trades_df:
                long_ratio = (trades_df["side"].str.upper() == "BUY").mean()

            st.metric(
                "Long ratio", _fmt_pct(long_ratio) if long_ratio is not None else "—"
            )
            st.metric("Positions", run_meta.get("Total positions", "—"))
            st.metric("Trades", len(trades_df) if not trades_df.empty else 0)

    # ① Price & Trades --------------------------------------------------------
    st.subheader("📉 Price & Trades")
    fig_pt = go.Figure()
    fig_pt = go.Figure()
    fig_pt.add_trace(
        go.Scatter(x=price_series.index, y=price_series, mode="lines", name="Price")
    )

    if not trades_df.empty:
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
    st.subheader("📈 Equity & Drawdown")
    col_eq, col_dd = st.columns(2)

    if not equity_df.empty:
        first_equity = (
            equity_df["equity"].dropna().iloc[0]
            if not equity_df["equity"].dropna().empty
            else 0.0
        )
        start_balance_series = pd.Series(first_equity, index=equity_df.index)
        eq_plot_df = pd.DataFrame(
            {"Equity": equity_df["equity"], "Start Balance": start_balance_series}
        ).dropna()

        col_eq.plotly_chart(
            px.line(
                eq_plot_df,
                x=eq_plot_df.index,
                y=["Equity", "Start Balance"],
                template=TPL,
                labels={"value": "Series value", "variable": "Series"},
            ),
            use_container_width=True,
        )
    else:
        col_eq.info("Equity data unavailable.")

    if not equity_df.empty:
        dd = (equity_df["equity"].cummax() - equity_df["equity"]) / equity_df[
            "equity"
        ].cummax()
        col_dd.plotly_chart(
            px.area(x=dd.index, y=dd.values, template=TPL),
            use_container_width=True,
        )
    else:
        col_dd.warning("Equity data unavailable.")

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

    with st.expander("Fees over time", expanded=False):
        if not fee_series.empty:
            st.plotly_chart(
                px.line(x=fee_series.index, y=fee_series.values, template=TPL),
                use_container_width=True,
            )
        else:
            st.info("No commissions recorded.")
    st.markdown("---")

    # ③ Risk & Seasonality ----------------------------------------------------
    st.subheader("📊 Risk & Seasonality")

    # окно скольжения берём на весь доступный период
    roll = len(strategy_returns) if not strategy_returns.empty else 1

    if strategy_returns.empty:
        st.info("Not enough data for risk calculations.")
    else:
        rvol = strategy_returns.rolling(roll).std(ddof=0).mul(np.sqrt(252)).dropna()
        cov = strategy_returns.rolling(roll).cov(benchmark_returns)
        rbeta = (cov / benchmark_returns.rolling(roll).var(ddof=0)).dropna()
        rsharp = strategy_returns.rolling(roll).apply(lambda s: sharpe(s)).dropna()

        with st.expander("Advanced analysis", expanded=False):
            risk_tabs = st.tabs(["Distribution & VaR", "Rolling metrics", "Seasonality"])

            # ── Distribution & VaR ───────────────────────────────────────────────
            var5 = np.percentile(strategy_returns, 5)
            hist = px.histogram(
                strategy_returns, nbins=60, template=TPL, title="Return distribution"
            )
            hist.add_vline(x=var5, line_color=NEG, annotation_text="VaR 5%")
            risk_tabs[0].plotly_chart(hist, use_container_width=True)

            # ── Rolling metrics (по факту – одно значение на конец периода) ─────
            fig_roll = go.Figure()
            fig_roll.add_trace(
                go.Scatter(x=rsharp.index, y=rsharp, name="Sharpe (full window)")
            )
            fig_roll.add_trace(
                go.Scatter(x=rvol.index, y=rvol, name="Volatility (full window)")
            )
            fig_roll.add_trace(
                go.Scatter(x=rbeta.index, y=rbeta, name="Beta (full window)")
            )
            fig_roll.update_layout(template=TPL, height=350, legend_orientation="h")
            risk_tabs[1].plotly_chart(fig_roll, use_container_width=True)

            # ── Seasonality ─────────────────────────────────────────────────────
            wd_ret = strategy_returns.groupby(strategy_returns.index.weekday).mean() * 100
            week_bar = px.bar(
                x=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
                y=wd_ret.reindex(range(7)).fillna(0),
                template=TPL,
                title="Average return by weekday",
            )

            monthly_heat = (
                strategy_returns.resample("M")
                .sum()
                .to_frame("ret")
                .assign(
                    Year=lambda d: d.index.year,
                    Month=lambda d: d.index.month_name().str[:3],
                )
            )
            pivot = monthly_heat.pivot(index="Year", columns="Month", values="ret").fillna(
                0
            )
            heatmap = px.imshow(
                pivot,
                color_continuous_scale="RdYlGn",
                template=TPL,
                title="Monthly return heatmap",
            )

            risk_tabs[2].plotly_chart(week_bar, use_container_width=True)
            risk_tabs[2].plotly_chart(heatmap, use_container_width=True)

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
                    trades_df,
                    x="duration_h",
                    nbins=40,
                    template=TPL,
                    title="Trade duration (h)",
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
    tab_metrics, tab_log, tab_extra = st.tabs(
        ["Key metrics", "Trades log", "Engine stats"]
    )
    tab_metrics.dataframe(
        pd.DataFrame.from_dict(
            {**kpi, **extra_stats}, orient="index", columns=["Value"]
        ).style.format("{:.4f}")
    )

    with tab_log:
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
            with st.expander("List of trades", expanded=False):
                st.dataframe(
                    style_trades(trades_df[cols_show]),
                    use_container_width=True,
                    height=350,
                )
        else:
            st.info("No trades recorded.")

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

    connector = DataConnector()

    # ── Data source tabs ────────────────────────────────────────────────
    st.markdown(
        "<h3 class='data-source-header'>Data source</h3>",
        unsafe_allow_html=True,
    )
    csv_path = None
    symbol = None
    exchange = None
    csv_exchs = connector.get_exchanges("CSV")
    csv_syms = connector.get_symbols("CSV", csv_exchs[0] if csv_exchs else None)
    csv_tfs = connector.get_timeframes("CSV")
    ch_exchs = connector.get_exchanges("ClickHouse")
    ch_tfs = connector.get_timeframes("ClickHouse")
    tf_csv = csv_tfs[0] if csv_tfs else ""
    tf_ch = ch_tfs[0] if ch_tfs else ""

    # Default date range based on available CSV data
    if csv_exchs and csv_syms and tf_csv:
        try:
            _default_path = connector.get_csv_path(csv_exchs[0], csv_syms[0], tf_csv)
            _df_info = load_ohlcv_csv(_default_path)
            start_csv = _df_info.index[0].date()
            end_csv = _df_info.index[-1].date()
        except Exception:
            start_csv = datetime.now(timezone.utc).date() - timedelta(days=30)
            end_csv = datetime.now(timezone.utc).date()
    else:
        start_csv = datetime.now(timezone.utc).date() - timedelta(days=30)
        end_csv = datetime.now(timezone.utc).date()
    start_ch = start_csv
    end_ch = end_csv
    data_src = st.radio("Data source", ["CSV", "ClickHouse"], horizontal=True, key="data_src_tab")
    tab_csv, tab_ch = st.tabs(["CSV", "ClickHouse"])
    with tab_csv:
        row1 = st.columns(3)
        exchange_csv = csv_exchs[0] if csv_exchs else ""
        symbol_csv = csv_syms[0] if csv_syms else ""

        row1[0].text_input(
            "Exchange",
            exchange_csv,
            disabled=True,
            key="csv_exch",
        )
        row1[1].text_input(
            "Symbol",
            symbol_csv,
            disabled=True,
            key="csv_sym",
        )
        tf_csv = row1[2].selectbox(
            "TimeFrame",
            csv_tfs,
            index=0,
            key="csv_tf",
        )

        row2 = st.columns(2)
        start_csv = row2[0].date_input(
            "Date from",
            start_csv,
            key="csv_start",
        )
        end_csv = row2[1].date_input(
            "Date to",
            end_csv,
            key="csv_end",
        )

        if csv_exchs and csv_syms:
            csv_path = connector.get_csv_path(csv_exchs[0], csv_syms[0], tf_csv)
        else:
            csv_path = ""
        st.write(f"Data file: **{csv_path}**")
    with tab_ch:
        row1 = st.columns(3)
        exchange = row1[0].selectbox("Exchange", ch_exchs, key="ch_exch")
        symbol = row1[1].text_input("Symbol", "BTCUSDT", key="ch_sym")
        tf_ch = row1[2].selectbox("TimeFrame", ch_tfs, key="ch_tf")
        row2 = st.columns(2)
        start_ch = row2[0].date_input("Date from", start_ch, key="ch_start")
        end_ch = row2[1].date_input("Date to", end_ch, key="ch_end")
    st.subheader("Parameters")
    params: Dict[str, Any] = {}
    for field, ann in info.cfg_cls.__annotations__.items():
        if field in ("instrument_id", "bar_type"):
            continue
        default = get_field_default(info.cfg_cls, field)
        label = field.replace("_", " ").title()

        if isinstance(default, bool) or issub(ann, bool):
            params[field] = st.checkbox(
                label, value=bool(default) if default is not None else False
            )
        elif isinstance(default, int) or issub(ann, int):
            params[field] = st.number_input(
                label,
                value=int(default) if default is not None else 0,
                step=1,
                format="%d",
            )
        elif isinstance(default, (float, Decimal)) or issub(ann, (float, Decimal)):
            params[field] = st.number_input(
                label,
                value=float(default) if default is not None else 0.0,
                format="%.6f",
            )
        else:
            params[field] = st.text_input(label, value=str(default or ""))

    theme = st.selectbox("Theme", ["Light", "Dark"], index=0)
    TPL = "plotly_dark" if theme == "Dark" else "plotly_white"
    ACCENT, NEG = ("#10B981", "#EF4444") if theme == "Light" else ("#22D3EE", "#F43F5E")

    st.markdown("---")
    run_bt = st.button("Run back‑test", key="run_backtest")
    if data_src == "CSV":
        data_source = "CSV"
        data_spec = csv_path
        start_dt = pd.to_datetime(start_csv, utc=True)
        end_dt = pd.to_datetime(end_csv, utc=True) + pd.Timedelta(days=1)
    else:
        data_source = "ClickHouse"
        data_spec = {
            "exchange": exchange,
            "symbol": symbol,
            "timeframe": (tf_ch[:-3] + "m") if tf_ch.endswith("min") else tf_ch,
            "start": datetime.combine(start_ch, datetime.min.time()),
            "end": datetime.combine(end_ch, datetime.min.time()),
        }
        start_dt = datetime.combine(start_ch, datetime.min.time())
        end_dt = datetime.combine(end_ch, datetime.min.time())
    with st.spinner("Running back‑test… please wait"):
        connector = DataConnector()
        data_df = connector.load(data_source, data_spec, start=start_dt, end=end_dt)

        if data_df.empty:
            st.error("No data found for the selected date range.")
            st.stop()

        log_stream = io.StringIO()
        with redirect_stdout(log_stream), redirect_stderr(log_stream):
            try:
                result = run_backtest(
                    info.strategy_cls,
                    info.cfg_cls,
                    params,
                    data_df,
                    actor_cls=DashboardPublisher,  # only if supported
                )
            except TypeError:  # actor_cls not accepted
                result = run_backtest(
                    info.strategy_cls,
                    info.cfg_cls,
                    params,
                    data_df,
                )
        log_text = log_stream.getvalue()

draw_dashboard(result, log_text, TPL, ACCENT, NEG)
