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
try:
    from lightweight_charts_v5 import lightweight_charts_v5_component as lwc
except ImportError:
    lwc = None
try:
    import extra_streamlit_components as stx
except ImportError:
    stx = None

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
    <style >
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
    /* make sidebar buttons and tab bar more compact */
    div[data-testid="stSidebar"] button {
        padding:0.25rem 0.75rem;
        font-size:0.85rem;
    }
    div[data-testid="stSidebar"] .stx-tab-bar button {
        padding-top:0.25rem;
        padding-bottom:0.25rem;
        margin-right:0.25rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("NautilusTrader — dashboard")

# Store back-test outputs so they persist across reruns
if "bt_result" not in st.session_state:
    st.session_state["bt_result"] = None
    st.session_state["bt_log"] = ""
    st.session_state["bt_tpl"] = "plotly_white"
    st.session_state["bt_accent"] = "#10B981"
    st.session_state["bt_neg"] = "#EF4444"


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
        styler = df.style.map(
            lambda v: f"color: {'#10B981' if v > 0 else '#EF4444'}",
            subset="profit",
        )
        return styler
    return df


def rebuild_equity_curve(
    price_index: pd.DatetimeIndex,
    trades_df: pd.DataFrame,
    start_balance: float = 10_000,
) -> pd.Series:
    """Construct a basic equity curve from trade profits."""
    price_index = pd.to_datetime(price_index)
    if getattr(price_index, "tz", None) is not None:
        price_index = price_index.tz_convert(None)

    if trades_df.empty:
        return pd.Series(start_balance, index=price_index)

    pnl = trades_df.sort_values("exit_time").set_index("exit_time")["profit"].cumsum()
    if getattr(pnl.index, "tz", None) is not None:
        pnl.index = pnl.index.tz_convert(None)
    pnl = pnl.reindex(price_index, method="ffill").fillna(0.0)
    equity = start_balance + pnl
    # Ensure all timestamps present
    equity = equity.reindex(price_index, method="ffill")
    equity.iloc[0] = start_balance
    return equity


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
        if not df.empty and getattr(df.index, "tz", None) is not None:
            df.index = df.index.tz_convert(None)

    for col in ("entry_time", "exit_time"):
        if col in trades_df.columns:
            trades_df[col] = pd.to_datetime(trades_df[col])
            if getattr(trades_df[col].dt, "tz", None) is not None:
                trades_df[col] = trades_df[col].dt.tz_convert(None)

    price_series = price_df["close"] if "close" in price_df else price_df.iloc[:, 0]

    if equity_df.empty and not price_series.empty:
        start_balance = (
            result.get("initial_balances", {}).get("USDT")
            or 10_000
        )
        equity_df = pd.DataFrame(
            {"equity": rebuild_equity_curve(price_df.index, trades_df, start_balance)}
        )

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

    def longest_dd_days(series: pd.Series) -> float:
        """Return the longest time in days between equity highs."""
        if series.empty:
            return np.nan
        peak_idx = series.index[0]
        peak_val = series.iloc[0]
        longest = 0
        for idx in series.index[1:]:
            val = series.loc[idx]
            if val >= peak_val:
                longest = max(longest, (idx - peak_idx).days)
                peak_idx = idx
                peak_val = val
        longest = max(longest, (series.index[-1] - peak_idx).days)
        return float(longest)

    def dd_periods(series: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
        """Return start and end timestamps for all drawdown phases."""
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

    def avg_dd_days(series: pd.Series) -> float:
        """Average drawdown duration in days."""
        durs = [(e - s).days for s, e in dd_periods(series)]
        return float(np.mean(durs)) if durs else np.nan

    def total_dd_days(series: pd.Series) -> float:
        """Total time spent in drawdowns in days."""
        durs = [(e - s).days for s, e in dd_periods(series)]
        return float(np.sum(durs)) if durs else np.nan

    def max_dd_abs(series: pd.Series) -> float:
        """Return the maximum drawdown in currency units."""
        if series.empty:
            return np.nan
        return float((series.cummax() - series).max())

    def longest_dd_span(series: pd.Series) -> tuple[pd.Timestamp, pd.Timestamp]:
        """Return start and end timestamps of the longest drawdown."""
        if series.empty:
            return pd.NaT, pd.NaT
        peak_idx = series.index[0]
        peak_val = series.iloc[0]
        longest = pd.Timedelta(0)
        start = peak_idx
        end = peak_idx
        for idx in series.index[1:]:
            val = series.loc[idx]
            if val >= peak_val:
                if idx - peak_idx > longest:
                    longest = idx - peak_idx
                    start = peak_idx
                    end = idx
                peak_idx = idx
                peak_val = val
        if series.index[-1] - peak_idx > longest:
            start = peak_idx
            end = series.index[-1]
        return start, end

    def current_dd_days(series: pd.Series) -> float:
        """Duration in days of the current drawdown."""
        if series.empty:
            return np.nan
        cummax = series.cummax()
        if series.iloc[-1] >= cummax.iloc[-1]:
            return 0.0
        peaks = series[cummax.diff().fillna(0) > 0]
        last_peak = peaks.index[-1] if not peaks.empty else series.index[0]
        return float((series.index[-1] - last_peak).days)

    def avg_dd_pct(series: pd.Series) -> float:
        """Average drawdown magnitude in percent."""
        dd = (series.cummax() - series) / series.cummax()
        dds = [dd.loc[s:e].max() for s, e in dd_periods(series)]
        return float(np.mean(dds) * 100) if dds else np.nan

    def dd_count(series: pd.Series) -> int:
        """Number of drawdown periods."""
        return len(dd_periods(series))


    def _fmt_pct(v: float | None) -> str:
        """Return value formatted as a percentage or an em dash."""
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "—"
        return f"{v:+.2%}"

    def _fmt_num(v: float | None, p: int = 2) -> str:
        """Return number with thousands separator or an em dash."""
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "—"
        return f"{v:,.{p}f}"

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

    # ╭──────────────────── 💹 ACCOUNT & Performance ────────────────────────────╮

    with st.container(border=True):
        st.subheader("💹 Account & Performance")

        start_default = price_df.index[0]
        end_default = price_df.index[-1]
        period_start, period_end = st.slider(
            "Analysis period",
            min_value=start_default.to_pydatetime(),
            max_value=end_default.to_pydatetime(),
            value=(start_default.to_pydatetime(), end_default.to_pydatetime()),
            format="YYYY-MM-DD",
        )

        price_df = price_df.loc[period_start:period_end]
        equity_df = equity_df.loc[period_start:period_end]
        trades_df = trades_df[
            (trades_df["entry_time"] >= period_start)
            & (trades_df["entry_time"] <= period_end)
        ]

        price_series = (
            price_df["close"] if "close" in price_df else price_df.iloc[:, 0]
        )
        returns_view = (
            equity_df["equity"].pct_change().dropna()
            if not equity_df.empty
            else pd.Series(dtype=float)
        )
        strategy_returns = returns_view
        benchmark_returns = price_series.pct_change().dropna()

        comm_total = sum(result.get("commissions", {}).values())

        period_seconds = (
            price_df.index[-1] - price_df.index[0]
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
        avg_trade_h = (
            (
                trades_df["exit_time"] - trades_df["entry_time"]
            ).dt.total_seconds().mean()
            / 3600
            if not trades_df.empty
            else np.nan
        )
        max_dd_pct = max_dd(equity_df["equity"]) * 100 if not equity_df.empty else np.nan
        if not np.isnan(max_dd_pct) and not np.isnan(total_return) and max_dd_pct != 0:
            pnl_dd_ratio = (total_return * 100) / abs(max_dd_pct)
        else:
            pnl_dd_ratio = np.nan
        if not equity_df.empty:
            bh_final = (
                price_series.reindex(equity_df.index, method="ffill").iloc[-1]
                / price_series.iloc[0]
                * equity_df["equity"].iloc[0]
            )
            bh_edge = (equity_df["equity"].iloc[-1] / bh_final - 1) * 100
        else:
            bh_edge = np.nan
        longest_dd_len = (
            longest_dd_days(equity_df["equity"]) if not equity_df.empty else np.nan
        )
        avg_dd_len = avg_dd_days(equity_df["equity"]) if not equity_df.empty else np.nan
        total_dd_len = total_dd_days(equity_df["equity"]) if not equity_df.empty else np.nan
        max_dd_dollars = max_dd_abs(equity_df["equity"]) if not equity_df.empty else np.nan
        dd_start, dd_end = (
            longest_dd_span(equity_df["equity"]) if not equity_df.empty else (pd.NaT, pd.NaT)
        )
        current_dd_len = current_dd_days(equity_df["equity"]) if not equity_df.empty else np.nan
        avg_dd_pct_val = avg_dd_pct(equity_df["equity"]) if not equity_df.empty else np.nan
        dd_total = dd_count(equity_df["equity"]) if not equity_df.empty else 0

        var5 = np.percentile(returns_view, 5) if not returns_view.empty else np.nan
        cvar5 = (
            returns_view[returns_view <= var5].mean()
            if not returns_view.empty
            else np.nan
        )
        calmar = (
            annual_return / (abs(max_dd_pct) / 100)
            if max_dd_pct not in (0, np.nan) and not np.isnan(annual_return)
            else np.nan
        )
        romad = (
            total_return / (abs(max_dd_pct) / 100)
            if max_dd_pct not in (0, np.nan) and not np.isnan(total_return)
            else np.nan
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
            "Sharpe": sharpe(returns_view),
            "Sortino": sortino(returns_view),
            "Max DD (%)": max_dd_pct,
            "Longest DD (days)": longest_dd_len,
            "Avg DD (days)": avg_dd_len,
            "Total DD (days)": total_dd_len,
            "Max DD ($)": max_dd_dollars,
            "Profit Factor": result.get("metrics", {}).get("profit_factor", np.nan),
            "Volatility (252d)": (
                returns_view.std(ddof=0) * np.sqrt(252)
                if not returns_view.empty
                else np.nan
            ),
            "Annual Return": annual_return,
            "Profit/DD": pnl_dd_ratio,
            "Time in Market": tim,
            "Avg Trade (h)": avg_trade_h,
            "Edge vs B&H (%)": bh_edge,
            "Current DD (days)": current_dd_len,
            "Avg DD (%)": avg_dd_pct_val,
            "Drawdowns": dd_total,
            "VaR 5%": var5,
            "CVaR 5%": cvar5,
            "Calmar": calmar,
            "RoMaD": romad,
        }
        KPI_ICONS = {
            "PnL ($)": "💰",
            "PnL (%)": "📈",
            "Win Rate": "🏆",
            "Sharpe": "⚖️",
            "Sortino": "📐",
            "Max DD (%)": "📉",
            "Longest DD (days)": "🕳️",
            "Avg DD (days)": "⏱️",
            "Total DD (days)": "🕒",
            "Max DD ($)": "💸",
            "Profit Factor": "🚀",
            "Volatility (252d)": "📊",
            "Annual Return": "📅",
            "Profit/DD": "⚡",
            "Time in Market": "⏱️",
            "Avg Trade (h)": "⏳",
            "Edge vs B&H (%)": "🏁",
            "Current DD (days)": "📉",
            "Avg DD (%)": "📉",
            "Drawdowns": "🔻",
            "VaR 5%": "🚩",
            "CVaR 5%": "🚩",
            "Calmar": "🏊",
            "RoMaD": "📏",
        }

        KPI_TOOLTIPS = {
            "PnL ($)": "Net profit in base currency",
            "PnL (%)": "Total return over the test period",
            "Annual Return": "Compound annual growth rate",
            "Profit/DD": "Profit to drawdown ratio",
            "Time in Market": "Percentage of time with open positions",
            "Max DD (%)": "Maximum equity drawdown in percent",
            "Longest DD (days)": "Longest drawdown period in days",
            "Avg DD (days)": "Average duration of drawdowns in days",
            "Total DD (days)": "Total time spent in drawdowns",
            "Max DD ($)": "Largest peak-to-trough drop in account balance",
            "Sharpe": "Risk-adjusted return ratio",
            "Sortino": "Downside risk-adjusted return",
            "Profit Factor": "Gross profit divided by gross loss",
            "Win Rate": "Share of profitable trades",
            "Avg Trade (h)": "Average trade duration in hours",
            "Edge vs B&H (%)": "Strategy performance relative to buy-and-hold",
            "Current DD (days)": "Days since last equity peak",
            "Avg DD (%)": "Average drawdown depth",
            "Drawdowns": "Number of drawdown periods",
            "VaR 5%": "Value at Risk at 5%",
            "CVaR 5%": "Conditional VaR at 5%",
            "Calmar": "CAGR divided by max drawdown",
            "RoMaD": "Return over maximum drawdown",
        }

        KPI_PCT_LABELS = {
            "PnL (%)",
            "Win Rate",
            "Max DD (%)",
            "Annual Return",
            "Time in Market",
            "Edge vs B&H (%)",
            "Avg DD (%)",
            "VaR 5%",
            "CVaR 5%",
        }

        # ── Tabs -----------------------------------------------------------
        perf_tabs = st.tabs(
            [
                "Summary",
                "Balances & Fees",
                "PnL",
                "Return & Risk",
                "General",
                "Time in Market",
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


            perf_order = [
                "PnL ($)",
                "PnL (%)",
                "Annual Return",
                "Profit/DD",
                "Edge vs B&H (%)",
                "Calmar",
                "RoMaD",
            ]
            trade_order = [
                "Win Rate",
                "Profit Factor",
                "Avg Trade (h)",
            ]
            risk_order = [
                "Sharpe",
                "Sortino",
                "Max DD (%)",
                "Longest DD (days)",
                "Avg DD (days)",
                "Total DD (days)",
                "Max DD ($)",
                "Volatility (252d)",
                "Current DD (days)",
                "Avg DD (%)",
                "Drawdowns",
                "VaR 5%",
                "CVaR 5%",
            ]

            groups = [
                ("Return", perf_order),
                ("Trade quality", trade_order),
                ("Risk", risk_order),
            ]

            for gtitle, order in groups:
                st.markdown(f"**{gtitle} metrics**")
                cols = st.columns(len(order))
                for label, col in zip(order, cols):
                    value = kpi.get(label)
                    icon = KPI_ICONS.get(label, "")
                    tip = KPI_TOOLTIPS.get(label, "")
                    is_pct = label in KPI_PCT_LABELS
                    precision = 0 if label in {"Longest DD (days)", "Avg DD (days)", "Total DD (days)"} else 2
                    text = _fmt_pct(value) if is_pct else _fmt_num(value, precision)
                    col.metric(f"{icon} {label}", text, help=tip)

                if gtitle == "Return" and "Edge vs B&H (%)" in order:
                    edge = kpi.get("Edge vs B&H (%)")
                    if edge is not None and not (isinstance(edge, float) and np.isnan(edge)):
                        ratio = min(max((edge + 100) / 200, 0), 1)
                        st.progress(ratio)

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

        # === Tab 5: Time in Market ===========================================
        with perf_tabs[5]:
            tim = kpi.get("Time in Market")
            if tim is not None and not (isinstance(tim, float) and np.isnan(tim)):
                st.metric("⏱️ Time in Market", _fmt_pct(tim))
                st.progress(float(tim) / 100)
            else:
                st.info("Time-in-market unavailable.")

    # ① Price & Trades --------------------------------------------------------
    st.subheader("📉 Price & Trades")

    controls = st.columns(3)
    show_long = controls[0].checkbox("Show longs", value=True)
    show_short = controls[1].checkbox("Show shorts", value=True)
    show_exit = controls[2].checkbox("Show exits", value=True)

    chart_opts = st.columns(3)
    price_style = chart_opts[0].selectbox(
        "Price style",
        ["Line", "Candlesticks"],
    )
    show_sma = chart_opts[1].checkbox("Show 50 SMA", value=False)
    show_ema = chart_opts[2].checkbox("Show 21 EMA", value=False)

    has_volume = "volume" in price_df.columns

    if lwc is None:
        st.error(
            "streamlit-lightweight-charts-v5 is required for this section."
        )
    else:
        series = []

        if price_style == "Candlesticks" and {"open", "high", "low", "close"}.issubset(price_df.columns):
            price_data = [
                {
                    "time": int(idx.timestamp()),
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                }
                for idx, row in price_df.iterrows()
            ]
            main_series = {"type": "Candlestick", "data": price_data}
        else:
            price_data = [
                {"time": int(idx.timestamp()), "value": float(val)}
                for idx, val in price_series.items()
            ]
            main_series = {"type": "Line", "data": price_data}

        markers = []
        if not trades_df.empty:
            buys = trades_df[trades_df.get("entry_side", "").str.upper() == "LONG"]
            sells = trades_df[trades_df.get("entry_side", "").str.upper() == "SELL"]

            if show_long and not buys.empty:
                for _, row in buys.iterrows():
                    markers.append(
                        {
                            "time": int(pd.to_datetime(row["entry_time"]).timestamp()),
                            "position": "belowBar",
                            "color": ACCENT,
                            "shape": "arrowUp",
                            "text": f"Buy\n@ {row['entry_price']:.2f}",
                        }
                    )
            if show_short and not sells.empty:
                for _, row in sells.iterrows():
                    markers.append(
                        {
                            "time": int(pd.to_datetime(row["entry_time"]).timestamp()),
                            "position": "aboveBar",
                            "color": NEG,
                            "shape": "arrowDown",
                            "text": f"Sell\n@ {row['entry_price']:.2f}",
                        }
                    )
            if show_exit and {"exit_time", "profit"}.issubset(trades_df.columns):
                for _, row in trades_df.iterrows():
                    markers.append(
                        {
                            "time": int(pd.to_datetime(row["exit_time"]).timestamp()),
                            "position": "aboveBar",
                            "color": "#6b7280",
                            "shape": "circle",
                            "text": f"Exit\n{row['profit']:+.2f}",
                        }
                    )

        if markers:
            main_series["markers"] = markers

        series.append(main_series)

        if show_sma:
            sma = price_series.rolling(50).mean()
            sma_data = [
                {"time": int(idx.timestamp()), "value": float(val)}
                for idx, val in sma.items()
            ]
            series.append({
                "type": "Line",
                "data": sma_data,
                "options": {"color": "#6366f1"},
            })

        if show_ema:
            ema = price_series.ewm(span=21, adjust=False).mean()
            ema_data = [
                {"time": int(idx.timestamp()), "value": float(val)}
                for idx, val in ema.items()
            ]
            series.append({
                "type": "Line",
                "data": ema_data,
                "options": {"color": "#fbbf24"},
            })

        if {"exit_time", "profit"}.issubset(trades_df.columns):
            pnl_cum = (
                trades_df.sort_values("exit_time")
                .set_index("exit_time")["profit"]
                .cumsum()
            )
            pnl_cum = pnl_cum.reindex(price_series.index, method="ffill").fillna(0)
            pnl_data = [
                {"time": int(idx.timestamp()), "value": float(val)}
                for idx, val in pnl_cum.items()
            ]
            series.append({
                "type": "Line",
                "data": pnl_data,
                "options": {"color": "#6b7280"},
            })

        if has_volume:
            volume_data = [
                {"time": int(idx.timestamp()), "value": float(row["volume"])}
                for idx, row in price_df.iterrows()
            ]
            series.append({
                "type": "Histogram",
                "data": volume_data,
                "options": {"color": "#d1d5db"},
            })

        visible_range = {
            "from": price_df.index[0].strftime("%Y-%m-%dT%H:%M:%SZ"),
            "to": price_df.index[-1].strftime("%Y-%m-%dT%H:%M:%SZ"),
        }

        time_scale_opts = {
            "minBarSpacing": 0.001,  # allow strong compression
            "rightOffset": 0,
            "fixLeftEdge": True,
            "fixRightEdge": True,
            "fitContent": True,
            "visibleRange": visible_range,
        }

        charts = [{
            "series": series,
            "height": 600 if has_volume else 420,
            "chart": {
                "layout": {"textColor": "#000" if TPL == "plotly_white" else "#fff"},
                "watermark": {"visible": False},
                "timeScale": time_scale_opts,
            },
        }]

        lwc(name="price_chart", charts=charts, height=600 if has_volume else 420)

    st.markdown("---")

    # ② Equity | Drawdown | Fees ---------------------------------------------
    st.subheader("📈 Equity & Drawdown")


    if not equity_df.empty:
        first_equity = (
            equity_df["equity"].dropna().iloc[0]
            if not equity_df["equity"].dropna().empty
            else 0.0
        )
        start_balance_series = pd.Series(first_equity, index=equity_df.index)
        buy_hold = (
            price_series.reindex(equity_df.index, method="ffill")
            / price_series.iloc[0]
            * first_equity
        )
        eq_plot_df = pd.DataFrame(
            {
                "Equity": equity_df["equity"],
                "Buy & Hold": buy_hold,
                "Start Balance": start_balance_series,
            }
        ).dropna()

        start, end = st.slider(
            "Select period",
            min_value=eq_plot_df.index[0].to_pydatetime(),
            max_value=eq_plot_df.index[-1].to_pydatetime(),
            value=(
                eq_plot_df.index[0].to_pydatetime(),
                eq_plot_df.index[-1].to_pydatetime(),
            ),
            format="YYYY-MM-DD",
        )
        view = eq_plot_df.loc[start:end]
        log_y = st.checkbox("Log scale", value=False)

        fig_eq = px.line(
            view,
            x=view.index,
            y=["Equity", "Buy & Hold", "Start Balance"],
            template=TPL,
            labels={"value": "Series value", "variable": "Series"},
        )
        if log_y:
            fig_eq.update_yaxes(type="log")
        if pd.notna(dd_start) and pd.notna(dd_end):
            fig_eq.add_vrect(
                x0=dd_start,
                x1=dd_end,
                fillcolor="grey",
                opacity=0.25,
                layer="below",
                annotation_text="Longest DD",
            )
            fig_eq.add_vline(x=dd_start, line_dash="dash", line_color=NEG)
            fig_eq.add_vline(x=dd_end, line_dash="dash", line_color=ACCENT)
        st.plotly_chart(fig_eq, use_container_width=True)
    else:
        st.info("Equity data unavailable.")

    if not equity_df.empty:
        dd_full = (equity_df["equity"].cummax() - equity_df["equity"]) / equity_df[
            "equity"
        ].cummax()
        dd = -dd_full.loc[start:end]
        fig_dd = px.area(x=dd.index, y=dd.values * 100, template=TPL)
        fig_dd.update_layout(title="Underwater plot", yaxis_title="Drawdown (%)")
        if pd.notna(dd_start) and pd.notna(dd_end):
            fig_dd.add_vrect(
                x0=dd_start,
                x1=dd_end,
                fillcolor="grey",
                opacity=0.25,
                layer="below",
                annotation_text="Longest DD",
            )
        st.plotly_chart(fig_dd, use_container_width=True)
    else:
        st.warning("Equity data unavailable.")

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

    # окно скольжения ~6 месяцев или меньше, если данных мало
    roll = min(len(strategy_returns), 126) if not strategy_returns.empty else 1

    if strategy_returns.empty:
        st.info("Not enough data for risk calculations.")
    else:
        rvol = strategy_returns.rolling(roll).std(ddof=0).mul(np.sqrt(252)).dropna()
        cov = strategy_returns.rolling(roll).cov(benchmark_returns)
        rbeta = (cov / benchmark_returns.rolling(roll).var(ddof=0)).dropna()
        rsharp = strategy_returns.rolling(roll).apply(lambda s: sharpe(s)).dropna()

        with st.expander("Advanced analysis", expanded=True):
            risk_tabs = st.tabs([
                "Distribution & VaR",
                "Rolling metrics",
                "Seasonality",
                "Risk radar",
            ])

            # ── Distribution & VaR ───────────────────────────────────────────────
            with risk_tabs[0]:
                bins = st.slider("Bins", min_value=20, max_value=120, value=60, step=10)
                level = st.slider("Confidence level (%)", min_value=90, max_value=99, value=95)
                var_thres = 100 - level
                var_val = np.percentile(strategy_returns, var_thres)
                cvar_val = strategy_returns[strategy_returns <= var_val].mean()

                hist = px.histogram(
                    strategy_returns,
                    nbins=bins,
                    template=TPL,
                    title="Return distribution",
                )
                hist.add_vline(x=var_val, line_color=NEG, annotation_text=f"VaR {100-level}%")
                st.plotly_chart(hist, use_container_width=True)

                gcol1, gcol2 = st.columns(2)
                gfig1 = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=var_val * 100,
                        number={"suffix": "%"},
                        title={"text": f"VaR {level}%"},
                        gauge={
                            "axis": {"range": [min(var_val * 100 * 1.5, -20), 0]},
                            "bar": {"color": NEG},
                        },
                    )
                )
                gfig2 = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=cvar_val * 100,
                        number={"suffix": "%"},
                        title={"text": f"CVaR {level}%"},
                        gauge={
                            "axis": {"range": [min(cvar_val * 100 * 1.5, -20), 0]},
                            "bar": {"color": NEG},
                        },
                    )
                )
                gcol1.plotly_chart(gfig1, use_container_width=True)
                gcol2.plotly_chart(gfig2, use_container_width=True)

            # ── Rolling metrics ─────────────────────────────────────────────
            fig_roll = go.Figure()
            fig_roll.add_trace(
                go.Scatter(x=rsharp.index, y=rsharp, name="Rolling Sharpe")
            )
            fig_roll.add_trace(
                go.Scatter(x=rvol.index, y=rvol, name="Rolling Volatility")
            )
            fig_roll.add_trace(
                go.Scatter(x=rbeta.index, y=rbeta, name="Rolling Beta")
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
                strategy_returns.resample("ME")
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
                text_auto=".2%",
            )

            with risk_tabs[2]:
                st.plotly_chart(week_bar, use_container_width=True)
                st.plotly_chart(heatmap, use_container_width=True)
                # Intraday seasonality
                median_diff = strategy_returns.index.to_series().diff().median()
                if pd.notna(median_diff) and median_diff < pd.Timedelta(days=1):
                    hr_ret = strategy_returns.groupby(strategy_returns.index.hour).mean() * 100
                    hour_bar = px.bar(
                        x=list(range(24)),
                        y=hr_ret.reindex(range(24)).fillna(0),
                        template=TPL,
                        title="Average return by hour",
                    )
                    st.plotly_chart(hour_bar, use_container_width=True)

                # Calendar heatmap by week/day
                daily_ret = strategy_returns.resample("D").sum() * 100
                if not daily_ret.empty:
                    cal = daily_ret.to_frame("ret")
                    cal["Week"] = cal.index.isocalendar().week.astype(int)
                    cal["Day"] = cal.index.weekday
                    cal["Year"] = cal.index.year
                    for year in cal["Year"].unique():
                        piv = cal[cal["Year"] == year].pivot(index="Day", columns="Week", values="ret")
                        cfig = px.imshow(
                            piv,
                            color_continuous_scale="RdYlGn",
                            template=TPL,
                            origin="lower",
                            aspect="auto",
                            title=f"Daily returns calendar {year}",
                            labels=dict(x="Week", y="Day"),
                            text_auto=".1f",
                        )
                        cfig.update_yaxes(
                            tickmode="array",
                            tickvals=list(range(7)),
                            ticktext=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
                        )
                        st.plotly_chart(cfig, use_container_width=True)

            # ── Risk radar ─────────────────────────────────────────────────
            with risk_tabs[3]:
                radar_vals = {
                    "Volatility": (rvol.iloc[-1] * 100) if not rvol.empty else np.nan,
                    "VaR": abs(var5 * 100) if not np.isnan(var5) else np.nan,
                    "CVaR": abs(cvar5 * 100) if not np.isnan(cvar5) else np.nan,
                    "Max DD": abs(max_dd_pct) if not np.isnan(max_dd_pct) else np.nan,
                }
                rfig = go.Figure(
                    go.Scatterpolar(
                        r=list(radar_vals.values()),
                        theta=list(radar_vals.keys()),
                        fill="toself",
                        name="Risk profile",
                    )
                )
                rfig.update_layout(
                    template=TPL,
                    polar=dict(radialaxis=dict(visible=True)),
                    showlegend=False,
                )
                st.plotly_chart(rfig, use_container_width=True)

    st.markdown("---")

    # ④ Trades statistics ----------------------------------------------------
    st.subheader("Trades statistics")
    if not trades_df.empty:
        # -- Filter by position side using custom styled pills ---------
        st.markdown(
            """
            <style>
            [data-testid="stBaseButton-pillsInactive"] {
                background:#e0e0e0;
                color:#555;
                border:1px solid #bdbdbd;
            }
            [data-testid="stBaseButton-pillsActive"] {
                color:#222;
            }
            div[data-testid="stBaseButton-pillsItem"]:nth-child(1)
              [data-testid="stBaseButton-pillsActive"]{
                background:#2ecc71;
                border-color:#2ecc71;
            }
            div[data-testid="stBaseButton-pillsItem"]:nth-child(1)
              [data-testid="stBaseButton-pillsInactive"]{
                background:#f0fff6;
                color:#2ecc71;
                border-color:#2ecc71;
            }
            div[data-testid="stBaseButton-pillsItem"]:nth-child(2)
              [data-testid="stBaseButton-pillsActive"]{
                background:#e74c3c;
                border-color:#e74c3c;
            }
            div[data-testid="stBaseButton-pillsItem"]:nth-child(2)
              [data-testid="stBaseButton-pillsInactive"]{
                background:#fff4f3;
                color:#e74c3c;
                border-color:#e74c3c;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        selected_sides = st.pills(
            "Сторона сделки",
            ["Long", "Short"],
            selection_mode="multi",
            default=["Long", "Short"],
            key="side_filter",
        )

        df_view = trades_df.copy()
        if selected_sides:
            df_view = df_view[df_view["entry_side"].str.capitalize().isin(selected_sides)]

        if df_view.empty:
            st.info("No trades for selected filter.")
        else:
            if "duration_h" not in df_view.columns:
                df_view["duration_h"] = (
                    df_view["exit_time"] - df_view["entry_time"]
                ).dt.total_seconds() / 3600.0

            start_balance = float(equity_df["equity"].iloc[0]) if not equity_df.empty else 0.0
            wins = (df_view["profit"] > 0).sum()
            losses = (df_view["profit"] <= 0).sum()
            long_count = (df_view["entry_side"].str.upper() == "LONG").sum()
            short_count = (df_view["entry_side"].str.upper() == "SHORT").sum()

            stats_tabs = st.tabs(["Distributions", "Breakdown", "Analysis"])

            # ── Tab 0: Distributions ------------------------------------
            with stats_tabs[0]:
                h_left, h_right = st.columns(2)
                profit_mean = df_view["profit"].mean()
                profit_med = df_view["profit"].median()
                skew_val = df_view["profit"].skew()
                kurt_val = df_view["profit"].kurt()
                with h_left:
                    fig_pnl = px.histogram(df_view, x="profit", nbins=40, template=TPL, title="PnL per trade")
                    fig_pnl.add_vline(x=float(profit_mean), line_dash="dash", line_color=ACCENT)
                    fig_pnl.add_vline(x=float(profit_med), line_dash="dot", line_color=NEG)
                    st.plotly_chart(fig_pnl, use_container_width=True)
                    st.caption(f"Skewness: {skew_val:.2f}, Kurtosis: {kurt_val:.2f}")
                with h_right:
                    st.plotly_chart(
                        px.histogram(df_view, x="duration_h", nbins=40, template=TPL, title="Trade duration (h)"),
                        use_container_width=True,
                    )
                st.plotly_chart(
                    px.scatter(df_view, x="duration_h", y="profit", template=TPL, title="PnL vs duration"),
                    use_container_width=True,
                )

            # ── Tab 1: Breakdown ----------------------------------------
            with stats_tabs[1]:
                p_left, p_right = st.columns(2)
                with p_left:
                    st.plotly_chart(
                        px.pie(values=[wins, losses], names=["Wins", "Losses"], template=TPL, title="Win / Loss"),
                        use_container_width=True,
                    )
                with p_right:
                    st.plotly_chart(
                        px.pie(values=[long_count, short_count], names=["Long", "Short"], template=TPL, title="Long / Short"),
                        use_container_width=True,
                    )
                if tim is not None and not (isinstance(tim, float) and np.isnan(tim)):
                    st.plotly_chart(
                        px.pie(values=[tim, 100 - tim], names=["In position", "Flat"], template=TPL, title="Capital usage"),
                        use_container_width=True,
                    )

            # ── Tab 2: Analysis -----------------------------------------
            with stats_tabs[2]:
                wins_df = df_view[df_view["profit"] > 0]
                losses_df = df_view[df_view["profit"] < 0]

                def _max_run(series, cond):
                    run = max_run = 0
                    for val in series:
                        if cond(val):
                            run += 1
                            max_run = max(max_run, run)
                        else:
                            run = 0
                    return max_run

                profit_factor = (
                    wins_df["profit"].sum() / abs(losses_df["profit"].sum())
                    if not losses_df.empty and losses_df["profit"].sum() != 0
                    else np.nan
                )

                num_trades = len(df_view)
                win_rate = wins / num_trades * 100 if num_trades > 0 else 0.0
                avg_hold = df_view["duration_h"].mean()
                avg_slippage = df_view["slippage"].mean() if "slippage" in df_view.columns else np.nan

                metrics = {
                    "Trades": num_trades,
                    "Long trades": long_count,
                    "Short trades": short_count,
                    "Win rate (%)": win_rate,
                    "Avg profit per trade": df_view["profit"].mean(),
                    "Avg winning trade": wins_df["profit"].mean() if not wins_df.empty else 0.0,
                    "Avg losing trade": losses_df["profit"].mean() if not losses_df.empty else 0.0,
                    "Max win": df_view["profit"].max(),
                    "Max loss": df_view["profit"].min(),
                    "Avg holding time (h)": avg_hold,
                    "Avg slippage": avg_slippage,
                    "Profit factor": profit_factor,
                    "Max consecutive wins": _max_run(df_view["profit"], lambda x: x > 0),
                    "Max consecutive losses": _max_run(df_view["profit"], lambda x: x <= 0),
                }

                st.table(pd.DataFrame.from_dict(metrics, orient="index", columns=["Value"]).style.format("{:.4f}"))

                step_df = df_view.sort_values("exit_time").copy()
                step_df["equity_after"] = start_balance + step_df["profit"].cumsum()
                fig_step = px.line(step_df, x="exit_time", y="equity_after", template=TPL, title="Equity after each trade")
                fig_step.update_traces(line_shape="hv")
                st.plotly_chart(fig_step, use_container_width=True)

                top_n = 5
                top_wins = df_view.nlargest(top_n, "profit")[["entry_time", "exit_time", "entry_side", "profit"]]
                top_losses = df_view.nsmallest(top_n, "profit")[["entry_time", "exit_time", "entry_side", "profit"]]
                with st.expander("Top trades", expanded=False):
                    col_win, col_loss = st.columns(2)
                    col_win.caption("Best trades")
                    col_win.dataframe(style_trades(top_wins), use_container_width=True)
                    col_loss.caption("Worst trades")
                    col_loss.dataframe(style_trades(top_losses), use_container_width=True)
    else:
        st.info("No trades to display.")
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

    if "data_tab" not in st.session_state:
        st.session_state["data_tab"] = "csv"

    if stx:
        data_tabs = [
            stx.TabBarItemData(id="csv", title="CSV", description="Load data from CSV"),
            stx.TabBarItemData(id="ch", title="ClickHouse", description="Load data from ClickHouse"),
        ]
        chosen_id = stx.tab_bar(
            data=data_tabs,
            default=st.session_state["data_tab"],
        )
        st.session_state["data_tab"] = chosen_id
    else:
        st.warning("extra-streamlit-components not installed")
        chosen_id = st.session_state["data_tab"]

    if chosen_id == "csv":
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
    else:
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

    st.markdown("---")

    run_bt = st.button("Run back‑test", key="run_bt")

# end sidebar ---------------------------------------------------------------

TPL = st.session_state["bt_tpl"]
ACCENT = st.session_state["bt_accent"]
NEG = st.session_state["bt_neg"]

data_source: str
data_spec: Any
start_dt: datetime
end_dt: datetime

run_csv = run_bt and st.session_state.get("data_tab") == "csv"
run_ch = run_bt and st.session_state.get("data_tab") == "ch"

if run_csv:
    data_source = "CSV"
    data_spec = csv_path
    start_dt = pd.to_datetime(start_csv, utc=True)
    end_dt = pd.to_datetime(end_csv, utc=True) + pd.Timedelta(days=1)
elif run_ch:
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

if run_bt:
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
            except TypeError:  # actor_cls not accepted or other init error
                result = run_backtest(
                    info.strategy_cls,
                    info.cfg_cls,
                    params,
                    data_df,
                    actor_cls=DashboardPublisher,
                )
        log_text = log_stream.getvalue()

    st.session_state["bt_result"] = result
    st.session_state["bt_log"] = log_text
    st.session_state["bt_tpl"] = TPL
    st.session_state["bt_accent"] = ACCENT
    st.session_state["bt_neg"] = NEG

if st.session_state.get("bt_result") is not None:
    draw_dashboard(
        st.session_state["bt_result"],
        st.session_state["bt_log"],
        st.session_state["bt_tpl"],
        st.session_state["bt_accent"],
        st.session_state["bt_neg"],
    )
