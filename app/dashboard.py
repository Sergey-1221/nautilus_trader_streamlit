"""
📊 UNIVERSAL BACKTEST DASHBOARD
Streamlit — версия 4
* Price & Trades – главный full‑width блок
* Новый full‑width блок “Risk & Seasonality”
* Встроены Rolling Volatility / Beta / Sharpe
Run:  streamlit run backtest_dashboard.py
"""

# ──────────────────────────────────────────────────────────────
#  Imports  ➜  set_page_config (первый Streamlit‑вызов)
# ──────────────────────────────────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config("Universal Backtest Dashboard", layout="wide")

# ──────────────────────────────────────────────────────────────
#  Theme & colours
# ──────────────────────────────────────────────────────────────
theme = st.sidebar.selectbox("Theme", ["Dark", "Light"], index=0)
TPL = "plotly_dark" if theme == "Dark" else "plotly_white"
ACCENT, NEG = "#10B981", "#EF4444"

# ──────────────────────────────────────────────────────────────
#  MOCK DATA  (замените реальным парсером логов)
# ──────────────────────────────────────────────────────────────
meta = {
    "Run ID":          "2f3e5a53-46cf-4acb-b144-26ec84429b91",
    "Run config ID":   None,
    "Run started":     "2025-05-08T10:51:32Z",
    "Run finished":    "2025-05-08T10:51:33Z",
    "Elapsed time":    "0 days 00:00:00.71",
    "Backtest start":  "2025-03-17",
    "Backtest end":    "2025-04-06",
    "Backtest range":  "20 days 11:11:00",
    "Iterations":      29246,
    "Total events":    388,
    "Total orders":    194,
    "Total positions": 97,
    "Start balance":   10_000,
    "End balance":     9_872.06,
    "Commissions":    -163.89,
}

start = pd.to_datetime(meta["Backtest start"], format="%Y-%m-%d")
end   = pd.to_datetime(meta["Backtest end"],   format="%Y-%m-%d")
dates = pd.date_range(start, end, freq="D")

np.random.seed(42)
strategy_eq  = meta["Start balance"] + np.cumsum(np.random.normal(0, 60, len(dates)))
benchmark_eq = meta["Start balance"] + np.cumsum(np.random.normal(0, 80, len(dates)))
comm_cum     = np.linspace(0, meta["Commissions"], len(dates))
price_series = 50_000 + np.cumsum(np.random.normal(0, 300, len(dates)))

eq_df = pd.DataFrame({"date": dates,
                      "Strategy": strategy_eq,
                      "Benchmark": benchmark_eq,
                      "Commissions": comm_cum,
                      "Price": price_series})

trades = pd.DataFrame(
    {
        "open":  pd.date_range(start, periods=meta["Total orders"], freq="3H"),
        "close": pd.date_range(start, periods=meta["Total orders"], freq="3H") + pd.Timedelta(hours=1),
        "side":  np.random.choice(["BUY", "SELL"], meta["Total orders"]),
        "price": np.random.uniform(50_000, 70_000, meta["Total orders"]),
        "pnl":   np.random.normal(0, 6, meta["Total orders"]),
    }
)
trades["duration_h"] = (trades["close"] - trades["open"]).dt.total_seconds() / 3600
allocation = pd.Series({"BTC": .55, "ETH": .25, "USDT": .15, "Others": .05})

# ──────────────────────────────────────────────────────────────
#  Sidebar filters
# ──────────────────────────────────────────────────────────────
st.sidebar.markdown("### Filters")
date_from, date_to = st.sidebar.date_input(
    "Date range", (dates.min(), dates.max()), min_value=dates.min(), max_value=dates.max()
)
side_filter = st.sidebar.multiselect("Side", ["BUY", "SELL"], ["BUY", "SELL"])

mask = (eq_df["date"].between(pd.to_datetime(date_from), pd.to_datetime(date_to)))
eq_f = eq_df.loc[mask].reset_index(drop=True)
tr_f = trades[
    (trades["open"].between(pd.to_datetime(date_from), pd.to_datetime(date_to)))
    & trades["side"].isin(side_filter)
].reset_index(drop=True)

rets = eq_f.set_index("date")[["Strategy", "Benchmark"]].pct_change().dropna()

# ──────────────────────────────────────────────────────────────
#  KPI helpers
# ──────────────────────────────────────────────────────────────
def sharpe(s):  return (s.mean() / s.std(ddof=0)) * np.sqrt(252) if s.std() else 0
def sortino(s): return (s.mean() / s[s<0].std(ddof=0)) * np.sqrt(252) if s[s<0].std() else 0
def max_dd(series): return ((series.cummax() - series) / series.cummax()).max()

kpi = {
    "PnL ($)": tr_f["pnl"].sum().round(2),
    "PnL (%)": (eq_f["Strategy"].iloc[-1] - eq_f["Strategy"].iloc[0]) / eq_f["Strategy"].iloc[0],
    "Win Rate": (tr_f["pnl"] > 0).mean(),
    "Sharpe": sharpe(rets["Strategy"]),
    "Sortino": sortino(rets["Strategy"]),
    "Max DD (%)": max_dd(eq_f["Strategy"]) * 100,
    "Profit Factor": tr_f.loc[tr_f["pnl"] > 0, "pnl"].sum()
                     / abs(tr_f.loc[tr_f["pnl"] < 0, "pnl"].sum()),
    "Volatility (252d)": rets["Strategy"].std(ddof=0) * np.sqrt(252),
}

# ──────────────────────────────────────────────────────────────
#  HEADER  +  KPI GRID
# ──────────────────────────────────────────────────────────────
st.markdown(f"## Backtest {meta['Run ID']}")
hdr = st.columns(4)
hdr[0].metric("Started",  meta["Run started"])
hdr[1].metric("Finished", meta["Run finished"])
hdr[2].metric("Elapsed",  meta["Elapsed time"])
hdr[3].metric("Orders",   meta["Total orders"])

kcols = st.columns(len(kpi))
for (lab, val), c in zip(kpi.items(), kcols):
    txt = f"{val*100:.2f}%" if ("%" in lab) and ("PnL" in lab or "DD" in lab) else f"{val:,.2f}"
    color = NEG if (("PnL" in lab and val < 0) or ("DD" in lab and val > 0)) else ACCENT
    c.markdown(f"<span style='font-size:1.1rem;color:{color}'><b>{txt}</b><br>{lab}</span>",
               unsafe_allow_html=True)
st.markdown("---")

# ──────────────────────────────────────────────────────────────
#  FULL‑WIDTH ①  Price & Trades
# ──────────────────────────────────────────────────────────────
st.subheader("📉 Price & Trades")
fig_pt = go.Figure()
fig_pt.add_trace(go.Scatter(x=eq_f["date"], y=eq_f["Price"],
                            mode="lines", name="Price"))
buys  = tr_f[tr_f["side"]=="BUY"]
sells = tr_f[tr_f["side"]=="SELL"]
fig_pt.add_trace(go.Scatter(x=buys["open"],  y=buys["price"],
                            mode="markers", marker_symbol="triangle-up",
                            marker_color=ACCENT, name="Buy"))
fig_pt.add_trace(go.Scatter(x=sells["open"], y=sells["price"],
                            mode="markers", marker_symbol="triangle-down",
                            marker_color=NEG,   name="Sell"))
fig_pt.update_layout(template=TPL, height=420, margin=dict(l=0,r=0,b=0,t=25))
st.plotly_chart(fig_pt, use_container_width=True)
st.markdown("---")

# ──────────────────────────────────────────────────────────────
#  FULL‑WIDTH ②  Equity / Drawdown / Fees
# ──────────────────────────────────────────────────────────────
st.subheader("📈 Equity | Drawdown | Fees")
tabs_eq = st.tabs(["Equity", "Drawdown", "Fees"])
tabs_eq[0].plotly_chart(
    px.line(eq_f, x="date", y=["Strategy","Benchmark"], template=TPL,
            labels={"value":"Equity","variable":"Series"}), use_container_width=True
)
dd = (eq_f["Strategy"].cummax() - eq_f["Strategy"]) / eq_f["Strategy"].cummax()
tabs_eq[1].plotly_chart(px.area(x=eq_f["date"], y=dd, template=TPL), use_container_width=True)
tabs_eq[2].plotly_chart(px.line(eq_f, x="date", y="Commissions", template=TPL),
                        use_container_width=True)
st.markdown("---")

# ──────────────────────────────────────────────────────────────
#  FULL‑WIDTH ③  Risk & Seasonality
# ──────────────────────────────────────────────────────────────
st.subheader("📊 Risk & Seasonality")
roll = st.slider("Rolling window (days)", 5, 60, 20, key="roll")
# Rolling metrics
rvol = rets["Strategy"].rolling(roll).std(ddof=0).mul(np.sqrt(252)).dropna()
cov  = rets["Strategy"].rolling(roll).cov(rets["Benchmark"])
rbeta = (cov / rets["Benchmark"].rolling(roll).var(ddof=0)).dropna()
rsharpe = rets["Strategy"].rolling(roll).apply(lambda s: sharpe(s)).dropna()

risk_tabs = st.tabs(["Distribution & VaR", "Rolling Metrics", "Seasonality"])

# --- Distribution with VaR
var5 = np.percentile(rets["Strategy"], 5)
hist = px.histogram(rets["Strategy"], nbins=60, template=TPL,
                    title="Return distribution")
hist.add_vline(x=var5, line_color=NEG, annotation_text="VaR 5%")
risk_tabs[0].plotly_chart(hist, use_container_width=True)

# --- Rolling metrics plot (Sharpe, Vol, Beta)
fig_roll = go.Figure()
fig_roll.add_trace(go.Scatter(x=rsharpe.index, y=rsharpe.values,
                              name="Rolling Sharpe"))
fig_roll.add_trace(go.Scatter(x=rvol.index, y=rvol.values,
                              name="Rolling Volatility"))
fig_roll.add_trace(go.Scatter(x=rbeta.index, y=rbeta.values,
                              name="Rolling Beta"))
fig_roll.update_layout(template=TPL, height=350, legend_orientation="h")
risk_tabs[1].plotly_chart(fig_roll, use_container_width=True)

# --- Seasonality (weekday bar + monthly heatmap)
wd_ret = rets["Strategy"].groupby(rets.index.weekday).mean()*100
week_bar = px.bar(x=["Mon","Tue","Wed","Thu","Fri","Sat","Sun"],
                  y=wd_ret.reindex(range(7)).fillna(0),
                  template=TPL, title="Average return by weekday")
heat = (rets["Strategy"].resample("M").sum().to_frame("ret")
        .assign(Year=lambda d: d.index.year,
                Month=lambda d: d.index.month_name().str[:3]))
piv = heat.pivot(index="Year", columns="Month", values="ret").fillna(0)
heatmap = px.imshow(piv, color_continuous_scale="RdYlGn",
                    template=TPL, title="Monthly return heatmap")
risk_tabs[2].plotly_chart(week_bar, use_container_width=True)
risk_tabs[2].plotly_chart(heatmap, use_container_width=True)

st.markdown("---")

# ──────────────────────────────────────────────────────────────
#  Bottom row: Trades Stats & Allocation
# ──────────────────────────────────────────────────────────────
left_bot, right_bot = st.columns(2)

with left_bot:
    st.subheader("Trades Stats")
    st.plotly_chart(px.histogram(tr_f, x="duration_h", nbins=40,
                                 template=TPL, title="Trade duration (h)"),
                    use_container_width=True)
    wins, losses = (tr_f["pnl"]>0).sum(), (tr_f["pnl"]<=0).sum()
    st.plotly_chart(px.pie(values=[wins, losses], names=["Wins","Losses"],
                           template=TPL, title="Win / Loss"),
                    use_container_width=True)

with right_bot:
    st.subheader("Portfolio Allocation")
    st.plotly_chart(px.pie(values=allocation.values, names=allocation.index,
                           template=TPL), use_container_width=True)
    st.table(allocation.rename("Weight"))

# ──────────────────────────────────────────────────────────────
#  Key Metrics  |  Trades Log
# ──────────────────────────────────────────────────────────────
tab_metrics, tab_log = st.tabs(["Key Metrics", "Trades Log"])

with tab_metrics:
    st.dataframe(pd.DataFrame.from_dict(kpi, orient="index", columns=["Value"]))

with tab_log:
    st.dataframe(tr_f[["open","close","side","price","pnl","duration_h"]],
                 use_container_width=True, height=350)
# ──────────────────────────────────────────────────────────────
#  End of script
# ──────────────────────────────────────────────────────────────
