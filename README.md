# NautilusTrader Streamlit
![DEMO](https://github.com/Sergey-1221/nautilus_trader_streamlit/raw/main/image/demo.png)
[DEMO](https://nautilustrader.streamlit.app/)

## 🎯 Project Goal

**Create a simple and convenient Streamlit-based extension for the NautilusTrader platform, enabling developers to quickly visualize, analyze, and convincingly demonstrate the performance of trading strategies to investors and teams — without wasting time on complex frontend development.**

NautilusTrader is a rapidly evolving open-source platform for algorithmic trading. This project emerged from the need to simplify and accelerate the visualization of strategy data created with it.

---

## 🌟 Why is it convenient?

* ✅ **Minimal setup**: just connect your strategy and specify the data source.
* ✅ **Fully interactive**: instantly get dynamic charts and metrics.
* ✅ **No frontend code**: work entirely in Python, no harder than Jupyter Notebook.
* ✅ **Great for presentations**: beautiful and clear visualizations for investors and team members.
* ✅ **Time-saving**: quickly identify bugs and strategy issues.
* ✅ **Improved visuals**: themed widgets, icons and styled data tables.
* ✅ **Price chart enhancements**: trade markers, optional volume bars and cumulative PnL overlay rendered with `streamlit-lightweight-charts`.
* ✅ **Detailed trade tooltips**: hover markers to see entry, exit and PnL info.
* ✅ **Chart options**: choose line or candlesticks and overlay SMA/EMA lines.
* ✅ **Fast charting**: uses `streamlit-lightweight-charts` to render thousands of bars smoothly.
* ✅ **Structured metrics**: grouped performance KPIs with a progress bar showing edge over buy‑and‑hold.

---

## 🛠️ Quick Start
Install the dependencies and run the app:

```bash
pip install -r requirements.txt
streamlit run app/main.py
```

If you plan to load data from ClickHouse, create a `.env` file or set the
environment variables `CH_HOST`, `CH_USER`, `CH_PASSWORD` and `CH_DATABASE`.

## 📈 Equity & Drawdown

The dashboard visualizes how the account balance evolves and how deep the drawdowns were.

* **Equity curve** – portfolio value over time. A steadily rising line indicates consistent profit.
* **Drawdown chart** – percentage fall from the latest equity peak. The maximum of this series is the *maximum drawdown*. The app also measures how many days it took to recover from the deepest slump – the *longest drawdown duration*.

The two charts are shown one under the other, making it easy to relate profit growth and dips. The equity chart includes a *Buy & Hold* line for instant comparison with simply holding the asset.

You can focus on any time range using the built‑in date slider and toggle logarithmic scaling. The dashboard also reports your edge over the benchmark as **Edge vs B&H (%)**.

The longest drawdown is highlighted on both charts. Additional metrics show its duration, the average and total time spent in drawdowns, and the deepest loss in dollars.

## 📊 Risk analysis

The **Risk & Seasonality** block aggregates several metrics:

* **Distribution & VaR** – interactive histogram of returns with adjustable bins. A slider lets you change the VaR confidence level (90–99%), and gauges display VaR and CVaR values.
* **Rolling metrics** – rolling Sharpe ratio, volatility and beta (six‑month window by default) reveal how the risk profile changes over time.
* **Seasonality** – average return by weekday and a monthly heatmap with return values on hover. For intraday data an additional chart shows average return by hour. A calendar heatmap visualizes daily returns across the year.
* **Risk radar** – polar chart comparing volatility, VaR, CVaR and max drawdown for a quick overview of the risk profile.
* **Additional risk KPIs** – current drawdown duration, average drawdown depth, drawdown count, VaR/CVaR at 5%, Calmar ratio and RoMaD. The drawdown chart doubles as an underwater plot highlighting equity dips.

---


## 📌 Roadmap

* [x] **v0.1.5**: First basic version for strategy visualization.
* [ ] **Next steps (depending on community interest):**

  * Real-time strategy monitoring.
  * Additional metrics and charts.
  * Portfolio evaluation features.
  * Improved flexibility and customization.
  * Better documentation and usage examples.
  * Enhanced compatibility with NautilusTrader.
  * UI/UX improvements.
  * Modular configuration support.

If you find this project helpful — give it a ⭐ and share your feedback. The more community interest, the faster the tool will evolve.

---

## 🤝 How to Contribute

* ⭐ **Star the repository**.
* 🐞 **Open Issues** if you find bugs.
* 🚀 **Submit Pull Requests** if you’d like to add new features.

---
