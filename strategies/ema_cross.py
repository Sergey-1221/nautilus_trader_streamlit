from __future__ import annotations

"""EMA Crossover Strategy for Nautilus‑Trader.

This implementation is **symmetrical** – it can go *long* or *short* depending on
which side of the crossover occurs.  Modify the SELL block if you prefer a
long‑only approach.

Trading rules
-------------
* **Bullish crossover** (*fast EMA* crosses **above** *slow EMA*)
    - If **flat** → BUY
    - If **net‑short** → close shorts, then BUY
* **Bearish crossover** (*fast EMA* crosses **below** *slow EMA*)
    - If **flat** → SELL (open short)
    - If **net‑long** → close longs, then SELL

The strategy trades only on the *first* bar where a crossover occurs, avoiding
whipsawing while both EMAs remain on the same side.
"""

from decimal import Decimal
from typing import Optional

from nautilus_trader.config import StrategyConfig
from nautilus_trader.indicators.average.ema import ExponentialMovingAverage
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.instruments import Instrument
from nautilus_trader.trading.strategy import Strategy

# ──────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────

class EMACrossConfig(StrategyConfig):
    instrument_id: InstrumentId
    bar_type: BarType
    trade_size: Decimal = Decimal("1000")
    fast_period: int = 12
    slow_period: int = 26

    class Config:
        frozen = True  # make it hashable / (de‑)serialisable

# ──────────────────────────────────────────────────────────────
# Strategy
# ──────────────────────────────────────────────────────────────

class EMACross(Strategy):
    """EMA crossover strategy supporting long *and* short positions."""

    def __init__(self, cfg: EMACrossConfig):
        super().__init__(cfg)

        if cfg.fast_period >= cfg.slow_period:
            raise ValueError("fast_period must be < slow_period")

        # Parameters
        self._iid = cfg.instrument_id
        self._bar_type = cfg.bar_type
        self._qty = cfg.trade_size

        # Indicators (manual update, so *don’t* register with engine)
        self.fast = ExponentialMovingAverage(cfg.fast_period)
        self.slow = ExponentialMovingAverage(cfg.slow_period)

        # State
        self._prev_fast_above: Optional[bool] = None
        self._instrument: Optional[Instrument] = None

    # ─────────── Lifecycle ───────────
    def on_start(self) -> None:
        self._instrument = self.cache.instrument(self._iid)
        if self._instrument is None:
            self.log.error(f"Instrument {self._iid} not found – stopping")
            self.stop(); return

        # Subscribe to bar stream
        self.subscribe_bars(self._bar_type)

    # ─────────── Bar handler ─────────
    def on_bar(self, bar: Bar) -> None:
        if bar.bar_type != self._bar_type:
            return  # ignore other resolutions

        price = bar.close.as_double()
        self.fast.update_raw(price)
        self.slow.update_raw(price)

        if not (self.fast.initialized and self.slow.initialized):
            return  # wait until both EMAs are warm

        fast_above = self.fast.value > self.slow.value

        # Skip first bar after warm‑up (no previous state)
        if self._prev_fast_above is None:
            self._prev_fast_above = fast_above
            return

        crossed_up   = fast_above and (self._prev_fast_above is False)
        crossed_down = (not fast_above) and (self._prev_fast_above is True)

        # Portfolio helpers
        flat      = self.portfolio.is_flat(self._iid)
        net_long  = self.portfolio.is_net_long(self._iid)
        net_short = self.portfolio.is_net_short(self._iid)

        # ‑‑‑ Bullish crossover ‑‑‑
        if crossed_up:
            if flat:
                self._market(OrderSide.BUY, self._qty)
            elif net_short:
                self.close_all_positions(self._iid)
                self._market(OrderSide.BUY, self._qty)

        # ‑‑‑ Bearish crossover ‑‑‑
        if crossed_down:
            if flat:
                self._market(OrderSide.SELL, self._qty)
            elif net_long:
                self.close_all_positions(self._iid)
                self._market(OrderSide.SELL, self._qty)

        self._prev_fast_above = fast_above

    # ─────────── Helpers ────────────
    def _market(self, side: OrderSide, size: Decimal) -> None:
        if self._instrument is None:
            return  # should not happen
        order = self.order_factory.market(
            instrument_id=self._iid,
            order_side=side,
            quantity=self._instrument.make_qty(size),  # type: ignore[arg-type]
        )
        self.submit_order(order)
        self.log.info(f"Submitted {side.name} {size}")
