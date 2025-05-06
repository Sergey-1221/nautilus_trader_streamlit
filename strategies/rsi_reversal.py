from __future__ import annotations

"""RSI‑Reversal strategy for Nautilus‑Trader.

Logic
-----
* **Long** when *RSI ≤ oversold*.
* **Short** when *RSI ≥ overbought*.
* If an opposite position exists we first flatten it, allowing
  LONG → FLAT → SHORT flips (and vice‑versa).

Changes
-------
* Removed `close_on_stop`, `subscribe_trade_ticks`, `subscribe_quote_ticks`
  to keep the config minimal.
* Duplicate indicator update removed – the engine now invokes `handle_bar`
  for registered indicators only once.
* `instrument.make_qty` ensures order quantities respect lot/precision.
* Added verbose logging for easier debugging.
"""

from decimal import Decimal
from typing import Optional

from nautilus_trader.config import StrategyConfig, PositiveInt
from nautilus_trader.indicators.base.indicator import Indicator
from nautilus_trader.model import Bar, BarType
from nautilus_trader.model.enums import OrderSide, TimeInForce
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.instruments import Instrument
from nautilus_trader.trading.strategy import Strategy


# ──────────────────────────── RSI INDICATOR ────────────────────────────────
class RelativeStrengthIndex(Indicator):
    """Minimal RSI implementation using Wilder's smoothing."""

    def __init__(self, period: int):
        super().__init__([period])
        self.period = period
        self._prev: Optional[Decimal] = None
        self._avg_gain = self._avg_loss = Decimal(0)
        self.count = 0
        self.value = Decimal(0)

    # Called automatically by the engine for each registered bar ----------
    def handle_bar(self, bar: Bar) -> None:
        self._update(bar.close.as_double())

    # Internal update routine ---------------------------------------------
    def _update(self, price_raw: float) -> None:
        price = Decimal(price_raw)
        if self._prev is None:  # seed first datapoint
            self._prev = price
            return

        change = price - self._prev
        gain   = change if change > 0 else Decimal(0)
        loss   = -change if change < 0 else Decimal(0)
        self._prev = price
        self.count += 1

        if self.count < self.period:        # warm‑up accumulation
            self._avg_gain += gain
            self._avg_loss += loss
            return
        elif self.count == self.period:     # initial Wilder averages
            self._avg_gain += gain
            self._avg_loss += loss
            self._avg_gain /= self.period
            self._avg_loss /= self.period
        else:                               # rolling Wilder averages
            self._avg_gain = ((self._avg_gain * (self.period - 1)) + gain) / self.period
            self._avg_loss = ((self._avg_loss * (self.period - 1)) + loss) / self.period

        # RSI calculation
        if self._avg_loss == 0:
            self.value = Decimal(100)
        else:
            rs = self._avg_gain / self._avg_loss
            self.value = Decimal(100) - (Decimal(100) / (Decimal(1) + rs))

    @property
    def ready(self) -> bool:
        return self.count >= self.period


# ───────────────────────────── CONFIG ─────────────────────────────────────
class RSIReversalConfig(StrategyConfig, frozen=True):
    instrument_id: InstrumentId
    bar_type:     BarType
    trade_size:   Decimal      = Decimal("0.01")  # default 0.01 BTC – adjust if needed
    rsi_period:   PositiveInt  = 14
    overbought:   Decimal      = Decimal("70")
    oversold:     Decimal      = Decimal("30")


# ──────────────────────────── STRATEGY ────────────────────────────────────
class RSIReversal(Strategy):
    """Reversal trading on RSI extremes."""

    def __init__(self, cfg: RSIReversalConfig):
        super().__init__(cfg)
        self.rsi = RelativeStrengthIndex(cfg.rsi_period)
        self.instrument: Instrument | None = None
        self.position_size = Decimal(0)          # signed net qty in instrument units

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def on_start(self) -> None:
        self.instrument = self.cache.instrument(self.config.instrument_id)
        if self.instrument is None:
            self.log.error(f"Instrument {self.config.instrument_id} not found; stopping.")
            self.stop()
            return

        # Register indicator BEFORE requesting bars so updates are automatic
        self.register_indicator_for_bars(self.config.bar_type, self.rsi)

        # Warm‑up with at least `period` bars so RSI is ready
        self.request_bars(self.config.bar_type, limit=self.rsi.period)
        self.subscribe_bars(self.config.bar_type)

    # ------------------------------------------------------------------
    # Market‑data events
    # ------------------------------------------------------------------
    def on_bar(self, bar: Bar) -> None:
        if not self.rsi.ready:
            self.log.debug(f"RSI warming‑up: count={self.rsi.count}/{self.rsi.period}")
            return

        rsi_val = self.rsi.value
        qty     = self.instrument.make_qty(self.config.trade_size)  # lot/precision safe

        # Verbose bar trace for debugging
        self.log.info(
            f"BAR ts={getattr(bar, 'ts_event', '<no ts>')} "
            f"close={bar.close} RSI={rsi_val} pos={self.position_size}"
        )

        # ---------------- trading rule ----------------
        if rsi_val <= self.config.oversold and self.position_size <= 0:
            self._submit(OrderSide.BUY, qty)
        elif rsi_val >= self.config.overbought and self.position_size >= 0:
            self._submit(OrderSide.SELL, qty)

    # ------------------------------------------------------------------
    # Orders & fills helpers
    # ------------------------------------------------------------------
    def _submit(self, side: OrderSide, qty) -> None:  # qty is already Quantity
        self.submit_order(
            self.order_factory.market(
                instrument_id=self.config.instrument_id,
                order_side=side,
                quantity=qty,
                time_in_force=TimeInForce.IOC,
            )
        )
        self.log.info(f"Submitted {side.name} {qty}")

    def on_order_filled(self, event) -> None:
        qty = Decimal(event.last_qty.as_double())
        self.position_size += qty if event.order_side is OrderSide.BUY else -qty
        self.log.info(f"FILL {event.order_side.name} {qty} → pos={self.position_size}")

    # ------------------------------------------------------------------
    # Graceful shutdown
    # ------------------------------------------------------------------
    def on_stop(self) -> None:
        # Always flatten any residual position on stop
        if self.position_size != 0:
            side = OrderSide.SELL if self.position_size > 0 else OrderSide.BUY
            qty  = self.instrument.make_qty(abs(self.position_size))
            self._submit(side, qty)
            self.log.info("Residual position closed on strategy stop.")
