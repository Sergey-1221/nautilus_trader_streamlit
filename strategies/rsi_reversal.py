from __future__ import annotations
"""RSI‑Reversal strategy for Nautilus‑Trader ≥ 1.150

Trading rules
-------------
* **Long** when *RSI ≤ oversold*.
* **Short** when *RSI ≥ overbought*.
* Opposite exposure is flattened first (LONG → FLAT → SHORT and vice‑versa).

Implementation notes
--------------------
* Private minimal RSI (Wilder’s smoothing).
* The new engine requires a full :class:`nautilus_trader.data.messages.RequestBars`
  object when requesting history – the old keyword shortcut was removed.
* At least one of *client_id* or *venue* must be supplied in the message; we use
  the instrument’s venue.
* Verbose logging makes back‑test debugging easier.
"""

import time
from decimal import Decimal
from typing import Optional

from nautilus_trader.config import StrategyConfig, PositiveInt
# The engine’s request_bars signature may vary across Nautilus‑Trader builds.
# We attempt to import RequestBars for ≥1.150, but gracefully handle absence.
try:
    from nautilus_trader.data.messages import RequestBars  # type: ignore
except ImportError:
    RequestBars = None  # type: ignore
except ImportError:  # fallback for older wheels (shouldn’t be needed)
    from nautilus_trader.data import RequestBars  # type: ignore
from nautilus_trader.core.uuid import UUID4
from nautilus_trader.indicators.base.indicator import Indicator
from nautilus_trader.model import Bar, BarType
from nautilus_trader.model.enums import OrderSide, TimeInForce
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.instruments import Instrument
from nautilus_trader.trading.strategy import Strategy

__all__ = ["RSIReversal", "RSIReversalConfig"]

# ─────────────────────────── RSI indicator ──────────────────────────────
class RelativeStrengthIndex(Indicator):
    """Lightweight RSI using Wilder’s rolling averages."""

    def __init__(self, period: int):
        super().__init__([period])
        self.period = period
        self._prev: Optional[Decimal] = None
        self._avg_gain = Decimal(0)
        self._avg_loss = Decimal(0)
        self.count = 0  # bars processed
        self.value = Decimal(0)  # last RSI value

    # Engine hook --------------------------------------------------------
    def handle_bar(self, bar: Bar) -> None:  # pragma: no cover
        self._update(bar.close.as_double())

    # Wilder calculation -------------------------------------------------
    def _update(self, price_raw: float) -> None:
        price = Decimal(price_raw)
        if self._prev is None:
            self._prev = price
            return

        change = price - self._prev
        gain = change if change > 0 else Decimal(0)
        loss = -change if change < 0 else Decimal(0)
        self._prev = price
        self.count += 1

        if self.count < self.period:
            self._avg_gain += gain
            self._avg_loss += loss
            return
        elif self.count == self.period:
            self._avg_gain += gain
            self._avg_loss += loss
            self._avg_gain /= self.period
            self._avg_loss /= self.period
        else:
            self._avg_gain = ((self._avg_gain * (self.period - 1)) + gain) / self.period
            self._avg_loss = ((self._avg_loss * (self.period - 1)) + loss) / self.period

        if self._avg_loss == 0:
            self.value = Decimal(100)
        else:
            rs = self._avg_gain / self._avg_loss
            self.value = Decimal(100) - (Decimal(100) / (Decimal(1) + rs))

    # Ready flag ---------------------------------------------------------
    @property
    def ready(self) -> bool:
        return self.count >= self.period


# ───────────────────────────── config ────────────────────────────────────
class RSIReversalConfig(StrategyConfig, frozen=True):
    """User‑tunable parameters for :class:`RSIReversal`."""

    instrument_id: InstrumentId
    bar_type:      BarType
    trade_size:    Decimal     = Decimal("0.01")  # in instrument units (e.g. BTC)
    rsi_period:    PositiveInt = 14
    overbought:    Decimal     = Decimal("70")
    oversold:      Decimal     = Decimal("30")


# ─────────────────────────── strategy ────────────────────────────────────
class RSIReversal(Strategy):
    """Mean‑reversion strategy trading RSI extremes."""

    def __init__(self, cfg: RSIReversalConfig):
        super().__init__(cfg)
        self.rsi = RelativeStrengthIndex(cfg.rsi_period)
        self.instrument: Instrument | None = None
        self.position_size = Decimal(0)  # signed net quantity

    # ───────────────────── lifecycle ─────────────────────
    def on_start(self) -> None:
        # Fetch instrument ------------------------------------------------
        self.instrument = self.cache.instrument(self.config.instrument_id)
        if self.instrument is None:
            self.log.error(f"Instrument {self.config.instrument_id} not found; stopping.")
            self.stop()
            return

        # Register indicator before history request ----------------------
        self.register_indicator_for_bars(self.config.bar_type, self.rsi)

        # Warm‑up history: call signature differs by version
        if RequestBars is not None:
            try:
                seed_req = RequestBars(
                    self.config.bar_type,
                    None,
                    None,
                    self.rsi.period,
                    None,
                    self.instrument.venue,
                    lambda _bars: None,
                    UUID4(),
                    int(time.time_ns()),
                    None,
                )
                self.request_bars(seed_req)
            except (TypeError, ValueError):  # fallback if wrong signature
                self.request_bars(self.config.bar_type)
        else:
            # Older build – simple call without kwargs (limit not allowed)
            self.request_bars(self.config.bar_type)

        # Subscribe to realtime bars -------------------------------------
        self.subscribe_bars(self.config.bar_type)

    # ───────────────────── market‑data events ────────────────────
    def on_bar(self, bar: Bar) -> None:  # pragma: no cover
        if not self.rsi.ready:
            self.log.debug(f"RSI warming‑up {self.rsi.count}/{self.rsi.period}")
            return

        rsi_val = self.rsi.value
        qty = self.instrument.make_qty(self.config.trade_size)  # type: ignore[arg-type]

        # Trading logic ---------------------------------------------------
        if rsi_val <= self.config.oversold and self.position_size <= 0:
            self._submit(OrderSide.BUY, qty)
        elif rsi_val >= self.config.overbought and self.position_size >= 0:
            self._submit(OrderSide.SELL, qty)

        self.log.info(
            f"BAR ts={getattr(bar, 'ts_event', '<no ts>')} close={bar.close} "
            f"RSI={rsi_val:.2f} pos={self.position_size}"
        )

    # ───────────────────── order helpers ─────────────────────
    def _submit(self, side: OrderSide, qty) -> None:  # qty already Quantity
        self.submit_order(
            self.order_factory.market(
                instrument_id=self.config.instrument_id,
                order_side=side,
                quantity=qty,
                time_in_force=TimeInForce.IOC,
            )
        )
        self.log.info(f"Submitted {side.name} {qty}")

    def on_order_filled(self, event) -> None:  # pragma: no cover
        fill_qty = Decimal(event.last_qty.as_double())
        self.position_size += fill_qty if event.order_side is OrderSide.BUY else -fill_qty
        self.log.info(f"FILL {event.order_side.name} {fill_qty} → pos={self.position_size}")

    # ───────────────────── shutdown ─────────────────────
    def on_stop(self) -> None:  # pragma: no cover
        if self.position_size != 0:
            side = OrderSide.SELL if self.position_size > 0 else OrderSide.BUY
            qty = self.instrument.make_qty(abs(self.position_size))
            self._submit(side, qty)
            self.log.info("Residual position closed on strategy stop.")
