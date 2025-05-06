from __future__ import annotations

from decimal import Decimal
from typing import Optional

from nautilus_trader.config import StrategyConfig
from nautilus_trader.model import Bar, BarType
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.instruments import Instrument
from nautilus_trader.trading.strategy import Strategy


class BuyAndHoldConfig(StrategyConfig):
    """Configuration for BuyAndHoldStrategy."""

    instrument_id: InstrumentId
    bar_type: BarType
    trade_size: Decimal = Decimal("1000")


class BuyAndHoldStrategy(Strategy):
    """Buy once at the earliest opportunity and hold indefinitely."""

    def __init__(self, config: BuyAndHoldConfig) -> None:
        super().__init__(config)
        self.instrument: Optional[Instrument] = None
        self._bought: bool = False

    # ───────────────────── lifecycle ─────────────────────
    def on_start(self) -> None:
        """Retrieve instrument and subscribe to bars."""
        self.instrument = self.cache.instrument(self.config.instrument_id)
        if self.instrument is None:
            self.log.error(f"{self.config.instrument_id} not found – stopping strategy")
            self.stop(); return

        self.request_bars(self.config.bar_type)
        self.subscribe_bars(self.config.bar_type)

    def on_bar(self, bar: Bar) -> None:
        """Buy on the first bar and then do nothing."""
        if bar.bar_type != self.config.bar_type:
            return
        if not self._bought and self.portfolio.is_flat(self.config.instrument_id):
            self._buy()

    # ───────────────────── helpers ──────────────────────
    def _buy(self) -> None:
        if self.instrument is None:
            return
        qty = self.instrument.make_qty(self.config.trade_size)   # type: ignore
        self.submit_order(
            self.order_factory.market(
                instrument_id=self.config.instrument_id,
                order_side=OrderSide.BUY,
                quantity=qty,
            )
        )
        self._bought = True
