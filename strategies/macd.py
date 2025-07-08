from __future__ import annotations

from nautilus_trader.core.message import Event
from nautilus_trader.indicators.macd import MovingAverageConvergenceDivergence
from nautilus_trader.model.enums import OrderSide, PositionSide, PriceType
from nautilus_trader.model.events import PositionOpened
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.position import Position
from nautilus_trader.model import Quantity, BarType
from nautilus_trader.model.data import QuoteTick
from nautilus_trader.trading.strategy import Strategy, StrategyConfig


class MACDConfig(StrategyConfig):
    """Configuration for the MACD strategy."""
    instrument_id: InstrumentId
    bar_type: BarType
    fast_period: int = 12
    slow_period: int = 26
    # Original examples used very large quantities which are unsuitable
    # for the modest balances provided by the default backtest engine.
    # Set a realistic size of 1 unit by default.
    trade_size: int = 1
    entry_threshold: float = 0.00010


class MACDStrategy(Strategy):
    """A MACD-based mean-reversion strategy using quote ticks and bars."""
    def __init__(self, config: MACDConfig):
        super().__init__(config)
        # Initialize MACD indicator on mid price
        self.macd = MovingAverageConvergenceDivergence(
            fast_period=config.fast_period,
            slow_period=config.slow_period,
            price_type=PriceType.MID,
        )
        # Strategy parameters
        self.entry_threshold = config.entry_threshold
        self.instrument_id = config.instrument_id
        self.trade_size = Quantity.from_int(config.trade_size)

        # Track current position
        self.position: Position | None = None

    def on_start(self):
        # Subscribe to quote ticks and bar data for our instrument
        self.subscribe_quote_ticks(instrument_id=self.instrument_id)
        self.subscribe_bars(self.config.bar_type)

    def on_stop(self):
        # Close any open positions and unsubscribe
        self.close_all_positions(self.instrument_id)
        self.unsubscribe_quote_ticks(instrument_id=self.instrument_id)
        self.unsubscribe_bars(self.config.bar_type)

    def on_quote_tick(self, tick: QuoteTick):
        # Update MACD with each incoming tick
        self.macd.handle_quote_tick(tick)
        if not self.macd.initialized:
            return  # wait for warm-up

        self.check_for_entry()
        self.check_for_exit()

    def on_event(self, event: Event):
        # Cache position when a new one is opened
        if isinstance(event, PositionOpened):
            self.position = self.cache.position(event.position_id)

    def check_for_entry(self):
        value = self.macd.value
        # Long entry
        if value > self.entry_threshold:
            if self.position is None or self.position.side != PositionSide.LONG:
                order = self.order_factory.market(
                    instrument_id=self.instrument_id,
                    order_side=OrderSide.BUY,
                    quantity=self.trade_size,
                )
                self.submit_order(order)

        # Short entry
        elif value < -self.entry_threshold:
            if self.position is None or self.position.side != PositionSide.SHORT:
                order = self.order_factory.market(
                    instrument_id=self.instrument_id,
                    order_side=OrderSide.SELL,
                    quantity=self.trade_size,
                )
                self.submit_order(order)

    def check_for_exit(self):
        value = self.macd.value
        if not self.position:
            return

        # Exit short when MACD crosses above zero
        if self.position.side == PositionSide.SHORT and value >= 0.0:
            self.close_position(self.position)
        # Exit long when MACD crosses below zero
        elif self.position.side == PositionSide.LONG and value <= 0.0:
            self.close_position(self.position)
