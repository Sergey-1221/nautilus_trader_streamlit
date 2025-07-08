from collections import deque
from datetime import datetime
from decimal import Decimal

from nautilus_trader.config import StrategyConfig
from nautilus_trader.indicators.base.indicator import Indicator
from nautilus_trader.model import Bar, BarType, InstrumentId
from nautilus_trader.model.instruments import Instrument
from nautilus_trader.model.enums import OrderSide, TimeInForce
from nautilus_trader.model.orders import MarketOrder
from nautilus_trader.trading.strategy import Strategy
# Import Money if it's not available via model.objects
# from nautilus_trader.model.objects import Money


# ATR (Average True Range) indicator for the trailing stop
class ATRIndicator(Indicator):
    def __init__(self, period: int):
        super().__init__(params=[period])
        self.period = period
        self._values: deque[Decimal] = deque(maxlen=period)
        self._current_atr: Decimal | None = None
        self._last_close: Decimal | None = None

    def handle_bar(self, bar: Bar) -> None:
        """Update ATR when a new bar arrives."""
        if not self.has_inputs:
            self._set_has_inputs(True)

        tr: Decimal
        if self._last_close is None:
            tr = bar.high - bar.low
        else:
            prev_close = self._last_close
            tr = max(bar.high - bar.low, abs(bar.high - prev_close), abs(bar.low - prev_close))

        self._values.append(tr)

        if len(self._values) == self.period:
            if self._current_atr is None:
                self._current_atr = sum(self._values) / Decimal(self.period)
            else:
                self._current_atr = (self._current_atr * Decimal(self.period - 1) + tr) / Decimal(self.period)

            if not self.initialized:
                self._set_initialized(True)

        self._last_close = bar.close

    @property
    def value(self) -> Decimal | None:
        return self._current_atr

    def _reset(self) -> None:
        self._values.clear()
        self._current_atr = None
        self._last_close = None


# Trend Following strategy configuration
class TrendFollowingConfig(StrategyConfig):
    instrument_id: InstrumentId
    bar_type: BarType
    N: int = 20
    atr_period: int = 14
    # A smaller default size keeps sample backtests running even with
    # the modest 10k USDT balance configured in ``backtest_runner``.
    trade_size: Decimal = Decimal("0.1")


# Trend following strategy based on Donchian channel breakout
class TrendFollowingStrategy(Strategy):
    def __init__(self, config: TrendFollowingConfig) -> None:
        super().__init__(config)
        self.instrument: Instrument | None = None
        self.atr = ATRIndicator(period=self.config.atr_period)
        self.recent_highs: deque[Decimal] = deque(maxlen=self.config.N)
        self.recent_lows: deque[Decimal] = deque(maxlen=self.config.N)
        self.position_side: str | None = None
        self.entry_price: Decimal | None = None
        self.trailing_stop: Decimal | None = None
        self.highest_since_entry: Decimal | None = None
        self.lowest_since_entry: Decimal | None = None
        self.time_started: datetime | None = None

    def on_start(self) -> None:
        self.instrument = self.cache.instrument(self.config.instrument_id)
        if self.instrument is None:
            self.log.error(f"Instrument {self.config.instrument_id} not found in cache")
            self.stop()
            return

        self.time_started = self.clock.utc_now()

        self.register_indicator_for_bars(self.config.bar_type, self.atr)

        self.request_bars(self.config.bar_type)

        self.subscribe_bars(self.config.bar_type)


    def on_bar(self, bar: Bar) -> None:
        """Handle each new bar (historical or real-time)."""
        if self.time_started:
            time_started_ns = int(self.time_started.timestamp() * 1_000_000_000)
            if bar.ts_event < time_started_ns:
                self.recent_highs.append(bar.high)
                self.recent_lows.append(bar.low)
                return

        self.recent_highs.append(bar.high)
        self.recent_lows.append(bar.low)

        if len(self.recent_highs) < self.config.N or not self.atr.initialized or self.atr.value is None:
            return

        atr_value: Decimal = self.atr.value
        highest_N = max(self.recent_highs)
        lowest_N = min(self.recent_lows)

        if self.position_side is None:
            if bar.close >= highest_N:
                qty = self.instrument.make_qty(self.config.trade_size)
                order: MarketOrder = self.order_factory.market(
                    instrument_id=self.config.instrument_id,
                    order_side=OrderSide.BUY,
                    quantity=qty,
                    time_in_force=TimeInForce.GTC
                )
                self.submit_order(order)
                self.position_side = 'LONG'
                self.entry_price = bar.close
                self.trailing_stop = bar.close - (Decimal("2") * atr_value)
                self.highest_since_entry = bar.close

            elif bar.close <= lowest_N:
                qty = self.instrument.make_qty(self.config.trade_size)
                order: MarketOrder = self.order_factory.market(
                    instrument_id=self.config.instrument_id,
                    order_side=OrderSide.SELL,
                    quantity=qty,
                    time_in_force=TimeInForce.GTC
                )
                self.submit_order(order)
                self.position_side = 'SHORT'
                self.entry_price = bar.close
                self.trailing_stop = bar.close + (Decimal("2") * atr_value)
                self.lowest_since_entry = bar.close
        else:
            if self.position_side == 'LONG':
                if self.highest_since_entry is None or bar.high > self.highest_since_entry:
                    self.highest_since_entry = bar.high

                if self.highest_since_entry is not None:
                    new_stop = self.highest_since_entry - (Decimal("2") * atr_value)
                    if self.trailing_stop is None or new_stop > self.trailing_stop:
                        self.trailing_stop = new_stop

                if self.trailing_stop is not None and bar.close <= self.trailing_stop:
                    qty = self.instrument.make_qty(self.config.trade_size)
                    exit_order: MarketOrder = self.order_factory.market(
                        instrument_id=self.config.instrument_id,
                        order_side=OrderSide.SELL,
                        quantity=qty,
                        time_in_force=TimeInForce.GTC
                    )
                    self.submit_order(exit_order)
                    self.position_side = None
                    self.entry_price = None
                    self.trailing_stop = None
                    self.highest_since_entry = None

            elif self.position_side == 'SHORT':
                if self.lowest_since_entry is None or bar.low < self.lowest_since_entry:
                    self.lowest_since_entry = bar.low

                if self.lowest_since_entry is not None:
                    new_stop = self.lowest_since_entry + (Decimal("2") * atr_value)
                    if self.trailing_stop is None or new_stop < self.trailing_stop:
                        self.trailing_stop = new_stop

                if self.trailing_stop is not None and bar.close >= self.trailing_stop:
                    qty = self.instrument.make_qty(self.config.trade_size)
                    exit_order: MarketOrder = self.order_factory.market(
                        instrument_id=self.config.instrument_id,
                        order_side=OrderSide.BUY,
                        quantity=qty,
                        time_in_force=TimeInForce.GTC
                    )
                    self.submit_order(exit_order)
                    self.position_side = None
                    self.entry_price = None
                    self.trailing_stop = None
                    self.lowest_since_entry = None

    def on_stop(self) -> None:
        # Cancel all active orders for the strategy instrument
        if self.instrument:
             self.cancel_all_orders(self.config.instrument_id)

        # If there is an open position, close it with a market order
        if self.instrument:
            net_exposure = self.portfolio.net_exposure(self.config.instrument_id)
            # Corrected comparison: use .as_decimal() to get a Decimal value
            if net_exposure.as_decimal() != Decimal("0"):
                # Use .as_decimal() for comparison and value retrieval
                side = OrderSide.SELL if net_exposure.as_decimal() > Decimal("0") else OrderSide.BUY
                qty = self.instrument.make_qty(abs(net_exposure.as_decimal())) # Use .as_decimal() here as well
                close_order: MarketOrder = self.order_factory.market(
                    instrument_id=self.config.instrument_id,
                    order_side=side,
                    quantity=qty,
                    time_in_force=TimeInForce.FOK
                )
                self.submit_order(close_order)

