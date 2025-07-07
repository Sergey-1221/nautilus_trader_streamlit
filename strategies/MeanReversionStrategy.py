from math import sqrt
from decimal import Decimal
from collections import deque

# Импорты из библиотеки Nautilus Trader
from nautilus_trader.model.data import Bar, BarType, QuoteTick, TradeTick
from nautilus_trader.core.nautilus_pyo3 import InstrumentId
from nautilus_trader.model.objects import Quantity # Price импортируется неявно через Bar
from nautilus_trader.model.enums import OrderSide, TimeInForce
from nautilus_trader.model.orders import MarketOrder
from nautilus_trader.trading.strategy import Strategy, StrategyConfig
from nautilus_trader.indicators.base.indicator import Indicator

# --- Пользовательский индикатор Bollinger Bands на Python ---
class BollingerBandsIndicator(Indicator):
    def __init__(self, period: int, std_factor: float = 2.0):
        params = [period, std_factor]
        super().__init__(params)

        self.period = period
        self.std_factor = std_factor
        self.values = deque(maxlen=self.period)
        self.middle: Decimal | None = None
        self.upper: Decimal | None = None
        self.lower: Decimal | None = None

    def handle_bar(self, bar: Bar) -> None:
        if not self.has_inputs:
            self._set_has_inputs(True)

        price = bar.close.as_decimal()
        self.values.append(price)

        if len(self.values) == self.period:
            prices_list = list(self.values)
            current_mean = sum(prices_list) / Decimal(self.period)
            variance = sum((p - current_mean) ** 2 for p in prices_list) / Decimal(self.period)
            current_stddev = Decimal(str(sqrt(float(max(Decimal('0'), variance)))))
            self.middle = current_mean
            self.upper = current_mean + (Decimal(str(self.std_factor)) * current_stddev)
            self.lower = current_mean - (Decimal(str(self.std_factor)) * current_stddev)
            if not self.initialized:
                self._set_initialized(True)

    def reset(self) -> None:
        self.values.clear()
        self.middle = None
        self.upper = None
        self.lower = None
        self._set_has_inputs(False)
        self._set_initialized(False)

    def handle_quote_tick(self, tick: QuoteTick) -> None:
        pass

    def handle_trade_tick(self, tick: TradeTick) -> None:
        pass

# --- Пользовательский индикатор RSI на Python ---
class RSIIndicator(Indicator):
    def __init__(self, period: int):
        params = [period]
        super().__init__(params)

        self.period = period
        self.last_close: Decimal | None = None
        self.avg_gain: float = 0.0
        self.avg_loss: float = 0.0
        self.count: int = 0
        self.value: Decimal | None = None

    def handle_bar(self, bar: Bar) -> None:
        if not self.has_inputs:
            self._set_has_inputs(True)

        price = bar.close.as_decimal()

        if self.last_close is None:
            self.last_close = price
            return

        change = price - self.last_close
        gain = float(change) if change > 0 else 0.0
        loss = float(-change) if change < 0 else 0.0

        if not self.initialized:
            self.avg_gain += gain
            self.avg_loss += loss
            self.count += 1
            if self.count == self.period:
                self.avg_gain /= self.period
                self.avg_loss /= self.period
                self._calculate_rsi_value()
                self._set_initialized(True)
        else:
            self.avg_gain = (self.avg_gain * (self.period - 1) + gain) / self.period
            self.avg_loss = (self.avg_loss * (self.period - 1) + loss) / self.period
            self._calculate_rsi_value()

        self.last_close = price

    def _calculate_rsi_value(self) -> None:
        if self.avg_loss == 0:
            self.value = Decimal("100.0")
        else:
            rs = self.avg_gain / self.avg_loss
            rsi = 100.0 - (100.0 / (1.0 + rs))
            self.value = Decimal(str(rsi))

    def reset(self) -> None:
        self.last_close = None
        self.avg_gain = 0.0
        self.avg_loss = 0.0
        self.count = 0
        self.value = None
        self._set_has_inputs(False)
        self._set_initialized(False)

    def handle_quote_tick(self, tick: QuoteTick) -> None:
        pass

    def handle_trade_tick(self, tick: TradeTick) -> None:
        pass


# Конфигурация стратегии Mean Reversion
class MeanReversionConfig(StrategyConfig):
    instrument_id: InstrumentId
    bar_type: BarType
    boll_period: int = 20
    boll_std: float = 2.0
    rsi_period: int = 14
    # Use a small default position size so the demo account can
    # open trades without exceeding its 10k USDT balance.
    trade_size: Decimal = Decimal("0.1")

# Стратегия возврата к среднему
class MeanReversionStrategy(Strategy):
    def __init__(self, config: MeanReversionConfig) -> None:
        super().__init__(config)
        self.boll = BollingerBandsIndicator(period=self.config.boll_period, std_factor=self.config.boll_std)
        self.rsi = RSIIndicator(period=self.config.rsi_period)
        self.position_side: str | None = None
        self.time_started = None

    def on_start(self) -> None:
        """Подписка на данные и инициализация индикаторов при старте."""
        self.instrument = self.cache.instrument(self.config.instrument_id)
        if self.instrument is None:
            self.log.error(f"Instrument {self.config.instrument_id} not found")
            self.stop()
            return

        # Преобразуем Timestamp, возвращаемый clock.utc_now(), в целое число наносекунд
        self.time_started = self.clock.utc_now().value

        # Регистрируем индикаторы
        self.register_indicator_for_bars(self.config.bar_type, self.boll)
        self.register_indicator_for_bars(self.config.bar_type, self.rsi)

        self.request_bars(self.config.bar_type)
        self.subscribe_bars(self.config.bar_type)

    def on_bar(self, bar: Bar) -> None:
        # Проверяем готовность индикаторов, используя их встроенный флаг initialized
        if not self.boll.initialized or not self.rsi.initialized:
            return

        # Игнорируем исторические бары после прогрева, используя ts_event
        if self.time_started and bar.ts_event < self.time_started:
            # Это исторический бар, который уже был использован для прогрева индикаторов.
            # Пропускаем его для генерации торговых сигналов.
            return

        mid = self.boll.middle
        upper = self.boll.upper
        lower = self.boll.lower
        rsi_val = self.rsi.value

        if mid is None or upper is None or lower is None or rsi_val is None:
            self.log.warning(
                f"Indicator values not ready at {bar.ts_event}: "
                f"BB({self.boll.middle}, {self.boll.upper}, {self.boll.lower}), "
                f"RSI({self.rsi.value})"
            )
            return

        current_close_price = bar.close.as_decimal()

        # === Логика входа (Entry signals) ===
        if self.position_side is None:
            if current_close_price < lower and rsi_val < Decimal("30"):
                qty = self.instrument.make_qty(self.config.trade_size)
                order = self.order_factory.market(
                    instrument_id=self.config.instrument_id,
                    order_side=OrderSide.BUY,
                    quantity=qty,
                    time_in_force=TimeInForce.GTC
                )
                self.submit_order(order)
                self.position_side = 'LONG'
                self.log.info(
                    f"Opened LONG position at {current_close_price} on bar {bar.ts_event}"
                )
            elif current_close_price > upper and rsi_val > Decimal("70"):
                qty = self.instrument.make_qty(self.config.trade_size)
                order = self.order_factory.market(
                    instrument_id=self.config.instrument_id,
                    order_side=OrderSide.SELL,
                    quantity=qty,
                    time_in_force=TimeInForce.GTC
                )
                self.submit_order(order)
                self.position_side = 'SHORT'
                self.log.info(
                    f"Opened SHORT position at {current_close_price} on bar {bar.ts_event}"
                )
        # === Логика выхода (Exit signals) ===
        else:
            if self.position_side == 'LONG' and current_close_price >= mid:
                qty = self.instrument.make_qty(self.config.trade_size)
                order = self.order_factory.market(
                    instrument_id=self.config.instrument_id,
                    order_side=OrderSide.SELL,
                    quantity=qty,
                    time_in_force=TimeInForce.GTC
                )
                self.submit_order(order)
                self.log.info(
                    f"Closed LONG position at {current_close_price} on bar {bar.ts_event}"
                )
                self.position_side = None
            elif self.position_side == 'SHORT' and current_close_price <= mid:
                qty = self.instrument.make_qty(self.config.trade_size)
                order = self.order_factory.market(
                    instrument_id=self.config.instrument_id,
                    order_side=OrderSide.BUY,
                    quantity=qty,
                    time_in_force=TimeInForce.GTC
                )
                self.submit_order(order)
                self.log.info(
                    f"Closed SHORT position at {current_close_price} on bar {bar.ts_event}"
                )
                self.position_side = None

    def on_stop(self) -> None:
        """Очистка ресурсов при остановке стратегии."""
        # Отменяем все активные заявки для инструмента стратегии
        self.cancel_all_orders(instrument_id=self.config.instrument_id)

        # Закрываем открытую позицию, если она осталась
        # Получаем net_exposure, который может быть Money или None
        net_exposure_money = self.portfolio.net_exposure(self.config.instrument_id)

        # Проверяем, что net_exposure_money не None И его значение в Decimal не равно нулю
        if net_exposure_money is not None and net_exposure_money.as_decimal() != Decimal("0"):
            side = (
                OrderSide.SELL if net_exposure_money.as_decimal() > Decimal("0") else OrderSide.BUY
            )
            # Получаем абсолютное значение количества в Decimal
            abs_qty_decimal = abs(net_exposure_money.as_decimal())
            qty = self.instrument.make_qty(abs_qty_decimal)

            # Создаем и отправляем ордер на закрытие
            close_order = self.order_factory.market(
                instrument_id=self.config.instrument_id,
                order_side=side,
                quantity=qty,
                time_in_force=TimeInForce.FOK,
            )
            self.submit_order(close_order)
            self.log.info(
                f"Submitted closing order for {self.config.instrument_id} with quantity {qty}"
            )

