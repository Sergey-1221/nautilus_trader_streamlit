# Adding Custom Strategies

This short guide explains how to implement and load your own Python strategies in **NautilusTrader** when using the Streamlit dashboard from this repository.

## 1. Define a strategy class

A strategy is a subclass of `Strategy`. Create a configuration dataclass that inherits from `StrategyConfig` and pass it to the strategy's constructor. Call `super().__init__(config)` so the base class can initialize the internals.

```python
from dataclasses import dataclass
from decimal import Decimal
from nautilus_trader.config import StrategyConfig
from nautilus_trader.trading.strategy import Strategy
from nautilus_trader.model.identifiers import InstrumentId

@dataclass
class MyStrategyConfig(StrategyConfig):
    instrument_id: InstrumentId
    trade_size: Decimal = Decimal("1")

class MyStrategy(Strategy):
    def __init__(self, config: MyStrategyConfig) -> None:
        super().__init__(config)

    def on_start(self) -> None:
        self.logger.info(f"Started on {self.config.instrument_id}")

    def on_bar(self, bar) -> None:
        pass  # trading logic goes here
```

## 2. Instantiate and add the strategy

When running a backtest, create the strategy and add it to the engine:

```python
config = MyStrategyConfig(instrument_id=instr.id)
strategy = MyStrategy(config)
engine.add_strategy(strategy)
```

Alternatively you can use `ImportableStrategyConfig` with the path to the module and class names to load it dynamically.

## 3. Run the backtest

After adding the strategy, simply call `engine.run()` or use the helper functions in `modules/backtest_runner.py`.

```
engine.run()
```

The Streamlit dashboard will display the results once the engine finishes.

## Tips

- Avoid heavy initialization in `__init__`; use `on_start()` to subscribe to data and set up state.
- Ensure each strategy instance has a unique `order_id_tag` in its config to prevent ID collisions.
- Strategies won't have a `logger` or `clock` available until `on_start()` is called.
