"""
DashboardPublisher
──────────────────

• Subscribes ONLY to Nautilus Trader event objects
  (BarEvent, OrderFillEvent, PortfolioUpdateEvent, etc.)—so it does not catch
  its own forwarded messages and avoid infinite recursion.

• Forwards a lightweight version of each event to the "DASHBOARD" topic:
  serializes via `.to_dict()` or falls back to `__dict__`, removes private
  fields, and ensures JSON compatibility.

• Adds the helper flag `_dash_forwarded = True` to avoid reprocessing forwarded events.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Type

from nautilus_trader.common.actor import Actor

# Attempt to import event classes from different Nautilus core versions
EVENT_TYPES: List[Type] = []
try:
    # Newer core versions
    from nautilus_trader.model.events import BarEvent, OrderFillEvent, PortfolioUpdateEvent
    EVENT_TYPES.extend([BarEvent, OrderFillEvent, PortfolioUpdateEvent])
except ImportError:
    try:
        # Older core versions
        from nautilus_trader.events.bar import BarEvent  # type: ignore
        from nautilus_trader.events.order import OrderFillEvent  # type: ignore
        from nautilus_trader.events.portfolio import PortfolioUpdateEvent  # type: ignore
        EVENT_TYPES.extend([BarEvent, OrderFillEvent, PortfolioUpdateEvent])
    except ImportError:
        # If no event types found, live streaming will be disabled
        EVENT_TYPES = []


def _json_safe(obj: Any) -> Any:
    """Ensure the object is JSON-serializable; fallback to str(obj)."""
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        return str(obj)


def _serialize(msg: Any) -> Dict[str, Any]:
    """Convert an event object into a lightweight JSON-safe dict."""
    # 1) If the message supports to_dict(), use it
    if hasattr(msg, "to_dict"):
        try:
            data = msg.to_dict()  # type: ignore[call-arg]
            if isinstance(data, dict):
                return {key: _json_safe(value) for key, value in data.items()}
        except Exception:
            pass

    # 2) Fallback: use public attributes from __dict__
    if hasattr(msg, "__dict__"):
        return {
            key: _json_safe(value)
            for key, value in msg.__dict__.items()
            if not key.startswith("_")
        }

    # 3) Last resort: represent via repr()
    return {"repr": repr(msg)}


class DashboardPublisher(Actor):
    """Publishes filtered, serialized events to the 'DASHBOARD' topic without recursion."""

    def on_start(self) -> None:
        """Subscribe only to supported Nautilus Trader event types."""
        if not EVENT_TYPES:
            self.log.warning(
                "DashboardPublisher: no EVENT_TYPES found; live streaming disabled"
            )
            return

        for ev in EVENT_TYPES:
            # Subscribe to each real event type to receive messages
            self.subscribe(ev)  # type: ignore[attr-defined]

    def on_message(self, message: Any) -> None:
        """Handle incoming messages: serialize, mark, and forward to 'DASHBOARD'."""
        # Skip messages that have already been forwarded
        if getattr(message, "_dash_forwarded", False):
            return

        payload = _serialize(message)
        payload["_dash_forwarded"] = True  # mark to prevent loops

        # Publish only once; since we never subscribe to raw dicts, no recursion
        self.msgbus.publish("DASHBOARD", payload)  # type: ignore[arg-type]
