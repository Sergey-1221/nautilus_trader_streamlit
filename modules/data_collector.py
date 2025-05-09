import queue

class DataCollector:
    """
    Thread-safe FIFO consumer of events from a MessageBus.
    Can be used in live mode or post-backtest to drain accumulated events.
    """
    def __init__(self, msgbus, topic: str = "DASHBOARD"):
        # Internal queue for storing incoming events
        self._q = queue.Queue()
        # Subscribe the queue's put method to the message bus topic
        self._bus = msgbus
        self._bus.subscribe(topic, self._q.put_nowait)

    def drain(self) -> list:
        """
        Return a list of all collected events and clear the queue.
        """
        events = []
        # Remove items until the queue is empty
        while not self._q.empty():
            events.append(self._q.get())
        return events
