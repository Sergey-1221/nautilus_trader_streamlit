import pickle
from pathlib import Path
from typing import Union

def save_backtest_result(res: dict, path: Union[str, Path]) -> None:
    """Save backtest results to the specified file using pickle."""
    with open(path, "wb") as f:
        pickle.dump(res, f)

def load_backtest_result(src: Union[str, Path]) -> dict:
    """
    Load backtest results from the given file.
    Returns the results as a dict, or None if the file is missing or cannot be unpickled.
    """
    try:
        with open(src, "rb") as f:
            return pickle.load(f)
    except Exception:
        # Return None on any error (e.g., file not found or pickle error)
        return None
