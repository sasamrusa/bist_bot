from abc import ABC, abstractmethod
import pandas as pd
from bist_bot.core.interfaces import IStrategy


class BaseStrategy(IStrategy):
    """Base class for all trading strategies, implementing the IStrategy interface."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def generate_signal(self, historical_data: pd.DataFrame, current_data: pd.Series, symbol: str) -> str:
        """
        Abstract method to generate a trading signal. Concrete strategies must implement this.

        Args:
            historical_data (pd.DataFrame): Historical OHLCV data.
            current_data (pd.Series): The latest OHLCV data point.
            symbol (str): The trading symbol.

        Returns:
            str: A trading signal ("BUY", "SELL", "HOLD").
        """
        pass

    def generate_signals_batch(
        self,
        historical_data: pd.DataFrame,
        symbol: str,
        interval: str | None = None,
    ) -> pd.Series | None:
        """
        Optional batch signal generation for backtests.
        Return a Series of signals aligned to historical_data index, or None to fall back to per-row logic.
        """
        return None

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"<BaseStrategy(name=\'{self.name}\')>"
