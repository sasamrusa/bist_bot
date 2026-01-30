from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Dict, Any
import pandas as pd


class IDataProvider(ABC):
    """Abstract Base Class for data providers."""

    @abstractmethod
    def get_historical_data(self, symbol: str, interval: str, start_date: datetime = None, end_date: datetime = None) -> pd.DataFrame:
        """
        Fetches historical data for a given symbol and interval.

        Args:
            symbol (str): The trading symbol (e.g., "AKBNK.IS").
            interval (str): The data interval (e.g., "1m", "5m", "1d").
            start_date (datetime, optional): The start date for historical data. Defaults to None.
            end_date (datetime, optional): The end date for historical data. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame with historical OHLCV data.
        """
        pass

    @abstractmethod
    def get_latest_data(self, symbol: str, interval: str) -> pd.DataFrame:
        """
        Fetches the latest data point for a given symbol and interval.

        Args:
            symbol (str): The trading symbol.
            interval (str): The data interval.

        Returns:
            pd.DataFrame: A DataFrame with the latest OHLCV data.
        """
        pass


class IBroker(ABC):
    """Abstract Base Class for brokerage interactions."""

    @abstractmethod
    def place_order(self, symbol: str, order_type: str, quantity: float, price: float = None) -> Dict[str, Any]:
        """
        Places an order with the broker.

        Args:
            symbol (str): The trading symbol.
            order_type (str): Type of order (e.g., "market", "limit").
            quantity (float): The quantity to trade.
            price (float, optional): The price for limit orders. Defaults to None.

        Returns:
            Dict[str, Any]: A dictionary containing order confirmation details.
        """
        pass

    @abstractmethod
    def get_account_balance(self) -> Dict[str, float]:
        """
        Retrieves the current account balance.

        Returns:
            Dict[str, float]: A dictionary with balance details (e.g., cash, total_value).
        """
        pass

    @abstractmethod
    def get_open_positions(self) -> List[Dict[str, Any]]:
        """
        Retrieves a list of currently open positions.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, each representing an open position.
        """
        pass
    
    @abstractmethod
    def get_asset_balance(self, symbol: str) -> float:
        """
        Retrieves the balance for a specific asset.

        Args:
            symbol (str): The trading symbol.

        Returns:
            float: The quantity of the asset held.
        """
        pass


class IStrategy(ABC):
    """Abstract Base Class for trading strategies."""

    @abstractmethod
    def generate_signal(self, historical_data: pd.DataFrame, current_data: pd.Series, symbol: str) -> str:
        """
        Generates a trading signal (BUY, SELL, HOLD) based on market data.

        Args:
            historical_data (pd.DataFrame): Historical OHLCV data.
            current_data (pd.Series): The latest OHLCV data point.
            symbol (str): The trading symbol.

        Returns:
            str: A trading signal ("BUY", "SELL", "HOLD").
        """
        pass
