from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf
from typing import List, Dict, Any

from bist_bot.core.interfaces import IDataProvider
from bist_bot.core.config import TICKERS, DATA_INTERVAL
from bist_bot.utils.logger import setup_logger


LOGGER = setup_logger("bist_bot.data")


class YFDataProvider(IDataProvider):
    """yfinance implementation of the IDataProvider interface."""

    def __init__(self):
        self.tickers = TICKERS
        self.interval = DATA_INTERVAL

    def get_historical_data(self, symbol: str, interval: str, start_date: datetime = None, end_date: datetime = None) -> pd.DataFrame:
        """
        Fetches historical data for a given symbol and interval using yfinance.
        If start_date and end_date are not provided, it fetches data for the last 60 days.
        """
        if start_date is None:
            start_date = datetime.now() - timedelta(days=60) # Default to last 60 days
        if end_date is None:
            end_date = datetime.now()

        LOGGER.info(
            "history_request symbol=%s interval=%s start=%s end=%s",
            symbol,
            interval,
            start_date,
            end_date,
        )
        try:
            ticker = yf.Ticker(symbol)
            history = ticker.history(interval=interval, start=start_date, end=end_date)
            LOGGER.info("history_result symbol=%s rows=%s", symbol, len(history))
            return history
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("history_error symbol=%s interval=%s err=%s", symbol, interval, exc)
            return pd.DataFrame()

    def get_latest_data(self, symbol: str, interval: str) -> pd.DataFrame:
        """
        Fetches the latest data point for a given symbol and interval using yfinance.
        """
        # Fetching a small period to get the latest data. 1 day period should give the last bar.
        # For intraday intervals, '1d' period might not return the latest '5m' bar.
        # Fetch a slightly larger period and take the last one.
        if interval in ["1m", "2m", "5m", "15m", "30m", "60m", "90m"]:
            period = "2d" # Fetch last 2 days to ensure we get the latest intraday bar
        else:
            period = "7d" # For daily or longer intervals, a week should be enough

        LOGGER.info("latest_request symbol=%s interval=%s period=%s", symbol, interval, period)
        try:
            ticker = yf.Ticker(symbol)
            latest_data = ticker.history(interval=interval, period=period)
            LOGGER.info("latest_result symbol=%s rows=%s", symbol, len(latest_data))
            if not latest_data.empty:
                return pd.DataFrame([latest_data.iloc[-1]]) # Return the last row as a DataFrame
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("latest_error symbol=%s interval=%s err=%s", symbol, interval, exc)
        return pd.DataFrame() # Return empty DataFrame if no data is found
