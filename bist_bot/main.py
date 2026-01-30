import time
import pandas as pd
from typing import List

from bist_bot.core.config import TICKERS, DATA_INTERVAL, DATA_POLLING_INTERVAL_SECONDS, ORDER_TYPE
from bist_bot.data.yf_provider import YFDataProvider
from bist_bot.execution.paper_broker import PaperBroker
from bist_bot.strategies.rsi_macd import RsiMacdStrategy
from bist_bot.utils.logger import setup_logger


def run_bot() -> None:
    """Main bot execution loop for polling, signal generation, and paper trading."""
    logger = setup_logger()
    data_provider = YFDataProvider()
    strategy = RsiMacdStrategy()
    broker = PaperBroker()

    logger.info("Starting BIST trading bot...")
    logger.info(f"Tracking tickers: {', '.join(TICKERS)}")
    logger.info(f"Interval: {DATA_INTERVAL}, Polling: {DATA_POLLING_INTERVAL_SECONDS}s")

    while True:
        logger.info("Fetching market data and generating signals...")

        for symbol in TICKERS:
            # Fetch historical data for indicators
            historical_data = data_provider.get_historical_data(symbol, DATA_INTERVAL)

            if historical_data.empty:
                logger.warning(f"No historical data available for {symbol}. Skipping.")
                continue

            # Fetch latest data point for current price
            latest_data = data_provider.get_latest_data(symbol, DATA_INTERVAL)

            if latest_data.empty:
                logger.warning(f"No latest data available for {symbol}. Skipping.")
                continue

            # Generate signal
            signal = strategy.generate_signal(historical_data, latest_data.iloc[0], symbol)
            current_price = latest_data.iloc[0]["Close"] if "Close" in latest_data.columns else latest_data.iloc[0]["close"]

            logger.info(f"{symbol}: Signal = {signal} at price {current_price:.2f}")

            # Execute trade
            if signal == "BUY":
                broker.place_order(symbol, ORDER_TYPE, price=current_price)
            elif signal == "SELL":
                # For simplicity, we sell the full position
                quantity = -broker.get_asset_balance(symbol)  # Use negative quantity to indicate sell
                if quantity != 0:
                    broker.place_order(symbol, ORDER_TYPE, quantity=quantity, price=current_price)

        logger.info("Cycle complete. Waiting for next polling interval...")
        time.sleep(DATA_POLLING_INTERVAL_SECONDS)


if __name__ == "__main__":
    run_bot()
