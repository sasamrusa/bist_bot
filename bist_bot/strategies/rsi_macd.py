import pandas as pd
import pandas_ta as ta
from typing import Tuple

from bist_bot.strategies.base_strategy import BaseStrategy
from bist_bot.core.config import (
    RSI_PERIOD, RSI_OVERBOUGHT, RSI_OVERSOLD,
    MACD_FAST_PERIOD, MACD_SLOW_PERIOD, MACD_SIGNAL_PERIOD,
    TREND_EMA_FAST_PERIOD, TREND_EMA_SLOW_PERIOD,
    BBANDS_PERIOD, BBANDS_STD,
    ATR_PERIOD,
    SIGNAL_SCORE_THRESHOLD,
)
from bist_bot.utils.logger import setup_logger


LOGGER = setup_logger("bist_bot.strategy")


class RsiMacdStrategy(BaseStrategy):
    """RSI + MACD + EMA trend + Bollinger + ATR scoring strategy."""

    def __init__(self):
        super().__init__("RSI_MACD_Trend")

    @staticmethod
    def _log_non_scalar(name: str, value: object, df: pd.DataFrame) -> None:
        if not pd.api.types.is_scalar(value):
            dup_cols = df.columns[df.columns.duplicated()].tolist()
            LOGGER.error(
                "strategy_non_scalar name=%s type=%s dup_cols=%s",
                name,
                type(value),
                dup_cols,
            )

    def _calculate_indicators(self, historical_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates RSI, MACD, and SMA indicators.

        Args:
            historical_data (pd.DataFrame): Historical OHLCV data.

        Returns:
            pd.DataFrame: DataFrame with calculated indicators.
        """
        if historical_data.empty:
            return historical_data

        # Ensure columns are in the correct format
        df = historical_data.copy()
        df.columns = [col.lower() for col in df.columns]

        # Calculate RSI
        df["rsi"] = ta.rsi(df["close"], length=RSI_PERIOD)

        # Calculate MACD
        macd = ta.macd(
            df["close"],
            fast=MACD_FAST_PERIOD,
            slow=MACD_SLOW_PERIOD,
            signal=MACD_SIGNAL_PERIOD,
        )
        macd_key = f"MACD_{MACD_FAST_PERIOD}_{MACD_SLOW_PERIOD}_{MACD_SIGNAL_PERIOD}"
        macdh_key = f"MACDh_{MACD_FAST_PERIOD}_{MACD_SLOW_PERIOD}_{MACD_SIGNAL_PERIOD}"
        macds_key = f"MACDs_{MACD_FAST_PERIOD}_{MACD_SLOW_PERIOD}_{MACD_SIGNAL_PERIOD}"
        if macd is None or macd.empty or macd_key not in macd.columns:
            df["macd"] = pd.NA
            df["macds"] = pd.NA
            df["macdh"] = pd.NA
        else:
            df["macd"] = macd[macd_key]
            df["macds"] = macd[macds_key]  # MACD Signal
            df["macdh"] = macd[macdh_key]  # MACD Histogram

        # Calculate EMAs for trend
        df["ema_fast"] = ta.ema(df["close"], length=TREND_EMA_FAST_PERIOD)
        df["ema_slow"] = ta.ema(df["close"], length=TREND_EMA_SLOW_PERIOD)

        # Bollinger Bands
        bbands = ta.bbands(df["close"], length=BBANDS_PERIOD, std=BBANDS_STD)
        bbu_key = f"BBU_{BBANDS_PERIOD}_{BBANDS_STD}"
        bbm_key = f"BBM_{BBANDS_PERIOD}_{BBANDS_STD}"
        bbl_key = f"BBL_{BBANDS_PERIOD}_{BBANDS_STD}"
        if bbands is None or bbands.empty or bbu_key not in bbands.columns:
            df["bbu"] = float("nan")
            df["bbm"] = float("nan")
            df["bbl"] = float("nan")
        else:
            df["bbu"] = bbands[bbu_key]
            df["bbm"] = bbands[bbm_key]
            df["bbl"] = bbands[bbl_key]

        # ATR for volatility filter
        df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=ATR_PERIOD)
        
        return df

    def generate_signal(self, historical_data: pd.DataFrame, current_data: pd.Series, symbol: str) -> str:
        """
        Generates a trading signal based on RSI, MACD, and a trend-following SMA.

        Signal Logic:
        - BUY: RSI oversold, MACD crossing above signal line, and price above SMA trend.
        - SELL: RSI overbought, MACD crossing below signal line, and price below SMA trend.
        - HOLD: Otherwise.
        """
        df = self._calculate_indicators(historical_data)

        if df.empty or len(df) < max(
            RSI_PERIOD,
            MACD_SLOW_PERIOD,
            TREND_EMA_SLOW_PERIOD,
            BBANDS_PERIOD,
            ATR_PERIOD,
        ) + 1:
            return "HOLD" # Not enough data for indicators

        if "macd" not in df.columns or "macds" not in df.columns:
            return "HOLD"

        # Get the latest indicator values
        last_row = df.iloc[-1]
        second_last_row = df.iloc[-2]

        rsi = last_row["rsi"]
        macd = last_row["macd"]
        macds = last_row["macds"]
        ema_fast = last_row["ema_fast"]
        ema_slow = last_row["ema_slow"]
        current_close = last_row["close"]
        bbu = last_row["bbu"]
        bbl = last_row["bbl"]
        atr = last_row["atr"]

        prev_macd = second_last_row["macd"]
        prev_macds = second_last_row["macds"]

        if df.columns.duplicated().any():
            LOGGER.warning("strategy_duplicate_columns columns=%s", df.columns[df.columns.duplicated()].tolist())

        self._log_non_scalar("rsi", rsi, df)
        self._log_non_scalar("macd", macd, df)
        self._log_non_scalar("macds", macds, df)
        self._log_non_scalar("prev_macd", prev_macd, df)
        self._log_non_scalar("prev_macds", prev_macds, df)
        self._log_non_scalar("ema_fast", ema_fast, df)
        self._log_non_scalar("ema_slow", ema_slow, df)
        self._log_non_scalar("close", current_close, df)
        self._log_non_scalar("bbu", bbu, df)
        self._log_non_scalar("bbl", bbl, df)
        self._log_non_scalar("atr", atr, df)

        has_macd = (
            pd.notna(macd)
            and pd.notna(macds)
            and pd.notna(prev_macd)
            and pd.notna(prev_macds)
        )
        if has_macd:
            macd_buy_signal = macd > macds and prev_macd <= prev_macds
            macd_sell_signal = macd < macds and prev_macd >= prev_macds
        else:
            macd_buy_signal = False
            macd_sell_signal = False

        # Trend Filter (EMA cross)
        if pd.notna(ema_fast) and pd.notna(ema_slow):
            uptrend = ema_fast > ema_slow
            downtrend = ema_fast < ema_slow
        else:
            uptrend = False
            downtrend = False

        score_buy = 0
        score_sell = 0

        if pd.notna(rsi) and rsi < RSI_OVERSOLD:
            score_buy += 1
        if pd.notna(rsi) and rsi > RSI_OVERBOUGHT:
            score_sell += 1

        if macd_buy_signal:
            score_buy += 1
        if macd_sell_signal:
            score_sell += 1

        if pd.notna(bbl) and pd.notna(current_close) and current_close <= bbl:
            score_buy += 1
        if pd.notna(bbu) and pd.notna(current_close) and current_close >= bbu:
            score_sell += 1

        if uptrend:
            score_buy += 1
        if downtrend:
            score_sell += 1

        # Volatility filter: only trade when ATR is present and non-zero
        if pd.isna(atr) or atr <= 0:
            return "HOLD"

        if score_buy >= SIGNAL_SCORE_THRESHOLD and score_buy >= score_sell:
            return "BUY"
        if score_sell >= SIGNAL_SCORE_THRESHOLD and score_sell > score_buy:
            return "SELL"
        return "HOLD"

    def generate_signals_batch(self, historical_data: pd.DataFrame, symbol: str) -> pd.Series | None:
        """
        Vectorized signal generation for backtests.
        Returns a Series of signals aligned to the historical_data index.
        """
        df = self._calculate_indicators(historical_data)
        if df.empty:
            return None

        required = max(
            RSI_PERIOD,
            MACD_SLOW_PERIOD,
            TREND_EMA_SLOW_PERIOD,
            BBANDS_PERIOD,
            ATR_PERIOD,
        ) + 1
        if len(df) < required:
            return None

        rsi = df["rsi"]
        if "macd" not in df.columns or "macds" not in df.columns:
            return None

        macd = df["macd"]
        macds = df["macds"]
        ema_fast = df["ema_fast"]
        ema_slow = df["ema_slow"]
        close = df["close"]
        bbu = pd.to_numeric(df["bbu"], errors="coerce")
        bbl = pd.to_numeric(df["bbl"], errors="coerce")
        atr = df["atr"]

        macd_buy = (macd > macds) & (macd.shift(1) <= macds.shift(1))
        macd_sell = (macd < macds) & (macd.shift(1) >= macds.shift(1))
        macd_buy = macd_buy.fillna(False)
        macd_sell = macd_sell.fillna(False)

        uptrend = (ema_fast > ema_slow).fillna(False)
        downtrend = (ema_fast < ema_slow).fillna(False)

        rsi_buy = (rsi < RSI_OVERSOLD).fillna(False)
        rsi_sell = (rsi > RSI_OVERBOUGHT).fillna(False)
        bb_buy = (close <= bbl).fillna(False)
        bb_sell = (close >= bbu).fillna(False)

        buy_score = (
            rsi_buy.astype(int)
            + macd_buy.astype(int)
            + bb_buy.astype(int)
            + uptrend.astype(int)
        )
        sell_score = (
            rsi_sell.astype(int)
            + macd_sell.astype(int)
            + bb_sell.astype(int)
            + downtrend.astype(int)
        )

        tradable = atr.notna() & (atr > 0)

        signals = pd.Series("HOLD", index=df.index)
        signals.loc[tradable & (buy_score >= SIGNAL_SCORE_THRESHOLD) & (buy_score >= sell_score)] = "BUY"
        signals.loc[tradable & (sell_score >= SIGNAL_SCORE_THRESHOLD) & (sell_score > buy_score)] = "SELL"
        return signals
