import numpy as np
import pandas as pd

try:
    import pandas_ta as ta
except Exception:
    class _TaFallback:
        @staticmethod
        def ema(series: pd.Series, length: int) -> pd.Series:
            return series.ewm(span=length, adjust=False, min_periods=length).mean()

        @staticmethod
        def rsi(series: pd.Series, length: int = 14) -> pd.Series:
            delta = series.diff()
            gain = delta.clip(lower=0.0)
            loss = -delta.clip(upper=0.0)
            avg_gain = gain.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()
            avg_loss = loss.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()
            rs = avg_gain / avg_loss.replace(0.0, np.nan)
            return 100.0 - (100.0 / (1.0 + rs))

        @staticmethod
        def macd(close: pd.Series, fast: int, slow: int, signal: int) -> pd.DataFrame:
            ema_fast = _TaFallback.ema(close, fast)
            ema_slow = _TaFallback.ema(close, slow)
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
            hist = macd_line - signal_line
            return pd.DataFrame(
                {
                    f"MACD_{fast}_{slow}_{signal}": macd_line,
                    f"MACDs_{fast}_{slow}_{signal}": signal_line,
                    f"MACDh_{fast}_{slow}_{signal}": hist,
                }
            )

        @staticmethod
        def bbands(close: pd.Series, length: int, std: float) -> pd.DataFrame:
            mid = close.rolling(length, min_periods=length).mean()
            sigma = close.rolling(length, min_periods=length).std(ddof=0)
            upper = mid + (sigma * std)
            lower = mid - (sigma * std)
            return pd.DataFrame(
                {
                    f"BBU_{length}_{std}": upper,
                    f"BBM_{length}_{std}": mid,
                    f"BBL_{length}_{std}": lower,
                }
            )

        @staticmethod
        def atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int) -> pd.Series:
            prev_close = close.shift(1)
            tr = pd.concat(
                [
                    (high - low).abs(),
                    (high - prev_close).abs(),
                    (low - prev_close).abs(),
                ],
                axis=1,
            ).max(axis=1)
            return tr.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()

        @staticmethod
        def adx(high: pd.Series, low: pd.Series, close: pd.Series, length: int) -> pd.DataFrame:
            up_move = high.diff()
            down_move = -low.diff()
            plus_dm = pd.Series(
                np.where((up_move > down_move) & (up_move > 0.0), up_move, 0.0),
                index=high.index,
            )
            minus_dm = pd.Series(
                np.where((down_move > up_move) & (down_move > 0.0), down_move, 0.0),
                index=high.index,
            )

            atr = _TaFallback.atr(high, low, close, length).replace(0.0, np.nan)
            plus_di = 100.0 * (plus_dm.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean() / atr)
            minus_di = 100.0 * (minus_dm.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean() / atr)
            dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)) * 100.0
            adx = dx.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()
            return pd.DataFrame({f"ADX_{length}": adx})

    ta = _TaFallback()

from bist_bot.strategies.base_strategy import BaseStrategy
from bist_bot.core.config import (
    DATA_INTERVAL,
    RSI_PERIOD, RSI_OVERBOUGHT, RSI_OVERSOLD,
    MACD_FAST_PERIOD, MACD_SLOW_PERIOD, MACD_SIGNAL_PERIOD,
    TREND_EMA_FAST_PERIOD, TREND_EMA_SLOW_PERIOD,
    BBANDS_PERIOD, BBANDS_STD,
    ATR_PERIOD,
    SIGNAL_SCORE_THRESHOLD,
    ENABLE_VOLUME_FILTER,
    VOLUME_SMA_PERIOD,
    VOLUME_MIN_RATIO,
    ENABLE_MULTI_TIMEFRAME_CONFIRMATION,
    ENABLE_ADX_FILTER,
    ADX_PERIOD,
    ADX_MIN_VALUE,
    ENABLE_MACD_HISTOGRAM_FILTER,
    SIGNAL_EDGE_MIN,
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

    @staticmethod
    def _higher_timeframe_rule(interval: str | None) -> str | None:
        if not interval:
            return "1h"

        mapping = {
            "1m": "15min",
            "2m": "15min",
            "5m": "1h",
            "15m": "1h",
            "30m": "4h",
            "60m": "1D",
            "90m": "1D",
            "1h": "1D",
            "1d": "1W",
            "5d": "1M",
            "1wk": "1M",
            "1mo": "3M",
        }
        return mapping.get(interval)

    @staticmethod
    def _volume_filter(df: pd.DataFrame) -> pd.Series:
        if not ENABLE_VOLUME_FILTER:
            return pd.Series(True, index=df.index, dtype=bool)
        if "volume" not in df.columns:
            return pd.Series(True, index=df.index, dtype=bool)

        volume = pd.to_numeric(df["volume"], errors="coerce")
        min_periods = max(2, min(VOLUME_SMA_PERIOD, 5))
        volume_sma = volume.rolling(VOLUME_SMA_PERIOD, min_periods=min_periods).mean()
        volume_ok = volume >= (volume_sma * VOLUME_MIN_RATIO)
        return volume_ok.fillna(False).astype(bool)

    def _multi_timeframe_filter(self, df: pd.DataFrame, interval: str | None) -> tuple[pd.Series, pd.Series]:
        all_true = pd.Series(True, index=df.index, dtype=bool)
        if not ENABLE_MULTI_TIMEFRAME_CONFIRMATION:
            return all_true, all_true
        if not isinstance(df.index, pd.DatetimeIndex):
            return all_true, all_true

        rule = self._higher_timeframe_rule(interval)
        if not rule:
            return all_true, all_true

        close_series = pd.to_numeric(df["close"], errors="coerce")
        higher_close = close_series.resample(rule).last().dropna()
        if higher_close.empty:
            return all_true, all_true

        higher_df = pd.DataFrame({"close": higher_close})
        higher_df["ema_fast"] = ta.ema(higher_df["close"], length=TREND_EMA_FAST_PERIOD)
        higher_df["ema_slow"] = ta.ema(higher_df["close"], length=TREND_EMA_SLOW_PERIOD)

        htf_up = (higher_df["ema_fast"] > higher_df["ema_slow"]).fillna(False)
        htf_down = (higher_df["ema_fast"] < higher_df["ema_slow"]).fillna(False)

        buy_ok = htf_up.reindex(df.index, method="ffill").fillna(False).astype(bool)
        sell_ok = htf_down.reindex(df.index, method="ffill").fillna(False).astype(bool)
        return buy_ok, sell_ok

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
        df = df.sort_index()
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

        # ADX trend strength
        adx = ta.adx(df["high"], df["low"], df["close"], length=ADX_PERIOD)
        adx_key = f"ADX_{ADX_PERIOD}"
        if adx is None or adx.empty or adx_key not in adx.columns:
            df["adx"] = float("nan")
        else:
            df["adx"] = adx[adx_key]
        
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
        adx = last_row["adx"] if "adx" in last_row else float("nan")
        macdh = last_row["macdh"] if "macdh" in last_row else float("nan")

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
        self._log_non_scalar("adx", adx, df)
        self._log_non_scalar("macdh", macdh, df)

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

        if ENABLE_ADX_FILTER:
            if pd.isna(adx) or adx < ADX_MIN_VALUE:
                return "HOLD"

        volume_ok = self._volume_filter(df).iloc[-1]
        if not volume_ok:
            return "HOLD"

        mtf_buy_ok, mtf_sell_ok = self._multi_timeframe_filter(df, DATA_INTERVAL)
        buy_allowed = bool(mtf_buy_ok.iloc[-1])
        sell_allowed = bool(mtf_sell_ok.iloc[-1])

        if ENABLE_MACD_HISTOGRAM_FILTER:
            hist_buy_ok = pd.notna(macdh) and macdh > 0
            hist_sell_ok = pd.notna(macdh) and macdh < 0
        else:
            hist_buy_ok = True
            hist_sell_ok = True

        buy_edge = score_buy - score_sell
        sell_edge = score_sell - score_buy

        if (
            score_buy >= SIGNAL_SCORE_THRESHOLD
            and buy_edge >= SIGNAL_EDGE_MIN
            and buy_allowed
            and hist_buy_ok
        ):
            return "BUY"
        if (
            score_sell >= SIGNAL_SCORE_THRESHOLD
            and sell_edge >= SIGNAL_EDGE_MIN
            and sell_allowed
            and hist_sell_ok
        ):
            return "SELL"
        return "HOLD"

    def generate_signals_batch(
        self,
        historical_data: pd.DataFrame,
        symbol: str,
        interval: str | None = None,
    ) -> pd.Series | None:
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
        adx = pd.to_numeric(df["adx"], errors="coerce") if "adx" in df.columns else pd.Series(float("nan"), index=df.index)
        macdh = pd.to_numeric(df["macdh"], errors="coerce") if "macdh" in df.columns else pd.Series(float("nan"), index=df.index)

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

        volume_ok = self._volume_filter(df)
        mtf_buy_ok, mtf_sell_ok = self._multi_timeframe_filter(df, interval or DATA_INTERVAL)
        tradable = atr.notna() & (atr > 0) & volume_ok
        if ENABLE_ADX_FILTER:
            tradable = tradable & adx.notna() & (adx >= ADX_MIN_VALUE)

        if ENABLE_MACD_HISTOGRAM_FILTER:
            hist_buy_ok = (macdh > 0).fillna(False)
            hist_sell_ok = (macdh < 0).fillna(False)
        else:
            hist_buy_ok = pd.Series(True, index=df.index, dtype=bool)
            hist_sell_ok = pd.Series(True, index=df.index, dtype=bool)

        buy_edge_ok = (buy_score - sell_score) >= SIGNAL_EDGE_MIN
        sell_edge_ok = (sell_score - buy_score) >= SIGNAL_EDGE_MIN

        signals = pd.Series("HOLD", index=df.index)
        signals.loc[
            tradable
            & mtf_buy_ok
            & hist_buy_ok
            & buy_edge_ok
            & (buy_score >= SIGNAL_SCORE_THRESHOLD)
        ] = "BUY"
        signals.loc[
            tradable
            & mtf_sell_ok
            & hist_sell_ok
            & sell_edge_ok
            & (sell_score >= SIGNAL_SCORE_THRESHOLD)
        ] = "SELL"
        return signals
