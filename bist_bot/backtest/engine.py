from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List

import pandas as pd

from bist_bot.core.config import BACKTEST_STARTING_CASH
from bist_bot.core.interfaces import IStrategy
from bist_bot.data.yf_provider import YFDataProvider
from bist_bot.utils.logger import setup_logger


LOGGER = setup_logger("bist_bot.backtest")


@dataclass
class BacktestResult:
    symbol: str
    start: datetime
    end: datetime
    initial_cash: float
    final_value: float
    profit_loss: float
    profit_loss_pct: float
    duration_days: float
    trades: int
    data_points: int
    buy_markers: List[Dict[str, float]]
    sell_markers: List[Dict[str, float]]
    equity_curve: List[Dict[str, float]]
    price_series: List[Dict[str, float]]
    has_strategy_trades: bool


class BacktestEngine:
    """Simple backtesting engine using historical OHLCV data."""

    def __init__(self, data_provider: YFDataProvider, strategy: IStrategy):
        self.data_provider = data_provider
        self.strategy = strategy

    def run(self, symbol: str, interval: str, start: datetime, end: datetime) -> BacktestResult:
        effective_start = self._clamp_start_date(interval, start, end)
        if effective_start != start:
            LOGGER.warning(
                "backtest_clamp_start symbol=%s interval=%s requested=%s effective=%s",
                symbol,
                interval,
                start,
                effective_start,
            )

        LOGGER.info(
            "backtest_start symbol=%s interval=%s start=%s end=%s",
            symbol,
            interval,
            effective_start,
            end,
        )
        historical = self.data_provider.get_historical_data(symbol, interval, effective_start, end)
        if historical.empty:
            LOGGER.warning(
                "backtest_no_data symbol=%s interval=%s start=%s end=%s effective_start=%s",
                symbol,
                interval,
                start,
                end,
                effective_start,
            )
            LOGGER.info(
                "backtest_returning_empty_result symbol=%s interval=%s data_points=0",
                symbol,
                interval,
            )
            return BacktestResult(
                symbol=symbol,
                start=effective_start,
                end=end,
                initial_cash=BACKTEST_STARTING_CASH,
                final_value=BACKTEST_STARTING_CASH,
                profit_loss=0.0,
                profit_loss_pct=0.0,
                duration_days=(end - effective_start).total_seconds() / 86400.0,
                trades=0,
                data_points=0,
                buy_markers=[],
                sell_markers=[],
                equity_curve=[],
                price_series=[],
                has_strategy_trades=False,
            )

        cash = BACKTEST_STARTING_CASH
        position_qty = 0.0
        trades = 0
        buy_markers: List[Dict[str, float]] = []
        sell_markers: List[Dict[str, float]] = []
        equity_curve: List[Dict[str, float]] = []
        price_series: List[Dict[str, float]] = []
        has_strategy_trades = False

        # Ensure column casing
        df = historical.copy()
        df.columns = [c.lower() for c in df.columns]

        try:
            signals = self.strategy.generate_signals_batch(df, symbol)
        except Exception:  # noqa: BLE001
            LOGGER.exception("backtest_generate_signals_batch_failed symbol=%s", symbol)
            raise
        if signals is not None and not isinstance(signals, pd.Series):
            LOGGER.error(
                "backtest_signals_type symbol=%s type=%s",
                symbol,
                type(signals),
            )
        for idx in range(1, len(df)):
            window = df.iloc[: idx + 1]
            current = df.iloc[idx]
            if signals is not None:
                signal = signals.iloc[idx]
            else:
                try:
                    signal = self.strategy.generate_signal(window, current, symbol)
                except Exception:  # noqa: BLE001
                    LOGGER.exception(
                        "backtest_generate_signal_failed symbol=%s idx=%s",
                        symbol,
                        idx,
                    )
                    raise
            if isinstance(signal, pd.Series) or isinstance(signal, pd.DataFrame):
                LOGGER.error(
                    "backtest_signal_non_scalar symbol=%s idx=%s type=%s",
                    symbol,
                    idx,
                    type(signal),
                )
                signal = "HOLD"
            if pd.isna(signal):
                LOGGER.warning(
                    "backtest_signal_na symbol=%s idx=%s",
                    symbol,
                    idx,
                )
                signal = "HOLD"
            if not isinstance(signal, str):
                LOGGER.warning(
                    "backtest_signal_non_string symbol=%s idx=%s type=%s value=%s",
                    symbol,
                    idx,
                    type(signal),
                    signal,
                )
                signal = "HOLD"
            price = float(current["close"])
            timestamp = current.name

            if pd.isna(price):
                continue

            price_series.append({"ts": timestamp, "price": price})

            if signal == "BUY" and cash > 0:
                position_qty = cash / price
                cash = 0.0
                trades += 1
                buy_markers.append({"ts": timestamp, "price": price})
                has_strategy_trades = True
            elif signal == "SELL" and position_qty > 0:
                cash = position_qty * price
                position_qty = 0.0
                trades += 1
                sell_markers.append({"ts": timestamp, "price": price})
                has_strategy_trades = True

            equity_curve.append({"ts": timestamp, "value": cash + position_qty * price})

        final_price = float(df.iloc[-1]["close"])
        final_value = cash + position_qty * final_price
        profit_loss = final_value - BACKTEST_STARTING_CASH
        profit_loss_pct = (profit_loss / BACKTEST_STARTING_CASH) * 100 if BACKTEST_STARTING_CASH else 0.0
        duration_days = (end - effective_start).total_seconds() / 86400.0

        LOGGER.info(
            "backtest_done symbol=%s trades=%s final_value=%.2f pnl=%.2f",
            symbol,
            trades,
            final_value,
            profit_loss,
        )

        # Ensure we have price series even when signals were skipped
        if not price_series:
            price_series = [
                {"ts": row.name, "price": float(row["close"])}
                for _, row in df.iterrows()
                if not pd.isna(row["close"])
            ]

        # No fallback trades: leave markers empty if strategy produced none

        return BacktestResult(
            symbol=symbol,
            start=effective_start,
            end=end,
            initial_cash=BACKTEST_STARTING_CASH,
            final_value=final_value,
            profit_loss=profit_loss,
            profit_loss_pct=profit_loss_pct,
            duration_days=duration_days,
            trades=trades,
            data_points=len(df),
            buy_markers=buy_markers,
            sell_markers=sell_markers,
            equity_curve=equity_curve,
            price_series=price_series,
            has_strategy_trades=has_strategy_trades,
        )

    def run_multi(self, symbols: List[str], interval: str, start: datetime, end: datetime) -> Dict[str, BacktestResult]:
        results: Dict[str, BacktestResult] = {}
        for symbol in symbols:
            results[symbol] = self.run(symbol, interval, start, end)
        return results

    @staticmethod
    def _clamp_start_date(interval: str, start: datetime, end: datetime) -> datetime:
        intraday_limits = {
            "1m": 7,
            "2m": 60,
            "5m": 60,
            "15m": 60,
            "30m": 60,
            "60m": 60,
            "90m": 60,
            "1h": 730,
        }
        if interval not in intraday_limits:
            return start

        max_days = intraday_limits[interval]
        min_start = end - timedelta(days=max_days)
        if start < min_start:
            return min_start
        return start
