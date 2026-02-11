from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List

import pandas as pd

from bist_bot.core.config import (
    ATR_PERIOD,
    BACKTEST_ATR_STOP_MULTIPLIER,
    BACKTEST_COMMISSION_BPS,
    BACKTEST_POSITION_SIZING_MODE,
    BACKTEST_RISK_PER_TRADE_PCT,
    BACKTEST_SLIPPAGE_BPS,
    BACKTEST_STARTING_CASH,
    BACKTEST_TAKE_PROFIT_MULTIPLIER,
    ENABLE_PROTECTIVE_EXITS,
)
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
    ohlc_series: List[Dict[str, float]]
    total_fees: float
    total_slippage_cost: float
    closed_trades: List[Dict[str, object]]
    open_trade: Dict[str, object] | None
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
                ohlc_series=[],
                total_fees=0.0,
                total_slippage_cost=0.0,
                closed_trades=[],
                open_trade=None,
                has_strategy_trades=False,
            )

        cash = BACKTEST_STARTING_CASH
        position_qty = 0.0
        trades = 0
        total_fees = 0.0
        total_slippage_cost = 0.0
        closed_trades: List[Dict[str, object]] = []
        open_trade: Dict[str, object] | None = None
        buy_markers: List[Dict[str, float]] = []
        sell_markers: List[Dict[str, float]] = []
        equity_curve: List[Dict[str, float]] = []
        price_series: List[Dict[str, float]] = []
        ohlc_series: List[Dict[str, float]] = []
        has_strategy_trades = False
        commission_rate = BACKTEST_COMMISSION_BPS / 10_000.0
        slippage_rate = BACKTEST_SLIPPAGE_BPS / 10_000.0
        open_entry: Dict[str, object] | None = None

        # Ensure column casing
        df = historical.copy()
        df.columns = [c.lower() for c in df.columns]
        if "open" not in df.columns:
            df["open"] = df["close"]
        if "high" not in df.columns:
            df["high"] = df["close"]
        if "low" not in df.columns:
            df["low"] = df["close"]

        for _, row in df.iterrows():
            close_price = row.get("close")
            if pd.isna(close_price):
                continue
            ohlc_series.append(
                {
                    "ts": row.name,
                    "open": float(row.get("open", close_price)),
                    "high": float(row.get("high", close_price)),
                    "low": float(row.get("low", close_price)),
                    "close": float(close_price),
                }
            )

        atr_series = self._compute_atr(df)

        try:
            try:
                signals = self.strategy.generate_signals_batch(df, symbol, interval)
            except TypeError:
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
            high_price = float(current.get("high", price)) if pd.notna(current.get("high", price)) else price
            low_price = float(current.get("low", price)) if pd.notna(current.get("low", price)) else price

            if pd.isna(price):
                continue

            price_series.append({"ts": timestamp, "price": price})

            if ENABLE_PROTECTIVE_EXITS and position_qty > 0 and open_entry is not None:
                stop_price = float(open_entry.get("stop_price", float("nan")))
                take_profit_price = float(open_entry.get("take_profit_price", float("nan")))
                stop_hit = pd.notna(stop_price) and low_price <= stop_price
                take_profit_hit = pd.notna(take_profit_price) and high_price >= take_profit_price

                if stop_hit or take_profit_hit:
                    if stop_hit:
                        exit_signal_price = stop_price
                        exit_reason = "STOP_LOSS"
                    else:
                        exit_signal_price = take_profit_price
                        exit_reason = "TAKE_PROFIT"

                    sell_fill_price = float(exit_signal_price) * (1.0 - slippage_rate)
                    gross_proceeds = position_qty * sell_fill_price
                    fee = gross_proceeds * commission_rate
                    net_proceeds = gross_proceeds - fee
                    sold_quantity = position_qty

                    cash += net_proceeds
                    total_fees += fee
                    total_slippage_cost += max(position_qty * (float(exit_signal_price) - sell_fill_price), 0.0)
                    position_qty = 0.0
                    trades += 1
                    sell_markers.append({"ts": timestamp, "price": float(exit_signal_price)})
                    has_strategy_trades = True

                    entry_fill_price = float(open_entry["entry_fill_price"])
                    entry_fee = float(open_entry["entry_fee"])
                    quantity = float(open_entry["quantity"])
                    gross_pnl = (sell_fill_price - entry_fill_price) * quantity
                    net_pnl = gross_pnl - entry_fee - fee
                    closed_trades.append(
                        {
                            "symbol": symbol,
                            "entry_ts": open_entry["entry_ts"],
                            "entry_signal_price": float(open_entry["entry_signal_price"]),
                            "entry_fill_price": entry_fill_price,
                            "exit_ts": timestamp,
                            "exit_signal_price": float(exit_signal_price),
                            "exit_fill_price": float(sell_fill_price),
                            "quantity": quantity,
                            "gross_pnl": float(gross_pnl),
                            "net_pnl": float(net_pnl),
                            "fees": float(entry_fee + fee),
                            "slippage_cost": float(
                                max(quantity * (entry_fill_price - float(open_entry["entry_signal_price"])), 0.0)
                                + max(sold_quantity * (float(exit_signal_price) - sell_fill_price), 0.0)
                            ),
                            "exit_reason": exit_reason,
                            "status": "CLOSED",
                        }
                    )
                    open_entry = None
                    equity_curve.append({"ts": timestamp, "value": cash + position_qty * price})
                    continue

            if signal == "BUY" and cash > 0 and position_qty <= 0:
                buy_fill_price = price * (1.0 + slippage_rate)
                atr_value = atr_series.iloc[idx] if idx < len(atr_series) else float("nan")
                if pd.isna(atr_value) or atr_value <= 0:
                    atr_value = max(price * 0.01, 0.01)

                if BACKTEST_POSITION_SIZING_MODE == "atr_risk":
                    risk_per_share = max(atr_value * BACKTEST_ATR_STOP_MULTIPLIER, price * 0.001)
                    risk_budget = max(cash * BACKTEST_RISK_PER_TRADE_PCT, 0.0)
                    quantity = risk_budget / risk_per_share if risk_per_share > 0 else 0.0
                    max_affordable_qty = cash / (buy_fill_price * (1.0 + commission_rate))
                    quantity = min(quantity, max_affordable_qty)
                else:
                    quantity = cash / (buy_fill_price * (1.0 + commission_rate))

                if quantity > 0:
                    gross_cost = quantity * buy_fill_price
                    fee = gross_cost * commission_rate
                    total_cost = gross_cost + fee
                    if total_cost > cash:
                        quantity = cash / (buy_fill_price * (1.0 + commission_rate))
                        gross_cost = quantity * buy_fill_price
                        fee = gross_cost * commission_rate
                        total_cost = gross_cost + fee

                    if quantity > 0 and total_cost <= cash:
                        cash -= total_cost
                        position_qty = quantity
                        total_fees += fee
                        total_slippage_cost += max(quantity * (buy_fill_price - price), 0.0)
                        open_entry = {
                            "symbol": symbol,
                            "entry_ts": timestamp,
                            "entry_signal_price": float(price),
                            "entry_fill_price": float(buy_fill_price),
                            "quantity": float(quantity),
                            "entry_fee": float(fee),
                            "atr_at_entry": float(atr_value),
                            "stop_price": float(buy_fill_price - (atr_value * BACKTEST_ATR_STOP_MULTIPLIER)),
                            "take_profit_price": float(buy_fill_price + (atr_value * BACKTEST_TAKE_PROFIT_MULTIPLIER)),
                        }
                        trades += 1
                        buy_markers.append({"ts": timestamp, "price": price})
                        has_strategy_trades = True

            elif signal == "SELL" and position_qty > 0:
                sell_fill_price = price * (1.0 - slippage_rate)
                gross_proceeds = position_qty * sell_fill_price
                fee = gross_proceeds * commission_rate
                net_proceeds = gross_proceeds - fee
                sold_quantity = position_qty

                cash += net_proceeds
                total_fees += fee
                total_slippage_cost += max(position_qty * (price - sell_fill_price), 0.0)
                position_qty = 0.0
                trades += 1
                sell_markers.append({"ts": timestamp, "price": price})
                has_strategy_trades = True

                if open_entry is not None:
                    entry_fill_price = float(open_entry["entry_fill_price"])
                    entry_fee = float(open_entry["entry_fee"])
                    quantity = float(open_entry["quantity"])
                    gross_pnl = (sell_fill_price - entry_fill_price) * quantity
                    net_pnl = gross_pnl - entry_fee - fee
                    closed_trades.append(
                        {
                            "symbol": symbol,
                            "entry_ts": open_entry["entry_ts"],
                            "entry_signal_price": float(open_entry["entry_signal_price"]),
                            "entry_fill_price": entry_fill_price,
                            "exit_ts": timestamp,
                            "exit_signal_price": float(price),
                            "exit_fill_price": float(sell_fill_price),
                            "quantity": quantity,
                            "gross_pnl": float(gross_pnl),
                            "net_pnl": float(net_pnl),
                            "fees": float(entry_fee + fee),
                            "slippage_cost": float(
                                max(quantity * (entry_fill_price - float(open_entry["entry_signal_price"])), 0.0)
                                + max(sold_quantity * (float(price) - sell_fill_price), 0.0)
                            ),
                            "exit_reason": "SIGNAL_SELL",
                            "status": "CLOSED",
                        }
                    )
                    open_entry = None

            equity_curve.append({"ts": timestamp, "value": cash + position_qty * price})

        final_price = float(df.iloc[-1]["close"])
        final_value = cash + position_qty * final_price
        profit_loss = final_value - BACKTEST_STARTING_CASH
        profit_loss_pct = (profit_loss / BACKTEST_STARTING_CASH) * 100 if BACKTEST_STARTING_CASH else 0.0
        duration_days = (end - effective_start).total_seconds() / 86400.0

        LOGGER.info(
            "backtest_done symbol=%s trades=%s final_value=%.2f pnl=%.2f fees=%.2f slippage=%.2f",
            symbol,
            trades,
            final_value,
            profit_loss,
            total_fees,
            total_slippage_cost,
        )

        # Ensure we have price series even when signals were skipped
        if not price_series:
            price_series = [
                {"ts": row.name, "price": float(row["close"])}
                for _, row in df.iterrows()
                if not pd.isna(row["close"])
            ]

        if position_qty > 0 and open_entry is not None:
            last_close = float(df.iloc[-1]["close"])
            entry_fill = float(open_entry["entry_fill_price"])
            qty = float(open_entry["quantity"])
            unrealized = (last_close - entry_fill) * qty
            open_trade = {
                "symbol": symbol,
                "entry_ts": open_entry["entry_ts"],
                "entry_signal_price": float(open_entry["entry_signal_price"]),
                "entry_fill_price": entry_fill,
                "quantity": qty,
                "entry_fee": float(open_entry["entry_fee"]),
                "mark_price": last_close,
                "unrealized_pnl": float(unrealized),
                "stop_price": float(open_entry.get("stop_price", 0.0)),
                "take_profit_price": float(open_entry.get("take_profit_price", 0.0)),
                "status": "OPEN",
            }

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
            ohlc_series=ohlc_series,
            total_fees=total_fees,
            total_slippage_cost=total_slippage_cost,
            closed_trades=closed_trades,
            open_trade=open_trade,
            has_strategy_trades=has_strategy_trades,
        )

    def run_multi(self, symbols: List[str], interval: str, start: datetime, end: datetime) -> Dict[str, BacktestResult]:
        results: Dict[str, BacktestResult] = {}
        for symbol in symbols:
            results[symbol] = self.run(symbol, interval, start, end)
        return results

    @staticmethod
    def _compute_atr(df: pd.DataFrame) -> pd.Series:
        if "high" not in df.columns or "low" not in df.columns or "close" not in df.columns:
            return pd.Series([float("nan")] * len(df), index=df.index)

        high = pd.to_numeric(df["high"], errors="coerce")
        low = pd.to_numeric(df["low"], errors="coerce")
        close = pd.to_numeric(df["close"], errors="coerce")
        prev_close = close.shift(1)

        tr = pd.concat(
            [
                (high - low).abs(),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        return tr.rolling(ATR_PERIOD, min_periods=1).mean()

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
