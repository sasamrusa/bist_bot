from datetime import datetime

import pandas as pd

from bist_bot.backtest.engine import BacktestEngine
from bist_bot.core.interfaces import IStrategy


class DummyProvider:
    def get_historical_data(self, symbol: str, interval: str, start_date=None, end_date=None) -> pd.DataFrame:
        return pd.DataFrame()


class DummyStrategy(IStrategy):
    def generate_signal(self, historical_data: pd.DataFrame, current_data: pd.Series, symbol: str) -> str:
        return "HOLD"

    def generate_signals_batch(self, historical_data: pd.DataFrame, symbol: str, interval: str | None = None):
        return None


def test_empty_data_backtest_returns_full_result():
    engine = BacktestEngine(DummyProvider(), DummyStrategy())
    result = engine.run(
        symbol="TEST.IS",
        interval="5m",
        start=datetime(2026, 1, 1),
        end=datetime(2026, 1, 2),
    )

    assert result.data_points == 0
    assert result.buy_markers == []
    assert result.sell_markers == []
    assert result.equity_curve == []
    assert result.price_series == []
    assert result.ohlc_series == []
    assert result.total_fees == 0.0
    assert result.total_slippage_cost == 0.0
    assert result.closed_trades == []
    assert result.open_trade is None
    assert result.has_strategy_trades is False
