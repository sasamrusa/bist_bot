import pandas as pd

from bist_bot.ai_pipeline.feature_store import BASE_FEATURE_COLUMNS, build_feature_frame
from bist_bot.ai_pipeline.risk_engine import RiskConfig, calculate_order_quantity
from bist_bot.ai_pipeline.signal_policy import PolicyConfig, probability_to_signal


def test_probability_to_signal_thresholds():
    cfg = PolicyConfig(buy_threshold=0.6, sell_threshold=0.4)
    assert probability_to_signal(0.65, has_position=False, config=cfg) == "BUY"
    assert probability_to_signal(0.35, has_position=True, config=cfg) == "SELL"
    assert probability_to_signal(0.52, has_position=False, config=cfg) == "HOLD"


def test_risk_engine_returns_positive_quantity():
    cfg = RiskConfig(risk_per_trade_pct=0.01, atr_stop_multiplier=2.0, max_position_pct=0.2)
    qty = calculate_order_quantity(equity=100_000.0, price=100.0, atr=2.5, config=cfg)
    assert qty > 0.0
    assert qty <= 200.0  # 20% position cap at 100 TRY price


def test_feature_store_has_expected_columns():
    idx = pd.date_range("2025-01-01", periods=80, freq="D")
    frame = pd.DataFrame(
        {
            "open": [100 + i * 0.2 for i in range(80)],
            "high": [101 + i * 0.2 for i in range(80)],
            "low": [99 + i * 0.2 for i in range(80)],
            "close": [100 + i * 0.2 for i in range(80)],
            "volume": [1_000_000 + i * 1_000 for i in range(80)],
        },
        index=idx,
    )
    features = build_feature_frame(frame, symbol="TEST.IS")
    for col in BASE_FEATURE_COLUMNS + ["close", "atr", "symbol"]:
        assert col in features.columns

