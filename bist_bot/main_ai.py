import time
import argparse
from datetime import datetime, timedelta
from pathlib import Path

from bist_bot.ai_pipeline.feature_store import build_feature_frame
from bist_bot.ai_pipeline.registry import load_latest_model_bundle
from bist_bot.ai_pipeline.risk_engine import RiskConfig, calculate_order_quantity
from bist_bot.ai_pipeline.signal_policy import PolicyConfig, probability_to_signal
from bist_bot.core.config import DATA_INTERVAL, DATA_POLLING_INTERVAL_SECONDS, ORDER_TYPE, TICKERS
from bist_bot.data.yf_provider import YFDataProvider
from bist_bot.execution.paper_broker import PaperBroker
from bist_bot.utils.logger import setup_logger


def run_ai_bot(model_dir: str = "models/ai_registry", lookback_days: int = 420, max_cycles: int = 0) -> None:
    logger = setup_logger("bist_bot.main_ai")
    bundle = load_latest_model_bundle(Path(model_dir))
    model = bundle["model"]
    feature_columns = list(bundle["feature_columns"])
    symbol_to_id = dict(bundle.get("symbol_to_id", {}))

    provider = YFDataProvider()
    broker = PaperBroker()
    policy = PolicyConfig()
    risk = RiskConfig()

    logger.info("starting_ai_bot symbols=%s interval=%s", len(TICKERS), DATA_INTERVAL)

    cycle_count = 0
    while True:
        cycle_now = datetime.now()
        start = cycle_now - timedelta(days=lookback_days)
        for symbol in TICKERS:
            historical = provider.get_historical_data(symbol, DATA_INTERVAL, start, cycle_now)
            if historical.empty:
                continue

            features = build_feature_frame(historical, symbol=symbol)
            if features.empty:
                continue
            features = features.dropna(subset=[col for col in feature_columns if col != "symbol_id"])
            if features.empty:
                continue

            latest = features.iloc[-1].copy()
            latest["symbol_id"] = int(symbol_to_id.get(symbol, -1))
            row = latest[feature_columns].to_numpy(dtype=float).reshape(1, -1)
            if hasattr(model, "predict_proba"):
                prob_up = float(model.predict_proba(row)[0, 1])
            else:
                prob_up = float(model.predict(row)[0])

            current_price = float(latest["close"])
            atr = float(latest["atr"]) if latest["atr"] == latest["atr"] else max(current_price * 0.01, 0.01)
            has_position = broker.get_asset_balance(symbol) > 0.0
            signal = probability_to_signal(prob_up, has_position=has_position, config=policy)

            logger.info("%s prob=%.3f signal=%s price=%.2f", symbol, prob_up, signal, current_price)
            if signal == "BUY":
                balance = broker.get_account_balance()
                qty = calculate_order_quantity(
                    equity=float(balance["total_value"]),
                    price=current_price,
                    atr=atr,
                    config=risk,
                )
                if qty > 0:
                    broker.place_order(symbol, ORDER_TYPE, quantity=qty, price=current_price)
            elif signal == "SELL" and has_position:
                qty = -broker.get_asset_balance(symbol)
                if qty != 0:
                    broker.place_order(symbol, ORDER_TYPE, quantity=qty, price=current_price)

        logger.info("ai_cycle_complete sleep=%ss", DATA_POLLING_INTERVAL_SECONDS)
        cycle_count += 1
        if max_cycles > 0 and cycle_count >= max_cycles:
            logger.info("ai_bot_stopped max_cycles=%s", max_cycles)
            break
        time.sleep(DATA_POLLING_INTERVAL_SECONDS)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AI-driven paper trading bot")
    parser.add_argument("--model-dir", type=str, default="models/ai_registry", help="Model registry directory")
    parser.add_argument("--lookback-days", type=int, default=420, help="Feature lookback window")
    parser.add_argument("--max-cycles", type=int, default=0, help="Run fixed number of cycles then stop (0=infinite)")
    cli_args = parser.parse_args()
    run_ai_bot(model_dir=cli_args.model_dir, lookback_days=cli_args.lookback_days, max_cycles=cli_args.max_cycles)
