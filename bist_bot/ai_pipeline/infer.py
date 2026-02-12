from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from bist_bot.ai_pipeline.drift_monitor import compute_drift_score
from bist_bot.ai_pipeline.feature_store import build_feature_frame
from bist_bot.ai_pipeline.registry import load_latest_model_bundle, load_model_bundle
from bist_bot.ai_pipeline.risk_engine import RiskConfig, calculate_order_quantity
from bist_bot.ai_pipeline.signal_policy import PolicyConfig, probability_to_signal
from bist_bot.core.config import TICKERS
from bist_bot.data.yf_provider import YFDataProvider


@dataclass
class InferenceDecision:
    symbol: str
    probability_up: float
    signal: str
    price: float
    atr: float
    suggested_qty: float
    drift_score: float


def _predict_probability(model: object, row_array: np.ndarray) -> float:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(row_array)
        return float(proba[0, 1])
    pred = model.predict(row_array)
    if isinstance(pred, np.ndarray):
        return float(pred[0])
    return float(pred)


def run_inference(
    bundle: Dict[str, object],
    symbols: Iterable[str],
    interval: str,
    lookback_days: int,
    equity: float,
    policy: PolicyConfig,
    risk: RiskConfig,
) -> List[InferenceDecision]:
    provider = YFDataProvider()
    model = bundle["model"]
    feature_columns: List[str] = list(bundle["feature_columns"])
    symbol_to_id: Dict[str, int] = dict(bundle.get("symbol_to_id", {}))
    baseline_stats = dict(bundle.get("feature_stats", {}))

    decisions: List[InferenceDecision] = []
    end = datetime.now()
    start = end - timedelta(days=lookback_days)

    for symbol in symbols:
        historical = provider.get_historical_data(symbol, interval, start, end)
        if historical.empty:
            continue

        frame = build_feature_frame(historical, symbol=symbol).dropna(subset=[col for col in feature_columns if col != "symbol_id"])
        if frame.empty:
            continue

        frame = frame.copy()
        frame["symbol_id"] = int(symbol_to_id.get(symbol, -1))
        latest = frame.iloc[-1]
        row_array = latest[feature_columns].to_numpy(dtype=float).reshape(1, -1)
        prob_up = _predict_probability(model, row_array)

        has_position = False
        signal = probability_to_signal(prob_up, has_position=has_position, config=policy)
        price = float(latest["close"])
        atr = float(latest["atr"]) if pd.notna(latest["atr"]) else max(price * 0.01, 0.01)
        qty = calculate_order_quantity(equity=equity, price=price, atr=atr, config=risk) if signal == "BUY" else 0.0

        recent_for_drift = frame.tail(80)
        drift_score = compute_drift_score(recent_for_drift, feature_columns, baseline_stats) if baseline_stats else 0.0
        decisions.append(
            InferenceDecision(
                symbol=symbol,
                probability_up=float(prob_up),
                signal=signal,
                price=price,
                atr=atr,
                suggested_qty=qty,
                drift_score=float(drift_score),
            )
        )

    decisions.sort(key=lambda item: abs(item.probability_up - 0.5), reverse=True)
    return decisions


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AI model inference for BIST symbols")
    parser.add_argument("--model-dir", type=str, default="models/ai_registry", help="Model registry directory")
    parser.add_argument("--model-path", type=str, default="", help="Specific model bundle path (optional)")
    parser.add_argument("--interval", type=str, default="1h", help="Market data interval")
    parser.add_argument("--lookback-days", type=int, default=420, help="Historical lookback window for features")
    parser.add_argument("--equity", type=float, default=100_000.0, help="Portfolio equity for suggested quantity")
    parser.add_argument("--buy-threshold", type=float, default=0.58, help="BUY probability threshold")
    parser.add_argument("--sell-threshold", type=float, default=0.42, help="SELL probability threshold")
    parser.add_argument("--symbols", type=str, default="", help="Comma-separated symbols; default uses config TICKERS")
    parser.add_argument("--output", type=str, default="reports/ai_inference_latest.json", help="Output JSON path")
    args = parser.parse_args()

    if args.buy_threshold <= args.sell_threshold:
        raise ValueError("buy-threshold must be greater than sell-threshold")

    if args.model_path:
        bundle = load_model_bundle(Path(args.model_path))
    else:
        bundle = load_latest_model_bundle(Path(args.model_dir))

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()] if args.symbols else list(TICKERS)
    policy = PolicyConfig(buy_threshold=args.buy_threshold, sell_threshold=args.sell_threshold)
    risk = RiskConfig()

    decisions = run_inference(
        bundle=bundle,
        symbols=symbols,
        interval=args.interval,
        lookback_days=args.lookback_days,
        equity=args.equity,
        policy=policy,
        risk=risk,
    )

    output = {
        "generated_at": datetime.now().isoformat(),
        "interval": args.interval,
        "count": len(decisions),
        "decisions": [decision.__dict__ for decision in decisions],
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print(f"inference count={len(decisions)} output={output_path}")
    for item in decisions[:10]:
        print(
            f"{item.symbol:10s} prob={item.probability_up:.3f} "
            f"signal={item.signal:4s} qty={item.suggested_qty:.2f} drift={item.drift_score:.2f}"
        )


if __name__ == "__main__":
    main()
