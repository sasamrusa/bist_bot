from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Dict, List, Tuple

from bist_bot.ai_pipeline.universe import resolve_universe
from bist_bot.backtest.engine import BacktestEngine, BacktestResult
from bist_bot.data.yf_provider import YFDataProvider
from bist_bot.strategies.ai_model_strategy import AIModelStrategy
from bist_bot.strategies.hybrid_ai_rsi_strategy import HybridAiRsiStrategy
from bist_bot.strategies.rsi_macd import RsiMacdStrategy


def _parse_date(text: str, arg_name: str) -> datetime:
    try:
        return datetime.strptime(text, "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError(f"{arg_name} must be YYYY-MM-DD, got: {text}") from exc


def _collect_metrics(results: Dict[str, BacktestResult]) -> Dict[str, object]:
    effective = [item for item in results.values() if item.data_points > 0]
    if not effective:
        return {
            "symbols_total": len(results),
            "symbols_with_data": 0,
            "active_symbols": 0,
            "profitable_symbols": 0,
            "mean_pnl_pct": 0.0,
            "median_pnl_pct": 0.0,
            "symbol_success_rate_pct": 0.0,
            "total_trades": 0,
            "avg_trades_per_active_symbol": 0.0,
            "total_profit_loss_try": 0.0,
            "total_fees_try": 0.0,
            "total_slippage_try": 0.0,
            "top_symbols": [],
            "bottom_symbols": [],
        }

    pnl_values = [float(item.profit_loss_pct) for item in effective]
    active = [item for item in effective if int(item.trades) > 0]
    profitable = [item for item in effective if float(item.profit_loss_pct) > 0.0]

    ranked = sorted(effective, key=lambda item: float(item.profit_loss_pct), reverse=True)
    top_symbols = [
        {"symbol": item.symbol, "pnl_pct": float(item.profit_loss_pct), "trades": int(item.trades)}
        for item in ranked[:10]
    ]
    bottom_symbols = [
        {"symbol": item.symbol, "pnl_pct": float(item.profit_loss_pct), "trades": int(item.trades)}
        for item in ranked[-10:]
    ]

    total_trades = int(sum(int(item.trades) for item in effective))
    total_pnl_try = float(sum(float(item.profit_loss) for item in effective))
    total_fees_try = float(sum(float(item.total_fees) for item in effective))
    total_slippage_try = float(sum(float(item.total_slippage_cost) for item in effective))

    return {
        "symbols_total": len(results),
        "symbols_with_data": len(effective),
        "active_symbols": len(active),
        "profitable_symbols": len(profitable),
        "mean_pnl_pct": float(sum(pnl_values) / len(pnl_values)),
        "median_pnl_pct": float(median(pnl_values)),
        "symbol_success_rate_pct": float((len(profitable) / len(effective)) * 100.0),
        "total_trades": total_trades,
        "avg_trades_per_active_symbol": float(total_trades / len(active)) if active else 0.0,
        "total_profit_loss_try": total_pnl_try,
        "total_fees_try": total_fees_try,
        "total_slippage_try": total_slippage_try,
        "top_symbols": top_symbols,
        "bottom_symbols": bottom_symbols,
    }


def _evaluate_strategy(
    strategy_name: str,
    strategy,
    provider: YFDataProvider,
    symbols: List[str],
    interval: str,
    start: datetime,
    end: datetime,
) -> Tuple[Dict[str, BacktestResult], Dict[str, object]]:
    engine = BacktestEngine(provider, strategy)
    results = engine.run_multi(symbols, interval, start, end)
    metrics = _collect_metrics(results)
    metrics["strategy"] = strategy_name
    return results, metrics


def _rank_key(metrics: Dict[str, object]) -> Tuple[float, float, float]:
    return (
        float(metrics.get("median_pnl_pct", 0.0)),
        float(metrics.get("mean_pnl_pct", 0.0)),
        float(metrics.get("symbol_success_rate_pct", 0.0)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare AI, classic and hybrid strategies on the same OOS window.")
    parser.add_argument("--interval", type=str, default="1h", help="Bar interval")
    parser.add_argument("--universe", type=str, default="bist100", help="Universe: config | bist100")
    parser.add_argument("--start-date", type=str, required=True, help="Out-of-sample start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default="", help="Out-of-sample end date (YYYY-MM-DD, default=today)")
    parser.add_argument("--model-dir", type=str, default="models/ai_registry", help="AI model registry directory")
    parser.add_argument("--model-path", type=str, default="", help="Optional explicit model path")
    parser.add_argument(
        "--hybrid-modes",
        type=str,
        default="consensus,ai_lead,classic_lead,weighted,probability_gate",
        help="Comma-separated hybrid modes",
    )
    parser.add_argument("--hybrid-margin", type=float, default=0.06, help="Hybrid probability margin")
    parser.add_argument("--buy-threshold", type=float, default=0.58, help="AI buy threshold")
    parser.add_argument("--sell-threshold", type=float, default=0.42, help="AI sell threshold")
    parser.add_argument("--output", type=str, default="", help="Optional explicit report path")
    args = parser.parse_args()

    start = _parse_date(args.start_date, "--start-date")
    end = _parse_date(args.end_date, "--end-date") if args.end_date else datetime.now()
    if end <= start:
        raise ValueError("end date must be after start date")

    symbols = resolve_universe(args.universe)
    if not symbols:
        raise RuntimeError("Universe is empty.")

    provider = YFDataProvider()
    reports: List[Dict[str, object]] = []

    print(f"running baseline RSI+MACD for {len(symbols)} symbols...")
    _, rsi_metrics = _evaluate_strategy(
        strategy_name="rsi_macd",
        strategy=RsiMacdStrategy(),
        provider=provider,
        symbols=symbols,
        interval=args.interval,
        start=start,
        end=end,
    )
    reports.append(rsi_metrics)

    print(f"running AI model for {len(symbols)} symbols...")
    _, ai_metrics = _evaluate_strategy(
        strategy_name="ai_model",
        strategy=AIModelStrategy(
            model_dir=args.model_dir,
            model_path=args.model_path,
            buy_threshold=args.buy_threshold,
            sell_threshold=args.sell_threshold,
        ),
        provider=provider,
        symbols=symbols,
        interval=args.interval,
        start=start,
        end=end,
    )
    reports.append(ai_metrics)

    hybrid_modes = [item.strip() for item in args.hybrid_modes.split(",") if item.strip()]
    for mode in hybrid_modes:
        print(f"running hybrid mode={mode} for {len(symbols)} symbols...")
        _, hybrid_metrics = _evaluate_strategy(
            strategy_name=f"hybrid_{mode}",
            strategy=HybridAiRsiStrategy(
                mode=mode,
                model_dir=args.model_dir,
                model_path=args.model_path,
                buy_threshold=args.buy_threshold,
                sell_threshold=args.sell_threshold,
                probability_margin=args.hybrid_margin,
            ),
            provider=provider,
            symbols=symbols,
            interval=args.interval,
            start=start,
            end=end,
        )
        reports.append(hybrid_metrics)

    ranked = sorted(reports, key=_rank_key, reverse=True)
    winner = ranked[0] if ranked else {}

    output_payload = {
        "meta": {
            "created_at": datetime.now().isoformat(),
            "interval": args.interval,
            "universe": args.universe,
            "symbol_count": len(symbols),
            "start_date": start.strftime("%Y-%m-%d"),
            "end_date": end.strftime("%Y-%m-%d"),
            "model_dir": args.model_dir,
            "model_path": args.model_path or "latest",
            "hybrid_modes": hybrid_modes,
            "hybrid_margin": args.hybrid_margin,
            "buy_threshold": args.buy_threshold,
            "sell_threshold": args.sell_threshold,
        },
        "ranked_results": ranked,
        "winner": winner,
    }

    if args.output:
        output_path = Path(args.output)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path("reports") / f"ai_hybrid_eval_{ts}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nRANKING")
    for rank, metrics in enumerate(ranked, start=1):
        print(
            f"{rank}. {metrics['strategy']}: "
            f"median={float(metrics['median_pnl_pct']):+.2f}% "
            f"mean={float(metrics['mean_pnl_pct']):+.2f}% "
            f"success={float(metrics['symbol_success_rate_pct']):.1f}% "
            f"trades={int(metrics['total_trades'])}"
        )
    print(f"\nreport={output_path}")


if __name__ == "__main__":
    main()
