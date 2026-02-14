from __future__ import annotations

import argparse
import json
import logging
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Dict, List, Tuple

import pandas as pd

from bist_bot.ai_pipeline.universe import resolve_universe
from bist_bot.backtest.engine import BacktestEngine, BacktestResult
from bist_bot.data.yf_provider import YFDataProvider
from bist_bot.strategies.ai_model_strategy import AIModelStrategy
from bist_bot.strategies.hybrid_ai_rsi_strategy import HybridAiRsiStrategy
from bist_bot.strategies.rsi_macd import RsiMacdStrategy


@dataclass(frozen=True)
class HybridCandidate:
    mode: str
    probability_margin: float
    buy_threshold: float
    sell_threshold: float


@dataclass(frozen=True)
class AiThresholdCandidate:
    buy_threshold: float
    sell_threshold: float


def _parse_date(text: str, arg_name: str) -> datetime:
    try:
        return datetime.strptime(text, "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError(f"{arg_name} must be YYYY-MM-DD, got: {text}") from exc


def _parse_float_list(text: str, arg_name: str) -> List[float]:
    values: List[float] = []
    for token in text.split(","):
        item = token.strip()
        if not item:
            continue
        try:
            values.append(float(item))
        except ValueError as exc:
            raise ValueError(f"{arg_name} contains non-float value: {item}") from exc
    if not values:
        raise ValueError(f"{arg_name} must contain at least one float value")
    return sorted(set(values))


def _parse_mode_list(text: str) -> List[str]:
    values = [item.strip() for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("--modes must contain at least one mode")
    valid = {"consensus", "ai_lead", "classic_lead", "weighted", "probability_gate"}
    invalid = [item for item in values if item not in valid]
    if invalid:
        raise ValueError(f"Unsupported hybrid mode(s): {invalid}")
    return values


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
        }

    pnl_values = [float(item.profit_loss_pct) for item in effective]
    active = [item for item in effective if int(item.trades) > 0]
    profitable = [item for item in effective if float(item.profit_loss_pct) > 0.0]

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
    }


def _score(metrics: Dict[str, object]) -> float:
    median_pnl = float(metrics.get("median_pnl_pct", 0.0))
    mean_pnl = float(metrics.get("mean_pnl_pct", 0.0))
    success = float(metrics.get("symbol_success_rate_pct", 0.0))
    trades = float(metrics.get("total_trades", 0))
    active = float(metrics.get("active_symbols", 0))
    trade_density = (trades / active) if active > 0 else 0.0
    activity_bonus = min(trade_density, 8.0) * 0.2
    raw = (0.52 * median_pnl) + (0.33 * mean_pnl) + (0.14 * success) + activity_bonus
    if trades < max(20.0, active * 0.5):
        raw -= 8.0
    return float(raw)


class _CachedProvider:
    def __init__(self, data: Dict[str, pd.DataFrame]):
        self.data = data

    def get_historical_data(self, symbol: str, interval: str, start_date=None, end_date=None) -> pd.DataFrame:
        frame = self.data.get(symbol)
        if frame is None or frame.empty:
            return pd.DataFrame()
        if start_date is None and end_date is None:
            return frame.copy()
        idx = frame.index
        mask = pd.Series(True, index=idx)
        if start_date is not None:
            mask &= idx >= pd.Timestamp(start_date).tz_localize(None)
        if end_date is not None:
            mask &= idx <= pd.Timestamp(end_date).tz_localize(None)
        return frame.loc[mask].copy()

    def get_latest_data(self, symbol: str, interval: str) -> pd.DataFrame:
        frame = self.data.get(symbol)
        if frame is None or frame.empty:
            return pd.DataFrame()
        return pd.DataFrame([frame.iloc[-1]])


def _evaluate_strategy(
    strategy_name: str,
    strategy,
    provider: _CachedProvider,
    symbols: List[str],
    interval: str,
    start: datetime,
    end: datetime,
) -> Dict[str, object]:
    engine = BacktestEngine(provider, strategy)
    results = engine.run_multi(symbols, interval, start, end)
    metrics = _collect_metrics(results)
    metrics["strategy"] = strategy_name
    metrics["score"] = _score(metrics)
    return metrics


def _candidate_to_dict(item: HybridCandidate) -> Dict[str, object]:
    data = asdict(item)
    data["key"] = f"{item.mode}|m={item.probability_margin:.3f}|b={item.buy_threshold:.3f}|s={item.sell_threshold:.3f}"
    return data


def _ai_candidate_to_dict(item: AiThresholdCandidate) -> Dict[str, object]:
    data = asdict(item)
    data["key"] = f"b={item.buy_threshold:.3f}|s={item.sell_threshold:.3f}"
    return data


def _build_hybrid_candidate_pool(
    modes: List[str],
    margins: List[float],
    buy_thresholds: List[float],
    sell_thresholds: List[float],
) -> List[HybridCandidate]:
    pool: Dict[str, HybridCandidate] = {}
    for mode in modes:
        margin_values = margins if mode == "probability_gate" else [0.0]
        for margin in margin_values:
            for buy in buy_thresholds:
                for sell in sell_thresholds:
                    if sell >= buy:
                        continue
                    candidate = HybridCandidate(
                        mode=mode,
                        probability_margin=float(max(0.0, margin if mode == "probability_gate" else 0.0)),
                        buy_threshold=float(buy),
                        sell_threshold=float(sell),
                    )
                    key = _candidate_to_dict(candidate)["key"]
                    pool[key] = candidate
    return list(pool.values())


def _build_ai_threshold_pool(buy_thresholds: List[float], sell_thresholds: List[float]) -> List[AiThresholdCandidate]:
    pairs: List[AiThresholdCandidate] = []
    for buy in buy_thresholds:
        for sell in sell_thresholds:
            if sell >= buy:
                continue
            pairs.append(AiThresholdCandidate(buy_threshold=float(buy), sell_threshold=float(sell)))
    if not pairs:
        raise ValueError("No valid AI threshold pairs found (sell must be lower than buy).")
    return pairs


def _preload_data(
    symbols: List[str],
    interval: str,
    start: datetime,
    end: datetime,
    min_rows: int,
) -> Dict[str, pd.DataFrame]:
    provider = YFDataProvider()
    loaded: Dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        frame = provider.get_historical_data(symbol, interval, start, end)
        if frame.empty or len(frame) < min_rows:
            continue
        frame = frame.sort_index().copy()
        if isinstance(frame.index, pd.DatetimeIndex) and frame.index.tz is not None:
            frame.index = frame.index.tz_convert(None)
        loaded[symbol] = frame
    return loaded


def _find_trial_by_key(items: List[Dict[str, object]], key: str) -> Dict[str, object] | None:
    for item in items:
        candidate = item.get("candidate", {})
        if str(candidate.get("key")) == key:
            return item
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize AI and Hybrid strategy parameters.")
    parser.add_argument("--interval", type=str, default="1d", help="Bar interval")
    parser.add_argument("--universe", type=str, default="bist100", help="Universe: config | bist100")
    parser.add_argument("--opt-start-date", type=str, required=True, help="Optimization start date YYYY-MM-DD")
    parser.add_argument("--opt-end-date", type=str, required=True, help="Optimization end date YYYY-MM-DD")
    parser.add_argument("--eval-start-date", type=str, default="", help="Out-of-sample test start date YYYY-MM-DD")
    parser.add_argument("--eval-end-date", type=str, default="", help="Out-of-sample test end date YYYY-MM-DD")
    parser.add_argument(
        "--selection-window",
        type=str,
        default="opt",
        choices=["opt", "eval"],
        help="Window used for selecting deployment parameters. Recommended: opt",
    )
    parser.add_argument("--modes", type=str, default="consensus,ai_lead,classic_lead,weighted,probability_gate")
    parser.add_argument("--margins", type=str, default="0.00,0.02,0.04,0.06,0.08,0.10")
    parser.add_argument("--buy-thresholds", type=str, default="0.52,0.54,0.56,0.58,0.60,0.62")
    parser.add_argument("--sell-thresholds", type=str, default="0.38,0.40,0.42,0.44,0.46,0.48,0.50")
    parser.add_argument("--trials", type=int, default=60, help="Hybrid random-search trial count")
    parser.add_argument("--coarse-size", type=int, default=35, help="Coarse symbol count")
    parser.add_argument("--stage2-top", type=int, default=15, help="Top coarse candidates re-evaluated on full set")
    parser.add_argument("--min-rows", type=int, default=120, help="Minimum rows per symbol to keep")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-dir", type=str, default="models/ai_registry")
    parser.add_argument("--model-path", type=str, default="")
    parser.add_argument(
        "--deploy-config-path",
        type=str,
        default="models/ai_registry/strategy_tuning_latest.json",
        help="Where to write tuned defaults for GUI/runtime",
    )
    parser.add_argument("--output", type=str, default="", help="Optional explicit JSON output path")
    parser.add_argument("--summary-md", type=str, default="", help="Optional explicit markdown summary path")
    args = parser.parse_args()

    if args.trials <= 0 or args.coarse_size <= 0 or args.stage2_top <= 0:
        raise ValueError("trials, coarse-size and stage2-top must be positive integers")

    modes = _parse_mode_list(args.modes)
    margins = _parse_float_list(args.margins, "--margins")
    buy_thresholds = _parse_float_list(args.buy_thresholds, "--buy-thresholds")
    sell_thresholds = _parse_float_list(args.sell_thresholds, "--sell-thresholds")

    opt_start = _parse_date(args.opt_start_date, "--opt-start-date")
    opt_end = _parse_date(args.opt_end_date, "--opt-end-date")
    if opt_end <= opt_start:
        raise ValueError("Optimization end date must be after start date")

    if args.eval_start_date:
        eval_start = _parse_date(args.eval_start_date, "--eval-start-date")
    else:
        eval_start = opt_start
    if args.eval_end_date:
        eval_end = _parse_date(args.eval_end_date, "--eval-end-date")
    else:
        eval_end = opt_end
    if eval_end <= eval_start:
        raise ValueError("Evaluation end date must be after start date")

    symbols = resolve_universe(args.universe)
    if not symbols:
        raise RuntimeError("Universe resolved to empty symbol set.")

    data_start = min(opt_start, eval_start)
    data_end = max(opt_end, eval_end)
    print(f"preloading data interval={args.interval} universe={args.universe} symbols={len(symbols)}")
    print(f"data window: {data_start.date()} -> {data_end.date()}")

    logging.getLogger("bist_bot.data").setLevel(logging.WARNING)
    logging.getLogger("bist_bot.backtest").setLevel(logging.WARNING)
    logging.getLogger("bist_bot.strategy").setLevel(logging.WARNING)
    logging.getLogger("bist_bot.ai_strategy").setLevel(logging.WARNING)

    data = _preload_data(
        symbols=symbols,
        interval=args.interval,
        start=data_start,
        end=data_end,
        min_rows=args.min_rows,
    )
    if not data:
        raise RuntimeError("No symbol data loaded for optimization.")

    symbols_loaded = sorted(data.keys(), key=lambda item: len(data[item]), reverse=True)
    coarse_symbols = symbols_loaded[: min(args.coarse_size, len(symbols_loaded))]
    print(
        f"loaded symbols={len(symbols_loaded)} coarse_symbols={len(coarse_symbols)} "
        f"rows_min={min(len(data[s]) for s in symbols_loaded)} rows_max={max(len(data[s]) for s in symbols_loaded)}"
    )

    provider = _CachedProvider(data)
    rng = random.Random(args.seed)

    # --- Hybrid search ---
    candidate_pool = _build_hybrid_candidate_pool(modes, margins, buy_thresholds, sell_thresholds)
    if not candidate_pool:
        raise RuntimeError("Hybrid candidate pool is empty.")

    trials = min(args.trials, len(candidate_pool))
    stage2_top = min(args.stage2_top, trials)
    sampled_candidates = rng.sample(candidate_pool, k=trials)
    print(f"hybrid candidates total={len(candidate_pool)} sampled={trials}")

    stage1_trials: List[Dict[str, object]] = []
    for idx, candidate in enumerate(sampled_candidates, start=1):
        strategy = HybridAiRsiStrategy(
            mode=candidate.mode,
            model_dir=args.model_dir,
            model_path=args.model_path,
            buy_threshold=candidate.buy_threshold,
            sell_threshold=candidate.sell_threshold,
            probability_margin=candidate.probability_margin,
        )
        metrics = _evaluate_strategy(
            strategy_name="hybrid",
            strategy=strategy,
            provider=provider,
            symbols=coarse_symbols,
            interval=args.interval,
            start=opt_start,
            end=opt_end,
        )
        stage1_trials.append({"candidate": _candidate_to_dict(candidate), "metrics": metrics})
        print(
            f"[hybrid stage1 {idx}/{trials}] "
            f"mode={candidate.mode} margin={candidate.probability_margin:.2f} "
            f"buy={candidate.buy_threshold:.2f} sell={candidate.sell_threshold:.2f} "
            f"score={float(metrics['score']):.3f}"
        )

    stage1_sorted = sorted(stage1_trials, key=lambda item: float(item["metrics"]["score"]), reverse=True)
    stage2_candidates = stage1_sorted[:stage2_top]
    print(f"hybrid stage2 candidates={len(stage2_candidates)}")

    stage2_trials: List[Dict[str, object]] = []
    for idx, payload in enumerate(stage2_candidates, start=1):
        candidate_data = payload["candidate"]
        candidate = HybridCandidate(
            mode=str(candidate_data["mode"]),
            probability_margin=float(candidate_data["probability_margin"]),
            buy_threshold=float(candidate_data["buy_threshold"]),
            sell_threshold=float(candidate_data["sell_threshold"]),
        )
        strategy = HybridAiRsiStrategy(
            mode=candidate.mode,
            model_dir=args.model_dir,
            model_path=args.model_path,
            buy_threshold=candidate.buy_threshold,
            sell_threshold=candidate.sell_threshold,
            probability_margin=candidate.probability_margin,
        )
        metrics = _evaluate_strategy(
            strategy_name="hybrid",
            strategy=strategy,
            provider=provider,
            symbols=symbols_loaded,
            interval=args.interval,
            start=opt_start,
            end=opt_end,
        )
        stage2_trials.append({"candidate": _candidate_to_dict(candidate), "metrics": metrics})
        print(
            f"[hybrid stage2 {idx}/{len(stage2_candidates)}] "
            f"mode={candidate.mode} margin={candidate.probability_margin:.2f} "
            f"buy={candidate.buy_threshold:.2f} sell={candidate.sell_threshold:.2f} "
            f"score={float(metrics['score']):.3f}"
        )

    stage2_sorted = sorted(stage2_trials, key=lambda item: float(item["metrics"]["score"]), reverse=True)
    best_hybrid_opt = stage2_sorted[0]

    holdout_hybrid_trials: List[Dict[str, object]] = []
    for idx, payload in enumerate(stage2_sorted, start=1):
        candidate_data = payload["candidate"]
        candidate = HybridCandidate(
            mode=str(candidate_data["mode"]),
            probability_margin=float(candidate_data["probability_margin"]),
            buy_threshold=float(candidate_data["buy_threshold"]),
            sell_threshold=float(candidate_data["sell_threshold"]),
        )
        strategy = HybridAiRsiStrategy(
            mode=candidate.mode,
            model_dir=args.model_dir,
            model_path=args.model_path,
            buy_threshold=candidate.buy_threshold,
            sell_threshold=candidate.sell_threshold,
            probability_margin=candidate.probability_margin,
        )
        metrics = _evaluate_strategy(
            strategy_name="hybrid",
            strategy=strategy,
            provider=provider,
            symbols=symbols_loaded,
            interval=args.interval,
            start=eval_start,
            end=eval_end,
        )
        holdout_hybrid_trials.append({"candidate": _candidate_to_dict(candidate), "metrics": metrics})
        print(
            f"[hybrid eval {idx}/{len(stage2_sorted)}] "
            f"mode={candidate.mode} margin={candidate.probability_margin:.2f} "
            f"buy={candidate.buy_threshold:.2f} sell={candidate.sell_threshold:.2f} "
            f"score={float(metrics['score']):.3f}"
        )

    holdout_hybrid_sorted = sorted(holdout_hybrid_trials, key=lambda item: float(item["metrics"]["score"]), reverse=True)
    best_hybrid_eval = holdout_hybrid_sorted[0]

    # --- AI threshold search ---
    ai_pool = _build_ai_threshold_pool(buy_thresholds, sell_thresholds)
    print(f"ai threshold candidate count={len(ai_pool)}")

    ai_opt_trials: List[Dict[str, object]] = []
    for idx, candidate in enumerate(ai_pool, start=1):
        metrics = _evaluate_strategy(
            strategy_name="ai_model",
            strategy=AIModelStrategy(
                model_dir=args.model_dir,
                model_path=args.model_path,
                buy_threshold=candidate.buy_threshold,
                sell_threshold=candidate.sell_threshold,
            ),
            provider=provider,
            symbols=symbols_loaded,
            interval=args.interval,
            start=opt_start,
            end=opt_end,
        )
        ai_opt_trials.append({"candidate": _ai_candidate_to_dict(candidate), "metrics": metrics})
        print(
            f"[ai opt {idx}/{len(ai_pool)}] "
            f"buy={candidate.buy_threshold:.2f} sell={candidate.sell_threshold:.2f} "
            f"score={float(metrics['score']):.3f}"
        )
    ai_opt_sorted = sorted(ai_opt_trials, key=lambda item: float(item["metrics"]["score"]), reverse=True)
    best_ai_opt = ai_opt_sorted[0]

    ai_eval_trials: List[Dict[str, object]] = []
    for idx, candidate in enumerate(ai_pool, start=1):
        metrics = _evaluate_strategy(
            strategy_name="ai_model",
            strategy=AIModelStrategy(
                model_dir=args.model_dir,
                model_path=args.model_path,
                buy_threshold=candidate.buy_threshold,
                sell_threshold=candidate.sell_threshold,
            ),
            provider=provider,
            symbols=symbols_loaded,
            interval=args.interval,
            start=eval_start,
            end=eval_end,
        )
        ai_eval_trials.append({"candidate": _ai_candidate_to_dict(candidate), "metrics": metrics})
        print(
            f"[ai eval {idx}/{len(ai_pool)}] "
            f"buy={candidate.buy_threshold:.2f} sell={candidate.sell_threshold:.2f} "
            f"score={float(metrics['score']):.3f}"
        )
    ai_eval_sorted = sorted(ai_eval_trials, key=lambda item: float(item["metrics"]["score"]), reverse=True)
    best_ai_eval = ai_eval_sorted[0]

    # --- Selection logic for deployment ---
    selected_hybrid = best_hybrid_opt if args.selection_window == "opt" else best_hybrid_eval
    selected_ai = best_ai_opt if args.selection_window == "opt" else best_ai_eval

    selected_hybrid_key = str(selected_hybrid["candidate"]["key"])
    selected_ai_key = str(selected_ai["candidate"]["key"])

    selected_hybrid_eval_trial = _find_trial_by_key(holdout_hybrid_trials, selected_hybrid_key)
    if selected_hybrid_eval_trial is None:
        raise RuntimeError("Selected hybrid candidate missing eval metrics.")

    selected_ai_eval_trial = _find_trial_by_key(ai_eval_trials, selected_ai_key)
    if selected_ai_eval_trial is None:
        raise RuntimeError("Selected AI threshold candidate missing eval metrics.")

    baseline_rsi_eval = _evaluate_strategy(
        strategy_name="rsi_macd",
        strategy=RsiMacdStrategy(),
        provider=provider,
        symbols=symbols_loaded,
        interval=args.interval,
        start=eval_start,
        end=eval_end,
    )
    baseline_ai_default_eval = _evaluate_strategy(
        strategy_name="ai_model_default",
        strategy=AIModelStrategy(model_dir=args.model_dir, model_path=args.model_path),
        provider=provider,
        symbols=symbols_loaded,
        interval=args.interval,
        start=eval_start,
        end=eval_end,
    )

    ai_selected_eval = dict(selected_ai_eval_trial["metrics"])
    ai_selected_eval["strategy"] = "ai_model_tuned_selected"

    hybrid_selected_eval = dict(selected_hybrid_eval_trial["metrics"])
    hybrid_selected_eval["strategy"] = "hybrid_tuned_selected"

    evaluation_ranked = sorted(
        [baseline_rsi_eval, baseline_ai_default_eval, ai_selected_eval, hybrid_selected_eval],
        key=lambda item: float(item["score"]),
        reverse=True,
    )

    deploy_config = {
        "created_at": datetime.now().isoformat(),
        "selection_window": args.selection_window,
        "interval": args.interval,
        "universe": args.universe,
        "opt_window": {"start": opt_start.strftime("%Y-%m-%d"), "end": opt_end.strftime("%Y-%m-%d")},
        "eval_window": {"start": eval_start.strftime("%Y-%m-%d"), "end": eval_end.strftime("%Y-%m-%d")},
        "ai": {
            "buy_threshold": float(selected_ai["candidate"]["buy_threshold"]),
            "sell_threshold": float(selected_ai["candidate"]["sell_threshold"]),
            "opt_score": float(best_ai_opt["metrics"]["score"]),
            "eval_score": float(selected_ai_eval_trial["metrics"]["score"]),
        },
        "hybrid": {
            "mode": str(selected_hybrid["candidate"]["mode"]),
            "probability_margin": float(selected_hybrid["candidate"]["probability_margin"]),
            "buy_threshold": float(selected_hybrid["candidate"]["buy_threshold"]),
            "sell_threshold": float(selected_hybrid["candidate"]["sell_threshold"]),
            "opt_score": float(best_hybrid_opt["metrics"]["score"]),
            "eval_score": float(selected_hybrid_eval_trial["metrics"]["score"]),
        },
    }

    deploy_config_path = Path(args.deploy_config_path)
    deploy_config_path.parent.mkdir(parents=True, exist_ok=True)
    deploy_config_path.write_text(json.dumps(deploy_config, indent=2, ensure_ascii=False), encoding="utf-8")

    payload = {
        "meta": {
            "created_at": datetime.now().isoformat(),
            "interval": args.interval,
            "universe": args.universe,
            "symbols_requested": len(symbols),
            "symbols_loaded": len(symbols_loaded),
            "opt_window": {"start": opt_start.strftime("%Y-%m-%d"), "end": opt_end.strftime("%Y-%m-%d")},
            "eval_window": {"start": eval_start.strftime("%Y-%m-%d"), "end": eval_end.strftime("%Y-%m-%d")},
            "selection_window": args.selection_window,
            "search": {
                "hybrid_trials": trials,
                "coarse_size": args.coarse_size,
                "stage2_top": stage2_top,
                "modes": modes,
                "margins": margins,
                "buy_thresholds": buy_thresholds,
                "sell_thresholds": sell_thresholds,
                "seed": args.seed,
            },
            "model": {"dir": args.model_dir, "path": args.model_path or "latest"},
            "deploy_config_path": str(deploy_config_path),
        },
        "hybrid_stage1_trials": stage1_sorted,
        "hybrid_stage2_trials_opt_window": stage2_sorted,
        "hybrid_stage2_trials_eval_window": holdout_hybrid_sorted,
        "ai_trials_opt_window": ai_opt_sorted,
        "ai_trials_eval_window": ai_eval_sorted,
        "best_hybrid_opt_window": best_hybrid_opt,
        "best_hybrid_eval_window": best_hybrid_eval,
        "best_ai_opt_window": best_ai_opt,
        "best_ai_eval_window": best_ai_eval,
        "selected_hybrid_for_deploy": selected_hybrid,
        "selected_ai_for_deploy": selected_ai,
        "evaluation_ranked": evaluation_ranked,
        "deploy_config": deploy_config,
    }

    if args.output:
        output_path = Path(args.output)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path("reports") / f"hybrid_param_optimization_{ts}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.summary_md:
        md_path = Path(args.summary_md)
    else:
        md_path = output_path.with_suffix(".md")
    md_lines = [
        "# AI + Hybrid Parameter Optimization Summary",
        "",
        f"- Selection window: `{args.selection_window}`",
        f"- Opt window: `{opt_start.date()}` -> `{opt_end.date()}`",
        f"- Eval window: `{eval_start.date()}` -> `{eval_end.date()}`",
        f"- Symbols loaded: `{len(symbols_loaded)}`",
        "",
        "## Selected For Deployment",
        "",
        f"- AI buy/sell: `{float(selected_ai['candidate']['buy_threshold']):.3f}` / `{float(selected_ai['candidate']['sell_threshold']):.3f}`",
        f"- Hybrid mode: `{selected_hybrid['candidate']['mode']}`",
        f"- Hybrid margin: `{float(selected_hybrid['candidate']['probability_margin']):.3f}`",
        f"- Hybrid buy/sell: `{float(selected_hybrid['candidate']['buy_threshold']):.3f}` / `{float(selected_hybrid['candidate']['sell_threshold']):.3f}`",
        f"- Deploy config: `{deploy_config_path}`",
        "",
        "## Eval Ranking",
        "",
        "| Rank | Strategy | Score | Median PnL % | Mean PnL % | Success % | Trades |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for rank, item in enumerate(evaluation_ranked, start=1):
        md_lines.append(
            f"| {rank} | `{item['strategy']}` | {float(item['score']):.3f} | "
            f"{float(item['median_pnl_pct']):+.2f} | {float(item['mean_pnl_pct']):+.2f} | "
            f"{float(item['symbol_success_rate_pct']):.1f} | {int(item['total_trades'])} |"
        )
    md_lines.append("")
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    print("\nSELECTED PARAMETERS")
    print(
        f"ai buy={float(selected_ai['candidate']['buy_threshold']):.2f} "
        f"sell={float(selected_ai['candidate']['sell_threshold']):.2f}"
    )
    print(
        f"hybrid mode={selected_hybrid['candidate']['mode']} "
        f"margin={float(selected_hybrid['candidate']['probability_margin']):.2f} "
        f"buy={float(selected_hybrid['candidate']['buy_threshold']):.2f} "
        f"sell={float(selected_hybrid['candidate']['sell_threshold']):.2f}"
    )
    print("\nEVALUATION RANKING")
    for rank, item in enumerate(evaluation_ranked, start=1):
        print(
            f"{rank}. {item['strategy']}: "
            f"score={float(item['score']):.3f} "
            f"median={float(item['median_pnl_pct']):+.2f}% "
            f"mean={float(item['mean_pnl_pct']):+.2f}% "
            f"success={float(item['symbol_success_rate_pct']):.1f}% "
            f"trades={int(item['total_trades'])}"
        )
    print(f"\nreport={output_path}")
    print(f"summary={md_path}")
    print(f"deploy_config={deploy_config_path}")


if __name__ == "__main__":
    main()
