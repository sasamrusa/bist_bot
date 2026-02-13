from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from bist_bot.ai_pipeline.dataset_builder import build_training_dataset
from bist_bot.ai_pipeline.drift_monitor import build_feature_stats
from bist_bot.ai_pipeline.registry import save_model_bundle
from bist_bot.ai_pipeline.signal_policy import PolicyConfig, evaluate_probability_policy
from bist_bot.ai_pipeline.universe import resolve_universe
from bist_bot.core.config import TICKERS
from bist_bot.data.yf_provider import YFDataProvider


def _select_model(random_state: int, backend: str) -> Tuple[object, str]:
    requested = backend.lower().strip()
    backends = ["lightgbm", "catboost", "sklearn_hgb"] if requested == "auto" else [requested]

    if "lightgbm" in backends:
        try:
            from lightgbm import LGBMClassifier  # type: ignore

            model = LGBMClassifier(
                n_estimators=350,
                learning_rate=0.04,
                num_leaves=31,
                subsample=0.85,
                colsample_bytree=0.85,
                random_state=random_state,
            )
            return model, "lightgbm"
        except Exception:
            if requested == "lightgbm":
                raise

    if "catboost" in backends:
        try:
            from catboost import CatBoostClassifier  # type: ignore

            model = CatBoostClassifier(
                iterations=450,
                depth=6,
                learning_rate=0.04,
                loss_function="Logloss",
                verbose=False,
                random_seed=random_state,
            )
            return model, "catboost"
        except Exception:
            if requested == "catboost":
                raise

    if requested not in {"auto", "sklearn_hgb"} and requested not in {"lightgbm", "catboost"}:
        raise ValueError(f"Unsupported model backend: {backend}")

    model = HistGradientBoostingClassifier(
        max_iter=400,
        learning_rate=0.05,
        max_depth=6,
        min_samples_leaf=40,
        random_state=random_state,
    )
    return model, "sklearn_hgb"


def _safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return 0.5
    return float(roc_auc_score(y_true, y_prob))


def _walk_forward_windows(
    start: pd.Timestamp,
    end: pd.Timestamp,
    train_years: int,
    test_months: int,
    step_months: int,
    gap_days: int,
) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    windows: List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]] = []
    cursor = pd.Timestamp(start)
    hard_end = pd.Timestamp(end)

    while cursor < hard_end:
        train_start = cursor
        train_end = (train_start + pd.DateOffset(years=train_years)) - pd.Timedelta(days=1)
        test_start = train_end + pd.Timedelta(days=1 + gap_days)
        test_end = (test_start + pd.DateOffset(months=test_months)) - pd.Timedelta(days=1)
        if test_end > hard_end:
            break

        windows.append((train_start, train_end, test_start, test_end))
        cursor = cursor + pd.DateOffset(months=step_months)

    return windows


def main() -> None:
    parser = argparse.ArgumentParser(description="Train AI-based BIST trading model")
    parser.add_argument("--years", type=int, default=2, help="History years to download")
    parser.add_argument("--start-date", type=str, default="", help="Optional explicit start date YYYY-MM-DD")
    parser.add_argument("--end-date", type=str, default="", help="Optional explicit end date YYYY-MM-DD")
    parser.add_argument("--interval", type=str, default="1h", help="Bar interval")
    parser.add_argument("--universe", type=str, default="config", help="Symbol universe: config | bist100")
    parser.add_argument("--model-backend", type=str, default="auto", help="auto | lightgbm | catboost | sklearn_hgb")
    parser.add_argument("--horizon-bars", type=int, default=5, help="Label horizon in bars")
    parser.add_argument("--min-abs-move", type=float, default=0.01, help="Meta-label absolute move threshold")
    parser.add_argument("--min-rows-per-symbol", type=int, default=500, help="Minimum rows per symbol")
    parser.add_argument("--train-years", type=int, default=5, help="Walk-forward train window in years")
    parser.add_argument("--test-months", type=int, default=12, help="Walk-forward test window in months")
    parser.add_argument("--step-months", type=int, default=6, help="Walk-forward step in months")
    parser.add_argument("--gap-days", type=int, default=5, help="Temporal embargo gap between train/test")
    parser.add_argument("--buy-threshold", type=float, default=0.58, help="Policy BUY threshold")
    parser.add_argument("--sell-threshold", type=float, default=0.42, help="Policy SELL threshold")
    parser.add_argument("--calibrate", action="store_true", help="Apply probability calibration")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max-symbols", type=int, default=0, help="Optional cap for faster experimentation")
    parser.add_argument("--model-dir", type=str, default="models/ai_registry", help="Model output directory")
    parser.add_argument("--report-path", type=str, default="", help="Optional explicit report path")
    args = parser.parse_args()

    if args.buy_threshold <= args.sell_threshold:
        raise ValueError("buy-threshold must be greater than sell-threshold")

    if args.end_date:
        end = datetime.strptime(args.end_date, "%Y-%m-%d")
    else:
        end = datetime.now()

    if args.start_date:
        start = datetime.strptime(args.start_date, "%Y-%m-%d")
    elif args.interval == "1h":
        # Yahoo intraday window is ~730 days; keep safe headroom.
        intraday_days = min(args.years * 365, 700)
        start = end - timedelta(days=intraday_days)
    else:
        start = end - timedelta(days=(args.years * 365) + 20)

    if end <= start:
        raise ValueError("end-date must be after start-date")

    if args.interval == "1h" and not args.start_date and args.years > 2:
        print("warning: 1h data on Yahoo has ~730 day limit, clamping years to 2")
        args.years = 2

    symbols = resolve_universe(args.universe)
    if args.universe == "config":
        symbols = list(TICKERS)
    if args.max_symbols > 0:
        symbols = symbols[: args.max_symbols]
    print(f"universe={args.universe} symbol_count={len(symbols)} interval={args.interval}")

    provider = YFDataProvider()
    dataset_result = build_training_dataset(
        data_provider=provider,
        symbols=symbols,
        interval=args.interval,
        start=start,
        end=end,
        horizon_bars=args.horizon_bars,
        min_abs_move=args.min_abs_move,
        min_rows_per_symbol=args.min_rows_per_symbol,
    )
    dataset = dataset_result.dataset
    if dataset.empty:
        raise RuntimeError("No dataset built for AI training.")

    feature_columns = dataset_result.feature_columns
    policy_cfg = PolicyConfig(
        buy_threshold=args.buy_threshold,
        sell_threshold=args.sell_threshold,
    )

    ts = pd.to_datetime(dataset["ts"])
    windows = _walk_forward_windows(
        start=ts.min().normalize(),
        end=ts.max().normalize(),
        train_years=args.train_years,
        test_months=args.test_months,
        step_months=args.step_months,
        gap_days=args.gap_days,
    )

    fold_results: List[Dict[str, object]] = []
    for fold_id, (train_start, train_end, test_start, test_end) in enumerate(windows, start=1):
        train_mask = (ts >= train_start) & (ts <= train_end)
        test_mask = (ts >= test_start) & (ts <= test_end)
        train_df = dataset.loc[train_mask].copy()
        test_df = dataset.loc[test_mask].copy()
        if train_df.empty or test_df.empty:
            continue

        x_train = train_df[feature_columns].to_numpy(dtype=float)
        y_train = train_df["y_dir"].to_numpy(dtype=int)
        x_test = test_df[feature_columns].to_numpy(dtype=float)
        y_test = test_df["y_dir"].to_numpy(dtype=int)

        model, backend = _select_model(args.seed + fold_id, args.model_backend)
        fitted_model: object
        if args.calibrate:
            calibrated = CalibratedClassifierCV(estimator=model, cv=3, method="sigmoid")
            calibrated.fit(x_train, y_train)
            fitted_model = calibrated
        else:
            model.fit(x_train, y_train)
            fitted_model = model

        if hasattr(fitted_model, "predict_proba"):
            test_proba = fitted_model.predict_proba(x_test)[:, 1]
        else:
            test_proba = np.asarray(fitted_model.predict(x_test), dtype=float)

        test_pred = (test_proba >= 0.5).astype(int)
        cls_metrics = {
            "accuracy": float(accuracy_score(y_test, test_pred)),
            "f1": float(f1_score(y_test, test_pred, zero_division=0)),
            "auc": _safe_auc(y_test, test_proba),
        }
        policy_metrics = evaluate_probability_policy(test_df, test_proba, policy_cfg)
        fold_results.append(
            {
                "fold_id": fold_id,
                "train_start": train_start.strftime("%Y-%m-%d"),
                "train_end": train_end.strftime("%Y-%m-%d"),
                "test_start": test_start.strftime("%Y-%m-%d"),
                "test_end": test_end.strftime("%Y-%m-%d"),
                "train_rows": int(len(train_df)),
                "test_rows": int(len(test_df)),
                "model_backend": backend,
                "classification": cls_metrics,
                "policy": policy_metrics,
            }
        )
        print(
            f"[fold {fold_id}] backend={backend} auc={cls_metrics['auc']:.3f} "
            f"policy_avg_symbol_return={policy_metrics['avg_symbol_return_pct']:.2f}% "
            f"trades={policy_metrics['trades']}"
        )

    # Final training on all data for deployment.
    x_all = dataset[feature_columns].to_numpy(dtype=float)
    y_all = dataset["y_dir"].to_numpy(dtype=int)
    final_model, final_backend = _select_model(args.seed, args.model_backend)
    if args.calibrate:
        final_fitted = CalibratedClassifierCV(estimator=final_model, cv=3, method="sigmoid")
        final_fitted.fit(x_all, y_all)
        final_model_obj = final_fitted
    else:
        final_model.fit(x_all, y_all)
        final_model_obj = final_model

    feature_stats = build_feature_stats(dataset, feature_columns)
    bundle = {
        "model": final_model_obj,
        "model_backend": final_backend,
        "feature_columns": feature_columns,
        "symbol_to_id": dataset_result.symbol_to_id,
        "feature_stats": feature_stats,
        "train_rows": int(len(dataset)),
        "symbols_used": sorted(dataset["symbol"].unique().tolist()),
        "config": {
            "interval": args.interval,
            "universe": args.universe,
            "model_backend_requested": args.model_backend,
            "years": args.years,
            "start_date": start.strftime("%Y-%m-%d"),
            "end_date": end.strftime("%Y-%m-%d"),
            "horizon_bars": args.horizon_bars,
            "min_abs_move": args.min_abs_move,
            "buy_threshold": args.buy_threshold,
            "sell_threshold": args.sell_threshold,
            "train_years": args.train_years,
            "test_months": args.test_months,
            "step_months": args.step_months,
            "gap_days": args.gap_days,
            "calibrate": args.calibrate,
        },
        "fold_results": fold_results,
        "created_at": datetime.now().isoformat(),
    }
    model_path = save_model_bundle(bundle, Path(args.model_dir))

    summary = {
        "model_path": str(model_path),
        "model_backend": final_backend,
        "rows": int(len(dataset)),
        "symbols": int(dataset["symbol"].nunique()),
        "folds": len(fold_results),
        "avg_fold_auc": float(np.mean([item["classification"]["auc"] for item in fold_results])) if fold_results else 0.0,
        "avg_fold_policy_return_pct": float(np.mean([item["policy"]["avg_symbol_return_pct"] for item in fold_results])) if fold_results else 0.0,
    }
    report = {
        "summary": summary,
        "folds": fold_results,
        "skipped_symbols": dataset_result.skipped_symbols,
        "symbol_to_id": dataset_result.symbol_to_id,
    }

    if args.report_path:
        report_path = Path(args.report_path)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = Path("reports") / f"ai_training_{timestamp}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"final_model={model_path}")
    print(f"report={report_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
