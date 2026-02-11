from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List

import pandas as pd

from bist_bot.ai_pipeline.feature_store import BASE_FEATURE_COLUMNS, build_feature_frame
from bist_bot.ai_pipeline.labeling import build_labels
from bist_bot.core.interfaces import IDataProvider


@dataclass
class DatasetBuildResult:
    dataset: pd.DataFrame
    feature_columns: List[str]
    symbol_to_id: Dict[str, int]
    skipped_symbols: List[str]


def build_training_dataset(
    data_provider: IDataProvider,
    symbols: Iterable[str],
    interval: str,
    start: datetime,
    end: datetime,
    horizon_bars: int,
    min_abs_move: float,
    min_rows_per_symbol: int,
) -> DatasetBuildResult:
    rows: List[pd.DataFrame] = []
    skipped: List[str] = []

    symbols_list = list(symbols)
    for symbol in symbols_list:
        historical = data_provider.get_historical_data(symbol, interval, start, end)
        if historical.empty:
            skipped.append(symbol)
            continue

        features = build_feature_frame(historical, symbol=symbol)
        if features.empty:
            skipped.append(symbol)
            continue

        labels = build_labels(features["close"], horizon_bars=horizon_bars, min_abs_move=min_abs_move)
        merged = features.join(labels, how="left")
        required = BASE_FEATURE_COLUMNS + ["close", "atr", "y_dir", "y_meta", "future_return"]
        merged = merged.dropna(subset=required)
        if len(merged) < min_rows_per_symbol:
            skipped.append(symbol)
            continue

        rows.append(merged)

    if not rows:
        return DatasetBuildResult(
            dataset=pd.DataFrame(),
            feature_columns=[],
            symbol_to_id={},
            skipped_symbols=skipped,
        )

    dataset = pd.concat(rows, axis=0).sort_index()
    dataset = dataset.rename_axis("ts").reset_index()

    unique_symbols = sorted(dataset["symbol"].unique().tolist())
    symbol_to_id = {symbol: idx for idx, symbol in enumerate(unique_symbols)}
    dataset["symbol_id"] = dataset["symbol"].map(symbol_to_id).astype(int)

    feature_columns = BASE_FEATURE_COLUMNS + ["symbol_id"]
    dataset = dataset.sort_values(["ts", "symbol"]).reset_index(drop=True)
    return DatasetBuildResult(
        dataset=dataset,
        feature_columns=feature_columns,
        symbol_to_id=symbol_to_id,
        skipped_symbols=sorted(set(skipped)),
    )

