from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd


def build_feature_stats(frame: pd.DataFrame, feature_columns: List[str]) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}
    for col in feature_columns:
        series = pd.to_numeric(frame[col], errors="coerce")
        stats[col] = {
            "mean": float(series.mean()) if not series.empty else 0.0,
            "std": float(series.std(ddof=0)) if not series.empty else 0.0,
        }
    return stats


def compute_drift_score(
    recent_frame: pd.DataFrame,
    feature_columns: List[str],
    baseline_stats: Dict[str, Dict[str, float]],
) -> float:
    scores = []
    for col in feature_columns:
        if col not in recent_frame.columns or col not in baseline_stats:
            continue
        recent_mean = float(pd.to_numeric(recent_frame[col], errors="coerce").mean())
        base_mean = float(baseline_stats[col].get("mean", 0.0))
        base_std = float(baseline_stats[col].get("std", 0.0))
        denom = max(base_std, 1e-6)
        scores.append(abs(recent_mean - base_mean) / denom)

    if not scores:
        return 0.0
    return float(np.mean(scores))

