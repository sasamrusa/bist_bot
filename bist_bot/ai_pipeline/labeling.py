from __future__ import annotations

import numpy as np
import pandas as pd


def build_labels(close: pd.Series, horizon_bars: int, min_abs_move: float) -> pd.DataFrame:
    """
    Direction label and trade-quality label.
    - y_dir: 1 if future return > 0 else 0
    - y_meta: 1 if |future return| >= min_abs_move else 0
    """
    close_series = pd.to_numeric(close, errors="coerce")
    future_return = (close_series.shift(-horizon_bars) / close_series) - 1.0

    labels = pd.DataFrame(index=close_series.index)
    labels["future_return"] = future_return
    labels["y_dir"] = (future_return > 0.0).astype(int)
    labels["y_meta"] = (future_return.abs() >= float(min_abs_move)).astype(int)

    # If return cannot be computed for tail rows, clear labels too.
    labels.loc[future_return.isna(), ["y_dir", "y_meta"]] = np.nan
    return labels

