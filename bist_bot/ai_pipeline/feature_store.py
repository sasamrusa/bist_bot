from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd


BASE_FEATURE_COLUMNS: List[str] = [
    "ret_1",
    "ret_3",
    "ret_5",
    "ret_10",
    "ret_20",
    "vol_10",
    "vol_20",
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_hist",
    "ema_ratio_12_26",
    "ema_slope_5",
    "bb_pos_20_2",
    "atr_pct_14",
    "range_pct",
    "volume_z_20",
    "volume_ratio_20",
]


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def build_feature_frame(ohlcv: pd.DataFrame, symbol: str) -> pd.DataFrame:
    df = ohlcv.copy()
    if df.empty:
        return pd.DataFrame()

    df = df.sort_index()
    df.columns = [str(col).lower() for col in df.columns]
    for col in ("open", "high", "low", "close", "volume"):
        if col not in df.columns:
            df[col] = np.nan

    close = pd.to_numeric(df["close"], errors="coerce")
    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    volume = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)

    returns = close.pct_change()
    ema_12 = _ema(close, 12)
    ema_26 = _ema(close, 26)
    macd = ema_12 - ema_26
    macd_signal = _ema(macd, 9)
    macd_hist = macd - macd_signal
    rsi_14 = _rsi(close, 14)
    atr_14 = _atr(high, low, close, 14)

    bb_mid = close.rolling(20, min_periods=20).mean()
    bb_std = close.rolling(20, min_periods=20).std(ddof=0)
    bb_upper = bb_mid + 2.0 * bb_std
    bb_lower = bb_mid - 2.0 * bb_std
    bb_width = (bb_upper - bb_lower).replace(0.0, np.nan)
    bb_pos = (close - bb_lower) / bb_width

    vol_mean_20 = volume.rolling(20, min_periods=20).mean()
    vol_std_20 = volume.rolling(20, min_periods=20).std(ddof=0).replace(0.0, np.nan)
    volume_z_20 = (volume - vol_mean_20) / vol_std_20
    volume_ratio_20 = volume / vol_mean_20.replace(0.0, np.nan)

    features = pd.DataFrame(index=df.index)
    features["ret_1"] = returns
    features["ret_3"] = close.pct_change(3)
    features["ret_5"] = close.pct_change(5)
    features["ret_10"] = close.pct_change(10)
    features["ret_20"] = close.pct_change(20)
    features["vol_10"] = returns.rolling(10, min_periods=10).std(ddof=0)
    features["vol_20"] = returns.rolling(20, min_periods=20).std(ddof=0)
    features["rsi_14"] = rsi_14
    features["macd"] = macd
    features["macd_signal"] = macd_signal
    features["macd_hist"] = macd_hist
    features["ema_ratio_12_26"] = ema_12 / ema_26.replace(0.0, np.nan)
    features["ema_slope_5"] = ema_12.pct_change(5)
    features["bb_pos_20_2"] = bb_pos
    features["atr_pct_14"] = atr_14 / close.replace(0.0, np.nan)
    features["range_pct"] = (high - low) / close.replace(0.0, np.nan)
    features["volume_z_20"] = volume_z_20
    features["volume_ratio_20"] = volume_ratio_20

    features["close"] = close
    features["atr"] = atr_14
    features["symbol"] = symbol
    return features

