from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from bs4 import BeautifulSoup


STARTING_CASH = 100_000.0
COMMISSION_BPS = 8.0
SLIPPAGE_BPS = 5.0


@dataclass
class StrategyParams:
    rsi_period: int
    rsi_oversold: float
    rsi_overbought: float
    macd_fast: int
    macd_slow: int
    macd_signal: int
    trend_ema_fast: int
    trend_ema_slow: int
    bbands_period: int
    bbands_std: float
    atr_period: int
    score_threshold: int
    signal_edge_min: int
    enable_volume_filter: bool
    volume_sma_period: int
    volume_min_ratio: float
    enable_mtf_filter: bool
    enable_adx_filter: bool
    adx_period: int
    adx_min_value: float
    enable_macdh_filter: bool
    risk_per_trade_pct: float
    stop_atr_multiplier: float
    take_profit_atr_multiplier: float
    enable_protective_exits: bool


@dataclass
class SymbolStats:
    symbol: str
    pnl_pct: float
    trades: int
    closed_trades: int
    wins: int
    losses: int
    win_rate: float
    profit_factor: float


@dataclass
class TrialStats:
    trial_id: int
    score: float
    mean_pnl_pct: float
    median_pnl_pct: float
    active_symbols: int
    total_symbols: int
    total_trades: int
    win_rate: float
    profit_factor: float
    params: StrategyParams


def fetch_bist100_symbols_from_cnbce() -> List[str]:
    """
    CNBCE BIST100 sayfasindan sembolleri alir.
    Sayfa bazen 100'den fazla sembol dondurebilir; sonradan veri kapsamina gore daraltiriz.
    """
    url = "https://www.cnbce.com/borsa/hisseler/bist-100-hisseleri"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    symbols: List[str] = []
    for anchor in soup.select("div.definition-table-symbol-name a.name"):
        href = anchor.get("href", "")
        if "/borsa/hisseler/" not in href:
            continue
        code = href.split("/borsa/hisseler/")[1].split("-")[0].upper()
        if not code or code == "BIST":
            continue
        if code not in symbols:
            symbols.append(code)
    return symbols


def download_daily_data(symbols: Iterable[str], start: datetime, end: datetime) -> Dict[str, pd.DataFrame]:
    tickers = [f"{symbol}.IS" for symbol in symbols]
    raw = yf.download(
        tickers,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        interval="1d",
        auto_adjust=True,
        group_by="ticker",
        progress=False,
        threads=True,
    )
    data: Dict[str, pd.DataFrame] = {}

    for symbol in symbols:
        ticker = f"{symbol}.IS"
        if ticker not in raw.columns.get_level_values(0):
            continue
        df = raw[ticker].copy()
        if df.empty:
            continue
        df = df.rename(columns=lambda column: str(column).lower())
        required = ["open", "high", "low", "close", "volume"]
        for col in required:
            if col not in df.columns:
                df[col] = np.nan
        df = df[required].dropna(subset=["close"]).sort_index()
        if len(df) < 400:
            continue
        data[symbol] = df
    return data


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def compute_rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def compute_macd(close: pd.Series, fast: int, slow: int, signal: int) -> tuple[pd.Series, pd.Series, pd.Series]:
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def compute_bbands(close: pd.Series, period: int, std: float) -> tuple[pd.Series, pd.Series, pd.Series]:
    basis = close.rolling(period, min_periods=period).mean()
    dev = close.rolling(period, min_periods=period).std(ddof=0)
    upper = basis + (dev * std)
    lower = basis - (dev * std)
    return upper, basis, lower


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
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


def compute_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0.0), up_move, 0.0),
        index=high.index,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0.0), down_move, 0.0),
        index=high.index,
    )

    atr = compute_atr(high, low, close, period).replace(0.0, np.nan)
    plus_di = 100.0 * plus_dm.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean() / atr
    minus_di = 100.0 * minus_dm.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean() / atr
    dx = (100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)).fillna(0.0)
    adx = dx.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    return adx


def higher_timeframe_filter(index: pd.DatetimeIndex, close: pd.Series, ema_fast_period: int, ema_slow_period: int) -> tuple[pd.Series, pd.Series]:
    weekly = close.resample("W-FRI").last().dropna()
    if weekly.empty:
        all_false = pd.Series(False, index=index)
        return all_false, all_false

    htf_fast = ema(weekly, ema_fast_period)
    htf_slow = ema(weekly, ema_slow_period)
    # Shift by 1 to avoid lookahead bias: daily bars can only use the prior completed higher-timeframe bar.
    htf_buy = (htf_fast > htf_slow).shift(1).reindex(index, method="ffill").fillna(False)
    htf_sell = (htf_fast < htf_slow).shift(1).reindex(index, method="ffill").fillna(False)
    return htf_buy.astype(bool), htf_sell.astype(bool)


def generate_signals(df: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
    close = pd.to_numeric(df["close"], errors="coerce")
    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    volume = pd.to_numeric(df["volume"], errors="coerce")

    rsi = compute_rsi(close, params.rsi_period)
    macd_line, macd_signal, macd_hist = compute_macd(close, params.macd_fast, params.macd_slow, params.macd_signal)
    ema_fast = ema(close, params.trend_ema_fast)
    ema_slow = ema(close, params.trend_ema_slow)
    bb_upper, _bb_mid, bb_lower = compute_bbands(close, params.bbands_period, params.bbands_std)
    atr = compute_atr(high, low, close, params.atr_period)
    adx = compute_adx(high, low, close, params.adx_period)

    macd_buy_cross = (macd_line > macd_signal) & (macd_line.shift(1) <= macd_signal.shift(1))
    macd_sell_cross = (macd_line < macd_signal) & (macd_line.shift(1) >= macd_signal.shift(1))
    uptrend = ema_fast > ema_slow
    downtrend = ema_fast < ema_slow

    rsi_buy = rsi < params.rsi_oversold
    rsi_sell = rsi > params.rsi_overbought
    bb_buy = close <= bb_lower
    bb_sell = close >= bb_upper

    buy_score = (
        rsi_buy.astype(int)
        + macd_buy_cross.astype(int)
        + bb_buy.astype(int)
        + uptrend.astype(int)
    )
    sell_score = (
        rsi_sell.astype(int)
        + macd_sell_cross.astype(int)
        + bb_sell.astype(int)
        + downtrend.astype(int)
    )

    tradable = atr.notna() & (atr > 0.0)

    if params.enable_volume_filter:
        vol_sma = volume.rolling(params.volume_sma_period, min_periods=max(2, min(5, params.volume_sma_period))).mean()
        volume_ok = volume >= (vol_sma * params.volume_min_ratio)
        tradable = tradable & volume_ok.fillna(False)

    if params.enable_adx_filter:
        tradable = tradable & adx.notna() & (adx >= params.adx_min_value)

    if params.enable_mtf_filter:
        mtf_buy_ok, mtf_sell_ok = higher_timeframe_filter(df.index, close, params.trend_ema_fast, params.trend_ema_slow)
    else:
        mtf_buy_ok = pd.Series(True, index=df.index)
        mtf_sell_ok = pd.Series(True, index=df.index)

    if params.enable_macdh_filter:
        hist_buy_ok = (macd_hist > 0.0).fillna(False)
        hist_sell_ok = (macd_hist < 0.0).fillna(False)
    else:
        hist_buy_ok = pd.Series(True, index=df.index)
        hist_sell_ok = pd.Series(True, index=df.index)

    buy_edge_ok = (buy_score - sell_score) >= params.signal_edge_min
    sell_edge_ok = (sell_score - buy_score) >= params.signal_edge_min

    signals = pd.Series("HOLD", index=df.index)
    signals.loc[
        tradable
        & mtf_buy_ok
        & hist_buy_ok
        & buy_edge_ok
        & (buy_score >= params.score_threshold)
    ] = "BUY"
    signals.loc[
        tradable
        & mtf_sell_ok
        & hist_sell_ok
        & sell_edge_ok
        & (sell_score >= params.score_threshold)
    ] = "SELL"

    out = pd.DataFrame(index=df.index)
    out["signal"] = signals
    out["atr"] = atr
    return out


def backtest_symbol(df: pd.DataFrame, params: StrategyParams) -> SymbolStats:
    fee_rate = COMMISSION_BPS / 10_000.0
    slippage_rate = SLIPPAGE_BPS / 10_000.0

    ind = generate_signals(df, params)
    cash = STARTING_CASH
    qty = 0.0
    open_trade: Dict[str, float] | None = None

    trades = 0
    closed = 0
    wins = 0
    losses = 0
    gross_profit = 0.0
    gross_loss = 0.0

    closes = pd.to_numeric(df["close"], errors="coerce")
    highs = pd.to_numeric(df["high"], errors="coerce")
    lows = pd.to_numeric(df["low"], errors="coerce")

    for i in range(1, len(df)):
        close_price = float(closes.iloc[i])
        high_price = float(highs.iloc[i]) if pd.notna(highs.iloc[i]) else close_price
        low_price = float(lows.iloc[i]) if pd.notna(lows.iloc[i]) else close_price
        signal = str(ind["signal"].iloc[i])
        atr_value = float(ind["atr"].iloc[i]) if pd.notna(ind["atr"].iloc[i]) else np.nan

        if params.enable_protective_exits and qty > 0 and open_trade is not None:
            stop_price = float(open_trade["stop_price"])
            take_profit_price = float(open_trade["take_profit_price"])
            stop_hit = low_price <= stop_price
            take_profit_hit = high_price >= take_profit_price

            if stop_hit or take_profit_hit:
                exit_signal_price = stop_price if stop_hit else take_profit_price
                exit_fill = exit_signal_price * (1.0 - slippage_rate)
                gross_proceeds = qty * exit_fill
                exit_fee = gross_proceeds * fee_rate
                cash += gross_proceeds - exit_fee

                entry_fill = float(open_trade["entry_fill"])
                entry_fee = float(open_trade["entry_fee"])
                trade_pnl = (exit_fill - entry_fill) * qty - entry_fee - exit_fee
                closed += 1
                trades += 1
                if trade_pnl > 0:
                    wins += 1
                    gross_profit += trade_pnl
                elif trade_pnl < 0:
                    losses += 1
                    gross_loss += abs(trade_pnl)

                qty = 0.0
                open_trade = None
                continue

        if signal == "BUY" and qty <= 0 and cash > 0:
            fill = close_price * (1.0 + slippage_rate)
            if pd.isna(atr_value) or atr_value <= 0:
                atr_value = max(close_price * 0.01, 0.01)

            risk_per_share = max(atr_value * params.stop_atr_multiplier, close_price * 0.001)
            risk_budget = max(cash * params.risk_per_trade_pct, 0.0)
            qty_by_risk = risk_budget / risk_per_share if risk_per_share > 0 else 0.0
            qty_by_cash = cash / (fill * (1.0 + fee_rate))
            buy_qty = min(qty_by_risk, qty_by_cash)
            if buy_qty <= 0:
                continue

            gross_cost = buy_qty * fill
            fee = gross_cost * fee_rate
            total_cost = gross_cost + fee
            if total_cost > cash:
                continue

            cash -= total_cost
            qty = buy_qty
            trades += 1
            open_trade = {
                "entry_fill": fill,
                "entry_fee": fee,
                "stop_price": fill - (atr_value * params.stop_atr_multiplier),
                "take_profit_price": fill + (atr_value * params.take_profit_atr_multiplier),
            }

        elif signal == "SELL" and qty > 0 and open_trade is not None:
            fill = close_price * (1.0 - slippage_rate)
            gross_proceeds = qty * fill
            exit_fee = gross_proceeds * fee_rate
            cash += gross_proceeds - exit_fee

            entry_fill = float(open_trade["entry_fill"])
            entry_fee = float(open_trade["entry_fee"])
            trade_pnl = (fill - entry_fill) * qty - entry_fee - exit_fee
            closed += 1
            trades += 1
            if trade_pnl > 0:
                wins += 1
                gross_profit += trade_pnl
            elif trade_pnl < 0:
                losses += 1
                gross_loss += abs(trade_pnl)

            qty = 0.0
            open_trade = None

    final_price = float(closes.iloc[-1])
    final_value = cash + (qty * final_price)
    pnl_pct = ((final_value - STARTING_CASH) / STARTING_CASH) * 100.0

    win_rate = (wins / closed) * 100.0 if closed > 0 else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (3.0 if gross_profit > 0 else 0.0)

    return SymbolStats(
        symbol="",
        pnl_pct=float(pnl_pct),
        trades=trades,
        closed_trades=closed,
        wins=wins,
        losses=losses,
        win_rate=float(win_rate),
        profit_factor=float(profit_factor),
    )


def evaluate_trial(
    trial_id: int,
    params: StrategyParams,
    dataset: Dict[str, pd.DataFrame],
) -> TrialStats:
    symbol_results: List[SymbolStats] = []

    total_wins = 0
    total_closed = 0
    total_profit = 0.0
    total_loss = 0.0
    total_trades = 0
    active_symbols = 0

    for symbol, df in dataset.items():
        stats = backtest_symbol(df, params)
        stats.symbol = symbol
        symbol_results.append(stats)
        total_wins += stats.wins
        total_closed += stats.closed_trades
        total_trades += stats.trades
        if stats.trades > 0:
            active_symbols += 1

        # Approximation: reconstruct global PF from symbol PF and closed trades
        # To keep runtime low, we only aggregate using per-symbol pnl sign split.
        if stats.profit_factor > 0 and stats.losses > 0:
            # Not exact, but keeps PF weight from trade-active symbols.
            total_profit += stats.profit_factor * stats.losses
            total_loss += stats.losses
        elif stats.wins > 0 and stats.losses == 0:
            total_profit += stats.wins

    pnl_values = [stats.pnl_pct for stats in symbol_results]
    mean_pnl = float(np.mean(pnl_values)) if pnl_values else 0.0
    median_pnl = float(np.median(pnl_values)) if pnl_values else 0.0
    win_rate = (total_wins / total_closed) * 100.0 if total_closed > 0 else 0.0
    profit_factor = (total_profit / total_loss) if total_loss > 0 else (3.0 if total_profit > 0 else 0.0)
    profit_factor = min(profit_factor, 5.0)

    activity_ratio = (active_symbols / len(dataset)) if dataset else 0.0
    score = (
        (0.55 * median_pnl)
        + (0.35 * mean_pnl)
        + (0.15 * win_rate)
        + (0.35 * min(profit_factor, 3.0))
        + (3.0 * activity_ratio)
    )

    return TrialStats(
        trial_id=trial_id,
        score=float(score),
        mean_pnl_pct=mean_pnl,
        median_pnl_pct=median_pnl,
        active_symbols=active_symbols,
        total_symbols=len(dataset),
        total_trades=total_trades,
        win_rate=float(win_rate),
        profit_factor=float(profit_factor),
        params=params,
    )


def sample_params(rng: random.Random) -> StrategyParams:
    while True:
        rsi_oversold = rng.choice([28.0, 30.0, 35.0, 40.0, 45.0])
        rsi_overbought = rng.choice([55.0, 60.0, 65.0, 70.0, 72.0])
        if rsi_overbought - rsi_oversold < 15.0:
            continue

        macd_fast = rng.choice([6, 8, 10, 12])
        macd_slow = rng.choice([18, 21, 26, 30])
        if macd_fast >= macd_slow:
            continue

        trend_fast = rng.choice([10, 20, 30])
        trend_slow = rng.choice([40, 50, 75, 100])
        if trend_fast >= trend_slow:
            continue

        return StrategyParams(
            rsi_period=14,
            rsi_oversold=rsi_oversold,
            rsi_overbought=rsi_overbought,
            macd_fast=macd_fast,
            macd_slow=macd_slow,
            macd_signal=rng.choice([5, 7, 9]),
            trend_ema_fast=trend_fast,
            trend_ema_slow=trend_slow,
            bbands_period=rng.choice([18, 20, 22]),
            bbands_std=rng.choice([1.8, 2.0, 2.2, 2.4]),
            atr_period=rng.choice([10, 14, 21]),
            score_threshold=rng.choice([2, 3]),
            signal_edge_min=rng.choice([0, 1, 2]),
            enable_volume_filter=rng.choice([True, False]),
            volume_sma_period=rng.choice([15, 20, 30]),
            volume_min_ratio=rng.choice([0.7, 0.8, 0.9, 1.0]),
            enable_mtf_filter=rng.choice([True, False]),
            enable_adx_filter=rng.choice([True, False]),
            adx_period=rng.choice([10, 14, 20]),
            adx_min_value=rng.choice([15.0, 18.0, 22.0, 25.0]),
            enable_macdh_filter=rng.choice([True, False]),
            risk_per_trade_pct=rng.choice([0.005, 0.01, 0.015]),
            stop_atr_multiplier=rng.choice([1.2, 1.5, 1.8, 2.0]),
            take_profit_atr_multiplier=rng.choice([2.0, 2.5, 3.0, 4.0]),
            enable_protective_exits=rng.choice([True, False]),
        )


def pick_top_100_by_coverage(dataset: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    ranked = sorted(dataset.items(), key=lambda item: len(item[1]), reverse=True)
    top = ranked[:100]
    return dict(top)


def slice_dataset_by_date(
    dataset: Dict[str, pd.DataFrame],
    start: pd.Timestamp,
    end: pd.Timestamp,
    min_rows: int,
) -> Dict[str, pd.DataFrame]:
    sliced: Dict[str, pd.DataFrame] = {}
    for symbol, df in dataset.items():
        window = df.loc[(df.index >= start) & (df.index <= end)].copy()
        if len(window) >= min_rows:
            sliced[symbol] = window
    return sliced


def build_walk_forward_windows(
    start: pd.Timestamp,
    end: pd.Timestamp,
    train_years: int,
    test_months: int,
    step_months: int,
) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    windows: List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]] = []
    cursor = pd.Timestamp(start)
    hard_end = pd.Timestamp(end)

    while cursor < hard_end:
        train_start = cursor
        train_end = (train_start + pd.DateOffset(years=train_years)) - pd.Timedelta(days=1)
        test_start = train_end + pd.Timedelta(days=1)
        test_end = (test_start + pd.DateOffset(months=test_months)) - pd.Timedelta(days=1)

        if test_end > hard_end:
            break

        windows.append((train_start, train_end, test_start, test_end))
        cursor = cursor + pd.DateOffset(months=step_months)

    return windows


def run_walk_forward_optimizer(
    data_full: Dict[str, pd.DataFrame],
    global_start: datetime,
    global_end: datetime,
    train_years: int,
    test_months: int,
    step_months: int,
    trials: int,
    seed: int,
    coarse_size: int,
    top_n_stage2: int,
    min_train_rows: int,
    min_test_rows: int,
    min_symbols: int,
) -> Dict[str, object]:
    windows = build_walk_forward_windows(
        start=pd.Timestamp(global_start),
        end=pd.Timestamp(global_end),
        train_years=train_years,
        test_months=test_months,
        step_months=step_months,
    )

    folds: List[Dict[str, object]] = []
    skipped = 0

    for fold_id, (train_start, train_end, test_start, test_end) in enumerate(windows, start=1):
        train_slice = slice_dataset_by_date(
            dataset=data_full,
            start=train_start,
            end=train_end,
            min_rows=min_train_rows,
        )
        test_slice = slice_dataset_by_date(
            dataset=data_full,
            start=test_start,
            end=test_end,
            min_rows=min_test_rows,
        )

        common_symbols = sorted(set(train_slice.keys()) & set(test_slice.keys()))
        if len(common_symbols) < min_symbols:
            skipped += 1
            print(
                f"[wf] fold={fold_id} skipped "
                f"(common_symbols={len(common_symbols)} < min_symbols={min_symbols})"
            )
            continue

        train_data = {symbol: train_slice[symbol] for symbol in common_symbols}
        test_data = {symbol: test_slice[symbol] for symbol in common_symbols}
        train_data = pick_top_100_by_coverage(train_data)
        test_data = {symbol: test_data[symbol] for symbol in train_data.keys() if symbol in test_data}

        if len(test_data) < min_symbols:
            skipped += 1
            print(
                f"[wf] fold={fold_id} skipped after top100 alignment "
                f"(test_symbols={len(test_data)} < min_symbols={min_symbols})"
            )
            continue

        print(
            f"[wf] fold={fold_id}/{len(windows)} "
            f"train={train_start.date()}->{train_end.date()} "
            f"test={test_start.date()}->{test_end.date()} "
            f"symbols={len(train_data)}"
        )

        train_report = run_optimizer(
            trials=trials,
            seed=seed + fold_id,
            data_full=train_data,
            coarse_size=min(coarse_size, len(train_data)),
            top_n_stage2=min(top_n_stage2, max(1, len(train_data))),
        )
        best_train = train_report.get("best")
        if not best_train:
            skipped += 1
            print(f"[wf] fold={fold_id} no best result on train window, skipped")
            continue

        best_params = StrategyParams(**best_train["params"])
        test_eval = evaluate_trial(
            trial_id=fold_id,
            params=best_params,
            dataset=test_data,
        )

        fold = {
            "fold_id": fold_id,
            "train_start": train_start.strftime("%Y-%m-%d"),
            "train_end": train_end.strftime("%Y-%m-%d"),
            "test_start": test_start.strftime("%Y-%m-%d"),
            "test_end": test_end.strftime("%Y-%m-%d"),
            "train_symbol_count": len(train_data),
            "test_symbol_count": len(test_data),
            "train_best": {
                "trial_id": best_train["trial_id"],
                "score": best_train["score"],
                "mean_pnl_pct": best_train["mean_pnl_pct"],
                "median_pnl_pct": best_train["median_pnl_pct"],
                "active_symbols": best_train["active_symbols"],
                "total_symbols": best_train["total_symbols"],
                "total_trades": best_train["total_trades"],
                "win_rate": best_train["win_rate"],
                "profit_factor": best_train["profit_factor"],
            },
            "test": asdict(test_eval),
            "selected_params": best_train["params"],
        }
        folds.append(fold)

        print(
            f"[wf] fold={fold_id} train_score={best_train['score']:.3f} "
            f"test_score={test_eval.score:.3f} "
            f"test_median={test_eval.median_pnl_pct:.2f}% "
            f"test_win_rate={test_eval.win_rate:.2f}%"
        )

    oos_scores = [fold["test"]["score"] for fold in folds]
    oos_mean_pnl = [fold["test"]["mean_pnl_pct"] for fold in folds]
    oos_median_pnl = [fold["test"]["median_pnl_pct"] for fold in folds]
    oos_win_rates = [fold["test"]["win_rate"] for fold in folds]
    oos_profit_factors = [fold["test"]["profit_factor"] for fold in folds]
    oos_trades = [fold["test"]["total_trades"] for fold in folds]

    summary: Dict[str, object] = {
        "folds_total": len(windows),
        "folds_completed": len(folds),
        "folds_skipped": skipped,
        "avg_test_score": float(np.mean(oos_scores)) if oos_scores else 0.0,
        "median_test_score": float(np.median(oos_scores)) if oos_scores else 0.0,
        "avg_test_mean_pnl_pct": float(np.mean(oos_mean_pnl)) if oos_mean_pnl else 0.0,
        "avg_test_median_pnl_pct": float(np.mean(oos_median_pnl)) if oos_median_pnl else 0.0,
        "avg_test_win_rate": float(np.mean(oos_win_rates)) if oos_win_rates else 0.0,
        "avg_test_profit_factor": float(np.mean(oos_profit_factors)) if oos_profit_factors else 0.0,
        "total_test_trades": int(sum(oos_trades)) if oos_trades else 0,
    }

    if folds:
        best_fold = max(folds, key=lambda item: float(item["test"]["score"]))
        summary["best_fold_id"] = int(best_fold["fold_id"])
        summary["best_fold_test_score"] = float(best_fold["test"]["score"])
        summary["best_fold_params"] = best_fold["selected_params"]

    return {
        "mode": "walk_forward",
        "walk_forward": {
            "config": {
                "train_years": train_years,
                "test_months": test_months,
                "step_months": step_months,
                "trials_per_fold": trials,
                "coarse_size": coarse_size,
                "stage2_top": top_n_stage2,
                "min_train_rows": min_train_rows,
                "min_test_rows": min_test_rows,
                "min_symbols": min_symbols,
            },
            "summary": summary,
            "folds": folds,
        },
        "symbol_count": len(data_full),
        "symbols": list(data_full.keys()),
    }


def run_optimizer(
    trials: int,
    seed: int,
    data_full: Dict[str, pd.DataFrame],
    coarse_size: int,
    top_n_stage2: int,
) -> Dict[str, object]:
    rng = random.Random(seed)

    symbols = list(data_full.keys())
    coarse_symbols = symbols[:coarse_size] if len(symbols) > coarse_size else symbols
    data_coarse = {symbol: data_full[symbol] for symbol in coarse_symbols}

    trial_results: List[TrialStats] = []
    for trial_id in range(1, trials + 1):
        params = sample_params(rng)
        trial = evaluate_trial(trial_id=trial_id, params=params, dataset=data_coarse)
        trial_results.append(trial)
        if trial_id % 5 == 0 or trial_id == 1:
            print(
                f"[coarse] trial={trial_id}/{trials} score={trial.score:.3f} "
                f"median_pnl={trial.median_pnl_pct:.2f}% active={trial.active_symbols}/{trial.total_symbols}"
            )

    top_coarse = sorted(trial_results, key=lambda item: item.score, reverse=True)[:top_n_stage2]
    final_results: List[TrialStats] = []
    for rank, trial in enumerate(top_coarse, start=1):
        full_eval = evaluate_trial(trial_id=trial.trial_id, params=trial.params, dataset=data_full)
        final_results.append(full_eval)
        print(
            f"[full] candidate={rank}/{len(top_coarse)} from_trial={trial.trial_id} "
            f"score={full_eval.score:.3f} median_pnl={full_eval.median_pnl_pct:.2f}% "
            f"active={full_eval.active_symbols}/{full_eval.total_symbols}"
        )

    final_sorted = sorted(final_results, key=lambda item: item.score, reverse=True)
    best = final_sorted[0] if final_sorted else None

    method_summary = []
    for item in final_sorted[:10]:
        method_summary.append(
            {
                "trial_id": item.trial_id,
                "score": item.score,
                "mean_pnl_pct": item.mean_pnl_pct,
                "median_pnl_pct": item.median_pnl_pct,
                "active_symbols": item.active_symbols,
                "total_trades": item.total_trades,
                "win_rate": item.win_rate,
                "profit_factor": item.profit_factor,
                "flags": {
                    "mtf": item.params.enable_mtf_filter,
                    "volume": item.params.enable_volume_filter,
                    "adx": item.params.enable_adx_filter,
                    "macdh": item.params.enable_macdh_filter,
                    "protective_exits": item.params.enable_protective_exits,
                },
            }
        )

    return {
        "best": asdict(best) if best else None,
        "top_trials": [asdict(item) for item in final_sorted[:10]],
        "method_summary": method_summary,
        "coarse_trials": len(trial_results),
        "stage2_trials": len(final_results),
        "symbol_count": len(data_full),
        "symbols": list(data_full.keys()),
    }


def save_report(report: Dict[str, object], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"bist100_optimization_{timestamp}.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Headless BIST100 10-year strategy optimizer")
    parser.add_argument("--years", type=int, default=10, help="Lookback years")
    parser.add_argument("--trials", type=int, default=50, help="Random search trials (coarse stage)")
    parser.add_argument("--coarse-size", type=int, default=35, help="Coarse stage symbol count")
    parser.add_argument("--stage2-top", type=int, default=12, help="Top coarse candidates to re-evaluate on full universe")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--walk-forward", action="store_true", help="Enable walk-forward optimization mode")
    parser.add_argument("--wf-train-years", type=int, default=5, help="Walk-forward train window in years")
    parser.add_argument("--wf-test-months", type=int, default=12, help="Walk-forward out-of-sample test window in months")
    parser.add_argument("--wf-step-months", type=int, default=6, help="Walk-forward step size in months")
    parser.add_argument("--wf-min-train-rows", type=int, default=500, help="Minimum bars per symbol in train fold")
    parser.add_argument("--wf-min-test-rows", type=int, default=120, help="Minimum bars per symbol in test fold")
    parser.add_argument("--wf-min-symbols", type=int, default=30, help="Minimum symbols to evaluate a fold")
    parser.add_argument("--output-dir", type=str, default="reports", help="Report directory")
    args = parser.parse_args()

    if args.trials <= 0 or args.coarse_size <= 0 or args.stage2_top <= 0:
        raise ValueError("trials, coarse-size and stage2-top must be positive integers.")
    if args.years <= 0:
        raise ValueError("years must be a positive integer.")
    if args.walk_forward:
        if args.wf_train_years <= 0 or args.wf_test_months <= 0 or args.wf_step_months <= 0:
            raise ValueError("wf-train-years, wf-test-months and wf-step-months must be positive integers.")
        if args.wf_min_train_rows <= 0 or args.wf_min_test_rows <= 0 or args.wf_min_symbols <= 0:
            raise ValueError("wf minimum row/symbol constraints must be positive integers.")

    end = datetime.now()
    start = end - timedelta(days=(args.years * 365) + 30)
    print(f"fetching BIST100 symbols and data for {start.date()} -> {end.date()}")

    symbols_raw = fetch_bist100_symbols_from_cnbce()
    print(f"source symbols: {len(symbols_raw)}")
    dataset = download_daily_data(symbols_raw, start, end)
    print(f"symbols with sufficient history: {len(dataset)}")

    if not dataset:
        raise RuntimeError("No symbol data downloaded.")

    dataset_top100 = pick_top_100_by_coverage(dataset)
    print(f"universe size used for optimization: {len(dataset_top100)}")

    if args.walk_forward:
        report = run_walk_forward_optimizer(
            data_full=dataset_top100,
            global_start=start,
            global_end=end,
            train_years=args.wf_train_years,
            test_months=args.wf_test_months,
            step_months=args.wf_step_months,
            trials=args.trials,
            seed=args.seed,
            coarse_size=args.coarse_size,
            top_n_stage2=args.stage2_top,
            min_train_rows=args.wf_min_train_rows,
            min_test_rows=args.wf_min_test_rows,
            min_symbols=args.wf_min_symbols,
        )
    else:
        report = run_optimizer(
            trials=args.trials,
            seed=args.seed,
            data_full=dataset_top100,
            coarse_size=args.coarse_size,
            top_n_stage2=args.stage2_top,
        )

    report["meta"] = {
        "mode": "walk_forward" if args.walk_forward else "single_pass",
        "years": args.years,
        "trials": args.trials,
        "coarse_size": args.coarse_size,
        "stage2_top": args.stage2_top,
        "seed": args.seed,
        "start_date": start.strftime("%Y-%m-%d"),
        "end_date": end.strftime("%Y-%m-%d"),
        "data_source": "CNBCE BIST100 page + Yahoo Finance daily OHLCV",
        "note": "Source list may contain >100 symbols; optimizer keeps top 100 by available history coverage.",
    }
    if args.walk_forward:
        report["meta"]["walk_forward"] = {
            "train_years": args.wf_train_years,
            "test_months": args.wf_test_months,
            "step_months": args.wf_step_months,
            "min_train_rows": args.wf_min_train_rows,
            "min_test_rows": args.wf_min_test_rows,
            "min_symbols": args.wf_min_symbols,
        }

    report_path = save_report(report, Path(args.output_dir))
    if args.walk_forward:
        wf = report.get("walk_forward", {})
        summary = wf.get("summary", {}) if isinstance(wf, dict) else {}
        print("\nWALK-FORWARD SUMMARY")
        print(
            f"folds={summary.get('folds_completed', 0)}/{summary.get('folds_total', 0)} "
            f"avg_test_score={summary.get('avg_test_score', 0.0):.3f} "
            f"avg_test_median_pnl={summary.get('avg_test_median_pnl_pct', 0.0):.2f}% "
            f"avg_test_win_rate={summary.get('avg_test_win_rate', 0.0):.2f}%"
        )
        best_fold_params = summary.get("best_fold_params")
        if best_fold_params:
            print("best fold params:")
            print(json.dumps(best_fold_params, indent=2, ensure_ascii=False))
    else:
        best = report.get("best")
        if best:
            print("\nBEST RESULT")
            print(
                f"score={best['score']:.3f} mean_pnl={best['mean_pnl_pct']:.2f}% "
                f"median_pnl={best['median_pnl_pct']:.2f}% win_rate={best['win_rate']:.2f}% "
                f"active={best['active_symbols']}/{best['total_symbols']}"
            )
            print("best params:")
            print(json.dumps(best["params"], indent=2, ensure_ascii=False))

    print(f"\nreport saved: {report_path}")


if __name__ == "__main__":
    main()
