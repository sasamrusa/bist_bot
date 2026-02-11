from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


@dataclass
class PolicyConfig:
    buy_threshold: float = 0.58
    sell_threshold: float = 0.42
    fee_bps: float = 8.0
    slippage_bps: float = 5.0


def probability_to_signal(prob_up: float, has_position: bool, config: PolicyConfig) -> str:
    if has_position:
        return "SELL" if prob_up <= config.sell_threshold else "HOLD"
    return "BUY" if prob_up >= config.buy_threshold else "HOLD"


def evaluate_probability_policy(
    dataset: pd.DataFrame,
    probabilities: np.ndarray,
    config: PolicyConfig,
) -> Dict[str, float]:
    """
    Simulates BUY/SELL policy per symbol based on probabilities and close prices.
    """
    if dataset.empty:
        return {
            "avg_symbol_return_pct": 0.0,
            "median_symbol_return_pct": 0.0,
            "trades": 0,
            "win_rate": 0.0,
            "avg_trade_return_pct": 0.0,
        }

    test_df = dataset.copy()
    test_df["prob_up"] = np.asarray(probabilities, dtype=float)
    fee_rate = config.fee_bps / 10_000.0
    slip_rate = config.slippage_bps / 10_000.0
    side_cost = fee_rate + slip_rate

    symbol_returns = []
    trade_returns = []

    for _symbol, frame in test_df.groupby("symbol", sort=False):
        group = frame.sort_values("ts")
        equity = 1.0
        in_position = False
        entry = 0.0

        for _, row in group.iterrows():
            price = float(row["close"])
            prob = float(row["prob_up"])
            if not in_position and prob >= config.buy_threshold:
                entry = price * (1.0 + side_cost)
                in_position = True
                continue

            if in_position and prob <= config.sell_threshold:
                exit_price = price * (1.0 - side_cost)
                trade_ret = (exit_price / entry) - 1.0
                equity *= (1.0 + trade_ret)
                trade_returns.append(trade_ret)
                in_position = False

        if in_position:
            last_price = float(group.iloc[-1]["close"])
            exit_price = last_price * (1.0 - side_cost)
            trade_ret = (exit_price / entry) - 1.0
            equity *= (1.0 + trade_ret)
            trade_returns.append(trade_ret)

        symbol_returns.append(equity - 1.0)

    wins = sum(1 for value in trade_returns if value > 0.0)
    trades = len(trade_returns)
    return {
        "avg_symbol_return_pct": float(np.mean(symbol_returns) * 100.0) if symbol_returns else 0.0,
        "median_symbol_return_pct": float(np.median(symbol_returns) * 100.0) if symbol_returns else 0.0,
        "trades": int(trades),
        "win_rate": float((wins / trades) * 100.0) if trades > 0 else 0.0,
        "avg_trade_return_pct": float(np.mean(trade_returns) * 100.0) if trade_returns else 0.0,
    }

