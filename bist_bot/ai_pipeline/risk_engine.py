from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RiskConfig:
    risk_per_trade_pct: float = 0.01
    atr_stop_multiplier: float = 1.5
    max_position_pct: float = 0.25
    min_qty: float = 0.0


def calculate_order_quantity(equity: float, price: float, atr: float, config: RiskConfig) -> float:
    if equity <= 0 or price <= 0:
        return 0.0

    stop_distance = max(atr * config.atr_stop_multiplier, price * 0.005)
    risk_budget = max(equity * config.risk_per_trade_pct, 0.0)
    qty_by_risk = (risk_budget / stop_distance) if stop_distance > 0 else 0.0
    qty_by_cap = (equity * config.max_position_pct) / price

    quantity = max(min(qty_by_risk, qty_by_cap), 0.0)
    if quantity < config.min_qty:
        return 0.0
    return float(quantity)

