from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

from bist_bot.core.interfaces import IBroker
from bist_bot.core.config import PAPER_TRADING_STARTING_CASH, ORDER_TYPE, POSITION_SIZING_MODE


class PaperBroker(IBroker):
    """Mock implementation of IBroker for simulated paper trading."""

    def __init__(
        self,
        starting_cash: float | None = None,
        state_path: str | None = None,
        auto_load: bool = False,
    ):
        self.initial_cash = float(starting_cash if starting_cash is not None else PAPER_TRADING_STARTING_CASH)
        self.cash = self.initial_cash
        self.positions: Dict[str, float] = defaultdict(float)  # symbol -> quantity
        self.position_costs: Dict[str, float] = defaultdict(float)  # symbol -> total cost basis for open quantity
        self.trade_history: List[Dict[str, Any]] = []
        self.performance_snapshots: List[Dict[str, Any]] = []
        self.started_at = datetime.now()
        self.state_path = Path(state_path) if state_path else None

        if auto_load and self.state_path is not None and self.state_path.exists():
            self.load_state()
        print(f"Paper trading started with initial cash: {self.initial_cash:.2f} TRY")

    def place_order(self, symbol: str, order_type: str, quantity: float = None, price: float = None) -> Dict[str, Any]:
        """
        Simulates placing a market order.
        In paper trading, we assume market orders are filled immediately at the given price.
        Quantity can be None for 'full_allocation' sizing.
        """
        if order_type.lower() not in ["market"]:
            raise ValueError("Only 'market' orders are supported in paper trading for now.")
        
        if price is None:
            raise ValueError("Price must be provided for market orders in paper trading simulation.")

        if POSITION_SIZING_MODE == "full_allocation":
            if self.cash <= 0 and quantity is None:
                return {"status": "REJECTED", "message": "No cash available for full allocation buy.", "symbol": symbol}
            
            if quantity is None: # This means it's a BUY with full_allocation
                quantity_to_buy = self.cash / price
                # Simple approach: assume we can buy fractional shares
                quantity = quantity_to_buy
            
            # If it's a SELL with full_allocation, quantity will be taken from self.positions
            if quantity is None and self.positions[symbol] > 0: # SELL with full_allocation
                quantity = self.positions[symbol]
            elif quantity is None and self.positions[symbol] == 0: # SELL with no position
                return {"status": "REJECTED", "message": f"No {symbol} to sell.", "symbol": symbol}

        if quantity is None: # If not full_allocation and quantity is still None, it's an error
            raise ValueError("Quantity must be provided if not using 'full_allocation' sizing.")

        cost = quantity * price
        transaction_type = "BUY" if quantity > 0 else "SELL"
        realized_pnl: float | None = None

        if transaction_type == "BUY":
            if self.cash >= cost:
                previous_qty = float(self.positions[symbol])
                self.cash -= cost
                self.positions[symbol] = previous_qty + quantity
                self.position_costs[symbol] = float(self.position_costs[symbol]) + (quantity * price)
                status = "FILLED"
                message = f"Bought {quantity:.2f} of {symbol} at {price} for {cost:.2f} TRY."
            else:
                status = "REJECTED"
                message = f"Insufficient cash to buy {quantity:.2f} of {symbol}. Needed {cost:.2f}, have {self.cash:.2f}."
        else: # SELL
            abs_quantity = abs(quantity) # Convert negative quantity to positive for calculations
            if self.positions[symbol] >= abs_quantity:
                previous_qty = float(self.positions[symbol])
                previous_cost = float(self.position_costs[symbol])
                average_cost = (previous_cost / previous_qty) if previous_qty > 0 else 0.0
                self.cash += abs_quantity * price
                self.positions[symbol] = previous_qty - abs_quantity
                remaining_cost = max(previous_cost - (average_cost * abs_quantity), 0.0)
                self.position_costs[symbol] = remaining_cost
                realized_pnl = (price - average_cost) * abs_quantity
                status = "FILLED"
                message = f"Sold {abs_quantity:.2f} of {symbol} at {price} for {abs_quantity * price:.2f} TRY."
                if self.positions[symbol] < 1e-9: # Handle floating point inaccuracies for zero
                    self.positions[symbol] = 0.0
                    self.position_costs[symbol] = 0.0
            else:
                status = "REJECTED"
                message = f"Insufficient {symbol} to sell. Have {self.positions[symbol]:.2f}, tried to sell {abs_quantity:.2f}."

        trade = {
            "timestamp": datetime.now(),
            "symbol": symbol,
            "order_type": order_type,
            "transaction_type": transaction_type,
            "quantity": quantity,
            "price": price,
            "cost": cost if transaction_type == "BUY" else -cost,
            "realized_pnl": realized_pnl,
            "status": status,
            "message": message,
            "cash_after": self.cash,
            "positions_after": dict(self.positions) # Store a copy of current positions
        }
        self.trade_history.append(trade)
        print(f"[TRADE] {message} Current Cash: {self.cash:.2f}")
        return {"status": status, "message": message, "trade": trade}

    def get_account_balance(self, current_prices: Dict[str, float] | None = None) -> Dict[str, float]:
        """
        Retrieves the current account balance and estimated total portfolio value.
        For simplicity, positions are valued at their last known price (if available), or 0.
        """
        total_asset_value = 0.0
        for symbol, quantity in self.positions.items():
            if quantity <= 0:
                continue
            last_trade_price = 0.0
            if current_prices is not None and symbol in current_prices:
                last_trade_price = float(current_prices[symbol])
            else:
                for trade in reversed(self.trade_history):
                    if trade.get("symbol") == symbol and trade.get("price") is not None:
                        last_trade_price = float(trade["price"])
                        break
            total_asset_value += quantity * last_trade_price

        total_value = self.cash + total_asset_value
        unrealized_pnl = total_value - self.initial_cash
        return {
            "cash": self.cash,
            "asset_value": total_asset_value,
            "total_value": total_value,
            "unrealized_pnl": unrealized_pnl,
        }

    def get_open_positions(self, current_prices: Dict[str, float] | None = None) -> List[Dict[str, Any]]:
        """
        Retrieves a list of currently open positions.
        """
        open_positions = []
        for symbol, quantity in self.positions.items():
            if quantity > 0:
                cost_basis = float(self.position_costs[symbol])
                avg_price = cost_basis / quantity if quantity > 0 else 0.0
                if current_prices is not None and symbol in current_prices:
                    mark_price = float(current_prices[symbol])
                else:
                    mark_price = self._lookup_last_trade_price(symbol)
                market_value = quantity * mark_price
                unrealized_pnl = market_value - cost_basis
                unrealized_pnl_pct = (unrealized_pnl / cost_basis * 100.0) if cost_basis > 0 else 0.0
                open_positions.append(
                    {
                        "symbol": symbol,
                        "quantity": quantity,
                        "avg_price": avg_price,
                        "cost_basis": cost_basis,
                        "last_price": mark_price,
                        "market_value": market_value,
                        "unrealized_pnl": unrealized_pnl,
                        "unrealized_pnl_pct": unrealized_pnl_pct,
                    }
                )
        open_positions.sort(key=lambda item: str(item.get("symbol", "")))
        return open_positions
    
    def get_asset_balance(self, symbol: str) -> float:
        """
        Retrieves the quantity of a specific asset held.
        """
        return self.positions.get(symbol, 0.0)

    def reset(self, starting_cash: float | None = None) -> None:
        self.initial_cash = float(starting_cash if starting_cash is not None else self.initial_cash)
        self.cash = self.initial_cash
        self.positions = defaultdict(float)
        self.position_costs = defaultdict(float)
        self.trade_history = []
        self.performance_snapshots = []
        self.started_at = datetime.now()
        self.save_state()

    def record_snapshot(self, balance: Dict[str, float]) -> None:
        snapshot = {
            "timestamp": datetime.now().isoformat(),
            "cash": float(balance.get("cash", self.cash)),
            "asset_value": float(balance.get("asset_value", 0.0)),
            "total_value": float(balance.get("total_value", self.cash)),
            "unrealized_pnl": float(balance.get("unrealized_pnl", 0.0)),
            "pnl_pct": (
                float(balance.get("total_value", self.cash) - self.initial_cash) / self.initial_cash * 100.0
                if self.initial_cash > 0
                else 0.0
            ),
        }
        self.performance_snapshots.append(snapshot)

    def save_state(self) -> None:
        if self.state_path is None:
            return
        self.state_path.parent.mkdir(parents=True, exist_ok=True)

        serialized_trades: List[Dict[str, Any]] = []
        for trade in self.trade_history:
            item = dict(trade)
            ts = item.get("timestamp")
            if isinstance(ts, datetime):
                item["timestamp"] = ts.isoformat()
            serialized_trades.append(item)

        payload = {
            "initial_cash": self.initial_cash,
            "cash": self.cash,
            "positions": dict(self.positions),
            "position_costs": dict(self.position_costs),
            "trade_history": serialized_trades,
            "performance_snapshots": self.performance_snapshots[-5000:],
            "started_at": self.started_at.isoformat(),
            "updated_at": datetime.now().isoformat(),
        }
        self.state_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    def load_state(self) -> None:
        if self.state_path is None or not self.state_path.exists():
            return
        payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        self.initial_cash = float(payload.get("initial_cash", self.initial_cash))
        self.cash = float(payload.get("cash", self.initial_cash))

        self.positions = defaultdict(float)
        for symbol, qty in dict(payload.get("positions", {})).items():
            self.positions[str(symbol)] = float(qty)

        self.position_costs = defaultdict(float)
        for symbol, total_cost in dict(payload.get("position_costs", {})).items():
            self.position_costs[str(symbol)] = float(total_cost)

        self.trade_history = []
        for trade in list(payload.get("trade_history", [])):
            item = dict(trade)
            ts_raw = item.get("timestamp")
            if isinstance(ts_raw, str):
                try:
                    item["timestamp"] = datetime.fromisoformat(ts_raw)
                except ValueError:
                    item["timestamp"] = ts_raw
            self.trade_history.append(item)

        self.performance_snapshots = [
            {
                "timestamp": str(s.get("timestamp", "")),
                "cash": float(s.get("cash", 0.0)),
                "asset_value": float(s.get("asset_value", 0.0)),
                "total_value": float(s.get("total_value", 0.0)),
                "unrealized_pnl": float(s.get("unrealized_pnl", 0.0)),
                "pnl_pct": float(s.get("pnl_pct", 0.0)),
            }
            for s in list(payload.get("performance_snapshots", []))
        ]
        started_at_raw = payload.get("started_at")
        if isinstance(started_at_raw, str):
            try:
                self.started_at = datetime.fromisoformat(started_at_raw)
            except ValueError:
                self.started_at = datetime.now()

        if not any(float(v) > 0 for v in self.position_costs.values()):
            self._rebuild_position_costs_from_history()

        for symbol, qty in list(self.positions.items()):
            if float(qty) <= 0:
                self.positions[symbol] = 0.0
                self.position_costs[symbol] = 0.0
            elif symbol not in self.position_costs:
                fallback_avg = self._lookup_last_buy_price(symbol)
                self.position_costs[symbol] = float(qty) * fallback_avg

    def _lookup_last_trade_price(self, symbol: str) -> float:
        for trade in reversed(self.trade_history):
            if trade.get("symbol") != symbol:
                continue
            if str(trade.get("status", "")).upper() != "FILLED":
                continue
            price = trade.get("price")
            if price is None:
                continue
            return float(price)
        return 0.0

    def _lookup_last_buy_price(self, symbol: str) -> float:
        for trade in reversed(self.trade_history):
            if trade.get("symbol") != symbol:
                continue
            if str(trade.get("status", "")).upper() != "FILLED":
                continue
            if str(trade.get("transaction_type", "")).upper() != "BUY":
                continue
            price = trade.get("price")
            if price is None:
                continue
            return float(price)
        return self._lookup_last_trade_price(symbol)

    def _rebuild_position_costs_from_history(self) -> None:
        rebuilt_qty: Dict[str, float] = defaultdict(float)
        rebuilt_costs: Dict[str, float] = defaultdict(float)
        for trade in self.trade_history:
            if str(trade.get("status", "")).upper() != "FILLED":
                continue

            symbol = str(trade.get("symbol", ""))
            quantity = float(trade.get("quantity", 0.0))
            price = float(trade.get("price", 0.0))
            side = str(trade.get("transaction_type", "")).upper()
            if not symbol or price <= 0:
                continue

            if side == "BUY" and quantity > 0:
                rebuilt_qty[symbol] += quantity
                rebuilt_costs[symbol] += quantity * price
            elif side == "SELL":
                sell_qty = abs(quantity)
                current_qty = rebuilt_qty[symbol]
                if current_qty <= 0 or sell_qty <= 0:
                    continue
                avg_cost = rebuilt_costs[symbol] / current_qty if current_qty > 0 else 0.0
                applied_qty = min(sell_qty, current_qty)
                rebuilt_qty[symbol] = current_qty - applied_qty
                rebuilt_costs[symbol] = max(rebuilt_costs[symbol] - (avg_cost * applied_qty), 0.0)

        self.position_costs = defaultdict(float)
        for symbol, qty in self.positions.items():
            if qty <= 0:
                self.position_costs[symbol] = 0.0
                continue
            rebuilt_symbol_qty = float(rebuilt_qty.get(symbol, 0.0))
            if rebuilt_symbol_qty > 0:
                avg_cost = float(rebuilt_costs.get(symbol, 0.0)) / rebuilt_symbol_qty
            else:
                avg_cost = self._lookup_last_buy_price(symbol)
            self.position_costs[symbol] = max(float(qty) * avg_cost, 0.0)
