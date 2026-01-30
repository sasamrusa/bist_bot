from typing import Dict, Any, List
from collections import defaultdict
from datetime import datetime

from bist_bot.core.interfaces import IBroker
from bist_bot.core.config import PAPER_TRADING_STARTING_CASH, ORDER_TYPE, POSITION_SIZING_MODE


class PaperBroker(IBroker):
    """Mock implementation of IBroker for simulated paper trading."""

    def __init__(self):
        self.cash = PAPER_TRADING_STARTING_CASH
        self.positions: Dict[str, float] = defaultdict(float)  # symbol -> quantity
        self.trade_history: List[Dict[str, Any]] = []
        print(f"Paper trading started with initial cash: {self.cash} TRY")

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

        if transaction_type == "BUY":
            if self.cash >= cost:
                self.cash -= cost
                self.positions[symbol] += quantity
                status = "FILLED"
                message = f"Bought {quantity:.2f} of {symbol} at {price} for {cost:.2f} TRY."
            else:
                status = "REJECTED"
                message = f"Insufficient cash to buy {quantity:.2f} of {symbol}. Needed {cost:.2f}, have {self.cash:.2f}."
        else: # SELL
            abs_quantity = abs(quantity) # Convert negative quantity to positive for calculations
            if self.positions[symbol] >= abs_quantity:
                self.cash += abs_quantity * price
                self.positions[symbol] -= abs_quantity
                status = "FILLED"
                message = f"Sold {abs_quantity:.2f} of {symbol} at {price} for {abs_quantity * price:.2f} TRY."
                if self.positions[symbol] < 1e-9: # Handle floating point inaccuracies for zero
                    self.positions[symbol] = 0.0
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
            "cost": cost if transaction_type == "BUY" else -cost, # Store cost as negative for sells
            "status": status,
            "message": message,
            "cash_after": self.cash,
            "positions_after": dict(self.positions) # Store a copy of current positions
        }
        self.trade_history.append(trade)
        print(f"[TRADE] {message} Current Cash: {self.cash:.2f}")
        return {"status": status, "message": message, "trade": trade}

    def get_account_balance(self) -> Dict[str, float]:
        """
        Retrieves the current account balance and estimated total portfolio value.
        For simplicity, positions are valued at their last known price (if available), or 0.
        """
        total_asset_value = 0.0
        for symbol, quantity in self.positions.items():
            if quantity <= 0:
                continue
            last_trade_price = 0.0
            for trade in reversed(self.trade_history):
                if trade.get("symbol") == symbol and trade.get("price") is not None:
                    last_trade_price = float(trade["price"])
                    break
            total_asset_value += quantity * last_trade_price

        total_value = self.cash + total_asset_value
        unrealized_pnl = total_value - PAPER_TRADING_STARTING_CASH
        return {
            "cash": self.cash,
            "asset_value": total_asset_value,
            "total_value": total_value,
            "unrealized_pnl": unrealized_pnl,
        }

    def get_open_positions(self) -> List[Dict[str, Any]]:
        """
        Retrieves a list of currently open positions.
        """
        open_positions = []
        for symbol, quantity in self.positions.items():
            if quantity > 0:
                # In a real scenario, you'd add average price, current market value, PnL, etc.
                open_positions.append({"symbol": symbol, "quantity": quantity, "avg_price": 0.0}) # TODO: track avg_price
        return open_positions
    
    def get_asset_balance(self, symbol: str) -> float:
        """
        Retrieves the quantity of a specific asset held.
        """
        return self.positions.get(symbol, 0.0)
