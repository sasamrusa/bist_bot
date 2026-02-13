from __future__ import annotations

from typing import Literal

import pandas as pd

from bist_bot.strategies.ai_model_strategy import AIModelStrategy
from bist_bot.strategies.base_strategy import BaseStrategy
from bist_bot.strategies.rsi_macd import RsiMacdStrategy


HybridMode = Literal["consensus", "ai_lead", "classic_lead", "weighted", "probability_gate"]


class HybridAiRsiStrategy(BaseStrategy):
    """Combines AI model and RSI/MACD strategy decisions."""

    def __init__(
        self,
        mode: HybridMode = "ai_lead",
        model_dir: str = "models/ai_registry",
        model_path: str = "",
        buy_threshold: float | None = None,
        sell_threshold: float | None = None,
        probability_margin: float = 0.06,
    ):
        super().__init__(f"HYBRID_AI_RSI_{mode.upper()}")
        if mode not in {"consensus", "ai_lead", "classic_lead", "weighted", "probability_gate"}:
            raise ValueError(f"Unsupported hybrid mode: {mode}")

        self.mode: HybridMode = mode
        self.probability_margin = max(0.0, float(probability_margin))
        self.ai_strategy = AIModelStrategy(
            model_dir=model_dir,
            model_path=model_path,
            buy_threshold=buy_threshold,
            sell_threshold=sell_threshold,
        )
        self.classic_strategy = RsiMacdStrategy()

    @staticmethod
    def _normalize(signal: object) -> str:
        text = str(signal).upper().strip()
        if text in {"BUY", "SELL"}:
            return text
        return "HOLD"

    def _combine(self, ai_signal: str, classic_signal: str, ai_prob: float | None, has_position: bool) -> str:
        ai = self._normalize(ai_signal)
        classic = self._normalize(classic_signal)

        if self.mode == "consensus":
            if ai == "BUY" and classic == "BUY" and not has_position:
                return "BUY"
            if ai == "SELL" and classic == "SELL" and has_position:
                return "SELL"
            return "HOLD"

        if self.mode == "ai_lead":
            if ai == "BUY" and classic != "SELL" and not has_position:
                return "BUY"
            if ai == "SELL" and classic != "BUY" and has_position:
                return "SELL"
            return "HOLD"

        if self.mode == "classic_lead":
            if classic == "BUY" and ai != "SELL" and not has_position:
                return "BUY"
            if classic == "SELL" and ai != "BUY" and has_position:
                return "SELL"
            return "HOLD"

        if self.mode == "probability_gate":
            buy_th = float(self.ai_strategy.policy.buy_threshold)
            sell_th = float(self.ai_strategy.policy.sell_threshold)
            prob = float(ai_prob) if ai_prob is not None else float("nan")

            classic_buy = classic == "BUY"
            classic_sell = classic == "SELL"
            ai_confident_buy = not pd.isna(prob) and prob >= (buy_th + self.probability_margin)
            ai_confident_sell = not pd.isna(prob) and prob <= (sell_th - self.probability_margin)
            ai_soft_buy = not pd.isna(prob) and prob >= buy_th
            ai_soft_sell = not pd.isna(prob) and prob <= sell_th

            if not has_position:
                if classic_buy and ai_soft_buy:
                    return "BUY"
                if ai_confident_buy and classic != "SELL":
                    return "BUY"
            else:
                if classic_sell and ai_soft_sell:
                    return "SELL"
                if ai_confident_sell and classic != "BUY":
                    return "SELL"
            return "HOLD"

        # weighted mode
        buy_score = (2 if ai == "BUY" else 0) + (1 if classic == "BUY" else 0)
        sell_score = (2 if ai == "SELL" else 0) + (1 if classic == "SELL" else 0)
        if buy_score >= 2 and buy_score > sell_score and not has_position:
            return "BUY"
        if sell_score >= 2 and sell_score > buy_score and has_position:
            return "SELL"
        return "HOLD"

    def generate_signals_batch(
        self,
        historical_data: pd.DataFrame,
        symbol: str,
        interval: str | None = None,
    ) -> pd.Series:
        if historical_data.empty:
            return pd.Series(dtype="object")

        ai_series = self.ai_strategy.generate_signals_batch(historical_data, symbol=symbol, interval=interval)
        classic_series = self.classic_strategy.generate_signals_batch(historical_data, symbol=symbol, interval=interval)
        if classic_series is None:
            classic_series = pd.Series("HOLD", index=historical_data.index, dtype="object")
        ai_prob_series = self.ai_strategy.predict_probability_series(historical_data, symbol=symbol)
        ai_prob_series = ai_prob_series.reindex(historical_data.index)

        ai_series = ai_series.reindex(historical_data.index, fill_value="HOLD")
        classic_series = classic_series.reindex(historical_data.index, fill_value="HOLD")

        out = pd.Series("HOLD", index=historical_data.index, dtype="object")
        has_position = False
        for idx, ai_signal, classic_signal, ai_prob in zip(
            out.index,
            ai_series.to_numpy(dtype=object),
            classic_series.to_numpy(dtype=object),
            ai_prob_series.to_numpy(dtype=float),
        ):
            prob_value = None if pd.isna(ai_prob) else float(ai_prob)
            signal = self._combine(str(ai_signal), str(classic_signal), prob_value, has_position)
            out.loc[idx] = signal
            if signal == "BUY":
                has_position = True
            elif signal == "SELL":
                has_position = False
        return out

    def generate_signal(self, historical_data: pd.DataFrame, current_data: pd.Series, symbol: str) -> str:
        if historical_data.empty:
            return "HOLD"
        signals = self.generate_signals_batch(historical_data, symbol=symbol, interval=None)
        if signals.empty:
            return "HOLD"
        value = signals.iloc[-1]
        return str(value) if isinstance(value, str) else "HOLD"
