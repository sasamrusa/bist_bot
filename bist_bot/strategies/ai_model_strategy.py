from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from bist_bot.ai_pipeline.feature_store import build_feature_frame
from bist_bot.ai_pipeline.registry import load_latest_model_bundle, load_model_bundle
from bist_bot.ai_pipeline.signal_policy import PolicyConfig, probability_to_signal
from bist_bot.strategies.base_strategy import BaseStrategy
from bist_bot.utils.logger import setup_logger


LOGGER = setup_logger("bist_bot.ai_strategy")


class AIModelStrategy(BaseStrategy):
    """Signal strategy backed by the latest trained AI model bundle."""

    def __init__(
        self,
        model_dir: str = "models/ai_registry",
        model_path: str = "",
        buy_threshold: float | None = None,
        sell_threshold: float | None = None,
    ):
        super().__init__("AI_Model")

        if model_path:
            bundle = load_model_bundle(Path(model_path))
        else:
            bundle = self._load_latest_bundle(model_dir)

        self.model = bundle["model"]
        self.feature_columns: List[str] = list(bundle["feature_columns"])
        self.symbol_to_id = dict(bundle.get("symbol_to_id", {}))
        config = dict(bundle.get("config", {}))

        buy = float(buy_threshold) if buy_threshold is not None else float(config.get("buy_threshold", 0.58))
        sell = float(sell_threshold) if sell_threshold is not None else float(config.get("sell_threshold", 0.42))
        self.policy = PolicyConfig(buy_threshold=buy, sell_threshold=sell)
        self.model_backend = str(bundle.get("model_backend", "unknown"))

        LOGGER.info(
            "ai_strategy_loaded backend=%s buy_th=%.3f sell_th=%.3f features=%s",
            self.model_backend,
            self.policy.buy_threshold,
            self.policy.sell_threshold,
            len(self.feature_columns),
        )

    def _load_latest_bundle(self, model_dir: str) -> dict:
        candidates = [
            Path(model_dir),
            Path(__file__).resolve().parents[1] / "models" / "ai_registry",
            Path.cwd() / "models" / "ai_registry",
            Path.cwd() / "bist_bot" / "models" / "ai_registry",
            Path("models/ai_registry"),
        ]
        seen = set()
        errors: List[str] = []
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            if not (candidate / "latest.json").exists():
                continue
            try:
                return load_latest_model_bundle(candidate)
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("ai_strategy_model_load_failed path=%s err=%s", candidate, exc)
                errors.append(f"{candidate}: {exc}")
        detail = " | ".join(errors) if errors else "latest.json not found in candidates"
        raise FileNotFoundError(f"No AI model found. Expected latest.json under models/ai_registry. Details: {detail}")

    def _predict_probabilities(self, rows: np.ndarray) -> np.ndarray:
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(rows)
            return np.asarray(proba, dtype=float)[:, 1]
        pred = self.model.predict(rows)
        pred_arr = np.asarray(pred, dtype=float).reshape(-1)
        return pred_arr

    def predict_probability_series(self, historical_data: pd.DataFrame, symbol: str) -> pd.Series:
        if historical_data.empty:
            return pd.Series(dtype="float64")

        features = build_feature_frame(historical_data, symbol=symbol)
        if features.empty:
            return pd.Series(np.nan, index=historical_data.index, dtype="float64")

        features = features.copy()
        features["symbol_id"] = int(self.symbol_to_id.get(symbol, -1))
        for col in self.feature_columns:
            if col not in features.columns:
                features[col] = np.nan

        valid_mask = ~features[self.feature_columns].isna().any(axis=1)
        out = pd.Series(np.nan, index=features.index, dtype="float64")
        if not bool(valid_mask.any()):
            return out.reindex(historical_data.index)

        row_data = features.loc[valid_mask, self.feature_columns].to_numpy(dtype=float)
        probabilities = self._predict_probabilities(row_data)
        valid_indices = list(np.flatnonzero(valid_mask.to_numpy()))
        for row_idx, prob_up in zip(valid_indices, probabilities):
            out.iat[row_idx] = float(prob_up)

        return out.reindex(historical_data.index)

    def generate_signals_batch(
        self,
        historical_data: pd.DataFrame,
        symbol: str,
        interval: str | None = None,
    ) -> pd.Series:
        if historical_data.empty:
            return pd.Series(dtype="object")

        probabilities_series = self.predict_probability_series(historical_data, symbol=symbol)
        if probabilities_series.empty:
            return pd.Series("HOLD", index=historical_data.index, dtype="object")

        signals = pd.Series("HOLD", index=probabilities_series.index, dtype="object")
        has_position = False
        for row_idx, prob_up in enumerate(probabilities_series.to_numpy(dtype=float)):
            if np.isnan(prob_up):
                continue
            signal = probability_to_signal(float(prob_up), has_position=has_position, config=self.policy)
            signals.iat[row_idx] = signal
            if signal == "BUY":
                has_position = True
            elif signal == "SELL":
                has_position = False

        return signals.reindex(historical_data.index, fill_value="HOLD").astype("object")

    def generate_signal(self, historical_data: pd.DataFrame, current_data: pd.Series, symbol: str) -> str:
        if historical_data.empty:
            return "HOLD"
        signals = self.generate_signals_batch(historical_data, symbol=symbol, interval=None)
        if signals.empty:
            return "HOLD"
        value = signals.iloc[-1]
        return str(value) if isinstance(value, str) else "HOLD"
