from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


DEFAULT_STRATEGY_TUNING: Dict[str, Any] = {
    "ai": {
        "buy_threshold": 0.54,
        "sell_threshold": 0.50,
    },
    "hybrid": {
        "mode": "probability_gate",
        "buy_threshold": 0.56,
        "sell_threshold": 0.48,
        "probability_margin": 0.04,
    },
}


def _candidate_paths(path: str = "") -> list[Path]:
    candidates: list[Path] = []
    if path:
        candidates.append(Path(path))
    candidates.extend(
        [
            Path("models/ai_registry/strategy_tuning_latest.json"),
            Path(__file__).resolve().parents[1] / "models" / "ai_registry" / "strategy_tuning_latest.json",
            Path.cwd() / "models" / "ai_registry" / "strategy_tuning_latest.json",
            Path.cwd() / "bist_bot" / "models" / "ai_registry" / "strategy_tuning_latest.json",
        ]
    )

    unique: list[Path] = []
    seen = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(candidate)
    return unique


def load_strategy_tuning(path: str = "") -> Dict[str, Any]:
    valid_modes = {"consensus", "ai_lead", "classic_lead", "weighted", "probability_gate"}
    fallback = json.loads(json.dumps(DEFAULT_STRATEGY_TUNING))

    for candidate in _candidate_paths(path):
        if not candidate.exists():
            continue
        try:
            raw = json.loads(candidate.read_text(encoding="utf-8"))
            ai_raw = dict(raw.get("ai", {}))
            hybrid_raw = dict(raw.get("hybrid", {}))

            ai_buy = float(ai_raw.get("buy_threshold", fallback["ai"]["buy_threshold"]))
            ai_sell = float(ai_raw.get("sell_threshold", fallback["ai"]["sell_threshold"]))
            if ai_sell >= ai_buy:
                continue

            hybrid_mode = str(hybrid_raw.get("mode", fallback["hybrid"]["mode"]))
            if hybrid_mode not in valid_modes:
                hybrid_mode = fallback["hybrid"]["mode"]

            hybrid_buy = float(hybrid_raw.get("buy_threshold", fallback["hybrid"]["buy_threshold"]))
            hybrid_sell = float(hybrid_raw.get("sell_threshold", fallback["hybrid"]["sell_threshold"]))
            if hybrid_sell >= hybrid_buy:
                hybrid_buy = float(fallback["hybrid"]["buy_threshold"])
                hybrid_sell = float(fallback["hybrid"]["sell_threshold"])

            hybrid_margin = max(0.0, float(hybrid_raw.get("probability_margin", fallback["hybrid"]["probability_margin"])))

            return {
                "ai": {
                    "buy_threshold": ai_buy,
                    "sell_threshold": ai_sell,
                },
                "hybrid": {
                    "mode": hybrid_mode,
                    "buy_threshold": hybrid_buy,
                    "sell_threshold": hybrid_sell,
                    "probability_margin": hybrid_margin,
                },
                "source_path": str(candidate),
            }
        except Exception:
            continue

    fallback["source_path"] = "default"
    return fallback
