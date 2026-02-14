# AI + Hybrid Parameter Optimization Summary

- Selection window: `opt`
- Opt window: `2012-01-01` -> `2024-12-31`
- Eval window: `2025-01-01` -> `2026-02-13`
- Symbols loaded: `97`

## Selected For Deployment

- AI buy/sell: `0.540` / `0.360`
- Hybrid mode: `probability_gate`
- Hybrid margin: `0.060`
- Hybrid buy/sell: `0.520` / `0.360`
- Deploy config: `bist_bot\models\ai_registry\strategy_tuning_latest.json`

## Eval Ranking

| Rank | Strategy | Score | Median PnL % | Mean PnL % | Success % | Trades |
|---|---|---:|---:|---:|---:|---:|
| 1 | `ai_model_default` | 20.944 | +8.41 | +9.51 | 84.5 | 2114 |
| 2 | `ai_model_tuned_selected` | 20.931 | +8.21 | +13.29 | 84.5 | 214 |
| 3 | `hybrid_tuned_selected` | 19.410 | +6.56 | +12.67 | 82.5 | 131 |
| 4 | `rsi_macd` | 14.851 | +3.73 | +7.82 | 70.1 | 240 |
