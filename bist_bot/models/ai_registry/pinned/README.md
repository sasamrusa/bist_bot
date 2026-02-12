# Pinned AI Models

This folder stores immutable model snapshots so `latest.json` updates do not overwrite your best versions.

Current pinned model:
- `bist100_1h_best_v1_20260212.joblib`
- Metadata: `bist100_1h_best_v1_20260212.json`

Run with this exact model:

```powershell
py -3.13 -m bist_bot.main_ai --model-path bist_bot/models/ai_registry/pinned/bist100_1h_best_v1_20260212.joblib --max-cycles 1 --buy-threshold 0.58 --sell-threshold 0.42
```
