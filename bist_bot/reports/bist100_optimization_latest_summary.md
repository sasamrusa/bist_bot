# BIST100 10Y Optimization Summary

## Latest Report
- File: `reports/bist100_optimization_20260211_141502.json`
- Universe used: 94 symbols (top coverage subset)

## In-sample Best (score-max)
- Trial: 8
- Score: 485.862
- Mean PnL %: 916.21
- Median PnL %: 276.82
- Win Rate %: 59.44
- Active symbols: 93/94

Params:
```json
{
  "rsi_period": 14,
  "rsi_oversold": 28.0,
  "rsi_overbought": 60.0,
  "macd_fast": 8,
  "macd_slow": 18,
  "macd_signal": 9,
  "trend_ema_fast": 20,
  "trend_ema_slow": 40,
  "bbands_period": 22,
  "bbands_std": 2.0,
  "atr_period": 21,
  "score_threshold": 2,
  "signal_edge_min": 0,
  "enable_volume_filter": true,
  "volume_sma_period": 30,
  "volume_min_ratio": 0.7,
  "enable_mtf_filter": true,
  "enable_adx_filter": true,
  "adx_period": 10,
  "adx_min_value": 22.0,
  "enable_macdh_filter": true,
  "risk_per_trade_pct": 0.015,
  "stop_atr_multiplier": 1.5,
  "take_profit_atr_multiplier": 3.0,
  "enable_protective_exits": false
}
```

## Robust Candidate (7Y train / 3Y test re-check)
- Trial: 15
- Test score: 46.03
- Test mean PnL %: 43.40
- Test median PnL %: 30.61
- Test win rate %: 66.36

Params:
```json
{
  "rsi_period": 14,
  "rsi_oversold": 40.0,
  "rsi_overbought": 72.0,
  "macd_fast": 12,
  "macd_slow": 18,
  "macd_signal": 9,
  "trend_ema_fast": 10,
  "trend_ema_slow": 40,
  "bbands_period": 18,
  "bbands_std": 1.8,
  "atr_period": 14,
  "score_threshold": 2,
  "signal_edge_min": 1,
  "enable_volume_filter": true,
  "volume_sma_period": 15,
  "volume_min_ratio": 0.9,
  "enable_mtf_filter": false,
  "enable_adx_filter": true,
  "adx_period": 14,
  "adx_min_value": 18.0,
  "enable_macdh_filter": true,
  "risk_per_trade_pct": 0.015,
  "stop_atr_multiplier": 1.2,
  "take_profit_atr_multiplier": 3.0,
  "enable_protective_exits": false
}
```

## Notes
- Source list returned >100 candidates; optimizer uses top history-coverage subset.
- Some symbols failed Yahoo lookup (delisted/renamed), excluded automatically.
- Results can include survivorship bias because current-constituent style universe is used.

## Walk-Forward Validation (new)
- File: `reports/bist100_optimization_20260211_143342.json`
- Mode: 3Y train / 6M test / 6M step
- Trials per fold: 3 (quick sanity run)
- Folds: 6/6 completed
- Avg OOS score: 5.66
- Avg OOS mean PnL %: 0.21
- Avg OOS median PnL %: 0.04
- Avg OOS win rate %: 26.69
- Total OOS trades: 596
- Best fold id: 5 (test score: 13.97)

Best fold params:
```json
{
  "rsi_period": 14,
  "rsi_oversold": 35.0,
  "rsi_overbought": 55.0,
  "macd_fast": 12,
  "macd_slow": 30,
  "macd_signal": 7,
  "trend_ema_fast": 30,
  "trend_ema_slow": 75,
  "bbands_period": 22,
  "bbands_std": 2.4,
  "atr_period": 14,
  "score_threshold": 2,
  "signal_edge_min": 1,
  "enable_volume_filter": true,
  "volume_sma_period": 15,
  "volume_min_ratio": 0.7,
  "enable_mtf_filter": false,
  "enable_adx_filter": false,
  "adx_period": 20,
  "adx_min_value": 15.0,
  "enable_macdh_filter": false,
  "risk_per_trade_pct": 0.015,
  "stop_atr_multiplier": 2.0,
  "take_profit_atr_multiplier": 2.5,
  "enable_protective_exits": true
}
```
