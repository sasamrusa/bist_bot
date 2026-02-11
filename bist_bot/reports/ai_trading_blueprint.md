# AI Trading Bot Blueprint (BIST)

## Goal
Build a second system (parallel to current rule-based bot) that:
- analyzes market data with ML/AI models,
- converts model outputs into BUY/SELL/HOLD orders,
- enforces hard risk controls before execution.

## Chosen Core Method
Use a **hybrid architecture**:
1. **Primary alpha model**: Gradient Boosted Trees (LightGBM/CatBoost).
2. **Probability calibration**: CalibratedClassifierCV.
3. **Signal-to-order policy**: thresholded probability + ATR-based risk sizing + hard stop/take-profit.
4. **Validation**: strict walk-forward with temporal gap.
5. **Monitoring**: drift + model registry + retrain workflow.

Reason:
- Tabular market features + limited local-market sample size are usually better served by GBDT than heavy deep models in first production version.
- Calibrated probabilities are required for stable position sizing and execution thresholds.
- Walk-forward and leakage controls are mandatory in financial time series.

## Target Labels
Start with two labels in parallel:
1. **Direction label**: next horizon return sign (e.g. +1/-1 over N bars).
2. **Meta label**: trade/no-trade quality filter (only take high-quality base signals).

Future:
- add triple-barrier style event labeling for more robust entry/exit supervision.

## Feature Set (v1)
- Price/return lags: 1, 3, 5, 10, 20 bars
- Volatility: ATR, rolling std, realized volatility
- Trend: EMA slopes, ADX-like trend strength
- Momentum/oscillators: RSI, MACD, Bollinger position
- Liquidity: volume z-score, rolling volume ratios
- Regime: trend/range regime flags, volatility regime flags
- Market context: BIST index features and relative strength vs symbol

## Training & Validation Protocol
- Time-aware split only (no random shuffle).
- Walk-forward folds (train window + OOS test window).
- Gap/embargo between train and test to reduce leakage.
- Objective metric: OOS profit-aware composite (PnL, drawdown, turnover, hit ratio, costs).
- Hyperparameter optimization: Optuna with pruning.

## Execution Policy (Order Engine)
Given `p_up = P(return > 0)`:
- BUY when `p_up >= buy_threshold`.
- SELL/EXIT when `p_up <= sell_threshold` or protective conditions hit.
- HOLD otherwise.

Position sizing:
- risk budget per trade (e.g. 0.5%-1.5% equity),
- ATR-based stop distance,
- cap by max exposure and symbol liquidity.

Hard guards:
- max daily loss,
- max concurrent positions,
- volatility/circuit-breaker filters,
- no-trade if drift alarm is high.

## MLOps Layer
- Experiment tracking: params, dataset version, fold metrics, backtest artifacts.
- Model registry: versioned model lifecycle (staging/champion).
- Drift monitoring: feature and prediction drift checks.
- Scheduled retraining + shadow deployment before live switch.

## Integration Plan with Existing Repo
Add new package:
- `ai_pipeline/`
  - `dataset_builder.py`
  - `feature_store.py`
  - `labeling.py`
  - `train.py`
  - `infer.py`
  - `signal_policy.py`
  - `risk_engine.py`
  - `registry.py`
  - `drift_monitor.py`

Keep existing modules:
- `data/` for fetching
- `backtest/engine.py` for execution simulation (extendable with model signals)
- `gui_app.py` for visualizing AI confidence + executed orders

## Phased Roadmap
1. **Phase 0 (1 week)**: Data contract + feature/label pipeline + leakage tests.
2. **Phase 1 (1-2 weeks)**: Baseline model (LightGBM/CatBoost) + calibrated probabilities + walk-forward evaluation.
3. **Phase 2 (1 week)**: AI signal policy + risk engine + backtest integration.
4. **Phase 3 (1 week)**: Registry/drift monitoring + retrain pipeline.
5. **Phase 4 (continuous)**: advanced models (TFT/PatchTST), ensemble, model stacking.

## Success Criteria
- Positive OOS expectancy net of fees/slippage.
- Stable performance across multiple walk-forward folds (not one lucky period).
- Max drawdown and turnover within predefined limits.
- Drift alarms handled with automatic fallback to safe mode.

## Notes
- This is an engineering blueprint, not investment advice.
- Start with robust and interpretable models first; deep models are phase-4 upgrades.
