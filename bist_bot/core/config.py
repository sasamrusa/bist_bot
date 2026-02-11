from typing import List

# --- General Configuration ---
PROJECT_NAME: str = "BIST Algorithmic Trading Bot"
VERSION: str = "0.1.0"

# --- Data Configuration ---
# BIST 30 tickers with '.IS' suffix for Yahoo Finance
BIST_30_TICKERS: List[str] = [
    "AKBNK.IS", "ARCLK.IS", "ASELS.IS", "BIMAS.IS", "DOHOL.IS",
    "EREGL.IS", "FROTO.IS", "GARAN.IS", "GOLTS.IS", "HALKB.IS",
    "ISCTR.IS", "KCHOL.IS", "KOZAL.IS", "KRDMD.IS", "PETKM.IS",
    "PGSUS.IS", "SAHOL.IS", "SASA.IS", "SISE.IS", "TAVHL.IS",
    "TKCMA.IS", "TCELL.IS", "THYAO.IS", "TKFEN.IS", "TTKOM.IS",
    "TUPRS.IS", "VAKBN.IS", "VESTL.IS", "YKBNK.IS", "ZOREN.IS",
]

EXCLUDED_TICKERS: List[str] = [
    # Symbols that frequently return no data on Yahoo Finance (delisted/renamed).
    "KOZAL.IS",
    "TKCMA.IS",
]

TICKERS: List[str] = [ticker for ticker in BIST_30_TICKERS if ticker not in EXCLUDED_TICKERS]
DATA_INTERVAL: str = "1d"  # e.g., "1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"
DATA_POLLING_INTERVAL_SECONDS: int = 60 # How often to fetch new data

# --- Paper Trading Configuration ---
PAPER_TRADING_STARTING_CASH: float = 100_000.0  # Starting capital in TRY
# Position sizing: 'full_allocation' means 1 position per ticker with full available cash
POSITION_SIZING_MODE: str = "full_allocation"
# Order type: For paper trading, we'll simulate market orders
ORDER_TYPE: str = "market" 

# --- Backtest Configuration ---
BACKTEST_STARTING_CASH: float = 100_000.0
BACKTEST_INTERVAL_OPTIONS = ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"]
BACKTEST_POSITION_SIZING_MODE: str = "atr_risk"  # atr_risk | full_allocation
BACKTEST_RISK_PER_TRADE_PCT: float = 0.015
BACKTEST_ATR_STOP_MULTIPLIER: float = 1.8
BACKTEST_TAKE_PROFIT_MULTIPLIER: float = 4.0
ENABLE_PROTECTIVE_EXITS: bool = False
BACKTEST_COMMISSION_BPS: float = 8.0  # 0.08%
BACKTEST_SLIPPAGE_BPS: float = 5.0  # 0.05%

# --- Strategy Configuration ---
# RSI + MACD strategy parameters (WFO best-fold candidate)
RSI_PERIOD: int = 14
RSI_OVERBOUGHT: int = 70
RSI_OVERSOLD: int = 30

# MACD Strategy Parameters
MACD_FAST_PERIOD: int = 6
MACD_SLOW_PERIOD: int = 26
MACD_SIGNAL_PERIOD: int = 9

# Trend Following (EMA)
TREND_EMA_FAST_PERIOD: int = 30
TREND_EMA_SLOW_PERIOD: int = 100

# Bollinger Bands Parameters
BBANDS_PERIOD: int = 20
BBANDS_STD: float = 2.4

# ATR Parameters
ATR_PERIOD: int = 14

# Signal Scoring Threshold (lower => more trades)
SIGNAL_SCORE_THRESHOLD: int = 2

# Volume Filter
ENABLE_VOLUME_FILTER: bool = False
VOLUME_SMA_PERIOD: int = 15
VOLUME_MIN_RATIO: float = 1.0

# Multi-timeframe Trend Confirmation
ENABLE_MULTI_TIMEFRAME_CONFIRMATION: bool = False

# Signal Quality Filters
ENABLE_ADX_FILTER: bool = False
ADX_PERIOD: int = 14
ADX_MIN_VALUE: float = 22.0
ENABLE_MACD_HISTOGRAM_FILTER: bool = False
SIGNAL_EDGE_MIN: int = 2
