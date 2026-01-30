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
DATA_INTERVAL: str = "5m"  # e.g., "1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"
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

# --- Strategy Configuration ---
# RSI Strategy Parameters (tuned for more frequent signals)
RSI_PERIOD: int = 14
RSI_OVERBOUGHT: int = 60
RSI_OVERSOLD: int = 40

# MACD Strategy Parameters (faster sensitivity)
MACD_FAST_PERIOD: int = 8
MACD_SLOW_PERIOD: int = 21
MACD_SIGNAL_PERIOD: int = 5

# Trend Following (EMA for faster reaction)
TREND_EMA_FAST_PERIOD: int = 20
TREND_EMA_SLOW_PERIOD: int = 50

# Bollinger Bands Parameters
BBANDS_PERIOD: int = 20
BBANDS_STD: float = 2.0

# ATR Parameters
ATR_PERIOD: int = 14

# Signal Scoring Threshold (lower => more trades)
SIGNAL_SCORE_THRESHOLD: int = 2
