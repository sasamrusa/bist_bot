from __future__ import annotations

import gc
import json
import os
import logging
from bisect import bisect_left
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Any

import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter, MaxNLocator
from PySide6.QtCore import QDate, QObject, QRunnable, QThreadPool, Qt, Signal, QTimer
from PySide6.QtGui import QBrush, QColor, QFont
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QDateEdit,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QPushButton,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from bist_bot.backtest.engine import BacktestEngine, BacktestResult
from bist_bot.ai_pipeline.universe import resolve_universe
from bist_bot.core.config import (
    BACKTEST_INTERVAL_OPTIONS,
    DATA_INTERVAL,
    DATA_POLLING_INTERVAL_SECONDS,
    ORDER_TYPE,
    TICKERS,
)
from bist_bot.data.yf_provider import YFDataProvider
from bist_bot.execution.paper_broker import PaperBroker
from bist_bot.strategies.ai_model_strategy import AIModelStrategy
from bist_bot.strategies.hybrid_ai_rsi_strategy import HybridAiRsiStrategy
from bist_bot.strategies.rsi_macd import RsiMacdStrategy
from bist_bot.utils.logger import setup_logger


LOGGER = setup_logger("bist_bot.gui")
AI_DEFAULT_BUY_THRESHOLD = 0.54
AI_DEFAULT_SELL_THRESHOLD = 0.50
HYBRID_DEFAULT_MODE = "weighted"
HYBRID_DEFAULT_BUY_THRESHOLD = 0.54
HYBRID_DEFAULT_SELL_THRESHOLD = 0.50
HYBRID_DEFAULT_PROBABILITY_MARGIN = 0.0
TRADE_TABLE_MAX_ROWS = 2500


class WorkerSignals(QObject):
    finished = Signal(object)
    error = Signal(str)


class FetchPricesWorker(QRunnable):
    def __init__(self, provider: YFDataProvider, tickers: List[str], interval: str):
        super().__init__()
        self.provider = provider
        self.tickers = tickers
        self.interval = interval
        self.signals = WorkerSignals()

    def run(self) -> None:
        results: List[Tuple[str, float | None, float | None, str]] = []
        try:
            for symbol in self.tickers:
                latest = self.provider.get_latest_data(symbol, self.interval)
                if latest.empty:
                    results.append((symbol, None, None, "No data"))
                    continue

                row = latest.iloc[-1]
                close_price = float(row.get("Close", row.get("close")))
                open_price_raw = row.get("Open", row.get("open", close_price))
                open_price = float(open_price_raw) if pd.notna(open_price_raw) else close_price
                if open_price:
                    change_pct = ((close_price - open_price) / open_price) * 100.0
                else:
                    change_pct = 0.0

                ts = row.name
                if hasattr(ts, "strftime"):
                    ts_text = ts.strftime("%Y-%m-%d %H:%M")
                else:
                    ts_text = str(ts)

                results.append((symbol, close_price, change_pct, ts_text))
            try:
                self.signals.finished.emit(results)
            except RuntimeError:
                return
        except Exception as exc:  # noqa: BLE001
            try:
                self.signals.error.emit(str(exc))
            except RuntimeError:
                return


class BacktestWorker(QRunnable):
    def __init__(
        self,
        symbol: str,
        interval: str,
        start: datetime,
        end: datetime,
        strategy_mode: str,
        symbols: List[str],
    ):
        super().__init__()
        self.symbol = symbol
        self.interval = interval
        self.start = start
        self.end = end
        self.strategy_mode = strategy_mode
        self.symbols = symbols
        self.signals = WorkerSignals()

    def run(self) -> None:
        try:
            # Keep all-tickers runs responsive by suppressing per-symbol INFO logs.
            logging.getLogger("bist_bot.data").setLevel(logging.WARNING)
            logging.getLogger("bist_bot.backtest").setLevel(logging.WARNING)
            logging.getLogger("bist_bot.strategy").setLevel(logging.WARNING)
            logging.getLogger("bist_bot.ai_strategy").setLevel(logging.WARNING)

            provider = YFDataProvider()
            if self.strategy_mode == "ai_model":
                strategy = AIModelStrategy(
                    buy_threshold=AI_DEFAULT_BUY_THRESHOLD,
                    sell_threshold=AI_DEFAULT_SELL_THRESHOLD,
                )
            elif self.strategy_mode == "hybrid_ai":
                strategy = HybridAiRsiStrategy(
                    mode=HYBRID_DEFAULT_MODE,
                    buy_threshold=HYBRID_DEFAULT_BUY_THRESHOLD,
                    sell_threshold=HYBRID_DEFAULT_SELL_THRESHOLD,
                    probability_margin=HYBRID_DEFAULT_PROBABILITY_MARGIN,
                )
            else:
                strategy = RsiMacdStrategy()
            engine = BacktestEngine(provider, strategy)
            if self.symbol == "__ALL__":
                results = engine.run_multi(self.symbols, self.interval, self.start, self.end)
                # Prevent UI freezes on repeated all-ticker runs by stripping heavy
                # chart payload from non-best symbols before crossing threads.
                candidates = [item for item in results.values() if item.data_points > 0]
                best_symbol = None
                if candidates:
                    best = max(candidates, key=lambda r: (r.profit_loss_pct, r.has_strategy_trades, r.trades))
                    best_symbol = best.symbol

                for sym, result in results.items():
                    if best_symbol is not None and sym == best_symbol:
                        continue
                    result.buy_markers = []
                    result.sell_markers = []
                    result.equity_curve = []
                    result.price_series = []
                    result.ohlc_series = []
                try:
                    self.signals.finished.emit(results)
                except RuntimeError:
                    return
            else:
                result = engine.run(self.symbol, self.interval, self.start, self.end)
                try:
                    self.signals.finished.emit(result)
                except RuntimeError:
                    return
        except Exception as exc:  # noqa: BLE001
            try:
                self.signals.error.emit(str(exc))
            except RuntimeError:
                return


class LiveSimWorker(QRunnable):
    def __init__(
        self,
        symbols: List[str],
        interval: str,
        strategy_mode: str,
        state_path: str,
        starting_cash: float,
        reset_state: bool,
    ):
        super().__init__()
        self.symbols = symbols
        self.interval = interval
        self.strategy_mode = strategy_mode
        self.state_path = state_path
        self.starting_cash = float(starting_cash)
        self.reset_state = reset_state
        self.signals = WorkerSignals()

    @staticmethod
    def _lookback_days(interval: str) -> int:
        if interval in {"1m", "2m", "5m", "15m", "30m", "60m", "90m"}:
            return 30
        if interval == "1h":
            return 120
        return 420

    @staticmethod
    def _resolve_price(frame: pd.DataFrame) -> float | None:
        if frame.empty:
            return None
        row = frame.iloc[-1]
        value = row.get("Close", row.get("close"))
        if pd.isna(value):
            return None
        return float(value)

    def run(self) -> None:
        try:
            provider = YFDataProvider()
            if self.strategy_mode == "ai_model":
                strategy = AIModelStrategy(
                    buy_threshold=AI_DEFAULT_BUY_THRESHOLD,
                    sell_threshold=AI_DEFAULT_SELL_THRESHOLD,
                )
            elif self.strategy_mode == "hybrid_ai":
                strategy = HybridAiRsiStrategy(
                    mode=HYBRID_DEFAULT_MODE,
                    buy_threshold=HYBRID_DEFAULT_BUY_THRESHOLD,
                    sell_threshold=HYBRID_DEFAULT_SELL_THRESHOLD,
                    probability_margin=HYBRID_DEFAULT_PROBABILITY_MARGIN,
                )
            else:
                strategy = RsiMacdStrategy()

            broker = PaperBroker(
                starting_cash=self.starting_cash,
                state_path=self.state_path,
                auto_load=not self.reset_state,
            )
            if self.reset_state:
                broker.reset(starting_cash=self.starting_cash)

            end = datetime.now()
            start = end - timedelta(days=self._lookback_days(self.interval))
            latest_prices: Dict[str, float] = {}
            scanned = 0
            filled_orders = 0

            for symbol in self.symbols:
                historical = provider.get_historical_data(symbol, self.interval, start, end)
                if historical.empty:
                    continue
                scanned += 1

                price = self._resolve_price(historical)
                if price is None or price <= 0:
                    continue
                latest_prices[symbol] = price

                current = historical.iloc[-1]
                signal = strategy.generate_signal(historical, current, symbol)
                has_position = broker.get_asset_balance(symbol) > 0.0

                if signal == "BUY" and not has_position:
                    cash = broker.cash
                    budget = cash * 0.2
                    quantity = budget / price if price > 0 else 0.0
                    if quantity > 0:
                        order = broker.place_order(symbol, ORDER_TYPE, quantity=quantity, price=price)
                        if str(order.get("status")) == "FILLED":
                            filled_orders += 1
                elif signal == "SELL" and has_position:
                    quantity = -broker.get_asset_balance(symbol)
                    if quantity != 0:
                        order = broker.place_order(symbol, ORDER_TYPE, quantity=quantity, price=price)
                        if str(order.get("status")) == "FILLED":
                            filled_orders += 1

            balance = broker.get_account_balance(current_prices=latest_prices)
            open_positions_detail = broker.get_open_positions(current_prices=latest_prices)
            broker.record_snapshot(balance)
            broker.save_state()

            pnl = float(balance["total_value"]) - broker.initial_cash
            pnl_pct = (pnl / broker.initial_cash * 100.0) if broker.initial_cash > 0 else 0.0
            running_days = (end - broker.started_at).total_seconds() / 86400.0
            summary: Dict[str, Any] = {
                "timestamp": end.isoformat(),
                "state_path": self.state_path,
                "strategy_mode": self.strategy_mode,
                "interval": self.interval,
                "symbols": len(self.symbols),
                "symbols_scanned": scanned,
                "orders_filled_cycle": filled_orders,
                "initial_cash": broker.initial_cash,
                "started_at": broker.started_at.isoformat(),
                "running_days": running_days,
                "trade_count_total": len(broker.trade_history),
                "open_positions": len([p for p in open_positions_detail if float(p.get("quantity", 0.0)) > 0]),
                "open_positions_detail": open_positions_detail,
                "latest_prices": latest_prices,
                "cash": float(balance["cash"]),
                "asset_value": float(balance["asset_value"]),
                "total_value": float(balance["total_value"]),
                "pnl": pnl,
                "pnl_pct": pnl_pct,
                "snapshot_count": len(broker.performance_snapshots),
            }
            try:
                self.signals.finished.emit(summary)
            except RuntimeError:
                return
        except Exception as exc:  # noqa: BLE001
            try:
                self.signals.error.emit(str(exc))
            except RuntimeError:
                return


class BistBotWindow(QMainWindow):
    def __init__(self, run_mode: str = "hybrid") -> None:
        super().__init__()
        self.run_mode = run_mode
        self.setWindowTitle("BIST Trading Terminal")
        self.setMinimumSize(1400, 820)

        self.thread_pool = QThreadPool()
        self.provider = YFDataProvider()
        self.current_result: BacktestResult | None = None
        self.chart_ax = None
        self.current_strategy_mode = "rsi_macd"
        self.current_universe = "config"
        self.active_symbols: List[str] = list(TICKERS)
        project_root = Path(__file__).resolve().parent
        preferred_state = project_root / "reports" / "live_sim_state.json"
        legacy_state = Path.cwd() / "reports" / "live_sim_state.json"
        if not preferred_state.exists() and legacy_state.exists():
            preferred_state.parent.mkdir(parents=True, exist_ok=True)
            legacy_state.replace(preferred_state)
        self.live_sim_state_path = str(preferred_state)
        self.live_sim_running = False
        self.live_sim_inflight = False
        self.live_sim_last_summary: Dict[str, Any] | None = None
        self.live_latest_prices: Dict[str, float] = {}
        self.live_open_positions: List[Dict[str, Any]] = []
        self.live_sim_timer = QTimer(self)
        self.live_sim_timer.setInterval(max(DATA_POLLING_INTERVAL_SECONDS, 15) * 1000)
        self.live_sim_timer.timeout.connect(self._run_live_sim_cycle)

        # Interactive chart drawing state
        self.draw_mode = "none"  # none | trend | hline
        self.user_drawings: List[Dict[str, float | str]] = []
        self._active_draw_start: Tuple[float, float] | None = None
        self._preview_line = None
        self.tool_buttons: Dict[str, QToolButton] = {}
        self._pan_active = False
        self._pan_start: Tuple[float, float] | None = None
        self._pan_xlim: Tuple[float, float] | None = None
        self._pan_ylim: Tuple[float, float] | None = None
        self._axis_zoom_active = False
        self._axis_zoom_mode: str | None = None  # x | y
        self._axis_zoom_start_px: Tuple[float, float] | None = None
        self._axis_zoom_xlim: Tuple[float, float] | None = None
        self._axis_zoom_ylim: Tuple[float, float] | None = None
        self._zoom_base = 1.18
        self._default_xlim: Tuple[float, float] | None = None
        self._default_ylim: Tuple[float, float] | None = None
        self._data_x_min: float | None = None
        self._data_x_max: float | None = None
        self._data_y_min: float | None = None
        self._data_y_max: float | None = None
        self._ts_values: List[pd.Timestamp] = []
        self._visible_highs: List[float] = []
        self._visible_lows: List[float] = []
        self._pan_autoscale_y = False

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        container = QWidget()
        root_layout = QVBoxLayout(container)
        root_layout.setContentsMargins(8, 8, 8, 8)
        root_layout.setSpacing(8)

        root_layout.addWidget(self._build_top_bar())

        body_layout = QHBoxLayout()
        body_layout.setSpacing(8)
        body_layout.addWidget(self._build_left_toolbar())
        body_layout.addWidget(self._build_chart_panel(), 1)
        body_layout.addWidget(self._build_watchlist_panel())
        root_layout.addLayout(body_layout, 1)

        self.setCentralWidget(container)
        self._apply_theme()
        self._set_draw_mode("none")
        self._apply_run_mode_profile()
        if not os.environ.get("PYTEST_CURRENT_TEST") and self.run_mode != "backtest":
            QTimer.singleShot(1200, self._auto_start_live_sim)

    def _apply_run_mode_profile(self) -> None:
        if self.run_mode == "backtest":
            self.setWindowTitle("BIST Backtest Terminal")
            self.bt_universe.setCurrentIndex(0)
            self.sim_initial_cash.hide()
            self.sim_toggle_button.hide()
            self.sim_reset_button.hide()
            self.live_sim_label.hide()
            self.open_positions_panel.hide()
            return

        if self.run_mode == "live_sim":
            self.setWindowTitle("BIST Live Simulation Terminal")
            self.bt_run.hide()
            self.bt_scope.setCurrentText("All Tickers")
            self.bt_scope.setEnabled(False)
            self.bt_symbol.setEnabled(False)

            strategy_index = self.bt_strategy.findData("ai_model")
            if strategy_index >= 0:
                self.bt_strategy.setCurrentIndex(strategy_index)

            universe_index = self.bt_universe.findData("bist100")
            if universe_index >= 0:
                self.bt_universe.setCurrentIndex(universe_index)
            self._initialize_live_chart_dates_from_state()
            self.open_positions_panel.show()
            self.watchlist_hint.setText(
                "Live mode: Start/End sadece grafik zaman aralığını belirler. Simülasyon state dosyasından kaldığı yerden devam eder."
            )
            return

        # hybrid/default profile
        self.bt_universe.setCurrentIndex(1)
        self.open_positions_panel.show()

    def _initialize_live_chart_dates_from_state(self) -> None:
        payload = self._load_live_sim_state()
        candidate_dates: List[pd.Timestamp] = []

        started_at = payload.get("started_at")
        if started_at:
            started_ts = self._normalize_ts(started_at)
            if not pd.isna(started_ts):
                candidate_dates.append(started_ts)

        for trade in list(payload.get("trade_history", [])):
            ts = self._normalize_ts(trade.get("timestamp"))
            if not pd.isna(ts):
                candidate_dates.append(ts)

        if not candidate_dates:
            return

        earliest = min(candidate_dates)
        self.bt_start.setDate(QDate(earliest.year, earliest.month, earliest.day))
        self.bt_end.setDate(QDate.currentDate())

    def _apply_theme(self) -> None:
        self.setStyleSheet(
            ""
            "QMainWindow { background-color: #f1f4f9; }\n"
            "QFrame#topBar { background: #ffffff; border: 1px solid #d9e0ea; border-radius: 8px; }\n"
            "QFrame#toolRail, QFrame#watchlistPanel, QFrame#chartPanel, QFrame#metricsPanel, QFrame#tradePanel { "
            "background: #ffffff; border: 1px solid #d9e0ea; border-radius: 8px; }\n"
            "QLabel#titleLabel { color: #111827; font-size: 18px; font-weight: 700; }\n"
            "QLabel#mutedLabel { color: #6b7280; }\n"
            "QLabel { color: #111827; }\n"
            "QPushButton { background: #2962ff; color: #ffffff; border: 0; border-radius: 6px; padding: 6px 12px; }\n"
            "QPushButton:disabled { background: #9ca3af; }\n"
            "QComboBox, QDateEdit, QDoubleSpinBox { background: #ffffff; color: #111827; border: 1px solid #cbd5e1; border-radius: 6px; padding: 4px 8px; }\n"
            "QToolButton { background: transparent; color: #334155; border: 1px solid #dbe3ee; border-radius: 6px; padding: 8px; min-width: 38px; }\n"
            "QToolButton:hover { background: #f3f7ff; }\n"
            "QToolButton:checked { background: #dbeafe; color: #1d4ed8; border-color: #93c5fd; }\n"
            "QTableWidget { background-color: #ffffff; color: #111827; gridline-color: #e5eaf2; border: none; }\n"
            "QHeaderView::section { background-color: #f8fafc; color: #334155; border: 1px solid #e2e8f0; padding: 6px; }\n"
            ""
        )

    def _build_top_bar(self) -> QWidget:
        bar = QFrame()
        bar.setObjectName("topBar")
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(8)

        title = QLabel("BIST Terminal")
        title.setObjectName("titleLabel")
        self.strategy_subtitle = QLabel("RSI + MACD + Trend Strategy")
        self.strategy_subtitle.setObjectName("mutedLabel")

        self.bt_scope = QComboBox()
        self.bt_scope.addItems(["Selected Ticker", "All Tickers"])
        self.bt_scope.currentTextChanged.connect(self._on_scope_changed)

        self.bt_universe = QComboBox()
        self.bt_universe.addItem("Config (BIST30)", "config")
        self.bt_universe.addItem("BIST100 (Live)", "bist100")
        self.bt_universe.currentIndexChanged.connect(self._on_universe_changed)

        self.bt_strategy = QComboBox()
        self.bt_strategy.addItem("RSI + MACD", "rsi_macd")
        self.bt_strategy.addItem("AI Model (Best)", "ai_model")
        self.bt_strategy.addItem("Hybrid (AI + RSI/MACD)", "hybrid_ai")
        self.bt_strategy.currentIndexChanged.connect(self._on_strategy_changed)

        self.bt_symbol = QComboBox()
        self.bt_symbol.addItems(self.active_symbols)

        self.bt_interval = QComboBox()
        self.bt_interval.addItems(BACKTEST_INTERVAL_OPTIONS)
        self.bt_interval.setCurrentText(DATA_INTERVAL)
        self.bt_interval.currentTextChanged.connect(self._on_live_chart_window_changed)
        self.live_interval = self.bt_interval

        today = QDate.currentDate()
        self.bt_start = QDateEdit(today.addDays(-30))
        self.bt_start.setCalendarPopup(True)
        self.bt_start.dateChanged.connect(self._on_live_chart_window_changed)
        self.bt_end = QDateEdit(today)
        self.bt_end.setCalendarPopup(True)
        self.bt_end.dateChanged.connect(self._on_live_chart_window_changed)

        self.bt_run = QPushButton("Run Analysis")
        self.bt_run.clicked.connect(self._run_backtest)
        self.refresh_button = QPushButton("Refresh Watchlist")
        self.refresh_button.clicked.connect(self._refresh_prices)
        self.sim_initial_cash = QDoubleSpinBox()
        self.sim_initial_cash.setDecimals(2)
        self.sim_initial_cash.setRange(100.0, 10_000_000.0)
        self.sim_initial_cash.setSingleStep(100.0)
        self.sim_initial_cash.setValue(5_000.0)
        self.sim_toggle_button = QPushButton("Start Live Sim")
        self.sim_toggle_button.clicked.connect(self._toggle_live_sim)
        self.sim_reset_button = QPushButton("Reset Sim")
        self.sim_reset_button.clicked.connect(self._reset_live_sim)

        self.live_status = QLabel("Ready")
        self.live_status.setObjectName("mutedLabel")

        layout.addWidget(title)
        layout.addWidget(self.strategy_subtitle)
        layout.addSpacing(16)
        layout.addWidget(QLabel("Universe"))
        layout.addWidget(self.bt_universe)
        layout.addWidget(QLabel("Strategy"))
        layout.addWidget(self.bt_strategy)
        layout.addWidget(QLabel("Scope"))
        layout.addWidget(self.bt_scope)
        layout.addWidget(QLabel("Ticker"))
        layout.addWidget(self.bt_symbol)
        layout.addWidget(QLabel("Interval"))
        layout.addWidget(self.bt_interval)
        layout.addWidget(QLabel("Start"))
        layout.addWidget(self.bt_start)
        layout.addWidget(QLabel("End"))
        layout.addWidget(self.bt_end)
        layout.addWidget(self.bt_run)
        layout.addWidget(self.refresh_button)
        layout.addWidget(QLabel("Sim Cash"))
        layout.addWidget(self.sim_initial_cash)
        layout.addWidget(self.sim_toggle_button)
        layout.addWidget(self.sim_reset_button)
        layout.addStretch(1)
        layout.addWidget(self.live_status)
        return bar

    def _build_left_toolbar(self) -> QWidget:
        rail = QFrame()
        rail.setObjectName("toolRail")
        rail.setFixedWidth(66)
        layout = QVBoxLayout(rail)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        mode_tools = [
            ("+", "Crosshair", "none"),
            ("TL", "Trendline", "trend"),
            ("SR", "Support/Resistance", "hline"),
        ]
        for text, tooltip, mode in mode_tools:
            button = QToolButton()
            button.setText(text)
            button.setToolTip(tooltip)
            button.setCheckable(True)
            button.clicked.connect(lambda checked, m=mode: self._set_draw_mode(m))
            layout.addWidget(button)
            self.tool_buttons[mode] = button

        zoom_in_button = QToolButton()
        zoom_in_button.setText("Z+")
        zoom_in_button.setToolTip("Zoom In")
        zoom_in_button.clicked.connect(lambda: self._zoom_chart(1.0 / self._zoom_base))
        layout.addWidget(zoom_in_button)

        zoom_out_button = QToolButton()
        zoom_out_button.setText("Z-")
        zoom_out_button.setToolTip("Zoom Out")
        zoom_out_button.clicked.connect(lambda: self._zoom_chart(self._zoom_base))
        layout.addWidget(zoom_out_button)

        fit_button = QToolButton()
        fit_button.setText("FT")
        fit_button.setToolTip("Fit Chart")
        fit_button.clicked.connect(self._fit_chart_view)
        layout.addWidget(fit_button)

        clear_button = QToolButton()
        clear_button.setText("CL")
        clear_button.setToolTip("Clear Drawings")
        clear_button.clicked.connect(self._clear_user_drawings)
        layout.addWidget(clear_button)

        reset_button = QToolButton()
        reset_button.setText("RS")
        reset_button.setToolTip("Reset Chart")
        reset_button.clicked.connect(self._redraw_current_result)
        layout.addWidget(reset_button)

        layout.addStretch(1)
        self._set_draw_mode("none")
        return rail

    def _build_chart_panel(self) -> QWidget:
        wrapper = QWidget()
        layout = QVBoxLayout(wrapper)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        panel = QFrame()
        panel.setObjectName("chartPanel")
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(12, 10, 12, 12)
        panel_layout.setSpacing(6)

        self.chart_symbol = QLabel("No analysis loaded")
        self.chart_symbol.setStyleSheet("font-size: 15px; font-weight: 700;")
        self.chart_stats = QLabel("Run analysis to render candlestick chart and signals.")
        self.chart_stats.setObjectName("mutedLabel")

        self.figure = Figure(figsize=(8, 5))
        self.canvas = FigureCanvasQTAgg(self.figure)
        panel_layout.addWidget(self.chart_symbol)
        panel_layout.addWidget(self.chart_stats)
        panel_layout.addWidget(self.canvas, 1)
        self.canvas.mpl_connect("button_press_event", self._on_chart_press)
        self.canvas.mpl_connect("motion_notify_event", self._on_chart_motion)
        self.canvas.mpl_connect("button_release_event", self._on_chart_release)
        self.canvas.mpl_connect("scroll_event", self._on_chart_scroll)

        metrics = QFrame()
        metrics.setObjectName("metricsPanel")
        metrics_layout = QGridLayout(metrics)
        metrics_layout.setContentsMargins(12, 10, 12, 10)
        metrics_layout.setHorizontalSpacing(16)
        metrics_layout.setVerticalSpacing(6)

        self.bt_initial_cash = QLabel("-")
        self.bt_final_value = QLabel("-")
        self.bt_profit = QLabel("-")
        self.bt_profit_pct = QLabel("-")
        self.bt_duration = QLabel("-")
        self.bt_trades = QLabel("-")

        metrics_layout.addWidget(QLabel("Initial Cash"), 0, 0)
        metrics_layout.addWidget(self.bt_initial_cash, 0, 1)
        metrics_layout.addWidget(QLabel("Final Value"), 0, 2)
        metrics_layout.addWidget(self.bt_final_value, 0, 3)
        metrics_layout.addWidget(QLabel("PnL"), 1, 0)
        metrics_layout.addWidget(self.bt_profit, 1, 1)
        metrics_layout.addWidget(QLabel("PnL %"), 1, 2)
        metrics_layout.addWidget(self.bt_profit_pct, 1, 3)
        metrics_layout.addWidget(QLabel("Duration"), 2, 0)
        metrics_layout.addWidget(self.bt_duration, 2, 1)
        metrics_layout.addWidget(QLabel("Trades"), 2, 2)
        metrics_layout.addWidget(self.bt_trades, 2, 3)

        trade_panel = QFrame()
        trade_panel.setObjectName("tradePanel")
        trade_layout = QVBoxLayout(trade_panel)
        trade_layout.setContentsMargins(12, 8, 12, 12)
        trade_layout.setSpacing(6)
        trade_layout.addWidget(QLabel("Trade Journal"))

        self.signal_table = QTableWidget(0, 10)
        self.signal_table.setHorizontalHeaderLabels(
            ["Ticker", "Entry Time", "Entry", "Exit Time", "Exit", "Qty", "Net PnL", "Fees", "Reason", "Status"]
        )
        self.signal_table.verticalHeader().setVisible(False)
        self.signal_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.signal_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.signal_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.signal_table.setSortingEnabled(True)
        self._configure_trade_table_columns()
        trade_layout.addWidget(self.signal_table)

        layout.addWidget(panel, 3)
        layout.addWidget(metrics)
        layout.addWidget(trade_panel, 1)
        return wrapper

    def _build_watchlist_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("watchlistPanel")
        panel.setFixedWidth(350)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        header = QLabel("Watchlist")
        header.setStyleSheet("font-size: 14px; font-weight: 700;")
        layout.addWidget(header)

        self.prices_table = QTableWidget(len(self.active_symbols), 4)
        self.prices_table.setHorizontalHeaderLabels(["Ticker", "Last", "Change %", "Time"])
        self.prices_table.verticalHeader().setVisible(False)
        self.prices_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.prices_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.prices_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.prices_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.prices_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.prices_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.prices_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self.prices_table.cellClicked.connect(self._on_watchlist_row_clicked)

        for row, symbol in enumerate(self.active_symbols):
            ticker_item = QTableWidgetItem(symbol)
            self.prices_table.setItem(row, 0, ticker_item)

        self.watchlist_hint = QLabel("Select a ticker row to sync chart ticker selector.")
        self.watchlist_hint.setWordWrap(True)
        self.watchlist_hint.setObjectName("mutedLabel")

        self.latest_price_label = QLabel("Last Price: -")
        self.latest_price_label.setStyleSheet("font-size: 16px; font-weight: 700;")
        self.live_sim_label = QLabel(
            "Live Sim: Stopped\n"
            "Initial: 5,000.00 TRY  |  Equity: -\n"
            "PnL: -  |  Trades: -  |  Period: -"
        )
        self.live_sim_label.setWordWrap(True)
        self.live_sim_label.setStyleSheet("font-size: 12px; color: #0f172a;")

        self.open_positions_panel = QFrame()
        self.open_positions_panel.setObjectName("metricsPanel")
        open_positions_layout = QVBoxLayout(self.open_positions_panel)
        open_positions_layout.setContentsMargins(8, 8, 8, 8)
        open_positions_layout.setSpacing(6)
        open_positions_layout.addWidget(QLabel("Open Positions (Live Sim)"))
        self.open_positions_table = QTableWidget(0, 6)
        self.open_positions_table.setHorizontalHeaderLabels(["Ticker", "Qty", "Avg Buy", "Last", "PnL", "PnL %"])
        self.open_positions_table.verticalHeader().setVisible(False)
        self.open_positions_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.open_positions_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.open_positions_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.open_positions_table.setSortingEnabled(True)
        self.open_positions_table.setMinimumHeight(170)
        self._configure_open_positions_columns()
        open_positions_layout.addWidget(self.open_positions_table)

        layout.addWidget(self.prices_table, 1)
        layout.addWidget(self.latest_price_label)
        layout.addWidget(self.live_sim_label)
        layout.addWidget(self.open_positions_panel)
        layout.addWidget(self.watchlist_hint)
        return panel

    def _configure_open_positions_columns(self) -> None:
        self.open_positions_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.open_positions_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.open_positions_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.open_positions_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.open_positions_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self.open_positions_table.horizontalHeader().setSectionResizeMode(5, QHeaderView.Stretch)

    def _render_open_positions_table(self, positions: List[Dict[str, Any]]) -> None:
        self.open_positions_table.setSortingEnabled(False)
        ordered = sorted(positions, key=lambda row: str(row.get("symbol", "")))
        self.open_positions_table.setRowCount(len(ordered))

        for row_idx, row in enumerate(ordered):
            symbol = str(row.get("symbol", ""))
            quantity = float(row.get("quantity", 0.0))
            avg_price = float(row.get("avg_price", 0.0))
            last_price = float(row.get("last_price", 0.0))
            pnl = float(row.get("unrealized_pnl", 0.0))
            pnl_pct = float(row.get("unrealized_pnl_pct", 0.0))

            symbol_item = QTableWidgetItem(symbol)
            qty_item = QTableWidgetItem(f"{quantity:,.2f}")
            avg_item = QTableWidgetItem(f"{avg_price:,.2f}")
            last_item = QTableWidgetItem(f"{last_price:,.2f}")
            pnl_item = QTableWidgetItem(f"{pnl:+,.2f}")
            pnl_pct_item = QTableWidgetItem(f"{pnl_pct:+.2f}%")

            if pnl > 0:
                pnl_color = QColor("#089981")
            elif pnl < 0:
                pnl_color = QColor("#f23645")
            else:
                pnl_color = QColor("#6b7280")
            pnl_item.setForeground(QBrush(pnl_color))
            pnl_pct_item.setForeground(QBrush(pnl_color))

            self.open_positions_table.setItem(row_idx, 0, symbol_item)
            self.open_positions_table.setItem(row_idx, 1, qty_item)
            self.open_positions_table.setItem(row_idx, 2, avg_item)
            self.open_positions_table.setItem(row_idx, 3, last_item)
            self.open_positions_table.setItem(row_idx, 4, pnl_item)
            self.open_positions_table.setItem(row_idx, 5, pnl_pct_item)

        self.open_positions_table.setSortingEnabled(True)
        self.open_positions_table.sortItems(4, Qt.DescendingOrder)

    def _on_scope_changed(self, text: str) -> None:
        self.bt_symbol.setEnabled(text == "Selected Ticker")

    def _on_universe_changed(self, _index: int) -> None:
        universe = str(self.bt_universe.currentData())
        self._apply_universe(universe, refresh_prices=True)

    def _on_strategy_changed(self, _index: int) -> None:
        mode = str(self.bt_strategy.currentData())
        self.current_strategy_mode = mode
        if mode == "ai_model":
            self.strategy_subtitle.setText("AI Model Strategy (latest trained model)")
            self.status_bar.showMessage("Strategy switched: AI Model (Best)", 3000)
        elif mode == "hybrid_ai":
            self.strategy_subtitle.setText("Hybrid Strategy (AI + RSI/MACD)")
            self.status_bar.showMessage("Strategy switched: Hybrid AI + RSI/MACD", 3000)
        else:
            self.strategy_subtitle.setText("RSI + MACD + Trend Strategy")
            self.status_bar.showMessage("Strategy switched: RSI + MACD", 3000)

    def _on_watchlist_row_clicked(self, row: int, _column: int) -> None:
        symbol_item = self.prices_table.item(row, 0)
        if not symbol_item:
            return
        symbol = symbol_item.text()
        self.bt_symbol.setCurrentText(symbol)
        if self.run_mode != "backtest" and self.live_sim_last_summary is not None:
            if self.run_mode == "live_sim" or self.current_result is None:
                self._refresh_live_chart_for_selected_symbol()

    def _on_live_chart_window_changed(self, _value: object = None) -> None:
        if self.run_mode != "live_sim":
            return
        if self.live_sim_last_summary is None:
            return
        self._refresh_live_chart_for_selected_symbol()

    def _auto_start_live_sim(self) -> None:
        if self.live_sim_running:
            return
        self._start_live_sim()

    def _toggle_live_sim(self) -> None:
        if self.live_sim_running:
            self._stop_live_sim()
        else:
            self._start_live_sim()

    def _start_live_sim(self) -> None:
        if not self.active_symbols:
            self.status_bar.showMessage("No symbols available for live simulation.", 5000)
            return
        self.live_sim_running = True
        self.sim_toggle_button.setText("Stop Live Sim")
        self.status_bar.showMessage("Live simulation started.", 3000)
        if not self.live_sim_timer.isActive():
            self.live_sim_timer.start()
        self._run_live_sim_cycle()

    def _stop_live_sim(self) -> None:
        self.live_sim_running = False
        if self.live_sim_timer.isActive():
            self.live_sim_timer.stop()
        self.sim_toggle_button.setText("Start Live Sim")
        self.status_bar.showMessage("Live simulation stopped.", 3000)

    def _reset_live_sim(self) -> None:
        if self.live_sim_inflight:
            self.status_bar.showMessage("Live simulation cycle is already running.", 3000)
            return
        state_path = Path(self.live_sim_state_path)
        if state_path.exists():
            state_path.unlink()
        self.live_sim_last_summary = None
        self.live_latest_prices = {}
        self.live_open_positions = []
        self._render_open_positions_table([])
        self.live_sim_label.setText(
            "Live Sim: Reset\n"
            f"Initial: {self.sim_initial_cash.value():,.2f} TRY  |  Equity: -\n"
            "PnL: -  |  Trades: 0  |  Period: 0.0d"
        )
        self.status_bar.showMessage("Live simulation state reset.", 4000)
        if self.live_sim_running:
            self._run_live_sim_cycle(reset_state=True)

    def _run_live_sim_cycle(self, reset_state: bool = False) -> None:
        if not self.live_sim_running and not reset_state:
            return
        if self.live_sim_inflight:
            return
        if not self.active_symbols:
            return

        self.live_sim_inflight = True
        interval = self.live_interval.currentText()
        strategy_mode = self.current_strategy_mode
        symbols = list(self.active_symbols)
        self.live_status.setText("Simulating...")

        worker = LiveSimWorker(
            symbols=symbols,
            interval=interval,
            strategy_mode=strategy_mode,
            state_path=self.live_sim_state_path,
            starting_cash=float(self.sim_initial_cash.value()),
            reset_state=reset_state,
        )
        worker.signals.finished.connect(self._on_live_sim_finished)
        worker.signals.error.connect(self._on_live_sim_error)
        self.thread_pool.start(worker)

    def _on_live_sim_finished(self, summary: Dict[str, Any]) -> None:
        self.live_sim_inflight = False
        self.live_sim_last_summary = summary
        self.live_latest_prices = {
            str(symbol): float(price)
            for symbol, price in dict(summary.get("latest_prices", {})).items()
            if price is not None
        }
        self.live_open_positions = list(summary.get("open_positions_detail", []))
        self.live_status.setText("Updated")
        self._render_live_sim_summary(summary)
        self._render_open_positions_table(self.live_open_positions)
        if self.run_mode == "live_sim":
            self._refresh_live_chart_for_selected_symbol()
        self.status_bar.showMessage(
            "Live sim updated: equity={0:,.2f} TRY, pnl={1:+.2f}% (trades={2})".format(
                float(summary.get("total_value", 0.0)),
                float(summary.get("pnl_pct", 0.0)),
                int(summary.get("trade_count_total", 0)),
            ),
            5000,
        )

    def _render_live_sim_summary(self, summary: Dict[str, Any]) -> None:
        pnl = float(summary.get("pnl", 0.0))
        pnl_pct = float(summary.get("pnl_pct", 0.0))
        equity = float(summary.get("total_value", 0.0))
        initial_cash = float(summary.get("initial_cash", self.sim_initial_cash.value()))
        trades = int(summary.get("trade_count_total", 0))
        running_days = float(summary.get("running_days", 0.0))
        open_positions = int(summary.get("open_positions", len(self.live_open_positions)))
        scanned = int(summary.get("symbols_scanned", 0))
        filled_cycle = int(summary.get("orders_filled_cycle", 0))

        pnl_color = "#089981" if pnl > 0 else "#f23645" if pnl < 0 else "#6b7280"
        self.live_sim_label.setStyleSheet(f"font-size: 12px; color: {pnl_color};")
        self.live_sim_label.setText(
            "Live Sim: Running\n"
            f"Initial: {initial_cash:,.2f} TRY  |  Equity: {equity:,.2f} TRY\n"
            f"PnL: {pnl:+,.2f} TRY ({pnl_pct:+.2f}%)  |  Trades: {trades}\n"
            f"Period: {running_days:.1f}d  |  Open: {open_positions}  |  Scanned: {scanned}  |  Filled(cycle): {filled_cycle}"
        )

    def _on_live_sim_error(self, message: str) -> None:
        self.live_sim_inflight = False
        LOGGER.error("live_sim_error %s", message)
        self.live_status.setText("Sim Error")
        self.status_bar.showMessage(f"Live Sim Error: {message}", 8000)

    def _apply_universe(self, universe: str, refresh_prices: bool) -> None:
        selected_before = self.bt_symbol.currentText() if hasattr(self, "bt_symbol") else ""
        symbols = resolve_universe(universe)
        dedup_symbols = list(dict.fromkeys(symbols))
        if not dedup_symbols:
            self.status_bar.showMessage("Universe load failed: empty symbol list.", 5000)
            return

        self.current_universe = universe
        self.active_symbols = dedup_symbols

        self.bt_symbol.blockSignals(True)
        self.bt_symbol.clear()
        self.bt_symbol.addItems(self.active_symbols)
        if selected_before in self.active_symbols:
            self.bt_symbol.setCurrentText(selected_before)
        self.bt_symbol.blockSignals(False)

        self.prices_table.setRowCount(len(self.active_symbols))
        for row, symbol in enumerate(self.active_symbols):
            self.prices_table.setItem(row, 0, QTableWidgetItem(symbol))
            self.prices_table.setItem(row, 1, QTableWidgetItem("-"))
            self.prices_table.setItem(row, 2, QTableWidgetItem("-"))
            self.prices_table.setItem(row, 3, QTableWidgetItem("-"))

        universe_name = "BIST100" if universe == "bist100" else "Config"
        self.watchlist_hint.setText(
            f"Universe: {universe_name} ({len(self.active_symbols)} symbols). Select a ticker row to sync chart ticker selector."
        )
        self.status_bar.showMessage(f"Universe loaded: {universe_name} ({len(self.active_symbols)})", 5000)

        if refresh_prices:
            self._refresh_prices()

    def _refresh_prices(self) -> None:
        if not self.active_symbols:
            self.status_bar.showMessage("No symbols available for selected universe.", 5000)
            return

        interval = self.live_interval.currentText()
        self.refresh_button.setEnabled(False)
        self.live_status.setText("Loading...")

        LOGGER.info(
            "gui_refresh_prices interval=%s universe=%s symbols=%s",
            interval,
            self.current_universe,
            len(self.active_symbols),
        )

        worker = FetchPricesWorker(self.provider, list(self.active_symbols), interval)
        worker.signals.finished.connect(self._on_prices_loaded)
        worker.signals.error.connect(self._on_error)
        self.thread_pool.start(worker)

    def _on_prices_loaded(self, results: List[Tuple[str, float | None, float | None, str]]) -> None:
        selected_symbol = self.bt_symbol.currentText()
        selected_price: float | None = None
        refreshed_prices: Dict[str, float] = {}
        self.prices_table.setRowCount(len(results))

        for row, (symbol, price, change_pct, ts_text) in enumerate(results):
            self.prices_table.setItem(row, 0, QTableWidgetItem(symbol))
            price_text = f"{price:.2f}" if price is not None else "N/A"
            change_text = f"{change_pct:+.2f}%" if change_pct is not None else "-"

            price_item = QTableWidgetItem(price_text)
            change_item = QTableWidgetItem(change_text)
            ts_item = QTableWidgetItem(ts_text)

            if change_pct is not None:
                if change_pct > 0:
                    color = QColor("#089981")
                elif change_pct < 0:
                    color = QColor("#f23645")
                else:
                    color = QColor("#6b7280")
                price_item.setForeground(QBrush(color))
                change_item.setForeground(QBrush(color))

            self.prices_table.setItem(row, 1, price_item)
            self.prices_table.setItem(row, 2, change_item)
            self.prices_table.setItem(row, 3, ts_item)

            if price is not None:
                refreshed_prices[symbol] = float(price)
            if symbol == selected_symbol and price is not None:
                selected_price = price

        if selected_price is not None:
            self.latest_price_label.setText(f"Last Price ({selected_symbol}): {selected_price:,.2f} TRY")
        if refreshed_prices:
            self.live_latest_prices.update(refreshed_prices)
        if self.live_open_positions:
            self.live_open_positions = self._mark_to_market_positions(self.live_open_positions, self.live_latest_prices)
            self._render_open_positions_table(self.live_open_positions)

        self.refresh_button.setEnabled(True)
        self.live_status.setText("Updated")
        LOGGER.info("gui_prices_loaded rows=%s", len(results))

    def _mark_to_market_positions(
        self,
        positions: List[Dict[str, Any]],
        latest_prices: Dict[str, float],
    ) -> List[Dict[str, Any]]:
        updated: List[Dict[str, Any]] = []
        for row in positions:
            item = dict(row)
            symbol = str(item.get("symbol", ""))
            quantity = float(item.get("quantity", 0.0))
            cost_basis = float(item.get("cost_basis", 0.0))
            avg_price = float(item.get("avg_price", 0.0))
            mark_price = float(latest_prices.get(symbol, item.get("last_price", avg_price)))
            market_value = quantity * mark_price
            unrealized_pnl = market_value - cost_basis
            unrealized_pnl_pct = (unrealized_pnl / cost_basis * 100.0) if cost_basis > 0 else 0.0
            item["last_price"] = mark_price
            item["market_value"] = market_value
            item["unrealized_pnl"] = unrealized_pnl
            item["unrealized_pnl_pct"] = unrealized_pnl_pct
            updated.append(item)
        return updated

    def _run_backtest(self) -> None:
        scope = self.bt_scope.currentText()
        symbol = self.bt_symbol.currentText()
        interval = self.bt_interval.currentText()
        strategy_mode = self.current_strategy_mode
        symbols = list(self.active_symbols)
        start = self._qdate_to_datetime(self.bt_start.date())
        end = self._qdate_to_datetime(self.bt_end.date(), end_of_day=True)

        if not symbols:
            self.status_bar.showMessage("No symbols available for selected universe.", 5000)
            return
        if scope != "All Tickers" and symbol not in symbols:
            self.status_bar.showMessage("Selected ticker is not in active universe.", 5000)
            return

        if end <= start:
            self.status_bar.showMessage("End date must be after start date.", 5000)
            return

        if interval in ["1m", "2m", "5m", "15m", "30m", "60m", "90m"]:
            max_days = 60
        elif interval == "1h":
            max_days = 730
        else:
            max_days = None

        if max_days is not None:
            min_start = end - timedelta(days=max_days)
            if start < min_start:
                self.status_bar.showMessage(
                    f"Intraday limit: {interval} supports ~{max_days} days. Start will be clamped.",
                    8000,
                )

        LOGGER.info(
            "gui_backtest_request strategy=%s universe=%s symbols=%s scope=%s symbol=%s interval=%s start=%s end=%s",
            strategy_mode,
            self.current_universe,
            len(symbols),
            scope,
            symbol,
            interval,
            start,
            end,
        )

        self.user_drawings.clear()
        self.bt_run.setEnabled(False)
        self.status_bar.showMessage("Running analysis...")
        if scope == "All Tickers":
            worker = BacktestWorker("__ALL__", interval, start, end, strategy_mode, symbols)
            worker.signals.finished.connect(self._on_backtest_multi_finished)
        else:
            worker = BacktestWorker(symbol, interval, start, end, strategy_mode, [symbol])
            worker.signals.finished.connect(self._on_backtest_finished)
        worker.signals.error.connect(self._on_error)
        self.thread_pool.start(worker)

    def _on_backtest_finished(self, result: BacktestResult) -> None:
        self.current_result = result
        LOGGER.info("gui_backtest_result symbol=%s pnl=%.2f trades=%s", result.symbol, result.profit_loss, result.trades)
        self._render_result_metrics(result)
        self._plot_signals(result)
        self._render_trade_table_single(result)
        if not result.has_strategy_trades:
            self.status_bar.showMessage("No strategy trades for this range.", 7000)
        self.bt_run.setEnabled(True)
        self.status_bar.showMessage("Analysis complete", 5000)

    def _on_backtest_multi_finished(self, results: Dict[str, BacktestResult]) -> None:
        if not results:
            self.status_bar.showMessage("No results returned.", 5000)
            self.bt_run.setEnabled(True)
            return

        candidates = [item for item in results.values() if item.data_points > 0]
        if not candidates:
            self.status_bar.showMessage("No historical data returned for selected universe/range.", 7000)
            self.bt_run.setEnabled(True)
            return

        best = max(candidates, key=lambda r: (r.profit_loss_pct, r.has_strategy_trades, r.trades))
        if not best.ohlc_series:
            with_chart = [item for item in candidates if item.ohlc_series]
            if with_chart:
                best = max(with_chart, key=lambda r: (r.profit_loss_pct, r.has_strategy_trades, r.trades))
        self.current_result = best
        LOGGER.info("gui_backtest_multi_best symbol=%s pnl=%.2f", best.symbol, best.profit_loss)
        self._render_result_metrics(best, best_mode=True)
        self._plot_signals(best)
        self._render_trade_table_multi(results)
        if not best.has_strategy_trades:
            self.status_bar.showMessage("No strategy trades for best ticker.", 7000)
        self.bt_run.setEnabled(True)
        self.status_bar.showMessage("Multi-ticker analysis complete", 5000)
        gc.collect()

    def _render_result_metrics(self, result: BacktestResult, best_mode: bool = False) -> None:
        self.bt_initial_cash.setText(f"{result.initial_cash:,.2f} TRY")
        self.bt_final_value.setText(f"{result.final_value:,.2f} TRY")
        self.bt_profit.setText(f"{result.profit_loss:,.2f} TRY")
        suffix = " (best)" if best_mode else ""
        self.bt_profit_pct.setText(f"{result.profit_loss_pct:.2f}%{suffix}")
        self.bt_duration.setText(f"{result.duration_days:.1f} days")
        self.bt_trades.setText(
            f"{result.trades} (rows: {result.data_points}, fees: {result.total_fees:,.2f}, slippage: {result.total_slippage_cost:,.2f})"
        )

        if result.profit_loss > 0:
            pnl_color = "#089981"
        elif result.profit_loss < 0:
            pnl_color = "#f23645"
        else:
            pnl_color = "#6b7280"
        self.bt_profit.setStyleSheet(f"color: {pnl_color}; font-weight: 700;")
        self.bt_profit_pct.setStyleSheet(f"color: {pnl_color}; font-weight: 700;")

    def _configure_trade_table_columns(self) -> None:
        self.signal_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.signal_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.signal_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.signal_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.signal_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self.signal_table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeToContents)
        self.signal_table.horizontalHeader().setSectionResizeMode(6, QHeaderView.ResizeToContents)
        self.signal_table.horizontalHeader().setSectionResizeMode(7, QHeaderView.ResizeToContents)
        self.signal_table.horizontalHeader().setSectionResizeMode(8, QHeaderView.ResizeToContents)
        self.signal_table.horizontalHeader().setSectionResizeMode(9, QHeaderView.Stretch)

    def _render_trade_table_single(self, result: BacktestResult) -> None:
        self._render_trade_table_rows([result])

    def _render_trade_table_multi(self, results: Dict[str, BacktestResult]) -> None:
        ordered = sorted(results.values(), key=lambda r: r.profit_loss_pct, reverse=True)
        self._render_trade_table_rows(ordered)

    def _render_trade_table_rows(self, result_list: List[BacktestResult]) -> None:
        rows: List[Dict[str, object]] = []
        for result in result_list:
            for trade in result.closed_trades:
                entry_ts = self._normalize_ts(trade.get("entry_ts"))
                exit_ts = self._normalize_ts(trade.get("exit_ts"))
                rows.append(
                    {
                        "symbol": trade.get("symbol", result.symbol),
                        "entry_ts": entry_ts,
                        "entry_fill_price": float(trade.get("entry_fill_price", 0.0)),
                        "exit_ts": exit_ts,
                        "exit_fill_price": float(trade.get("exit_fill_price", 0.0)),
                        "quantity": float(trade.get("quantity", 0.0)),
                        "net_pnl": float(trade.get("net_pnl", 0.0)),
                        "fees": float(trade.get("fees", 0.0)),
                        "reason": str(trade.get("exit_reason", "SIGNAL_SELL")),
                        "status": "CLOSED",
                    }
                )
            if result.open_trade:
                open_trade = result.open_trade
                entry_ts = self._normalize_ts(open_trade.get("entry_ts"))
                rows.append(
                    {
                        "symbol": open_trade.get("symbol", result.symbol),
                        "entry_ts": entry_ts,
                        "entry_fill_price": float(open_trade.get("entry_fill_price", 0.0)),
                        "exit_ts": None,
                        "exit_fill_price": float(open_trade.get("mark_price", 0.0)),
                        "quantity": float(open_trade.get("quantity", 0.0)),
                        "net_pnl": float(open_trade.get("unrealized_pnl", 0.0)),
                        "fees": float(open_trade.get("entry_fee", 0.0)),
                        "reason": "OPEN",
                        "status": "OPEN",
                    }
                )

        def _sort_key(row: Dict[str, object]) -> Tuple[pd.Timestamp, str]:
            exit_ts = row.get("exit_ts")
            entry_ts = row.get("entry_ts")
            primary = exit_ts if isinstance(exit_ts, pd.Timestamp) and not pd.isna(exit_ts) else entry_ts
            if not isinstance(primary, pd.Timestamp) or pd.isna(primary):
                primary = pd.Timestamp.min
            return primary, str(row.get("symbol", ""))

        rows.sort(key=_sort_key, reverse=True)
        max_rows = TRADE_TABLE_MAX_ROWS
        truncated = len(rows) > max_rows
        if truncated:
            rows = rows[:max_rows]

        self.signal_table.setUpdatesEnabled(False)
        self.signal_table.setSortingEnabled(False)
        self.signal_table.clearContents()
        self.signal_table.setRowCount(len(rows))
        for row_idx, row in enumerate(rows):
            symbol_item = QTableWidgetItem(str(row["symbol"]))
            entry_ts = row["entry_ts"]
            entry_item = QTableWidgetItem(
                entry_ts.strftime("%Y-%m-%d %H:%M")
                if isinstance(entry_ts, pd.Timestamp) and not pd.isna(entry_ts)
                else "-"
            )
            entry_price_item = QTableWidgetItem(f"{float(row['entry_fill_price']):,.2f}")

            exit_ts = row["exit_ts"]
            exit_item = QTableWidgetItem(
                exit_ts.strftime("%Y-%m-%d %H:%M")
                if isinstance(exit_ts, pd.Timestamp) and not pd.isna(exit_ts)
                else "-"
            )
            exit_price_item = QTableWidgetItem(
                f"{float(row['exit_fill_price']):,.2f}" if row["status"] != "OPEN" else f"{float(row['exit_fill_price']):,.2f} (mark)"
            )

            qty_item = QTableWidgetItem(f"{float(row['quantity']):,.2f}")
            pnl_item = QTableWidgetItem(f"{float(row['net_pnl']):,.2f}")
            fees_item = QTableWidgetItem(f"{float(row['fees']):,.2f}")
            reason_item = QTableWidgetItem(str(row["reason"]))
            status_item = QTableWidgetItem(str(row["status"]))

            pnl_value = float(row["net_pnl"])
            if pnl_value > 0:
                pnl_color = QColor("#089981")
            elif pnl_value < 0:
                pnl_color = QColor("#f23645")
            else:
                pnl_color = QColor("#6b7280")
            pnl_item.setForeground(QBrush(pnl_color))

            if row["status"] == "OPEN":
                status_item.setForeground(QBrush(QColor("#d97706")))
            else:
                status_item.setForeground(QBrush(QColor("#2563eb")))

            self.signal_table.setItem(row_idx, 0, symbol_item)
            self.signal_table.setItem(row_idx, 1, entry_item)
            self.signal_table.setItem(row_idx, 2, entry_price_item)
            self.signal_table.setItem(row_idx, 3, exit_item)
            self.signal_table.setItem(row_idx, 4, exit_price_item)
            self.signal_table.setItem(row_idx, 5, qty_item)
            self.signal_table.setItem(row_idx, 6, pnl_item)
            self.signal_table.setItem(row_idx, 7, fees_item)
            self.signal_table.setItem(row_idx, 8, reason_item)
            self.signal_table.setItem(row_idx, 9, status_item)

        self.signal_table.setSortingEnabled(True)
        self.signal_table.setUpdatesEnabled(True)
        if truncated:
            self.status_bar.showMessage(
                f"Trade journal truncated to latest {max_rows} rows for performance.",
                8000,
            )

    def _load_live_sim_state(self) -> Dict[str, Any]:
        state_path = Path(self.live_sim_state_path)
        if not state_path.exists():
            return {}
        try:
            return json.loads(state_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            LOGGER.exception("live_sim_state_read_failed path=%s", state_path)
            return {}

    def _extract_live_trade_markers(
        self,
        symbol: str,
        state_payload: Dict[str, Any],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        buy_markers: List[Dict[str, Any]] = []
        sell_markers: List[Dict[str, Any]] = []
        trade_history = list(state_payload.get("trade_history", []))
        for trade in trade_history:
            if str(trade.get("symbol", "")) != symbol:
                continue
            if str(trade.get("status", "")).upper() != "FILLED":
                continue
            ts = self._normalize_ts(trade.get("timestamp"))
            if pd.isna(ts):
                continue
            price_raw = trade.get("price")
            if price_raw is None:
                continue
            marker = {"ts": ts, "price": float(price_raw)}
            side = str(trade.get("transaction_type", "")).upper()
            if side == "BUY":
                buy_markers.append(marker)
            elif side == "SELL":
                sell_markers.append(marker)

        buy_markers.sort(key=lambda row: pd.to_datetime(row["ts"]))
        sell_markers.sort(key=lambda row: pd.to_datetime(row["ts"]))
        return buy_markers, sell_markers

    def _refresh_live_chart_for_selected_symbol(self) -> None:
        symbol = self.bt_symbol.currentText().strip()
        if not symbol:
            return
        interval = self.live_interval.currentText()
        now = datetime.now()
        start = self._qdate_to_datetime(self.bt_start.date())
        end = self._qdate_to_datetime(self.bt_end.date(), end_of_day=True)
        if end <= start:
            end = start + timedelta(days=1)
        if end > now:
            end = now

        # Keep chart usable when chosen range returns empty data.
        fallback_used = False
        historical = self.provider.get_historical_data(symbol, interval, start, end)
        if historical.empty:
            fallback_used = True
            end = now
            start = end - timedelta(days=LiveSimWorker._lookback_days(interval))
            historical = self.provider.get_historical_data(symbol, interval, start, end)
        if historical.empty:
            self.status_bar.showMessage(
                f"Live chart: no data for {symbol} in selected range.",
                5000,
            )
            return

        state_payload = self._load_live_sim_state()
        buy_markers, sell_markers = self._extract_live_trade_markers(symbol, state_payload)
        self._plot_live_symbol(symbol, interval, historical, buy_markers, sell_markers, start=start, end=end)
        if fallback_used:
            self.status_bar.showMessage(
                f"Live chart range had no data; fallback range loaded for {symbol}.",
                5000,
            )

    def _plot_live_symbol(
        self,
        symbol: str,
        interval: str,
        historical: pd.DataFrame,
        buy_markers: List[Dict[str, Any]],
        sell_markers: List[Dict[str, Any]],
        start: datetime,
        end: datetime,
    ) -> None:
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        self.figure.subplots_adjust(left=0.055, right=0.94, top=0.93, bottom=0.14)
        self.chart_ax = ax
        ax.set_facecolor("#ffffff")
        self.figure.patch.set_facecolor("#ffffff")

        df = historical.copy()
        df.columns = [str(c).lower() for c in df.columns]
        if "close" not in df.columns:
            ax.set_title("No data to plot")
            self.canvas.draw_idle()
            return
        if "open" not in df.columns:
            df["open"] = df["close"]
        if "high" not in df.columns:
            df["high"] = df["close"]
        if "low" not in df.columns:
            df["low"] = df["close"]
        df = df[["open", "high", "low", "close"]].copy()
        df.dropna(inplace=True)
        if df.empty:
            ax.set_title("No data to plot")
            self.canvas.draw_idle()
            return

        candles = [
            {
                "ts": ts,
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
            }
            for ts, row in df.iterrows()
        ]
        ts_values = [self._normalize_ts(c["ts"]) for c in candles]
        if not ts_values or any(pd.isna(ts) for ts in ts_values):
            ax.set_title("No data to plot")
            self.canvas.draw_idle()
            return
        self._ts_values = ts_values
        ts_values_ns = [int(ts.value) for ts in ts_values]
        ts_min = ts_values[0]
        ts_max = ts_values[-1]
        x_values = list(range(len(candles)))
        opens = [float(c["open"]) for c in candles]
        highs = [float(c["high"]) for c in candles]
        lows = [float(c["low"]) for c in candles]
        closes = [float(c["close"]) for c in candles]
        self._visible_highs = highs
        self._visible_lows = lows

        candle_width = self._resolve_candle_width(x_values)
        price_span = max(highs) - min(lows) if highs and lows else 0.0
        min_body = max(price_span * 0.0007, 0.01)

        for x, open_price, high_price, low_price, close_price in zip(x_values, opens, highs, lows, closes):
            color = "#089981" if close_price >= open_price else "#f23645"
            ax.vlines(x, low_price, high_price, color=color, linewidth=1.0, alpha=0.95, zorder=2)
            lower = min(open_price, close_price)
            body_height = max(abs(close_price - open_price), min_body)
            rect = Rectangle(
                (x - candle_width / 2, lower),
                candle_width,
                body_height,
                facecolor=color,
                edgecolor=color,
                linewidth=0.8,
                zorder=3,
            )
            ax.add_patch(rect)

        visible_buy_markers = [m for m in buy_markers if ts_min <= self._normalize_ts(m["ts"]) <= ts_max]
        visible_sell_markers = [m for m in sell_markers if ts_min <= self._normalize_ts(m["ts"]) <= ts_max]

        if visible_buy_markers:
            buy_x = [self._resolve_bar_index(ts_values_ns, self._normalize_ts(marker["ts"])) for marker in visible_buy_markers]
            buy_y = [float(marker["price"]) for marker in visible_buy_markers]
            ax.scatter(buy_x, buy_y, marker="^", s=82, color="#2962ff", edgecolors="#ffffff", linewidths=0.9, zorder=4)
            for x, y in zip(buy_x[-8:], buy_y[-8:]):
                ax.annotate(
                    "AL",
                    (x, y),
                    textcoords="offset points",
                    xytext=(0, -16),
                    ha="center",
                    color="#1d4ed8",
                    fontsize=8,
                    bbox={"boxstyle": "round,pad=0.22", "fc": "#dbeafe", "ec": "#93c5fd"},
                )

        if visible_sell_markers:
            sell_x = [self._resolve_bar_index(ts_values_ns, self._normalize_ts(marker["ts"])) for marker in visible_sell_markers]
            sell_y = [float(marker["price"]) for marker in visible_sell_markers]
            ax.scatter(sell_x, sell_y, marker="v", s=82, color="#d946ef", edgecolors="#ffffff", linewidths=0.9, zorder=4)
            for x, y in zip(sell_x[-8:], sell_y[-8:]):
                ax.annotate(
                    "SAT",
                    (x, y),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha="center",
                    color="#9d174d",
                    fontsize=8,
                    bbox={"boxstyle": "round,pad=0.22", "fc": "#fce7f3", "ec": "#f9a8d4"},
                )

        self._draw_user_overlays(ax)

        ax.grid(color="#e5ebf3", linewidth=0.8)
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
        ax.spines["right"].set_visible(True)
        ax.spines["left"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_color("#cbd5e1")
        ax.spines["bottom"].set_color("#cbd5e1")
        ax.tick_params(axis="x", colors="#475569")
        ax.tick_params(axis="y", colors="#475569", labelright=True, right=True, labelleft=False, left=False, pad=4)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.2f}"))

        ax.set_xlim(-1, len(candles))
        y_pad = price_span * 0.08 if price_span > 0 else 1.0
        ax.set_ylim(min(lows) - y_pad, max(highs) + y_pad)
        self._data_x_min = -1.0
        self._data_x_max = float(len(candles))
        self._data_y_min = float(min(lows) - y_pad)
        self._data_y_max = float(max(highs) + y_pad)
        self._default_xlim = ax.get_xlim()
        self._default_ylim = ax.get_ylim()
        self._update_x_ticks()

        last = candles[-1]
        last_close = float(last["close"])
        summary = self.live_sim_last_summary or {}
        pnl_pct = float(summary.get("pnl_pct", 0.0))
        total_trades = int(summary.get("trade_count_total", 0))
        open_positions = len([row for row in self.live_open_positions if float(row.get("quantity", 0.0)) > 0])
        strategy_mode = str(summary.get("strategy_mode", self.current_strategy_mode))
        if strategy_mode == "ai_model":
            strategy_label = "AI Model"
        elif strategy_mode == "hybrid_ai":
            strategy_label = "Hybrid AI+RSI"
        else:
            strategy_label = "RSI+MACD"
        range_text = f"{start.strftime('%Y-%m-%d')} -> {end.strftime('%Y-%m-%d')}"

        self.chart_symbol.setText(f"{symbol} - {interval} [{strategy_label} | Live Sim | Range {range_text}]")
        self.chart_stats.setText(
            "O {0:,.2f}  H {1:,.2f}  L {2:,.2f}  C {3:,.2f}   |   Sim PnL {4:+.2f}%   Filled Trades {5}   Open {6}".format(
                float(last["open"]),
                float(last["high"]),
                float(last["low"]),
                last_close,
                pnl_pct,
                total_trades,
                open_positions,
            )
        )
        self.latest_price_label.setText(f"Last Price ({symbol}): {last_close:,.2f} TRY")
        self.canvas.draw_idle()

    def _plot_signals(self, result: BacktestResult) -> None:
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        self.figure.subplots_adjust(left=0.055, right=0.94, top=0.93, bottom=0.14)
        self.chart_ax = ax
        ax.set_facecolor("#ffffff")
        self.figure.patch.set_facecolor("#ffffff")

        if not result.ohlc_series:
            ax.set_title("No data to plot")
            interval = self.bt_interval.currentText()
            self.chart_symbol.setText(f"{result.symbol} - {interval} [No data]")
            self.chart_stats.setText(
                f"No OHLC data in selected range ({result.start.strftime('%Y-%m-%d')} -> {result.end.strftime('%Y-%m-%d')})."
            )
            self.latest_price_label.setText(f"Last Price ({result.symbol}): -")
            self._ts_values = []
            self._visible_highs = []
            self._visible_lows = []
            self.canvas.draw()
            return

        candles = result.ohlc_series
        ts_values = [pd.to_datetime(c["ts"]) for c in candles]
        self._ts_values = ts_values
        ts_values_ns = [int(ts.value) for ts in ts_values]
        x_values = list(range(len(candles)))
        opens = [float(c["open"]) for c in candles]
        highs = [float(c["high"]) for c in candles]
        lows = [float(c["low"]) for c in candles]
        closes = [float(c["close"]) for c in candles]
        self._visible_highs = highs
        self._visible_lows = lows

        candle_width = self._resolve_candle_width(x_values)
        price_span = max(highs) - min(lows) if highs and lows else 0.0
        min_body = max(price_span * 0.0007, 0.01)

        for x, open_price, high_price, low_price, close_price in zip(x_values, opens, highs, lows, closes):
            color = "#089981" if close_price >= open_price else "#f23645"
            ax.vlines(x, low_price, high_price, color=color, linewidth=1.0, alpha=0.95, zorder=2)

            lower = min(open_price, close_price)
            body_height = max(abs(close_price - open_price), min_body)
            rect = Rectangle(
                (x - candle_width / 2, lower),
                candle_width,
                body_height,
                facecolor=color,
                edgecolor=color,
                linewidth=0.8,
                zorder=3,
            )
            ax.add_patch(rect)

        if result.buy_markers:
            buy_x = [self._resolve_bar_index(ts_values_ns, pd.to_datetime(m["ts"])) for m in result.buy_markers]
            buy_y = [float(m["price"]) for m in result.buy_markers]
            ax.scatter(buy_x, buy_y, marker="^", s=82, color="#2962ff", edgecolors="#ffffff", linewidths=0.9, zorder=4)
            for x, y in zip(buy_x[-8:], buy_y[-8:]):
                ax.annotate(
                    "AL",
                    (x, y),
                    textcoords="offset points",
                    xytext=(0, -16),
                    ha="center",
                    color="#1d4ed8",
                    fontsize=8,
                    bbox={"boxstyle": "round,pad=0.22", "fc": "#dbeafe", "ec": "#93c5fd"},
                )

        if result.sell_markers:
            sell_x = [self._resolve_bar_index(ts_values_ns, pd.to_datetime(m["ts"])) for m in result.sell_markers]
            sell_y = [float(m["price"]) for m in result.sell_markers]
            ax.scatter(sell_x, sell_y, marker="v", s=82, color="#d946ef", edgecolors="#ffffff", linewidths=0.9, zorder=4)
            for x, y in zip(sell_x[-8:], sell_y[-8:]):
                ax.annotate(
                    "SAT",
                    (x, y),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha="center",
                    color="#9d174d",
                    fontsize=8,
                    bbox={"boxstyle": "round,pad=0.22", "fc": "#fce7f3", "ec": "#f9a8d4"},
                )

        self._draw_sl_tp_overlays(ax, result, candle_width, ts_values_ns)
        self._draw_user_overlays(ax)

        ax.grid(color="#e5ebf3", linewidth=0.8)
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
        ax.spines["right"].set_visible(True)
        ax.spines["left"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_color("#cbd5e1")
        ax.spines["bottom"].set_color("#cbd5e1")
        ax.tick_params(axis="x", colors="#475569")
        ax.tick_params(axis="y", colors="#475569", labelright=True, right=True, labelleft=False, left=False, pad=4)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.2f}"))

        ax.set_xlim(-1, len(candles))
        y_pad = price_span * 0.08 if price_span > 0 else 1.0
        ax.set_ylim(min(lows) - y_pad, max(highs) + y_pad)
        self._data_x_min = -1.0
        self._data_x_max = float(len(candles))
        self._data_y_min = float(min(lows) - y_pad)
        self._data_y_max = float(max(highs) + y_pad)
        self._default_xlim = ax.get_xlim()
        self._default_ylim = ax.get_ylim()
        self._update_x_ticks()

        last = candles[-1]
        last_close = float(last["close"])
        if self.current_strategy_mode == "ai_model":
            strategy_label = "AI Model"
        elif self.current_strategy_mode == "hybrid_ai":
            strategy_label = "Hybrid AI+RSI"
        else:
            strategy_label = "RSI+MACD"
        self.chart_symbol.setText(f"{result.symbol} - {self.bt_interval.currentText()} [{strategy_label}]")
        self.chart_stats.setText(
            "O {0:,.2f}  H {1:,.2f}  L {2:,.2f}  C {3:,.2f}   |   PnL {4:+.2f}%   Trades {5}   |   SL 1.5x ATR / TP 3x ATR".format(
                float(last["open"]),
                float(last["high"]),
                float(last["low"]),
                last_close,
                result.profit_loss_pct,
                result.trades,
            )
        )
        self.latest_price_label.setText(f"Last Price ({result.symbol}): {last_close:,.2f} TRY")

        self.canvas.draw()

    def _redraw_current_result(self) -> None:
        if self.current_result is not None:
            self._plot_signals(self.current_result)

    def _fit_chart_view(self) -> None:
        if self.chart_ax is None:
            return
        if self._default_xlim is None or self._default_ylim is None:
            return
        self.chart_ax.set_xlim(self._default_xlim)
        self.chart_ax.set_ylim(self._default_ylim)
        self._update_x_ticks()
        self.canvas.draw_idle()

    def _zoom_chart(
        self,
        scale: float,
        center_x: float | None = None,
    ) -> None:
        if self.chart_ax is None:
            return

        x0, x1 = self.chart_ax.get_xlim()
        if center_x is None:
            center_x = (x0 + x1) / 2.0

        x_span = (x1 - x0) * scale
        min_x_span = 15.0
        x_span = max(min_x_span, x_span)

        new_x0 = center_x - x_span / 2.0
        new_x1 = center_x + x_span / 2.0

        new_x0, new_x1 = self._clamp_limits(new_x0, new_x1, self._data_x_min, self._data_x_max)

        self.chart_ax.set_xlim(new_x0, new_x1)
        self._autoscale_y_for_visible_x()
        self._update_x_ticks()
        self.canvas.draw_idle()

    def _zoom_y(
        self,
        scale: float,
        center_y: float | None = None,
    ) -> None:
        if self.chart_ax is None:
            return
        y0, y1 = self.chart_ax.get_ylim()
        if center_y is None:
            center_y = (y0 + y1) / 2.0
        y_span = y1 - y0
        total_y_span = max((self._data_y_max or y1) - (self._data_y_min or y0), 1.0)
        min_y_span = total_y_span * 0.03
        new_span = max(min_y_span, y_span * scale)
        new_y0 = center_y - new_span / 2.0
        new_y1 = center_y + new_span / 2.0
        new_y0, new_y1 = self._clamp_limits(new_y0, new_y1, self._data_y_min, self._data_y_max)
        self.chart_ax.set_ylim(new_y0, new_y1)
        self.canvas.draw_idle()

    def _zoom_xy(
        self,
        scale: float,
        center_x: float | None = None,
        center_y: float | None = None,
    ) -> None:
        if self.chart_ax is None:
            return

        x0, x1 = self.chart_ax.get_xlim()
        y0, y1 = self.chart_ax.get_ylim()
        if center_x is None:
            center_x = (x0 + x1) / 2.0
        if center_y is None:
            center_y = (y0 + y1) / 2.0

        x_span = max(15.0, (x1 - x0) * scale)
        total_y_span = max((self._data_y_max or y1) - (self._data_y_min or y0), 1.0)
        min_y_span = total_y_span * 0.03
        y_span = max(min_y_span, (y1 - y0) * scale)

        new_x0 = center_x - x_span / 2.0
        new_x1 = center_x + x_span / 2.0
        new_y0 = center_y - y_span / 2.0
        new_y1 = center_y + y_span / 2.0

        new_x0, new_x1 = self._clamp_limits(new_x0, new_x1, self._data_x_min, self._data_x_max)
        new_y0, new_y1 = self._clamp_limits(new_y0, new_y1, self._data_y_min, self._data_y_max)
        self.chart_ax.set_xlim(new_x0, new_x1)
        self.chart_ax.set_ylim(new_y0, new_y1)
        self._update_x_ticks()
        self.canvas.draw_idle()

    def _zoom_x_from_base(
        self,
        base_xlim: Tuple[float, float],
        scale: float,
        center_x: float | None = None,
    ) -> None:
        if self.chart_ax is None:
            return
        x0, x1 = base_xlim
        if center_x is None:
            center_x = (x0 + x1) / 2.0
        x_span = max(15.0, (x1 - x0) * scale)
        new_x0 = center_x - x_span / 2.0
        new_x1 = center_x + x_span / 2.0
        new_x0, new_x1 = self._clamp_limits(new_x0, new_x1, self._data_x_min, self._data_x_max)
        self.chart_ax.set_xlim(new_x0, new_x1)
        self._autoscale_y_for_visible_x()
        self._update_x_ticks()
        self.canvas.draw_idle()

    def _zoom_y_from_base(
        self,
        base_ylim: Tuple[float, float],
        scale: float,
        center_y: float | None = None,
    ) -> None:
        if self.chart_ax is None:
            return
        y0, y1 = base_ylim
        if center_y is None:
            center_y = (y0 + y1) / 2.0
        y_span = y1 - y0
        total_y_span = max((self._data_y_max or y1) - (self._data_y_min or y0), 1.0)
        min_y_span = total_y_span * 0.03
        new_span = max(min_y_span, y_span * scale)
        new_y0 = center_y - new_span / 2.0
        new_y1 = center_y + new_span / 2.0
        new_y0, new_y1 = self._clamp_limits(new_y0, new_y1, self._data_y_min, self._data_y_max)
        self.chart_ax.set_ylim(new_y0, new_y1)
        self.canvas.draw_idle()

    def _detect_axis_hit(self, event) -> str | None:
        if self.chart_ax is None:
            return None
        if event.x is None or event.y is None:
            return None

        bbox = self.chart_ax.get_window_extent()
        x_px = float(event.x)
        y_px = float(event.y)
        margin = 18.0

        near_y_axis = abs(x_px - bbox.x1) <= margin and (bbox.y0 - margin) <= y_px <= (bbox.y1 + margin)
        near_x_axis = abs(y_px - bbox.y0) <= margin and (bbox.x0 - margin) <= x_px <= (bbox.x1 + margin)

        if near_y_axis:
            return "y"
        if near_x_axis:
            return "x"
        return None

    def _update_x_ticks(self) -> None:
        if self.chart_ax is None or not self._ts_values:
            return

        x0, x1 = self.chart_ax.get_xlim()
        left = max(0, int(x0))
        right = min(len(self._ts_values) - 1, int(x1))
        if right <= left:
            right = min(len(self._ts_values) - 1, left + 1)

        span_hours = (self._ts_values[right] - self._ts_values[left]).total_seconds() / 3600.0
        if span_hours <= 72:
            tick_fmt = "%d %b\n%H:%M"
        elif span_hours <= 24 * 90:
            tick_fmt = "%d %b"
        else:
            tick_fmt = "%b %Y"

        def _fmt(value: float, _pos: int) -> str:
            idx = int(round(value))
            if idx < 0 or idx >= len(self._ts_values):
                return ""
            return self._ts_values[idx].strftime(tick_fmt)

        self.chart_ax.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True, min_n_ticks=4))
        self.chart_ax.xaxis.set_major_formatter(FuncFormatter(_fmt))

    def _autoscale_y_for_visible_x(self) -> None:
        if self.chart_ax is None:
            return
        if not self._visible_highs or not self._visible_lows:
            return

        x0, x1 = self.chart_ax.get_xlim()
        left = max(0, int(x0))
        right = min(len(self._visible_highs) - 1, int(x1))
        if right <= left:
            right = min(len(self._visible_highs) - 1, left + 1)

        visible_high = max(self._visible_highs[left : right + 1])
        visible_low = min(self._visible_lows[left : right + 1])
        span = visible_high - visible_low
        pad = max(span * 0.08, 0.35)

        new_y0 = visible_low - pad
        new_y1 = visible_high + pad
        new_y0, new_y1 = self._clamp_limits(new_y0, new_y1, self._data_y_min, self._data_y_max)
        self.chart_ax.set_ylim(new_y0, new_y1)

    @staticmethod
    def _clamp_limits(
        low: float,
        high: float,
        bound_low: float | None,
        bound_high: float | None,
    ) -> Tuple[float, float]:
        if bound_low is None or bound_high is None:
            return low, high

        total_span = bound_high - bound_low
        view_span = high - low
        if total_span <= 0:
            return low, high
        if view_span >= total_span:
            return bound_low, bound_high

        if low < bound_low:
            shift = bound_low - low
            low += shift
            high += shift
        if high > bound_high:
            shift = high - bound_high
            low -= shift
            high -= shift
        return low, high

    def _set_draw_mode(self, mode: str) -> None:
        self.draw_mode = mode
        if hasattr(self, "canvas"):
            if mode == "none":
                self.canvas.setCursor(Qt.ArrowCursor)
            else:
                self.canvas.setCursor(Qt.CrossCursor)
        for key, button in self.tool_buttons.items():
            button.blockSignals(True)
            button.setChecked(key == mode)
            button.blockSignals(False)
        mode_name = {
            "none": "Crosshair",
            "trend": "Trendline",
            "hline": "Support/Resistance",
        }.get(mode, mode)
        self.status_bar.showMessage(f"Tool: {mode_name}", 2500)

    def _clear_user_drawings(self) -> None:
        self.user_drawings.clear()
        self._active_draw_start = None
        self._preview_line = None
        self._redraw_current_result()
        self.status_bar.showMessage("Drawings cleared", 2500)

    def _on_chart_press(self, event) -> None:
        if self.chart_ax is None:
            return

        button = getattr(event, "button", None)
        button_text = str(button)
        is_left = button == 1 or button_text.endswith("LEFT")
        is_right = button == 3 or button_text.endswith("RIGHT")

        axis_hit = self._detect_axis_hit(event)
        if axis_hit is not None and is_left:
            self._axis_zoom_active = True
            self._axis_zoom_mode = axis_hit
            self._axis_zoom_start_px = (float(event.x), float(event.y))
            self._axis_zoom_xlim = self.chart_ax.get_xlim()
            self._axis_zoom_ylim = self.chart_ax.get_ylim()
            if axis_hit == "x":
                self.canvas.setCursor(Qt.SizeHorCursor)
            else:
                self.canvas.setCursor(Qt.SizeVerCursor)
            return

        if event.inaxes != self.chart_ax:
            return
        if event.xdata is None or event.ydata is None:
            return

        if is_right:
            self._pan_active = True
            self._pan_autoscale_y = False
            self._pan_start = (float(event.xdata), float(event.ydata))
            self._pan_xlim = self.chart_ax.get_xlim()
            self._pan_ylim = self.chart_ax.get_ylim()
            self.canvas.setCursor(Qt.ClosedHandCursor)
            return

        if not is_left:
            return

        if self.draw_mode == "trend":
            self._active_draw_start = (float(event.xdata), float(event.ydata))
            if self._preview_line is not None:
                self._preview_line.remove()
            (self._preview_line,) = self.chart_ax.plot(
                [event.xdata, event.xdata],
                [event.ydata, event.ydata],
                color="#334155",
                linestyle="--",
                linewidth=1.2,
                zorder=6,
            )
            self.canvas.draw_idle()
        elif self.draw_mode == "hline":
            self.user_drawings.append({"type": "hline", "y": float(event.ydata)})
            self._redraw_current_result()
        elif self.draw_mode == "none":
            self._pan_active = True
            self._pan_autoscale_y = False
            self._pan_start = (float(event.xdata), float(event.ydata))
            self._pan_xlim = self.chart_ax.get_xlim()
            self._pan_ylim = self.chart_ax.get_ylim()
            self.canvas.setCursor(Qt.ClosedHandCursor)

    def _on_chart_motion(self, event) -> None:
        if self._axis_zoom_active:
            if self.chart_ax is None:
                return
            if self._axis_zoom_start_px is None:
                return
            if self._axis_zoom_xlim is None or self._axis_zoom_ylim is None:
                return
            if event.x is None or event.y is None:
                return

            start_x, start_y = self._axis_zoom_start_px
            dx = float(event.x) - start_x
            dy = float(event.y) - start_y
            if self._axis_zoom_mode == "x":
                scale = 1.0 / (self._zoom_base ** (dx / 40.0))
                self._zoom_x_from_base(self._axis_zoom_xlim, scale)
            elif self._axis_zoom_mode == "y":
                scale = self._zoom_base ** (dy / 40.0)
                self._zoom_y_from_base(self._axis_zoom_ylim, scale)
            return

        if self._pan_active:
            if self.chart_ax is None or event.inaxes != self.chart_ax:
                return
            if event.xdata is None or event.ydata is None:
                return
            if self._pan_start is None or self._pan_xlim is None or self._pan_ylim is None:
                return

            start_x, _start_y = self._pan_start
            dx = float(event.xdata) - start_x
            dy = float(event.ydata) - self._pan_start[1]
            new_x0 = self._pan_xlim[0] - dx
            new_x1 = self._pan_xlim[1] - dx
            new_y0 = self._pan_ylim[0] - dy
            new_y1 = self._pan_ylim[1] - dy

            new_x0, new_x1 = self._clamp_limits(new_x0, new_x1, self._data_x_min, self._data_x_max)
            new_y0, new_y1 = self._clamp_limits(new_y0, new_y1, self._data_y_min, self._data_y_max)
            self.chart_ax.set_xlim(new_x0, new_x1)
            if self._pan_autoscale_y:
                self._autoscale_y_for_visible_x()
            else:
                self.chart_ax.set_ylim(new_y0, new_y1)
            self._update_x_ticks()
            self.canvas.draw_idle()
            return

        if self.draw_mode != "trend":
            return
        if self._active_draw_start is None or self._preview_line is None:
            return
        if self.chart_ax is None or event.inaxes != self.chart_ax:
            return
        if event.xdata is None or event.ydata is None:
            return

        x0, y0 = self._active_draw_start
        self._preview_line.set_data([x0, float(event.xdata)], [y0, float(event.ydata)])
        self.canvas.draw_idle()

    def _on_chart_release(self, event) -> None:
        if self._axis_zoom_active:
            self._axis_zoom_active = False
            self._axis_zoom_mode = None
            self._axis_zoom_start_px = None
            self._axis_zoom_xlim = None
            self._axis_zoom_ylim = None
            if self.draw_mode == "none":
                self.canvas.setCursor(Qt.ArrowCursor)
            else:
                self.canvas.setCursor(Qt.CrossCursor)
            return

        if self._pan_active:
            self._pan_active = False
            self._pan_start = None
            self._pan_xlim = None
            self._pan_ylim = None
            self._pan_autoscale_y = False
            if self.draw_mode == "none":
                self.canvas.setCursor(Qt.ArrowCursor)
            else:
                self.canvas.setCursor(Qt.CrossCursor)
            return

        if self.draw_mode != "trend":
            return
        if self._active_draw_start is None:
            return

        start = self._active_draw_start
        self._active_draw_start = None

        if self._preview_line is not None:
            self._preview_line.remove()
            self._preview_line = None

        if self.chart_ax is None or event.inaxes != self.chart_ax:
            self.canvas.draw_idle()
            return
        if event.xdata is None or event.ydata is None:
            self.canvas.draw_idle()
            return

        x0, y0 = start
        x1 = float(event.xdata)
        y1 = float(event.ydata)
        if abs(x1 - x0) < 1e-9 and abs(y1 - y0) < 1e-9:
            self.canvas.draw_idle()
            return

        self.user_drawings.append(
            {"type": "trend", "x1": x0, "y1": y0, "x2": x1, "y2": y1}
        )
        self._redraw_current_result()

    def _on_chart_scroll(self, event) -> None:
        if self.chart_ax is None or event.inaxes != self.chart_ax:
            return

        button = getattr(event, "button", None)
        if button == "up":
            scale = 1.0 / self._zoom_base
        elif button == "down":
            scale = self._zoom_base
        else:
            return
        key_text = str(getattr(event, "key", "") or "").lower()
        center_x = float(event.xdata) if event.xdata is not None else None
        center_y = float(event.ydata) if event.ydata is not None else None

        has_shift = "shift" in key_text
        has_alt = "alt" in key_text

        if has_shift and not has_alt:
            self._zoom_chart(scale, center_x=center_x)
            return
        if has_alt and not has_shift:
            self._zoom_y(scale, center_y=center_y)
            return
        self._zoom_xy(scale, center_x=center_x, center_y=center_y)

    def _on_error(self, message: str) -> None:
        LOGGER.error("gui_error %s", message)
        self.refresh_button.setEnabled(True)
        self.bt_run.setEnabled(True)
        self.status_bar.showMessage(f"Error: {message}", 8000)

    def closeEvent(self, event) -> None:  # noqa: N802
        try:
            self._stop_live_sim()
        except Exception:  # noqa: BLE001
            pass
        super().closeEvent(event)

    @staticmethod
    def _normalize_ts(value: Any) -> pd.Timestamp:
        ts = pd.to_datetime(value, errors="coerce")
        if pd.isna(ts):
            return pd.NaT
        try:
            if getattr(ts, "tzinfo", None) is not None:
                ts = ts.tz_convert(None)
        except (TypeError, AttributeError):
            try:
                ts = ts.tz_localize(None)
            except (TypeError, AttributeError):
                pass
        return ts

    def _draw_sl_tp_overlays(self, ax, result: BacktestResult, candle_width: float, ts_values_ns: List[int]) -> None:
        if not result.ohlc_series:
            return

        ohlc_df = pd.DataFrame(result.ohlc_series).copy()
        ohlc_df["ts"] = pd.to_datetime(ohlc_df["ts"])
        ohlc_df.sort_values("ts", inplace=True)
        atr_series = self._compute_atr_series(ohlc_df)

        markers: List[Dict[str, object]] = []
        for marker in result.buy_markers:
            markers.append({"side": "BUY", "ts": pd.to_datetime(marker["ts"]), "price": float(marker["price"])})
        for marker in result.sell_markers:
            markers.append({"side": "SELL", "ts": pd.to_datetime(marker["ts"]), "price": float(marker["price"])})
        if not markers:
            return

        markers.sort(key=lambda item: item["ts"])
        markers = markers[-8:]
        last_ts = ohlc_df["ts"].iloc[-1]

        for idx, marker in enumerate(markers):
            side = str(marker["side"])
            start_ts = pd.to_datetime(marker["ts"])
            entry = float(marker["price"])

            end_ts = last_ts
            for next_idx in range(idx + 1, len(markers)):
                next_marker = markers[next_idx]
                if str(next_marker["side"]) != side:
                    end_ts = pd.to_datetime(next_marker["ts"])
                    break

            atr_value = self._lookup_atr(atr_series, start_ts)
            if pd.isna(atr_value) or float(atr_value) <= 0:
                atr_value = max(entry * 0.01, 0.01)
            risk = float(atr_value) * 1.5
            reward = risk * 2.0

            if side == "BUY":
                sl = entry - risk
                tp = entry + reward
            else:
                sl = entry + risk
                tp = entry - reward

            x0 = self._resolve_bar_index(ts_values_ns, start_ts)
            x1 = self._resolve_bar_index(ts_values_ns, end_ts)
            if x1 <= x0:
                x1 = x0 + (candle_width * 3)

            ax.hlines(entry, x0, x1, colors="#64748b", linestyles=":", linewidth=1.0, alpha=0.9, zorder=3)
            ax.hlines(sl, x0, x1, colors="#ef4444", linestyles="--", linewidth=1.0, alpha=0.9, zorder=3)
            ax.hlines(tp, x0, x1, colors="#22c55e", linestyles="--", linewidth=1.0, alpha=0.9, zorder=3)

            max_x = (self._data_x_max - 0.5) if self._data_x_max is not None else (x1 + candle_width)
            label_x = min(x1 + candle_width, max_x)

            ax.text(
                label_x,
                sl,
                "SL",
                color="#ef4444",
                fontsize=8,
                va="center",
                ha="left",
                clip_on=True,
                zorder=4,
            )
            ax.text(
                label_x,
                tp,
                "TP",
                color="#16a34a",
                fontsize=8,
                va="center",
                ha="left",
                clip_on=True,
                zorder=4,
            )

    def _draw_user_overlays(self, ax) -> None:
        for drawing in self.user_drawings:
            draw_type = drawing.get("type")
            if draw_type == "trend":
                ax.plot(
                    [float(drawing["x1"]), float(drawing["x2"])],
                    [float(drawing["y1"]), float(drawing["y2"])],
                    color="#334155",
                    linewidth=1.25,
                    alpha=0.95,
                    zorder=5,
                )
            elif draw_type == "hline":
                ax.axhline(
                    y=float(drawing["y"]),
                    color="#1d4ed8",
                    linestyle="-.",
                    linewidth=1.0,
                    alpha=0.8,
                    zorder=4,
                )

    @staticmethod
    def _compute_atr_series(ohlc_df: pd.DataFrame, period: int = 14) -> pd.Series:
        prev_close = ohlc_df["close"].shift(1)
        high_low = ohlc_df["high"] - ohlc_df["low"]
        high_prev = (ohlc_df["high"] - prev_close).abs()
        low_prev = (ohlc_df["low"] - prev_close).abs()
        true_range = pd.concat([high_low, high_prev, low_prev], axis=1).max(axis=1)
        atr = true_range.rolling(window=period, min_periods=1).mean()
        atr.index = ohlc_df["ts"]
        return atr

    @staticmethod
    def _lookup_atr(atr_series: pd.Series, ts: pd.Timestamp) -> float:
        try:
            candidates = atr_series.loc[:ts]
            if len(candidates) == 0:
                return float(atr_series.iloc[0])
            return float(candidates.iloc[-1])
        except Exception:  # noqa: BLE001
            if len(atr_series) == 0:
                return 0.0
            return float(atr_series.iloc[-1])

    @staticmethod
    def _resolve_candle_width(times: List[float]) -> float:
        if len(times) < 2:
            return 0.6
        return 0.72

    @staticmethod
    def _resolve_bar_index(ts_values_ns: List[int], ts: pd.Timestamp) -> int:
        if not ts_values_ns:
            return 0
        target = int(pd.to_datetime(ts).value)
        pos = bisect_left(ts_values_ns, target)
        if pos <= 0:
            return 0
        if pos >= len(ts_values_ns):
            return len(ts_values_ns) - 1
        if ts_values_ns[pos] == target:
            return pos
        return pos - 1

    @staticmethod
    def _qdate_to_datetime(date: QDate, end_of_day: bool = False) -> datetime:
        if end_of_day:
            return datetime(date.year(), date.month(), date.day(), 23, 59, 59)
        return datetime(date.year(), date.month(), date.day(), 0, 0, 0)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="BIST GUI")
    parser.add_argument(
        "--mode",
        type=str,
        default="hybrid",
        choices=["hybrid", "backtest", "live_sim"],
        help="UI profile mode",
    )
    args = parser.parse_args()

    app = QApplication([])
    app.setFont(QFont("Segoe UI", 10))
    window = BistBotWindow(run_mode=args.mode)
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
