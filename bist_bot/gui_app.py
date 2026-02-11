from __future__ import annotations

from bisect import bisect_left
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter, MaxNLocator
from PySide6.QtCore import QDate, QObject, QRunnable, QThreadPool, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QDateEdit,
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
from bist_bot.core.config import BACKTEST_INTERVAL_OPTIONS, DATA_INTERVAL, TICKERS
from bist_bot.data.yf_provider import YFDataProvider
from bist_bot.strategies.rsi_macd import RsiMacdStrategy
from bist_bot.utils.logger import setup_logger


LOGGER = setup_logger("bist_bot.gui")


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
    def __init__(self, symbol: str, interval: str, start: datetime, end: datetime):
        super().__init__()
        self.symbol = symbol
        self.interval = interval
        self.start = start
        self.end = end
        self.signals = WorkerSignals()

    def run(self) -> None:
        try:
            provider = YFDataProvider()
            strategy = RsiMacdStrategy()
            engine = BacktestEngine(provider, strategy)
            if self.symbol == "__ALL__":
                results = engine.run_multi(TICKERS, self.interval, self.start, self.end)
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


class BistBotWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("BIST Trading Terminal")
        self.setMinimumSize(1400, 820)

        self.thread_pool = QThreadPool()
        self.provider = YFDataProvider()
        self.current_result: BacktestResult | None = None
        self.chart_ax = None

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
        self._refresh_prices()

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
            "QComboBox, QDateEdit { background: #ffffff; color: #111827; border: 1px solid #cbd5e1; border-radius: 6px; padding: 4px 8px; }\n"
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
        subtitle = QLabel("RSI + MACD + Trend Strategy")
        subtitle.setObjectName("mutedLabel")

        self.bt_scope = QComboBox()
        self.bt_scope.addItems(["Selected Ticker", "All Tickers"])
        self.bt_scope.currentTextChanged.connect(self._on_scope_changed)

        self.bt_symbol = QComboBox()
        self.bt_symbol.addItems(TICKERS)

        self.bt_interval = QComboBox()
        self.bt_interval.addItems(BACKTEST_INTERVAL_OPTIONS)
        self.bt_interval.setCurrentText(DATA_INTERVAL)
        self.live_interval = self.bt_interval

        today = QDate.currentDate()
        self.bt_start = QDateEdit(today.addDays(-30))
        self.bt_start.setCalendarPopup(True)
        self.bt_end = QDateEdit(today)
        self.bt_end.setCalendarPopup(True)

        self.bt_run = QPushButton("Run Analysis")
        self.bt_run.clicked.connect(self._run_backtest)
        self.refresh_button = QPushButton("Refresh Watchlist")
        self.refresh_button.clicked.connect(self._refresh_prices)

        self.live_status = QLabel("Ready")
        self.live_status.setObjectName("mutedLabel")

        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addSpacing(16)
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

        self.prices_table = QTableWidget(len(TICKERS), 4)
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

        for row, symbol in enumerate(TICKERS):
            ticker_item = QTableWidgetItem(symbol)
            self.prices_table.setItem(row, 0, ticker_item)

        self.watchlist_hint = QLabel("Select a ticker row to sync chart ticker selector.")
        self.watchlist_hint.setWordWrap(True)
        self.watchlist_hint.setObjectName("mutedLabel")

        self.latest_price_label = QLabel("Last Price: -")
        self.latest_price_label.setStyleSheet("font-size: 16px; font-weight: 700;")

        layout.addWidget(self.prices_table, 1)
        layout.addWidget(self.latest_price_label)
        layout.addWidget(self.watchlist_hint)
        return panel

    def _on_scope_changed(self, text: str) -> None:
        self.bt_symbol.setEnabled(text == "Selected Ticker")

    def _on_watchlist_row_clicked(self, row: int, _column: int) -> None:
        symbol_item = self.prices_table.item(row, 0)
        if not symbol_item:
            return
        self.bt_symbol.setCurrentText(symbol_item.text())

    def _refresh_prices(self) -> None:
        interval = self.live_interval.currentText()
        self.refresh_button.setEnabled(False)
        self.live_status.setText("Loading...")

        LOGGER.info("gui_refresh_prices interval=%s", interval)

        worker = FetchPricesWorker(self.provider, TICKERS, interval)
        worker.signals.finished.connect(self._on_prices_loaded)
        worker.signals.error.connect(self._on_error)
        self.thread_pool.start(worker)

    def _on_prices_loaded(self, results: List[Tuple[str, float | None, float | None, str]]) -> None:
        selected_symbol = self.bt_symbol.currentText()
        selected_price: float | None = None

        for row, (symbol, price, change_pct, ts_text) in enumerate(results):
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

            if symbol == selected_symbol and price is not None:
                selected_price = price

        if selected_price is not None:
            self.latest_price_label.setText(f"Last Price ({selected_symbol}): {selected_price:,.2f} TRY")

        self.refresh_button.setEnabled(True)
        self.live_status.setText("Updated")
        LOGGER.info("gui_prices_loaded rows=%s", len(results))

    def _run_backtest(self) -> None:
        scope = self.bt_scope.currentText()
        symbol = self.bt_symbol.currentText()
        interval = self.bt_interval.currentText()
        start = self._qdate_to_datetime(self.bt_start.date())
        end = self._qdate_to_datetime(self.bt_end.date(), end_of_day=True)

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
            "gui_backtest_request scope=%s symbol=%s interval=%s start=%s end=%s",
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
            worker = BacktestWorker("__ALL__", interval, start, end)
            worker.signals.finished.connect(self._on_backtest_multi_finished)
        else:
            worker = BacktestWorker(symbol, interval, start, end)
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

        best = max(results.values(), key=lambda r: r.profit_loss_pct)
        self.current_result = best
        LOGGER.info("gui_backtest_multi_best symbol=%s pnl=%.2f", best.symbol, best.profit_loss)
        self._render_result_metrics(best, best_mode=True)
        self._plot_signals(best)
        self._render_trade_table_multi(results)
        if not best.has_strategy_trades:
            self.status_bar.showMessage("No strategy trades for best ticker.", 7000)
        self.bt_run.setEnabled(True)
        self.status_bar.showMessage("Multi-ticker analysis complete", 5000)

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
                rows.append(
                    {
                        "symbol": trade.get("symbol", result.symbol),
                        "entry_ts": pd.to_datetime(trade.get("entry_ts")),
                        "entry_fill_price": float(trade.get("entry_fill_price", 0.0)),
                        "exit_ts": pd.to_datetime(trade.get("exit_ts")),
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
                rows.append(
                    {
                        "symbol": open_trade.get("symbol", result.symbol),
                        "entry_ts": pd.to_datetime(open_trade.get("entry_ts")),
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

        rows.sort(
            key=lambda row: (
                row["exit_ts"] if row["exit_ts"] is not None else row["entry_ts"],
                row["symbol"],
            ),
            reverse=True,
        )

        self.signal_table.setSortingEnabled(False)
        self.signal_table.setRowCount(len(rows))
        for row_idx, row in enumerate(rows):
            symbol_item = QTableWidgetItem(str(row["symbol"]))
            entry_ts = row["entry_ts"]
            entry_item = QTableWidgetItem(entry_ts.strftime("%Y-%m-%d %H:%M") if hasattr(entry_ts, "strftime") else "-")
            entry_price_item = QTableWidgetItem(f"{float(row['entry_fill_price']):,.2f}")

            exit_ts = row["exit_ts"]
            exit_item = QTableWidgetItem(exit_ts.strftime("%Y-%m-%d %H:%M") if hasattr(exit_ts, "strftime") else "-")
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
        self.signal_table.sortItems(1, Qt.DescendingOrder)

    def _plot_signals(self, result: BacktestResult) -> None:
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        self.figure.subplots_adjust(left=0.055, right=0.94, top=0.93, bottom=0.14)
        self.chart_ax = ax
        ax.set_facecolor("#ffffff")
        self.figure.patch.set_facecolor("#ffffff")

        if not result.ohlc_series:
            ax.set_title("No data to plot")
            self.canvas.draw_idle()
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
        self.chart_symbol.setText(f"{result.symbol} - {self.bt_interval.currentText()}")
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

        self.canvas.draw_idle()

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
    app = QApplication([])
    app.setFont(QFont("Segoe UI", 10))
    window = BistBotWindow()
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
