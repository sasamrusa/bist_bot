from __future__ import annotations

from datetime import datetime, timedelta
from dataclasses import asdict
from typing import List, Tuple

import pandas as pd

from PySide6.QtCore import Qt, QObject, Signal, QRunnable, QThreadPool, QDate
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDateEdit,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QStatusBar,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from bist_bot.core.config import TICKERS, DATA_INTERVAL, BACKTEST_INTERVAL_OPTIONS
from bist_bot.data.yf_provider import YFDataProvider
from bist_bot.strategies.rsi_macd import RsiMacdStrategy
from bist_bot.backtest.engine import BacktestEngine, BacktestResult
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
        results: List[Tuple[str, float | None, str]] = []
        try:
            for symbol in self.tickers:
                latest = self.provider.get_latest_data(symbol, self.interval)
                if latest.empty:
                    results.append((symbol, None, "No data"))
                    continue
                row = latest.iloc[-1]
                price = float(row.get("Close", row.get("close")))
                ts = row.name
                ts_text = ts.strftime("%Y-%m-%d %H:%M") if hasattr(ts, "strftime") else str(ts)
                results.append((symbol, price, ts_text))
            self.signals.finished.emit(results)
        except Exception as exc:  # noqa: BLE001
            self.signals.error.emit(str(exc))


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
                self.signals.finished.emit(results)
            else:
                result = engine.run(self.symbol, self.interval, self.start, self.end)
                self.signals.finished.emit(result)
        except Exception as exc:  # noqa: BLE001
            self.signals.error.emit(str(exc))


class BistBotWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("BIST Trading Bot Dashboard")
        self.setMinimumSize(1000, 640)

        self.thread_pool = QThreadPool()
        self.provider = YFDataProvider()

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_live_tab(), "Live Prices")
        self.tabs.addTab(self._build_backtest_tab(), "Backtest")

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.addWidget(self.tabs)
        self.setCentralWidget(container)

        self._apply_theme()
        self._refresh_prices()

    def _apply_theme(self) -> None:
        self.setStyleSheet(
            ""
            "QMainWindow { background-color: #0f172a; }\n"
            "QLabel { color: #e2e8f0; }\n"
            "QGroupBox { border: 1px solid #1e293b; margin-top: 12px; color: #e2e8f0; }\n"
            "QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }\n"
            "QPushButton { background-color: #2563eb; color: #ffffff; padding: 6px 12px; border-radius: 6px; }\n"
            "QPushButton:disabled { background-color: #475569; }\n"
            "QComboBox, QDateEdit { background-color: #0b1220; color: #e2e8f0; padding: 4px; border: 1px solid #1e293b; }\n"
            "QTableWidget { background-color: #0b1220; color: #e2e8f0; gridline-color: #1e293b; }\n"
            "QHeaderView::section { background-color: #111827; color: #e2e8f0; padding: 6px; border: 1px solid #1f2937; }\n"
            ""
        )

    def _build_live_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        header_layout = QHBoxLayout()
        self.live_interval = QComboBox()
        self.live_interval.addItems(BACKTEST_INTERVAL_OPTIONS)
        self.live_interval.setCurrentText(DATA_INTERVAL)
        self.refresh_button = QPushButton("Refresh Prices")
        self.refresh_button.clicked.connect(self._refresh_prices)
        self.live_status = QLabel("Ready")
        self.live_status.setStyleSheet("color: #94a3b8;")

        header_layout.addWidget(QLabel("Interval:"))
        header_layout.addWidget(self.live_interval)
        header_layout.addWidget(self.refresh_button)
        header_layout.addStretch(1)
        header_layout.addWidget(self.live_status)

        self.prices_table = QTableWidget(len(TICKERS), 3)
        self.prices_table.setHorizontalHeaderLabels(["Ticker", "Last Price", "Timestamp"])
        self.prices_table.verticalHeader().setVisible(False)
        self.prices_table.setAlternatingRowColors(True)

        for row, symbol in enumerate(TICKERS):
            item = QTableWidgetItem(symbol)
            item.setFlags(item.flags() ^ Qt.ItemIsEditable)
            self.prices_table.setItem(row, 0, item)

        layout.addLayout(header_layout)
        layout.addWidget(self.prices_table)
        return widget

    def _build_backtest_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        controls_group = QGroupBox("Backtest Settings")
        form = QFormLayout(controls_group)

        self.bt_scope = QComboBox()
        self.bt_scope.addItems(["Selected Ticker", "All Tickers"])
        self.bt_scope.currentTextChanged.connect(self._on_scope_changed)

        self.bt_symbol = QComboBox()
        self.bt_symbol.addItems(TICKERS)

        self.bt_interval = QComboBox()
        self.bt_interval.addItems(BACKTEST_INTERVAL_OPTIONS)
        self.bt_interval.setCurrentText(DATA_INTERVAL)

        today = QDate.currentDate()
        self.bt_start = QDateEdit(today.addDays(-30))
        self.bt_start.setCalendarPopup(True)
        self.bt_end = QDateEdit(today)
        self.bt_end.setCalendarPopup(True)

        self.bt_run = QPushButton("Run Backtest")
        self.bt_run.clicked.connect(self._run_backtest)

        form.addRow("Scope", self.bt_scope)
        form.addRow("Ticker", self.bt_symbol)
        form.addRow("Interval", self.bt_interval)
        form.addRow("Start Date", self.bt_start)
        form.addRow("End Date", self.bt_end)
        form.addRow("", self.bt_run)

        results_group = QGroupBox("Results")
        results_layout = QFormLayout(results_group)
        self.bt_initial_cash = QLabel("-")
        self.bt_final_value = QLabel("-")
        self.bt_profit = QLabel("-")
        self.bt_profit_pct = QLabel("-")
        self.bt_duration = QLabel("-")
        self.bt_trades = QLabel("-")

        results_layout.addRow("Initial Cash", self.bt_initial_cash)
        results_layout.addRow("Final Value", self.bt_final_value)
        results_layout.addRow("Profit/Loss", self.bt_profit)
        results_layout.addRow("Profit/Loss %", self.bt_profit_pct)
        results_layout.addRow("Duration (days)", self.bt_duration)
        results_layout.addRow("Trades", self.bt_trades)

        chart_group = QGroupBox("Signals Chart")
        chart_layout = QVBoxLayout(chart_group)
        self.figure = Figure(figsize=(6, 4), tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        chart_layout.addWidget(self.canvas)

        layout.addWidget(controls_group)
        layout.addWidget(results_group)
        layout.addWidget(chart_group)
        layout.addStretch(1)
        return widget

    def _on_scope_changed(self, text: str) -> None:
        self.bt_symbol.setEnabled(text == "Selected Ticker")

    def _refresh_prices(self) -> None:
        interval = self.live_interval.currentText()
        self.refresh_button.setEnabled(False)
        self.live_status.setText("Loading...")

        LOGGER.info("gui_refresh_prices interval=%s", interval)

        worker = FetchPricesWorker(self.provider, TICKERS, interval)
        worker.signals.finished.connect(self._on_prices_loaded)
        worker.signals.error.connect(self._on_error)
        self.thread_pool.start(worker)

    def _on_prices_loaded(self, results: List[Tuple[str, float | None, str]]) -> None:
        for row, (symbol, price, ts_text) in enumerate(results):
            price_text = f"{price:.2f}" if price is not None else "N/A"
            price_item = QTableWidgetItem(price_text)
            price_item.setFlags(price_item.flags() ^ Qt.ItemIsEditable)
            ts_item = QTableWidgetItem(ts_text)
            ts_item.setFlags(ts_item.flags() ^ Qt.ItemIsEditable)
            self.prices_table.setItem(row, 1, price_item)
            self.prices_table.setItem(row, 2, ts_item)

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
                    f"Intraday limit: {interval} supports ~{max_days} days. Start was adjusted.",
                    8000,
                )

        LOGGER.info("gui_backtest_request scope=%s symbol=%s interval=%s start=%s end=%s", scope, symbol, interval, start, end)

        self.bt_run.setEnabled(False)
        self.status_bar.showMessage("Running backtest... (this may take a while)")
        LOGGER.info("gui_backtest_start_button scope=%s symbol=%s", scope, symbol)

        if scope == "All Tickers":
            worker = BacktestWorker("__ALL__", interval, start, end)
            worker.signals.finished.connect(self._on_backtest_multi_finished)
        else:
            worker = BacktestWorker(symbol, interval, start, end)
            worker.signals.finished.connect(self._on_backtest_finished)
        worker.signals.error.connect(self._on_error)
        self.thread_pool.start(worker)

    def _on_backtest_finished(self, result: BacktestResult) -> None:
        if isinstance(result, pd.Series):
            LOGGER.error("gui_backtest_result_series type=%s", type(result))
        LOGGER.info("gui_backtest_result symbol=%s pnl=%.2f trades=%s", result.symbol, result.profit_loss, result.trades)
        self.bt_initial_cash.setText(f"{result.initial_cash:,.2f} TRY")
        self.bt_final_value.setText(f"{result.final_value:,.2f} TRY")
        self.bt_profit.setText(f"{result.profit_loss:,.2f} TRY")
        self.bt_profit_pct.setText(f"{result.profit_loss_pct:.2f}%")
        self.bt_duration.setText(f"{result.duration_days:.1f}")
        self.bt_trades.setText(f"{result.trades} (rows: {result.data_points})")
        if not result.has_strategy_trades:
            self.status_bar.showMessage("No strategy trades for this range.", 7000)
        self._plot_signals(result)
        self.bt_run.setEnabled(True)
        self.status_bar.showMessage("Backtest complete", 5000)

    def _on_backtest_multi_finished(self, results: dict) -> None:
        if not results:
            self.status_bar.showMessage("No results returned.", 5000)
            self.bt_run.setEnabled(True)
            return

        best = max(results.values(), key=lambda r: r.profit_loss_pct)
        LOGGER.info("gui_backtest_multi_best symbol=%s pnl=%.2f", best.symbol, best.profit_loss)
        self.bt_initial_cash.setText(f"{best.initial_cash:,.2f} TRY")
        self.bt_final_value.setText(f"{best.final_value:,.2f} TRY")
        self.bt_profit.setText(f"{best.profit_loss:,.2f} TRY")
        self.bt_profit_pct.setText(f"{best.profit_loss_pct:.2f}% (best)")
        self.bt_duration.setText(f"{best.duration_days:.1f}")
        self.bt_trades.setText(f"{best.trades} (rows: {best.data_points})")
        if not best.has_strategy_trades:
            self.status_bar.showMessage("No strategy trades for best ticker.", 7000)
        self._plot_signals(best)
        self.bt_run.setEnabled(True)
        self.status_bar.showMessage("Multi-ticker backtest complete", 5000)

    def _on_error(self, message: str) -> None:
        LOGGER.error("gui_error %s", message)
        self.refresh_button.setEnabled(True)
        self.bt_run.setEnabled(True)
        self.status_bar.showMessage(f"Error: {message}", 8000)

    def _plot_signals(self, result: BacktestResult) -> None:
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        if not result.price_series:
            ax.set_title("No data to plot")
            self.canvas.draw()
            return

        x = [point["ts"] for point in result.price_series]
        y = [point["price"] for point in result.price_series]
        ax.plot(x, y, label="Price", color="#60a5fa")

        if result.buy_markers:
            bx = [m["ts"] for m in result.buy_markers]
            by = [m["price"] for m in result.buy_markers]
            ax.scatter(bx, by, marker="^", color="#22c55e", label="Buy")
        if result.sell_markers:
            sx = [m["ts"] for m in result.sell_markers]
            sy = [m["price"] for m in result.sell_markers]
            ax.scatter(sx, sy, marker="v", color="#ef4444", label="Sell")

        if not result.has_strategy_trades:
            ax.text(
                0.5,
                0.5,
                "No strategy trades",
                transform=ax.transAxes,
                ha="center",
                va="center",
                color="#f97316",
                fontsize=12,
            )

        ax.set_title(f"{result.symbol} Buy/Sell Signals")
        ax.grid(True, alpha=0.3)
        ax.legend()
        self.canvas.draw()


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
