from __future__ import annotations

from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication

from bist_bot.gui_app import BistBotWindow


def main() -> None:
    app = QApplication([])
    app.setFont(QFont("Segoe UI", 10))
    window = BistBotWindow(run_mode="backtest")
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
