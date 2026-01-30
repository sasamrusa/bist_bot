# Borsa Istanbul Algorithmic Trading Bot

This project implements a modular, high-performance algorithmic trading bot for Borsa Istanbul (BIST) using free APIs for data and supporting real-time simulation (paper trading).

## Environment Setup

1.  **Initialize a new Python project structure:**
    ```bash
    # This step is already done by the agent.
    # cd bist_bot
    ```

2.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    ```

3.  **Activate the virtual environment:**
    *   **Windows:**
        ```bash
        .\venv\Scripts\activate
        ```
    *   **macOS/Linux:**
        ```bash
        source venv/bin/activate
        ```

4.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Core Architecture (SOLID Principles)

The project is structured with the following key components:

*   `core/`: Contains core functionalities like configuration and interfaces.
*   `data/`: Handles data provision, with `yf_provider.py` for yfinance integration.
*   `strategies/`: Houses trading strategies, including a base class and an example RSI+MACD strategy.
*   `execution/`: Manages trade execution, with `paper_broker.py` for simulated trading.
*   `utils/`: Provides utility functions, such as structured logging.
*   `main.py`: The entry point for the application, handling the runtime loop and CLI.

## GUI Dashboard

The GUI provides live prices and a backtesting tab with start/end date pickers.

1. Install GUI dependency:
    ```bash
    pip install -r requirements.txt
    ```
2. Run the GUI from the Desktop (parent of `bist_bot`):
    ```bash
    py -m bist_bot.gui_app
    ```

### Intraday Data Limits (Yahoo)
Yahoo Finance restricts intraday history. The backtest will clamp the start date based on interval:

*   `1m`: ~7 days
*   `2m`, `5m`, `15m`, `30m`, `60m`, `90m`: ~60 days
*   `1h`: ~730 days

If your selected start date is too old, the GUI will warn and the engine will auto‑clamp.

### Entrypoint Notes
Run via module paths to avoid import errors:

*   GUI: `py -m bist_bot.gui_app`
*   CLI bot loop: `py -m bist_bot.main`
