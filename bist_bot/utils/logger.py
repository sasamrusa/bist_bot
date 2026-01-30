from rich.console import Console
from rich.logging import RichHandler
import logging


def setup_logger(name: str = "bist_bot", level: int = logging.INFO) -> logging.Logger:
    """
    Sets up and returns a logger with RichHandler for structured logging.

    Args:
        name (str): The name of the logger.
        level (int): The logging level.

    Returns:
        logging.Logger: A configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        # Avoid duplicate handlers if called multiple times
        console = Console()
        handler = RichHandler(console=console, rich_tracebacks=True)
        formatter = logging.Formatter("%(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
