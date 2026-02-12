from __future__ import annotations

from typing import List

import requests
from bs4 import BeautifulSoup

from bist_bot.core.config import TICKERS


def fetch_bist100_symbols_from_cnbce() -> List[str]:
    url = "https://www.cnbce.com/borsa/hisseler/bist-100-hisseleri"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    symbols: List[str] = []
    for anchor in soup.select("div.definition-table-symbol-name a.name"):
        href = anchor.get("href", "")
        if "/borsa/hisseler/" not in href:
            continue
        code = href.split("/borsa/hisseler/")[1].split("-")[0].upper()
        if not code or code == "BIST":
            continue
        ticker = f"{code}.IS"
        if ticker not in symbols:
            symbols.append(ticker)
    return symbols


def resolve_universe(universe: str) -> List[str]:
    scope = universe.lower().strip()
    if scope == "config":
        return list(TICKERS)
    if scope == "bist100":
        try:
            symbols = fetch_bist100_symbols_from_cnbce()
            return symbols if symbols else list(TICKERS)
        except Exception:
            return list(TICKERS)
    raise ValueError(f"Unsupported universe: {universe}")

