# BIST Real-Time Data Source Research (2026-02-12)

## Objective
Find faster, more real-time BIST market data sources than current Yahoo/yfinance-based flow.

## Findings

### 1) Borsa Istanbul official data dissemination (best for true real-time)
- Borsa Istanbul states market data is disseminated in real-time, delayed, and EOD through licensed data vendors.
- Data is delivered via BISTECH infrastructure; direct vendor connectivity and co-location are supported.
- TIP feed is used for data vendors (MBP). ITCH (MBO, lower latency) is for co-location customers and not for redistribution.
- Official data vendor/distributor directories are published by Borsa Istanbul.

Practical meaning:
- If target is truly low-latency "anlik" BIST feed for a trading bot, this is the correct/compliant route.
- Requires commercial agreement and licensing (either with Borsa Istanbul directly or via a licensed distributor).

### 2) Licensed terminal/data providers (practical middle path)
- Matriks products emphasize live market monitoring and algorithmic features for BIST/VIOP.
- Borsa Istanbul data rights and redistribution constraints are explicitly shown on provider pages (licensing matters).

Practical meaning:
- Good operational option if provider gives API/socket stream suitable for your bot.
- You still need proper exchange/data usage permissions.

### 3) EODHD (easy API, but delayed for stocks)
- EODHD supports Istanbul exchange symbols (`.IS`) and exchange metadata.
- Their "Live (Delayed)" stock API notes stock prices are delayed (15-20 minutes).

Practical meaning:
- Useful for watchlists/backtests/light analytics.
- Not suitable for truly real-time execution logic on BIST.

### 4) Twelve Data (not suitable for real-time BIST in this context)
- Twelve Data support docs describe EOD pricing as end-of-day.
- Their EU real-time materials focus on Cboe Europe context, not BIST direct exchange feed in these docs.

Practical meaning:
- Can be useful for EOD/historical tasks.
- Not a direct answer for low-latency BIST real-time bot data.

## Recommended path for this project

1. **Primary recommendation:** move to a Borsa-licensed real-time feed path (TIP via licensed distributor).
2. **If ultra-low latency is needed later:** evaluate co-location + ITCH model constraints with Borsa Istanbul.
3. **Fallback (quick integration):** keep yfinance/EODHD for non-execution analytics only; do not use as "real-time execution" source.

## Suggested technical architecture

- `PrimaryFeed`: licensed stream (WebSocket/TCP/vendor SDK) -> normalized internal tick/bar bus
- `SecondaryFeed`: yfinance/EODHD as fallback/validation only
- `Clock/TradingHours`: use exchange session calendar + heartbeat watchdog
- `OrderGate`: block order generation if primary feed stale (e.g., >2-3 seconds)

## Sources
- Borsa Istanbul Data Dissemination: https://www.borsaistanbul.com/en/data/data-dissemination
- Borsa Istanbul Technical Side FAQ: https://www.borsaistanbul.com/en/faq/technical-side-data-dissemination
- Borsa Istanbul Data Vendors Directory (TR): https://www.borsaistanbul.com/tr/sayfa/3224
- Borsa Istanbul Nasdaq/TIP-ITCH notes: https://www.borsaistanbul.com/en/sayfa/3730/nasdaq-support
- Borsa Istanbul Technology Services: https://www.borsaistanbul.com/en/technology-services
- EODHD Live (Delayed) API docs: https://eodhd.com/financial-apis/live-realtime-stocks-api/
- EODHD Istanbul exchange page: https://eodhd.com/exchange/is
- Twelve Data EOD pricing: https://support.twelvedata.com/en/articles/12682324-end-of-day-eod-pricing-market-data
- Twelve Data EU equities note: https://support.twelvedata.com/en/articles/12656239-european-equities-market-data
- Matriks product/info pages: https://www.matriksdata.com/website/urunlerimiz/kullanici-platformlari/matriksiq-veri-terminali
