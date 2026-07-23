from __future__ import annotations

import os
import time
from datetime import date, timedelta

import pandas as pd
import requests
import yfinance as yf
from flask import Flask, jsonify, render_template, request

from analysis import compute_signals

app = Flask(__name__)

# Simple in-memory cache: {(symbol, period): (timestamp, payload)}
_cache: dict[tuple[str, str], tuple[float, dict]] = {}
_CACHE_TTL = 900  # 15 minutes


def _clean_number(value):
    if value is None:
        return None
    return pd.to_numeric(str(value).replace("$", "").replace(",", "").strip(), errors="coerce")


def fetch_nasdaq_history(symbol: str, period: str) -> pd.DataFrame:
    history_days = {"3mo": 110, "6mo": 220, "1y": 390, "2y": 760}
    end = date.today()
    start = end - timedelta(days=history_days.get(period, 390))
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/124 Safari/537.36",
        "Accept": "application/json, text/plain, */*",
        "Origin": "https://www.nasdaq.com",
        "Referer": "https://www.nasdaq.com/",
    }
    for asset_class in ("stocks", "etf"):
        resp = requests.get(
            f"https://api.nasdaq.com/api/quote/{symbol}/historical",
            params={
                "assetclass": asset_class,
                "fromdate": start.isoformat(),
                "todate": end.isoformat(),
                "limit": 5000,
            },
            headers=headers,
            timeout=18,
        )
        resp.raise_for_status()
        payload = resp.json()
        rows = (((payload.get("data") or {}).get("tradesTable") or {}).get("rows") or [])
        if rows:
            df = pd.DataFrame(rows).rename(columns={
                "date": "Date", "open": "Open", "high": "High",
                "low": "Low", "close": "Close", "volume": "Volume",
            })
            for col in ["Open", "High", "Low", "Close", "Volume"]:
                df[col] = df[col].map(_clean_number)
            return df[["Date", "Open", "High", "Low", "Close", "Volume"]]
    raise ValueError(f"Nasdaq returned no data for {symbol}")


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/api/signals")
def signals():
    symbol = request.args.get("symbol", "SPY").strip().upper()
    period = request.args.get("period", "1y")

    if period not in {"3mo", "6mo", "1y", "2y"}:
        period = "1y"
    if not symbol or len(symbol) > 10 or not all(ch.isalnum() or ch in ".-^=" for ch in symbol):
        return jsonify({"error": "Enter a valid ticker symbol."}), 400

    cache_key = (symbol, period)
    cached = _cache.get(cache_key)
    if cached and (time.time() - cached[0]) < _CACHE_TTL:
        return jsonify(cached[1])

    try:
        try:
            df = fetch_nasdaq_history(symbol, period)
        except Exception as e:
            app.logger.info("Nasdaq failed for %s (%s); trying yfinance", symbol, e)
            df = yf.Ticker(symbol).history(period=period, interval="1d", auto_adjust=True)
            if df.empty:
                raise ValueError(f"No data found for {symbol}.")

        result = compute_signals(df, symbol)
        _cache[cache_key] = (time.time(), result)
        return jsonify(result)

    except Exception as exc:
        app.logger.warning("Signal computation failed for %s: %s", symbol, exc)
        return jsonify({"error": str(exc)}), 422


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
