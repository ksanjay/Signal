"""Signal computation engine for the 3-out-of-5 hedging strategy.

Indicators:
  RSI(18)         — oversold(<30)=bullish, overbought(>70)=bearish
  MACD(12,26,9)   — line > signal=bullish, else bearish
  Bollinger(20,2.5) — above upper=bearish(overbought), below lower=bullish(oversold), inside=neutral
  Volume(vs 20 SMA) — high-vol + price up=bullish, high-vol + price down=bearish, else neutral
  Trend(vs 50 SMA)  — price > SMA=bullish, price < SMA=bearish

Composite: 3+ bullish = LONG, 3+ bearish = HEDGE.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MIN_ROWS = 60


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = [str(c[0]) for c in d.columns]
    d.columns = [str(c).strip().lower() for c in d.columns]

    aliases = {
        "date": "date", "datetime": "date", "timestamp": "date",
        "open": "open", "high": "high", "low": "low",
        "close": "close", "adj close": "close", "adj_close": "close",
        "volume": "volume",
    }
    d = d.rename(columns={c: aliases[c] for c in d.columns if c in aliases})

    if "date" not in d.columns:
        d = d.reset_index()
        d.columns = [str(c).strip().lower() for c in d.columns]
        first = d.columns[0]
        if first not in {"open", "high", "low", "close", "volume"}:
            d = d.rename(columns={first: "date"})

    required = ["date", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in d.columns]
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")

    d = d[required].copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce", utc=True)
    for col in required[1:]:
        d[col] = pd.to_numeric(d[col], errors="coerce")
    d = d.dropna(subset=["date", "close", "volume"])
    d = d.sort_values("date").drop_duplicates("date").reset_index(drop=True)

    if len(d) < MIN_ROWS:
        raise ValueError(f"Need at least {MIN_ROWS} rows; got {len(d)}.")
    return d


def _rsi(close: pd.Series, period: int = 18) -> pd.Series:
    # Wilder's smoothing via EWM with alpha = 1/period
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line


def _bollinger(close: pd.Series, period: int = 20, mult: float = 2.5):
    basis = close.rolling(period).mean()
    std = close.rolling(period).std(ddof=0)
    return basis, basis + mult * std, basis - mult * std


def _f(v) -> float | None:
    if v is None:
        return None
    try:
        fv = float(v)
        return None if np.isnan(fv) else round(fv, 4)
    except (TypeError, ValueError):
        return None


def compute_signals(df: pd.DataFrame, symbol: str) -> dict:
    d = _normalize(df)
    close = d["close"]
    volume = d["volume"]

    rsi = _rsi(close, 18)
    macd_line, signal_line = _macd(close, 12, 26, 9)
    bb_basis, bb_upper, bb_lower = _bollinger(close, 20, 2.5)
    sma50 = close.rolling(50).mean()
    vol_sma20 = volume.rolling(20).mean()

    # RSI state
    rsi_state = pd.Series("neutral", index=d.index, dtype=str)
    rsi_state[rsi < 30] = "bullish"
    rsi_state[rsi > 70] = "bearish"

    # MACD state — only defined where both lines are non-null
    valid_macd = macd_line.notna() & signal_line.notna()
    macd_state = pd.Series("neutral", index=d.index, dtype=str)
    macd_state[valid_macd] = np.where(macd_line[valid_macd] > signal_line[valid_macd], "bullish", "bearish")

    # Bollinger Band state — above upper band = overbought (bearish), below lower = oversold (bullish)
    bb_state = pd.Series("neutral", index=d.index, dtype=str)
    bb_state[bb_upper.notna() & (close > bb_upper)] = "bearish"
    bb_state[bb_lower.notna() & (close < bb_lower)] = "bullish"

    # Volume state — direction matters: high-vol up = bullish, high-vol down = bearish
    price_change = close.diff()
    above_avg_vol = volume > vol_sma20
    vol_state = pd.Series("neutral", index=d.index, dtype=str)
    vol_state[above_avg_vol & (price_change > 0)] = "bullish"
    vol_state[above_avg_vol & (price_change < 0)] = "bearish"

    # Trend state vs 50 SMA
    trend_state = pd.Series("neutral", index=d.index, dtype=str)
    trend_state[sma50.notna() & (close > sma50)] = "bullish"
    trend_state[sma50.notna() & (close < sma50)] = "bearish"

    # Composite 3-out-of-5 signal
    states = pd.DataFrame({
        "rsi": rsi_state, "macd": macd_state, "bb": bb_state,
        "vol": vol_state, "trend": trend_state,
    })
    bullish_count = (states == "bullish").sum(axis=1)
    bearish_count = (states == "bearish").sum(axis=1)
    comp_signal = pd.Series("neutral", index=d.index, dtype=str)
    comp_signal[bullish_count >= 3] = "long"
    comp_signal[bearish_count >= 3] = "hedge"

    # Build chart data array
    chart = []
    for i in range(len(d)):
        chart.append({
            "date": d["date"].iloc[i].strftime("%Y-%m-%d"),
            "close": _f(close.iloc[i]),
            "volume": int(volume.iloc[i]),
            "rsi": _f(rsi.iloc[i]),
            "macd_line": _f(macd_line.iloc[i]),
            "signal_line": _f(signal_line.iloc[i]),
            "bb_upper": _f(bb_upper.iloc[i]),
            "bb_lower": _f(bb_lower.iloc[i]),
            "bb_basis": _f(bb_basis.iloc[i]),
            "sma50": _f(sma50.iloc[i]),
            "vol_sma20": _f(vol_sma20.iloc[i]),
            "rsi_state": rsi_state.iloc[i],
            "macd_state": macd_state.iloc[i],
            "bb_state": bb_state.iloc[i],
            "vol_state": vol_state.iloc[i],
            "trend_state": trend_state.iloc[i],
            "signal": comp_signal.iloc[i],
        })

    last = chart[-1]
    rsi_val = rsi.iloc[-1]
    ml = macd_line.iloc[-1]
    sl = signal_line.iloc[-1]
    vol_sma_val = vol_sma20.iloc[-1]
    sma50_val = sma50.iloc[-1]

    return {
        "symbol": symbol,
        "as_of": d["date"].iloc[-1].strftime("%b %d, %Y"),
        "row_count": len(d),
        "latest": {
            "signal": last["signal"],
            "close": _f(close.iloc[-1]),
            "indicators": {
                "rsi": {
                    "state": last["rsi_state"],
                    "value": f"{rsi_val:.2f}" if _f(rsi_val) is not None else "...",
                },
                "macd": {
                    "state": last["macd_state"],
                    "value": f"L:{ml:.2f}, S:{sl:.2f}" if (_f(ml) is not None and _f(sl) is not None) else "...",
                },
                "bb": {
                    "state": last["bb_state"],
                    "value": (
                        f"P:{last['close']}, B:{last['bb_lower']:.2f}–{last['bb_upper']:.2f}"
                        if last["bb_upper"] else "..."
                    ),
                },
                "vol": {
                    "state": last["vol_state"],
                    "value": (
                        f"V:{last['volume']:,}, A:{int(vol_sma_val):,}"
                        if _f(vol_sma_val) is not None else "..."
                    ),
                },
                "trend": {
                    "state": last["trend_state"],
                    "value": (
                        f"P:{last['close']}, A:{sma50_val:.2f}"
                        if _f(sma50_val) is not None else "..."
                    ),
                },
            },
        },
        "chart": chart,
    }
