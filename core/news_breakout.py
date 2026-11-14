# core/news_breakout.py
# v5.10.0 — POST-NEWS BREAKOUT setup (CPI by default). ALERT-ONLY: nothing in
# here can place, queue or even offer a trade. Signals are sent without Accept
# buttons and recorded under their own setup name so the outcome tracker can
# prove (or kill) the idea before any real money goes near it.
#
# THE RULES (mine — written down so they can be changed):
#
#   1. During the blackout the bot only WATCHES. It never trades inside the window.
#   2. The "reaction range" = the highest high and lowest low of the 1-minute
#      candles from the release (T) until the blackout ends (T + after).
#   3. Normal size = 5-minute ATR(14) measured BEFORE the blackout began.
#      If the reaction range is under 2.0 x ATR the news fizzled -> no setup.
#   4. After the blackout ends, the FIRST completed 1-minute candle that CLOSES
#      beyond the range (by at least 0.1 x ATR) is the breakout:
#         close above the range high -> BUY      close below the range low -> SELL
#   5. Entry = that candle's close. Stop = the MIDDLE of the reaction range (if
#      price drops back to the middle of the news range, the breakout failed).
#      Target = 2R (same 1:2 the rest of the bot uses, so results are comparable).
#   6. Skipped (and logged) when: the breakout candle closed more than 3 minutes
#      ago ("late"), it closed more than 1.0 x ATR beyond the level ("chased"),
#      the stop would be more than 3.0 x ATR away ("too wide"), or no breakout
#      happens within 60 minutes ("expired").
#   7. Each breakout is tagged WITH or AGAINST the first move (the net direction of
#      the reaction range), so the stats can show which one actually works.

import numpy as np
import pandas as pd
from typing import Dict, Optional


def _naive_utc(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return t.tz_convert("UTC").tz_localize(None) if t.tzinfo is not None else t


def _atr_5m_before(d: pd.DataFrame, cutoff: pd.Timestamp, n: int = 14) -> Optional[float]:
    hist = d[d["timestamp"] < cutoff]
    if len(hist) < 5 * (n + 2):
        return None
    h5 = (hist.set_index("timestamp")
              .resample("5min")
              .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
              .dropna())
    if len(h5) < n + 1:
        return None
    prev_c = h5["close"].shift(1)
    tr = pd.concat([h5["high"] - h5["low"], (h5["high"] - prev_c).abs(), (h5["low"] - prev_c).abs()], axis=1).max(axis=1)
    a = float(tr.tail(n).mean())
    return a if np.isfinite(a) and a > 0 else None


def evaluate(df: Optional[pd.DataFrame], event_time, after_min: float, now, *,
             before_min: float = 8.0, min_range_atr: float = 2.0, buffer_atr: float = 0.1,
             max_chase_atr: float = 1.0, max_risk_atr: float = 3.0,
             max_age_min: float = 3.0, watch_min: float = 60.0) -> Dict:
    """
    df: 1-minute candles (columns timestamp/open/high/low/close, naive UTC), the
        last row may still be forming.
    Returns a dict with "status" = waiting | no_data | fizzled | watching |
    breakout | late | chased | too_wide | expired. Only "breakout" carries a trade.
    """
    T = _naive_utc(event_time)
    now = _naive_utc(now)
    if df is None or len(df) < 100:
        return {"status": "no_data", "why": "not enough 1m candles"}

    d = df[["timestamp", "open", "high", "low", "close"]].copy()
    d["timestamp"] = pd.to_datetime(d["timestamp"])
    d = d.sort_values("timestamp").reset_index(drop=True)
    d = d[d["timestamp"] + pd.Timedelta(minutes=1) <= now]          # completed candles only

    T0 = T.floor("min")
    react_end = T0 + pd.Timedelta(minutes=after_min)
    if now < react_end:
        return {"status": "waiting", "why": "blackout still running"}

    react = d[(d["timestamp"] >= T0) & (d["timestamp"] < react_end)]
    pre = d[d["timestamp"] < T0].tail(1)
    if len(react) < max(3, int(after_min * 0.75)) or pre.empty:
        return {"status": "no_data", "why": "candles missing around the release"}
    atr = _atr_5m_before(d, T0 - pd.Timedelta(minutes=before_min))
    if atr is None:
        return {"status": "no_data", "why": "not enough history for ATR"}

    rh, rl = float(react["high"].max()), float(react["low"].min())
    rng = rh - rl
    pre_close = float(pre["close"].iloc[0])
    impulse = float(react["close"].iloc[-1]) - pre_close
    first_move = "up" if impulse > 0 else ("down" if impulse < 0 else "flat")
    base = {"atr": atr, "range_high": rh, "range_low": rl, "range_atr": round(rng / atr, 2),
            "pre_close": pre_close, "first_move": first_move}
    if rng < min_range_atr * atr:
        return {**base, "status": "fizzled"}

    watch_end = react_end + pd.Timedelta(minutes=watch_min)
    buf = buffer_atr * atr
    mid = (rh + rl) / 2
    for _, row in d[d["timestamp"] >= react_end].iterrows():
        if row["timestamp"] >= watch_end:
            break
        c = float(row["close"])
        if c > rh + buf:
            direction, level = "BUY", rh
        elif c < rl - buf:
            direction, level = "SELL", rl
        else:
            continue
        close_time = row["timestamp"] + pd.Timedelta(minutes=1)
        out = {**base, "direction": direction, "entry": c, "level": level,
               "breakout_close_time": close_time.isoformat()}
        if (now - close_time) > pd.Timedelta(minutes=max_age_min):
            return {**out, "status": "late"}
        if abs(c - level) > max_chase_atr * atr:
            return {**out, "status": "chased"}
        risk = abs(c - mid)
        if risk > max_risk_atr * atr:
            return {**out, "status": "too_wide"}
        tp = c + 2 * risk if direction == "BUY" else c - 2 * risk
        with_first = (direction == "BUY" and first_move == "up") or (direction == "SELL" and first_move == "down")
        return {**out, "status": "breakout", "sl": mid, "tp": tp, "risk": risk, "with_first_move": with_first}

    return {**base, "status": "expired" if now >= watch_end else "watching"}