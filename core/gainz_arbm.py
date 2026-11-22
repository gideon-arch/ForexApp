# core/gainz_arbm.py
# v5.13.0 — Python port of "Adaptive Regression Breakout Map | GainzAlgo" (Pine v6,
# Mozilla Public License 2.0, source supplied by the user). ALERT-ONLY.
#
# WHAT THE INDICATOR DOES (ported line by line):
#   * regression channel: linreg(close, 50) +/- 2 x stdev(close, 50)   (stdev = population, Pine default)
#   * bandwidth = upper - lower; "squeeze" = bandwidth within the lowest 20% of its own
#     last-100-bar range:  (bw - lowest100) / (highest100 - lowest100) * 100 <= 20
#   * LONG trigger : close crosses ABOVE the upper band while the PREVIOUS bar was in a squeeze
#     SHORT trigger: close crosses BELOW the lower band while the PREVIOUS bar was in a squeeze
#   * on a trigger (only if no trade is already open in the indicator's own state):
#         entry = close | stop = opposite band of the previous bar |
#         TP1/TP2/TP3 = entry +/- previous bar's bandwidth x 1 / 2 / 3
#   * the indicator then manages that trade bar by bar (TP1 -> stop to entry, TP2 -> stop to TP1,
#     TP3 or stop -> closed). A new signal can only appear once the previous trade is closed,
#     so the whole state machine is replayed over the candles we have.
#
# WHAT IS ADDED BY THE BOT (not in the indicator):
#   * only completed candles are used (the forming one is dropped) -> no repainting
#   * the tracker needs ONE target, so it records TP2 by default (GAINZ_ARBM_TP=TP1|TP2|TP3);
#     all three are shown on the alert
#   * stops closer than 0.3 x ATR or further than 4 x ATR are skipped (ATR is the bot's, not the indicator's)
#
# FIDELITY NOTES: the replay starts at the first candle we have (about 700 bars) instead of the
# start of the chart's history, so the very first trades of the window can differ from TradingView.
# After the first closed trade the state is identical. Compare against the free script on a
# TradingView chart to see signals on the same candles.

import numpy as np
import pandas as pd
from typing import Dict, Optional

DEFAULTS = dict(length=50, dev_mult=2.0, sqz_len=100, sqz_pct=20.0, tp1_mult=1.0, tp2_mult=2.0, tp3_mult=3.0)


def atr_rma(h, l, c, n: int = 14):
    """Wilder ATR (Pine's ta.atr): RMA of true range, seeded with the simple mean of the first n."""
    size = len(h)
    out = np.full(size, np.nan)
    if size < n:
        return out
    tr = np.empty(size)
    tr[0] = h[0] - l[0]
    for i in range(1, size):
        tr[i] = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
    out[n - 1] = tr[:n].mean()
    for i in range(n, size):
        out[i] = (out[i - 1] * (n - 1) + tr[i]) / n
    return out


def compute_series(close, p) -> Dict[str, np.ndarray]:
    L = int(p["length"])
    n = len(close)
    reg = np.full(n, np.nan)
    std = np.full(n, np.nan)
    if n >= L:
        x = np.arange(L, dtype=float)
        xm = x.mean()
        sxx = ((x - xm) ** 2).sum()
        for i in range(L - 1, n):
            w = close[i - L + 1:i + 1]
            ym = w.mean()
            slope = ((x - xm) * (w - ym)).sum() / sxx
            reg[i] = ym + slope * (L - 1 - xm)       # value of the fitted line at the newest bar (ta.linreg offset 0)
            std[i] = w.std()                          # population std (ta.stdev biased=true)
    upper = reg + p["dev_mult"] * std
    lower = reg - p["dev_mult"] * std
    bw = upper - lower
    s = pd.Series(bw)
    mn = s.rolling(int(p["sqz_len"])).min().values
    mx = s.rolling(int(p["sqz_len"])).max().values
    width_pct = (bw - mn) / np.maximum(mx - mn, 0.000001) * 100.0
    squeeze = np.where(np.isnan(width_pct), False, width_pct <= p["sqz_pct"])
    return {"upper": upper, "lower": lower, "bw": bw, "width_pct": width_pct, "squeeze": squeeze}


def simulate(h, l, c, s, p) -> Dict:
    """Replay the indicator's trade state machine. Returns {"signals": {bar_index: {...}}, "trade_dir": int}."""
    n = len(c)
    up, lo, bw, sq = s["upper"], s["lower"], s["bw"], s["squeeze"]
    signals: Dict[int, Dict] = {}
    td, entry, sl, t1, t2, t3 = 0, np.nan, np.nan, np.nan, np.nan, np.nan
    h1 = h2 = False
    for i in range(1, n):
        if td == 0 and np.isfinite(up[i]) and np.isfinite(up[i - 1]) and np.isfinite(bw[i - 1]):
            bull = c[i] > up[i] and c[i - 1] <= up[i - 1] and bool(sq[i - 1])
            bear = c[i] < lo[i] and c[i - 1] >= lo[i - 1] and bool(sq[i - 1])
            if bull or bear:
                w = bw[i - 1]
                d = 1 if bull else -1
                td, entry = d, float(c[i])
                sl = float(lo[i - 1] if bull else up[i - 1])
                t1, t2, t3 = (entry + d * w * p["tp1_mult"], entry + d * w * p["tp2_mult"], entry + d * w * p["tp3_mult"])
                h1 = h2 = False
                signals[i] = {"bar": i, "direction": "BUY" if bull else "SELL", "entry": entry, "sl": sl,
                              "tp1": float(t1), "tp2": float(t2), "tp3": float(t3), "bandwidth": float(w),
                              "width_pct": float(s["width_pct"][i - 1]), "upper": float(up[i]), "lower": float(lo[i])}
        if td != 0:                                       # same bar as the entry, exactly like the Pine code
            hit_sl = (td == 1 and l[i] <= sl) or (td == -1 and h[i] >= sl)
            hit1 = (td == 1 and h[i] >= t1) or (td == -1 and l[i] <= t1)
            hit2 = (td == 1 and h[i] >= t2) or (td == -1 and l[i] <= t2)
            hit3 = (td == 1 and h[i] >= t3) or (td == -1 and l[i] <= t3)
            if hit1 and not h1:
                h1, sl = True, entry
            if hit2 and not h2:
                h2, sl = True, t1
            if hit3:
                td = 0
            elif hit_sl:
                td = 0
    return {"signals": signals, "trade_dir": td}


def run_indicator(df: pd.DataFrame, **kw) -> Dict:
    p = {**DEFAULTS, **kw}
    h, l, c = (df[k].to_numpy(dtype=float) for k in ("high", "low", "close"))
    s = compute_series(c, p)
    res = simulate(h, l, c, s, p)
    res["series"] = s
    return res


def _naive_utc(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return t.tz_convert("UTC").tz_localize(None) if t.tzinfo is not None else t


def evaluate(df: Optional[pd.DataFrame], now, *, timeframe_min: float, max_age_min: float = 10.0,
             tp_pick: str = "TP2", min_risk_atr: float = 0.3, max_risk_atr: float = 4.0, **kw) -> Dict:
    """df: candles (timestamp/open/high/low/close, naive UTC); the last row may still be forming.
    status = no_data | no_event | stale | too_tight | too_wide | signal"""
    if df is None or len(df) < 220:
        return {"status": "no_data", "why": "not enough candles"}
    d = df.iloc[:-1].reset_index(drop=True)                  # completed candles only
    res = run_indicator(d, **kw)
    ev = res["signals"].get(len(d) - 1)
    if ev is None:
        return {"status": "no_event", "trade_dir": res["trade_dir"]}
    ts = pd.Timestamp(d["timestamp"].iloc[-1])
    bar_close = ts + pd.Timedelta(minutes=timeframe_min)
    age = (_naive_utc(now) - bar_close).total_seconds() / 60.0
    out = {**ev, "bar_ts": ts.isoformat(), "bar_close": bar_close.isoformat(), "age_min": round(age, 1)}
    if age > max_age_min:
        return {**out, "status": "stale"}
    atr = atr_rma(d["high"].to_numpy(float), d["low"].to_numpy(float), d["close"].to_numpy(float))[-1]
    if not np.isfinite(atr) or atr <= 0:
        return {**out, "status": "no_data", "why": "ATR unavailable"}
    risk = abs(ev["entry"] - ev["sl"])
    out.update(atr=float(atr), risk=float(risk), risk_atr=round(risk / atr, 2))
    if risk < min_risk_atr * atr:
        return {**out, "status": "too_tight"}
    if risk > max_risk_atr * atr:
        return {**out, "status": "too_wide"}
    tp = ev[{"TP1": "tp1", "TP2": "tp2", "TP3": "tp3"}.get(tp_pick.upper(), "tp2")]
    side = "above the upper band" if ev["direction"] == "BUY" else "below the lower band"
    out.update(status="signal", tp=float(tp), tp_pick=tp_pick.upper(),
               lines=[f"Squeeze breakout: closed {side} (channel was at the {ev['width_pct']:.0f}th percentile of its 100-bar width range)",
                      f"Indicator targets: TP1 {ev['tp1']:,.2f} | TP2 {ev['tp2']:,.2f} | TP3 {ev['tp3']:,.2f}"])
    return out