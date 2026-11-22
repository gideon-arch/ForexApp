# core/gainz_mlsmc.py
# v5.14.0 — Python port of "Machine Learning Smart Money Concepts | GainzAlgo" (Pine v6,
# Mozilla Public License 2.0, source supplied by the user). ALERT-ONLY.
#
# WHAT THE INDICATOR DOES (ported line by line):
#   * pivots (5 bars each side) give the last swing high / swing low
#   * a CHoCH ("change of character") fires when price CLOSES through the last swing high
#     while the trend state is neutral/bearish (bullish CHoCH), or through the last swing
#     low while neutral/bullish (bearish CHoCH). It alternates, one per structure flip.
#   * at each CHoCH it measures 3 raw features: average volume delta over the leg (max 50
#     bars), displacement (leg size / ATR) and velocity (leg size / bars)
#   * "memory": every past CHoCH is stored once it is 20 bars old, labelled with what
#     happened in those 20 bars (did price run further in the CHoCH direction than against
#     it?) and how far the best run went
#   * the K=5 most similar past CHoCHs in the SAME direction (plain Euclidean distance on
#     the 3 raw, un-normalised features, only those within the last 1500 bars) vote:
#     score = % of them that followed through. TP1/TP2/TP3 = their mean x 0.5 / median /
#     75th percentile run, measured from the CHoCH close.
#   * the indicator's own "valid setup" bar is score >= 60
#
# WHAT IS ADDED BY THE BOT (not in the indicator):
#   * the indicator has NO stop-loss. Here (GAINZ_MLSMC_STOP):
#       "level" (default) = 1 x ATR beyond the swing level that was just broken (the retest level)
#       "swing"           = 0.1 x ATR beyond the OPPOSITE swing (the whole leg; usually 4-7 x ATR away,
#                           so most signals are skipped by the 4 x ATR risk cap)
#     the tracker's target is 2R by default (GAINZ_MLSMC_TP = 2R | TP1 | TP2 | TP3). The
#     indicator's own three targets are shown on the alert.
#   * a signal needs all 5 neighbours (GAINZ_MLSMC_MIN_MATCHES) so a score is never built on 1-2 past events
#   * only completed candles are used; the forming one is dropped
#   * memory: the bot only gets ~720 candles per request, so every matured CHoCH it sees is
#     saved in Redis and fed back in as "seeds" next time. The indicator only looks 1500
#     bars back anyway, so after ~1500 bars of running (about 15 days on 15m) the memory is
#     the same depth as on the chart. Until then scores rest on fewer past events.
#   * the first 100 candles of each run are a warm-up (structure state is unknown): no
#     events are created there, but saved seeds cover that stretch.
#
# FIDELITY NOTES: pivots need the centre bar strictly higher/lower than all 5 bars on each
# side (TradingView's treatment of exact ties may differ in rare cases). The features are
# NOT normalised in the original, so price "velocity" dominates the distance; that is
# reproduced as-is. Compare against the free script on a TradingView chart.

import math
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .gainz_arbm import atr_rma, _naive_utc

DEFAULTS = dict(lookahead=20, window_len=1500, k=5, atr_len=14, swing_len=5, target_scalar=0.5, db_cap=2000, warmup=100)


def vol_delta(h: float, l: float, c: float, v: float) -> float:
    if not (v > 0):
        return 0.0
    rng = max(h - l, 1e-6)
    return ((c - l) - (h - c)) / rng            # (buying - selling) / volume, volume cancels out


def is_pivot(arr, center: int, left: int, right: int, high: bool = True) -> bool:
    if center - left < 0 or center + right >= len(arr):
        return False
    v = arr[center]
    lo, hi = arr[center - left:center], arr[center + 1:center + right + 1]
    return bool(v > lo.max() and v > hi.max()) if high else bool(v < lo.min() and v < hi.min())


def knn_predict(db: List[Dict], vol_f: float, disp_f: float, vel_f: float, is_bull: bool, bar: int,
                window_len: int, k: int, scalar: float, close: float) -> Dict:
    out = dict(score=50.0, matches=0, successes=0, tp1=0.0, tp2=0.0, tp3=0.0)
    if not db:
        return out
    cand = [(math.sqrt((vol_f - r["vol"]) ** 2 + (disp_f - r["disp"]) ** 2 + (vel_f - r["vel"]) ** 2), i)
            for i, r in enumerate(db) if bar - r["bar"] <= window_len and r["bull"] == is_bull]
    if not cand:
        return out
    n = len(cand)
    lim = min(k, n)
    used = [False] * n
    picked = []
    for _ in range(lim):                         # same repeated min-selection as the Pine code (first of equals wins)
        best, best_i = 1e38, 0
        for s in range(n):
            if not used[s] and cand[s][0] < best:
                best, best_i = cand[s][0], s
        used[best_i] = True
        picked.append(best_i)
    runs, succ = [], 0
    for s in picked:
        rec = db[cand[s][1]]
        succ += 1 if rec["outcome"] > 0 else 0
        runs.append(rec["run"])
    runs.sort()
    nr = len(runs)
    med = runs[int(nr / 2)]
    mean = sum(runs) / nr
    p75 = min((int(math.floor(nr * 0.75 + 0.5)) - 1) if nr > 1 else 0, nr - 1)   # Pine math.round: halves round up
    d = 1.0 if is_bull else -1.0
    out.update(score=succ / lim * 100.0, matches=lim, successes=succ,
               tp1=close + d * mean * scalar, tp2=close + d * med, tp3=close + d * runs[p75])
    return out


def run_indicator(df: pd.DataFrame, *, timeframe_min: float = 15.0, seeds: Optional[List[Dict]] = None, **kw) -> Dict:
    p = {**DEFAULTS, **kw}
    d = df.reset_index(drop=True)
    o, h, l, c = (d[x].to_numpy(float) for x in ("open", "high", "low", "close"))
    v = d["volume"].to_numpy(float) if "volume" in d else np.zeros(len(d))
    n = len(c)
    ts = [pd.Timestamp(x) for x in d["timestamp"]]
    atr = atr_rma(h, l, c, p["atr_len"])
    vd = np.array([vol_delta(h[i], l[i], c[i], v[i]) for i in range(n)])
    L = p["swing_len"]
    la = p["lookahead"]
    step = timeframe_min * 60.0
    t0 = ts[0]

    db: List[Dict] = []
    pending: Dict[int, List[Dict]] = {}
    for s in sorted(seeds or [], key=lambda s: s["ts"]):          # saved memory from earlier runs
        rel = int(round((pd.Timestamp(s["ts"]) - t0).total_seconds() / step))
        if rel >= p["warmup"]:
            continue                                               # this stretch is recomputed below
        rec = dict(bar=rel, vol=float(s["vol"]), disp=float(s["disp"]), vel=float(s["vel"]),
                   bull=bool(s["bull"]), outcome=float(s["outcome"]), run=float(s["run"]))
        (db if rel + la <= 0 else pending.setdefault(rel + la, [])).append(rec)

    trend = 0
    last_sh = last_sl = None
    sh_idx = sl_idx = None
    events: Dict[int, Dict] = {}
    new_matured: List[Dict] = []
    for t in range(n):
        ctr = t - L
        if ctr - L >= 0:
            if is_pivot(h, ctr, L, L, True):
                last_sh, sh_idx = float(h[ctr]), ctr
            if is_pivot(l, ctr, L, L, False):
                last_sl, sl_idx = float(l[ctr]), ctr
        bull = bear = False
        if last_sh is not None and trend <= 0 and c[t] > last_sh:
            bull, trend = True, 1
        if last_sl is not None and trend >= 0 and c[t] < last_sl:
            bear, trend = True, -1
        if (bull or bear) and t >= p["warmup"]:
            is_bull = bull
            start = sh_idx if is_bull else sl_idx
            duration = min(max(1, t - start), 4990)
            back = min(duration, 50)
            if t - (back - 1) >= 0:
                vol_f = float(vd[t - back + 1:t + 1].sum()) / max(1, back)
                total = abs(c[t] - o[t - duration])
                a = atr[t]
                disp = total / a if (np.isfinite(a) and a > 0) else 1.0
                vel = total / duration
                pred = knn_predict(db, vol_f, disp, vel, is_bull, t, p["window_len"], p["k"], p["target_scalar"], float(c[t]))
                events[t] = dict(bar=t, ts=ts[t].isoformat(), bull=is_bull, close=float(c[t]), atr=float(a) if np.isfinite(a) else float("nan"),
                                 swing_low=last_sl, swing_high=last_sh, both=bool(bull and bear), vol=vol_f, disp=float(disp), vel=float(vel), db_size=len(db), **pred)
        te = t - la                                                # an event that is now `lookahead` bars old becomes memory
        if te >= 0 and te in events:
            ev = events[te]
            ref = c[te]
            fav = adv = 0.0
            for j in range(te + 1, t + 1):
                if ev["bull"]:
                    fav, adv = max(fav, h[j] - ref), max(adv, ref - l[j])
                else:
                    fav, adv = max(fav, ref - l[j]), max(adv, h[j] - ref)
            rec = dict(bar=te, vol=ev["vol"], disp=ev["disp"], vel=ev["vel"], bull=ev["bull"], outcome=1.0 if fav > adv else -1.0, run=fav)
            db.append(rec)
            new_matured.append(dict(ts=ev["ts"], vol=ev["vol"], disp=ev["disp"], vel=ev["vel"], bull=ev["bull"], outcome=rec["outcome"], run=fav))
            if len(db) > p["db_cap"]:
                db.pop(0)
        for rec in pending.pop(t, []):
            db.append(rec)
            if len(db) > p["db_cap"]:
                db.pop(0)
    return {"events": events, "new_matured": new_matured, "db_size": len(db), "trend": trend}


def evaluate(df: Optional[pd.DataFrame], now, *, timeframe_min: float, seeds: Optional[List[Dict]] = None,
             min_score: float = 60.0, min_matches: int = 5, max_age_min: float = 10.0, tp_pick: str = "2R",
             min_risk_atr: float = 0.3, max_risk_atr: float = 4.0, stop_mode: str = "level", **kw) -> Dict:
    """status = no_data | no_event | stale | thin_memory | low_score | no_stop | too_tight | too_wide | signal.
    Every result carries new_matured (events to save as memory) except no_data."""
    if df is None or len(df) < 150:
        return {"status": "no_data", "why": "not enough candles"}
    d = df.iloc[:-1].reset_index(drop=True)                        # completed candles only
    res = run_indicator(d, timeframe_min=timeframe_min, seeds=seeds, **kw)
    base = {"new_matured": res["new_matured"], "db_size": res["db_size"]}
    ev = res["events"].get(len(d) - 1)
    if ev is None:
        return {**base, "status": "no_event"}
    ts = pd.Timestamp(d["timestamp"].iloc[-1])
    bar_close = ts + pd.Timedelta(minutes=timeframe_min)
    age = (_naive_utc(now) - bar_close).total_seconds() / 60.0
    out = {**base, **ev, "bar_ts": ts.isoformat(), "bar_close": bar_close.isoformat(), "age_min": round(age, 1),
           "direction": "BUY" if ev["bull"] else "SELL", "entry": ev["close"]}
    if age > max_age_min:
        return {**out, "status": "stale"}
    if ev["matches"] < min_matches:
        return {**out, "status": "thin_memory"}
    if ev["score"] < min_score:
        return {**out, "status": "low_score"}
    a, entry = ev["atr"], ev["close"]
    if not np.isfinite(a) or a <= 0:
        return {**out, "status": "no_data", "why": "ATR unavailable"}
    mode = "swing" if stop_mode.lower() == "swing" else "level"
    if mode == "swing":
        swing, buf, sl_note = (ev["swing_low"] if ev["bull"] else ev["swing_high"]), 0.1 * a, "just beyond the opposite swing"
    else:
        swing, buf, sl_note = (ev["swing_high"] if ev["bull"] else ev["swing_low"]), 1.0 * a, "1 ATR beyond the level that was broken"
    sl = None if swing is None else (swing - buf if ev["bull"] else swing + buf)
    if sl is None or (ev["bull"] and sl >= entry) or ((not ev["bull"]) and sl <= entry):
        return {**out, "status": "no_stop"}
    risk = abs(entry - sl)
    out.update(sl=float(sl), risk=float(risk), risk_atr=round(risk / a, 2), stop_mode=mode, sl_note=sl_note)
    if risk < min_risk_atr * a:
        return {**out, "status": "too_tight"}
    if risk > max_risk_atr * a:
        return {**out, "status": "too_wide"}
    tp = entry + 2 * risk if ev["bull"] else entry - 2 * risk
    pick = tp_pick.upper()
    if pick in ("TP1", "TP2", "TP3"):
        cand = ev[pick.lower()]
        if (cand - entry) * (1 if ev["bull"] else -1) > 0:
            tp = cand
        else:
            pick = "2R"
    else:
        pick = "2R"
    word = "Bullish" if ev["bull"] else "Bearish"
    out.update(status="signal", tp=float(tp), tp_pick=pick,
               lines=[f"{word} CHoCH (change of structure) — price closed through the last swing {'high' if ev['bull'] else 'low'}",
                      f"Indicator score {ev['score']:.0f}%: {ev['successes']} of its {ev['matches']} most similar past breaks followed through",
                      f"Indicator's own targets: TP1 {ev['tp1']:,.2f} | TP2 {ev['tp2']:,.2f} | TP3 {ev['tp3']:,.2f}",
                      f"Memory: {ev['db_size']} past breaks on file"],
               context={"choch": word.lower(), "score": ev["score"], "matches": ev["matches"], "successes": ev["successes"],
                        "db_size": ev["db_size"], "tp1": ev["tp1"], "tp2": ev["tp2"], "tp3": ev["tp3"]})
    return out