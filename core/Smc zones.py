# core/smc_zones.py
# v5.9.0 — Support/Resistance levels, Order Blocks, Supply/Demand zones and
# CRT (Candle Range Theory), computed straight from OHLC candles.
#
# These are CONFIRMATIONS ONLY. They are shown on the Telegram alert and
# recorded with every signal so /api/stats?breakdown=true can show whether
# signals that line up with them actually win more often. Nothing here can
# block or create a trade.
#
# THE RULES (chosen by me, written down here so they can be changed):
#
#  Swing point   a candle whose high (or low) beats the 3 candles on each side.
#  ATR           14-candle average true range of the completed candles.
#
#  Support / Resistance
#                swing highs and lows within max(0.5 ATR, 0.07% of price) of
#                each other are merged into one level. A level needs 2 or more
#                touches to count. Aligned = price is within 1 ATR of a level
#                on the correct side (support under a BUY, resistance over a SELL).
#
#  Order Block   a break of structure (a close beyond the most recent swing
#                high/low) driven by an impulse that travels 1.5 ATR or more.
#                The block is the LAST opposite-colour candle at the origin of
#                that impulse (down candle for a bullish block, up candle for a
#                bearish one). Zone = that candle's full high-low range.
#                Dead once any candle closes through the far side.
#                "Fresh" = price has not touched the zone since it formed.
#
#  Supply/Demand 1-3 small candles (each 0.9 ATR or less) = the base, followed by
#                one strong departure candle (range 1.5 ATR or more, body 60%+
#                of range) that closes beyond the base. Demand = departs up,
#                supply = departs down. Same dead/fresh rules as order blocks.
#
#  CRT (1H)      C1 = a completed 1H candle (the range). C2 = the next candle
#                sweeps ONE side of C1's range and closes back inside it.
#                Sweep of the low -> bullish, target = C1 high.
#                Sweep of the high -> bearish, target = C1 low.
#                "confirmed" = C2 is complete. "forming" = the live 1H candle
#                is the one sweeping. Aligned = same direction as the trade and
#                price still has room (more than 0.5 ATR) before the target.
#
#  A zone counts as "inside" when price is within 0.25 ATR of it.
#  Obstacle      the nearest zone or level standing in the way of the trade
#                (supply / bearish block / resistance over a BUY, and the
#                mirror for a SELL). "Near" = under 1 ATR away.

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

SWING_N = 3
DISPLACEMENT_ATR = 1.5
MAX_BARS = 300
ZONE_PAD_ATR = 0.25


def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    prev_c = df["close"].astype(float).shift(1)
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=5).mean()


def swing_points(df: pd.DataFrame, n: int = SWING_N) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    highs, lows = [], []
    for i in range(n, len(df) - n):
        if h[i] > h[i - n:i].max() and h[i] >= h[i + 1:i + n + 1].max():
            highs.append((i, float(h[i])))
        if l[i] < l[i - n:i].min() and l[i] <= l[i + 1:i + n + 1].min():
            lows.append((i, float(l[i])))
    return highs, lows


def find_levels(df: pd.DataFrame, atr_last: float, price: float):
    """Returns (supports, resistances): levels with 2+ touches, nearest first."""
    highs, lows = swing_points(df)
    prices = sorted([p for _, p in highs] + [p for _, p in lows])
    if not prices:
        return [], []
    tol = max(0.5 * atr_last, price * 0.0007)
    clusters = [[prices[0]]]
    for p in prices[1:]:
        if p - clusters[-1][-1] <= tol:
            clusters[-1].append(p)
        else:
            clusters.append([p])
    levels = [{"price": float(np.mean(c)), "touches": len(c)} for c in clusters if len(c) >= 2]
    supports = sorted([x for x in levels if x["price"] <= price], key=lambda x: price - x["price"])
    resistances = sorted([x for x in levels if x["price"] > price], key=lambda x: x["price"] - price)
    return supports, resistances


def _finish_zones(zones: List[Dict], df: pd.DataFrame) -> List[Dict]:
    """Drops dead zones, marks fresh/touched, adds age."""
    c = df["close"].to_numpy(float)
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    out = []
    for z in zones:
        start = z["formed"] + 1
        bullish = z["type"] in ("bull", "demand")
        if bullish:
            broken = bool((c[start:] < z["low"]).any())
        else:
            broken = bool((c[start:] > z["high"]).any())
        if broken:
            continue
        touched = int(((l[start:] <= z["high"]) & (h[start:] >= z["low"])).sum())
        z["touches"] = touched
        z["fresh"] = touched == 0
        z["age"] = len(df) - 1 - z["formed"]
        out.append(z)
    return out


def _origin_candle(o, c, t: int, bullish: bool) -> Optional[int]:
    """
    Walk back through the impulse leg (up to 5 same-colour candles ending at t).
    The order block is the opposite-colour candle just before that leg — the
    LAST down candle before a bullish impulse, the last up candle before a
    bearish one.
    """
    k, steps = t, 0
    while k - 1 >= 0 and steps < 5 and ((c[k - 1] > o[k - 1]) if bullish else (c[k - 1] < o[k - 1])):
        k -= 1
        steps += 1
    j = k - 1
    if j >= 0 and ((c[j] < o[j]) if bullish else (c[j] > o[j])):
        return j
    return None


def find_order_blocks(df: pd.DataFrame, atr: pd.Series) -> List[Dict]:
    o = df["open"].to_numpy(float)
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    A = atr.to_numpy(float)
    highs, lows = swing_points(df)
    n = SWING_N
    found: Dict[Tuple[str, int], Dict] = {}
    for t in range(n + 2, len(df)):
        a = A[t - 1]
        if not np.isfinite(a) or a <= 0:
            continue
        sh = [p for i, p in highs if i + n <= t - 1]
        sl = [p for i, p in lows if i + n <= t - 1]
        if sh and c[t] > o[t] and c[t] > sh[-1] >= c[t - 1]:       # bullish candle breaks structure up
            j = _origin_candle(o, c, t, bullish=True)
            if j is not None and h[t] - l[j] >= DISPLACEMENT_ATR * a:
                found.setdefault(("bull", j), {"type": "bull", "low": float(l[j]),
                                               "high": float(h[j]), "idx": j, "formed": t})
        if sl and c[t] < o[t] and c[t] < sl[-1] <= c[t - 1]:       # bearish candle breaks structure down
            j = _origin_candle(o, c, t, bullish=False)
            if j is not None and h[j] - l[t] >= DISPLACEMENT_ATR * a:
                found.setdefault(("bear", j), {"type": "bear", "low": float(l[j]),
                                               "high": float(h[j]), "idx": j, "formed": t})
    return _finish_zones(list(found.values()), df)


def find_supply_demand(df: pd.DataFrame, atr: pd.Series) -> List[Dict]:
    o = df["open"].to_numpy(float)
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    A = atr.to_numpy(float)
    found: Dict[Tuple[str, int], Dict] = {}
    for i in range(2, len(df)):
        a = A[i - 1]
        if not np.isfinite(a) or a <= 0:
            continue
        rng = h[i] - l[i]
        body = abs(c[i] - o[i])
        if rng < DISPLACEMENT_ATR * a or body < 0.6 * rng:
            continue
        base = []
        k = i - 1
        while k >= 0 and len(base) < 3 and (h[k] - l[k]) <= 0.9 * a:
            base.append(k)
            k -= 1
        if not base:
            continue
        zl = min(l[b] for b in base)
        zh = max(h[b] for b in base)
        if zh - zl > 2.5 * a:
            continue
        if c[i] > o[i] and c[i] > zh:
            found.setdefault(("demand", base[-1]), {"type": "demand", "low": float(zl),
                                                    "high": float(zh), "idx": base[-1], "formed": i})
        elif c[i] < o[i] and c[i] < zl:
            found.setdefault(("supply", base[-1]), {"type": "supply", "low": float(zl),
                                                    "high": float(zh), "idx": base[-1], "formed": i})
    return _finish_zones(list(found.values()), df)


def crt_setup(df_htf: Optional[pd.DataFrame]) -> Optional[Dict]:
    """CRT on the higher-timeframe candles (1H). The last row is the live candle."""
    if df_htf is None or len(df_htf) < 30:
        return None
    d = df_htf.reset_index(drop=True)
    a = float(_atr(d.iloc[:-1]).iloc[-1])
    if not np.isfinite(a) or a <= 0:
        return None
    live, c2, c1 = d.iloc[-1], d.iloc[-2], d.iloc[-3]

    def check(rng_c, sweep_c, state):
        H, L = float(rng_c["high"]), float(rng_c["low"])
        if H - L < 0.5 * a:
            return None
        swept_low = float(sweep_c["low"]) < L
        swept_high = float(sweep_c["high"]) > H
        if swept_low == swept_high:               # swept neither side, or both = no clean setup
            return None
        if not (L <= float(sweep_c["close"]) <= H):   # must close back inside the range
            return None
        if swept_low:
            return {"direction": "BUY", "state": state, "range_high": H, "range_low": L,
                    "target": H, "swept": "low", "sweep_extreme": float(sweep_c["low"])}
        return {"direction": "SELL", "state": state, "range_high": H, "range_low": L,
                "target": L, "swept": "high", "sweep_extreme": float(sweep_c["high"])}

    return check(c1, c2, "confirmed") or check(c2, live, "forming")


def _fmt(x: float) -> str:
    return f"{x:,.0f}" if abs(x) >= 100 else (f"{x:.2f}" if abs(x) >= 10 else f"{x:.5f}")


_EMPTY = {"available": False, "confirmations": 0, "sr_aligned": False, "ob_aligned": False,
          "sd_aligned": False, "crt_aligned": False, "crt_state": None, "crt_zone": None,
          "obstacle_atr": None, "obstacle_near": False}


def analyze_smc(df_ltf: Optional[pd.DataFrame], df_htf: Optional[pd.DataFrame],
                direction: str, _raise: bool = False) -> Tuple[Dict, List[str]]:
    """
    df_ltf: the 5m candles (last row = still-forming candle, price = its close).
    df_htf: the 1H candles (last row = live candle), or None.
    Returns (features, lines). Never raises unless _raise=True (used by tests).
    """
    try:
        if df_ltf is None or len(df_ltf) < 80:
            return dict(_EMPTY), []
        price = float(df_ltf["close"].iloc[-1])
        d = df_ltf.iloc[:-1].tail(MAX_BARS).reset_index(drop=True)   # completed candles only
        atr_s = _atr(d)
        a = float(atr_s.iloc[-1])
        if not np.isfinite(a) or a <= 0:
            return dict(_EMPTY), []

        buy = direction == "BUY"
        pad = ZONE_PAD_ATR * a
        supports, resistances = find_levels(d, a, price)
        obs = find_order_blocks(d, atr_s)
        sds = find_supply_demand(d, atr_s)

        want_ob, opp_ob = ("bull", "bear") if buy else ("bear", "bull")
        want_sd, opp_sd = ("demand", "supply") if buy else ("supply", "demand")

        def inside(z):
            return z["low"] - pad <= price <= z["high"] + pad

        def rank(z):
            return (not z["fresh"], z["age"])

        ob_in = sorted([z for z in obs if z["type"] == want_ob and inside(z)], key=rank)
        sd_in = sorted([z for z in sds if z["type"] == want_sd and inside(z)], key=rank)
        lvl_pool = supports if buy else resistances
        sr = next((x for x in lvl_pool
                   if ((price - x["price"]) if buy else (x["price"] - price)) <= a), None)

        crt = crt_setup(df_htf)
        crt_ok, crt_zone, crt_room = False, None, None
        if crt:
            mid = (crt["range_high"] + crt["range_low"]) / 2
            crt_zone = "discount" if price < mid else "premium"
            crt_room = (crt["target"] - price) if crt["direction"] == "BUY" else (price - crt["target"])
            crt_ok = crt["direction"] == direction and crt_room > 0.5 * a

        # nearest obstacle standing between price and the target
        cands: List[Tuple[str, float, float]] = []      # (label, level, distance in price)
        for z in obs:
            if z["type"] == opp_ob:
                lab = "Bearish order block" if buy else "Bullish order block"
                if buy and z["high"] >= price - pad:
                    cands.append((lab, z["low"], max(0.0, z["low"] - price)))
                if (not buy) and z["low"] <= price + pad:
                    cands.append((lab, z["high"], max(0.0, price - z["high"])))
        for z in sds:
            if z["type"] == opp_sd:
                lab = "Supply zone" if buy else "Demand zone"
                if buy and z["high"] >= price - pad:
                    cands.append((lab, z["low"], max(0.0, z["low"] - price)))
                if (not buy) and z["low"] <= price + pad:
                    cands.append((lab, z["high"], max(0.0, price - z["high"])))
        for x in (resistances if buy else supports):
            cands.append(("Resistance" if buy else "Support", x["price"], abs(x["price"] - price)))
        obstacle = min(cands, key=lambda t: t[2]) if cands else None
        obstacle_atr = round(obstacle[2] / a, 2) if obstacle else None
        obstacle_near = bool(obstacle_atr is not None and obstacle_atr < 1.0)

        flags = [sr is not None, bool(ob_in), bool(sd_in), crt_ok]
        feats = {
            "available": True,
            "confirmations": int(sum(flags)),
            "sr_aligned": bool(sr is not None),
            "ob_aligned": bool(ob_in),
            "sd_aligned": bool(sd_in),
            "crt_aligned": bool(crt_ok),
            "crt_state": (f"{crt['direction']}_{crt['state']}" if crt else None),
            "crt_zone": crt_zone,
            "obstacle_atr": obstacle_atr,
            "obstacle_near": obstacle_near,
        }

        lines = [f"🔎 SMC confirmations: {feats['confirmations']}/4"]
        if ob_in:
            z = ob_in[0]
            lines.append(f"🧱 In {'bullish' if buy else 'bearish'} order block {_fmt(z['low'])}–{_fmt(z['high'])} "
                         f"({'fresh' if z['fresh'] else 'retested'})")
        if sd_in:
            z = sd_in[0]
            lines.append(f"{'🟩 Demand' if buy else '🟥 Supply'} zone {_fmt(z['low'])}–{_fmt(z['high'])} "
                         f"({'fresh' if z['fresh'] else 'retested'})")
        if crt_ok:
            lines.append(f"🕯 CRT (1H, {crt['state']}): swept the {crt['swept']}, back inside — "
                         f"target {_fmt(crt['target'])} ({_fmt(crt_room)} pts away, price in {crt_zone})")
        if sr is not None:
            lines.append(f"📍 {'Support' if buy else 'Resistance'} {_fmt(sr['price'])} "
                         f"({sr['touches']} touches), {abs(price - sr['price']) / a:.1f}x ATR away")
        if obstacle_near:
            lines.append(f"⚠️ {obstacle[0]} {_fmt(obstacle[1])} only {obstacle_atr:.1f}x ATR ahead — in the way")
        return feats, lines
    except Exception:
        if _raise:
            raise
        return dict(_EMPTY), []