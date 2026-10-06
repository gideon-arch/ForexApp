# core/multi_timeframe.py
# Multi-timeframe alignment engine
# v5.1: Morning-bird boost + 2-layer fallback + bug fix (primary_signal NameError)
# v5.2: PER-ASSET THRESHOLD OVERRIDES — each asset can use its own gates
# v5.3: TIGHTER 2-LAYER GATE — primary must ACTIVELY agree (own signal matches
#       or strong trend), and entry layer must not oppose. Fixes v5.1 buying
#       the top on a WEAK lean (see Sept 13 incident: primary HOLD/WEAK_BULLISH
#       fired BUY right into a dump).
# v5.4: EARLY TREND-START LAYER — EMA9/21 cross + swing-structure break on the
#       entry TF catches trends at BIRTH instead of waiting for candle-pattern
#       confirmation (which lagged 1-2h: trend started ~7:00, signal ~9:00).
#       Also: TREND-START entries when primary is still SIDEWAYS but entry TF
#       structure has broken AND primary momentum shifted. Fully symmetric —
#       BUY and SELL use identical logic, so both pairs fire both directions.
# BTCUSD: 15m -> 5m -> 1m  |  V10(1s): 5m -> 1m -> 10s

import os
import hashlib
import logging
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd

from .technical_analysis import TechnicalAnalyzer

logger = logging.getLogger(__name__)


class SignalTracker:
    """Prevents duplicate signals for the same active setup."""

    def __init__(self, cooldown_minutes: int = 10, price_tolerance_pct: float = 0.005):
        self.cooldown = timedelta(minutes=cooldown_minutes)
        self.price_tolerance = price_tolerance_pct
        self.active_signals: Dict[str, dict] = {}

    def _make_setup_hash(self, support, resistance, trend: str, signal: str) -> str:
        s = round(support, 2) if support else 0
        r = round(resistance, 2) if resistance else 0
        raw = f"{signal}_{trend}_{s}_{r}"
        return hashlib.md5(raw.encode()).hexdigest()[:12]

    def is_duplicate(self, pair: str, direction: str, setup_hash: str,
                     entry_price: float) -> Tuple[bool, str]:
        key = pair.upper()
        if key not in self.active_signals:
            return False, "No active signal"

        active = self.active_signals[key]
        elapsed = datetime.utcnow() - active["timestamp"]

        if elapsed < self.cooldown:
            if active["direction"] != direction:
                return False, f"Different direction (active: {active['direction']})"

            if active["setup_hash"] == setup_hash:
                price_diff = abs(entry_price - active["entry_price"]) / active["entry_price"]
                if price_diff < self.price_tolerance:
                    return True, f"Duplicate: same setup active for {elapsed.seconds // 60}m"
                else:
                    return False, f"Price moved {price_diff * 100:.2f}% — new setup"
            else:
                return False, "Setup changed — allowing new signal"

        return False, "Cooldown expired"

    def record_signal(self, pair: str, direction: str, setup_hash: str,
                      entry_price: float):
        self.active_signals[pair.upper()] = {
            "direction": direction,
            "setup_hash": setup_hash,
            "entry_price": entry_price,
            "timestamp": datetime.utcnow()
        }

    def clear_old(self, max_age_minutes: int = 30):
        cutoff = datetime.utcnow() - timedelta(minutes=max_age_minutes)
        to_remove = [k for k, v in self.active_signals.items() if v["timestamp"] < cutoff]
        for k in to_remove:
            del self.active_signals[k]


class MultiTimeframeEngine:
    """
    3-Layer Analysis:
    - Primary (direction): confirms overall trend/bias
    - Secondary (setup): validates trading setup
    - Entry (timing): precise entry moment

    v5.3 2-layer fallback quality gate:
    Fires without entry-layer confirmation ONLY when:
      1. Primary layer ACTIVELY agrees — its own smc signal matches the
         direction OR the trend is strongly directional (BULLISH/BEARISH).
         A WEAK_* lean with primary signal HOLD is NOT enough.
      2. Entry layer does not signal the OPPOSITE direction.
      3. Secondary confluence score clears the (morning-adjusted) bar.
      4. Fakeout risk is low.
    Otherwise HOLD and wait for real alignment.

    v5.4 early trend-start layer (symmetric BUY/SELL):
    - Running trend + quiet entry layer -> EMA9/21 cross + swing-structure
      break on the entry TF confirms the entry NOW (no pattern wait).
    - Primary still SIDEWAYS -> if entry structure broke AND primary momentum
      shifted AND secondary doesn't oppose, fire a TREND-START entry.
    Hard requirements (alignment + structure break) keep the v5.3 lesson:
    no firing against structure, only with its birth.

    v5.4.1 REVERSAL OVERRIDE (root-cause fix for "bot stuck buying while
    market is selling"):
    Previously, once `primary_trend` (15m, EMA200-based) locked in a
    direction, it stayed locked until the 15m EMA200 relationship itself
    flipped — which lags a real reversal by many candles. During that lag,
    the hard trend block in filters.py additionally vetoed any opposite
    signal on the SAME stale logic (5m version), so even if this engine
    somehow produced a counter-signal, filters.py killed it again. Two
    independent layers of the same lag.
    Now: after establishing trend_direction from primary as before, the
    engine separately checks whether the ENTRY timeframe shows a strong,
    fresh reversal (EMA9/21 cross + structure break, score >= 0.75) against
    that stale direction, confirmed by a primary momentum shift (price vs
    EMA21, which reacts far faster than EMA200) and a secondary layer that
    is not actively opposing. Only when all three agree does the direction
    flip early, tagged `reversal_override=True`. That flag is passed all the
    way to filters.py so the stale-structure hard block doesn't veto the
    very reversal this logic just confirmed.
    """

    def __init__(self):
        self.analyzer = TechnicalAnalyzer()
        self.signal_tracker = SignalTracker(
            cooldown_minutes=int(os.getenv("SIGNAL_COOLDOWN_MINUTES", "10")),
            price_tolerance_pct=0.005
        )

        # ===== STANDARD THRESHOLDS =====
        self.min_confidence = float(os.getenv("MIN_SIGNAL_CONFIDENCE", "0.60"))
        self.min_adx = float(os.getenv("MIN_ADX", "18"))
        self.min_atr_pct = float(os.getenv("MIN_ATR_PCT", "0.02"))

        # ===== 2-LAYER FALLBACK — REMOVED in v5.5.0 =====
        # Kept as dead config (harmless if still set in .env) so an old
        # TWO_LAYER_MIN_SCORE value doesn't error out — it's simply unused
        # now. The fallback itself (firing with zero entry-timeframe
        # confirmation) was cut entirely, not just tightened further; see
        # the entry-confirmation block below for why.
        self.allow_two_layer = False
        self.two_layer_min_score = float(os.getenv("TWO_LAYER_MIN_SCORE", "0.45"))

        # ===== MORNING-BIRD BOOST (times are UTC) =====
        self.morning_start_utc = int(os.getenv("MORNING_BOOST_START_UTC", "6"))
        self.morning_end_utc = int(os.getenv("MORNING_BOOST_END_UTC", "10"))
        self.morning_min_confidence = float(os.getenv("MORNING_MIN_CONFIDENCE", "0.55"))
        self.morning_min_adx = float(os.getenv("MORNING_MIN_ADX", "15"))
        self.morning_min_atr_pct = float(os.getenv("MORNING_MIN_ATR_PCT", "0.015"))
        self.morning_two_layer_min_score = float(os.getenv("MORNING_TWO_LAYER_MIN_SCORE", "0.35"))

        # ===== REVERSAL OVERRIDE (v5.4.1) =====
        self.reversal_min_score = float(os.getenv("REVERSAL_MIN_SCORE", "0.75"))

        logger.info(
            f"MultiTimeframeEngine v5.5.5 (SIMPLIFIED) | Min confidence: {self.min_confidence} | "
            f"Min ADX: {self.min_adx} | Min ATR%: {self.min_atr_pct}% | "
            f"Entry confirmation: full-confluence OR early-trend-start ONLY | "
            f"2-layer fallback: REMOVED | EMA-quiet-entry catch: REMOVED | "
            f"Reversal override: ON (score>={self.reversal_min_score}) | "
            f"Morning boost: {self.morning_start_utc}:00-{self.morning_end_utc}:00 UTC | "
            f"Per-asset overrides: supported"
        )

    def _morning_boost_active(self) -> bool:
        """True during the configured morning window (UTC)."""
        hour = datetime.utcnow().hour
        if self.morning_start_utc <= self.morning_end_utc:
            return self.morning_start_utc <= hour < self.morning_end_utc
        # Window wraps midnight (e.g. 22 -> 6)
        return hour >= self.morning_start_utc or hour < self.morning_end_utc

    # ===== v5.4: EARLY TREND-START DETECTION =====
    # Catches trends at the START (EMA9/21 cross + swing-structure break on
    # the entry TF) instead of waiting for confirmed candle patterns, which
    # by definition lag the move they describe.

    def _detect_early_trend(self, df: pd.DataFrame, direction: str,
                            lookback: int = 30) -> Tuple[bool, float, str]:
        """
        Score-based early trend detection on the entry timeframe.
        Requires EMA alignment AND a break of the recent swing (the structure
        break is the actual birth of a trend). A fresh cross adds confidence.
        Fully symmetric: BUY and SELL run identical math mirrored.
        Returns (detected, score 0..1, reason).
        """
        if df is None or len(df) < max(60, lookback + 10):
            return False, 0.0, "insufficient data"

        close = df["close"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)

        ema_fast = close.ewm(span=9, adjust=False).mean()
        ema_slow = close.ewm(span=21, adjust=False).mean()
        cur_f, cur_s = ema_fast.iloc[-1], ema_slow.iloc[-1]

        if direction == "BUY":
            cross = (ema_fast.shift(1) <= ema_slow.shift(1)) & (ema_fast > ema_slow)
            aligned = cur_f > cur_s
            swing = high.iloc[-lookback - 1:-1].max()
            broke_structure = close.iloc[-1] > swing
            momentum = close.iloc[-1] > close.iloc[-6]
        else:
            cross = (ema_fast.shift(1) >= ema_slow.shift(1)) & (ema_fast < ema_slow)
            aligned = cur_f < cur_s
            swing = low.iloc[-lookback - 1:-1].min()
            broke_structure = close.iloc[-1] < swing
            momentum = close.iloc[-1] < close.iloc[-6]

        score, reasons = 0.0, []
        if bool(cross.iloc[-10:].any()):
            score += 0.45
            reasons.append("fresh EMA9/21 cross")
        elif aligned:
            score += 0.30
            reasons.append("EMA alignment")
        if broke_structure:
            score += 0.35
            reasons.append(f"structure break vs {lookback}-bar swing")
        if momentum:
            score += 0.20
            reasons.append("momentum")

        # Hard requirements: EMA alignment + structure break. Cross/momentum
        # only modulate confidence — keeps v5.3's lesson (no blind early
        # fires against structure; only with its break).
        detected = aligned and broke_structure and score >= 0.60
        return detected, min(score, 1.0), "+".join(reasons) or "no early trend"

    def _primary_momentum_shift(self, df: pd.DataFrame, direction: str) -> Tuple[bool, str]:
        """
        PRIMARY timeframe: has momentum shifted in `direction`? Used to allow
        trend-start entries while market_structure is still SIDEWAYS — requires
        price on the right side of a RISING (BUY) / FALLING (SELL) EMA21.
        """
        if df is None or len(df) < 40:
            return False, "insufficient primary data"
        close = df["close"].astype(float)
        ema21 = close.ewm(span=21, adjust=False).mean()
        cur, e, e_prev = close.iloc[-1], ema21.iloc[-1], ema21.iloc[-6]

        if direction == "BUY":
            ok = cur > e and e > e_prev
            desc = "price above rising EMA21"
        else:
            ok = cur < e and e < e_prev
            desc = "price below falling EMA21"
        return ok, desc

    async def analyze_asset(self, pair: str, source: str, timeframes: dict,
                            data_feed, threshold_overrides: Optional[dict] = None) -> Optional[Dict]:
        """
        timeframes = {
            "primary":   {"tf": "15m", "deriv_granularity": 900},
            "secondary": {"tf": "5m",  "deriv_granularity": 300},
            "entry":     {"tf": "1m",  "deriv_granularity": 60}
        }

        threshold_overrides (v5.2+) = optional per-asset gates, e.g.:
            {"min_atr_pct": 0.008, "min_adx": 12,
             "min_confidence": 0.55, "two_layer_min_score": 0.40}
        """
        self.signal_tracker.clear_old()
        morning = self._morning_boost_active()

        # ===== EFFECTIVE THRESHOLDS =====
        # 1. Start from per-asset override if given, else global default
        # 2. Morning boost relaxes further (looser of the two wins)
        base_conf = float(threshold_overrides.get("min_confidence", self.min_confidence)) if threshold_overrides else self.min_confidence
        base_adx = float(threshold_overrides.get("min_adx", self.min_adx)) if threshold_overrides else self.min_adx
        base_atr = float(threshold_overrides.get("min_atr_pct", self.min_atr_pct)) if threshold_overrides else self.min_atr_pct
        base_2l = float(threshold_overrides.get("two_layer_min_score", self.two_layer_min_score)) if threshold_overrides else self.two_layer_min_score

        if morning:
            eff_min_conf = min(base_conf, self.morning_min_confidence)
            eff_min_adx = min(base_adx, self.morning_min_adx)
            eff_min_atr = min(base_atr, self.morning_min_atr_pct)
            eff_two_layer_score = min(base_2l, self.morning_two_layer_min_score)
        else:
            eff_min_conf, eff_min_adx, eff_min_atr, eff_two_layer_score = base_conf, base_adx, base_atr, base_2l

        if threshold_overrides:
            logger.debug(f"{pair} using per-asset thresholds | ATR>={eff_min_atr}% ADX>={eff_min_adx} "
                         f"conf>={eff_min_conf} 2L>={eff_two_layer_score}"
                         f"{' (morning)' if morning else ''}")

        # Fetch all 3 timeframes
        frames = {}
        for layer, cfg in timeframes.items():
            try:
                if source == "deriv":
                    df = await data_feed.fetch_latest(
                        pair=pair,
                        timeframe=cfg["deriv_granularity"],
                        source="deriv",
                        deriv_symbol=pair
                    )
                else:
                    df = await data_feed.fetch_latest(
                        pair=pair,
                        timeframe=cfg["tf"],
                        source="kraken"
                    )

                if df is None or len(df) < 50:
                    logger.warning(f"{pair} [{layer}]: insufficient data ({len(df) if df is not None else 0} candles)")
                    return None
                frames[layer] = df
            except Exception as e:
                logger.error(f"{pair} [{layer}] fetch error: {e}")
                return None

        # Analyze each timeframe
        analyses = {}
        for layer, df in frames.items():
            try:
                analyses[layer] = self.analyzer.calculate_indicators(df)
            except Exception as e:
                logger.error(f"{pair} [{layer}] analysis error: {e}")
                return None

        primary = analyses["primary"]
        secondary = analyses["secondary"]
        entry = analyses["entry"]

        current_price = entry["current_price"]

        # ===== VOLATILITY FILTER =====
        atr_pct = (secondary.get("atr", 0) / current_price) * 100
        adx = secondary.get("adx", 0)

        if atr_pct < eff_min_atr or adx < eff_min_adx:
            reason = []
            if atr_pct < eff_min_atr:
                reason.append(f"ATR {atr_pct:.3f}% < {eff_min_atr}%")
            if adx < eff_min_adx:
                reason.append(f"ADX {adx:.1f} < {eff_min_adx}")
            tag = "MORNING " if morning else ""
            logger.info(f"{pair} {tag}VOLATILITY BLOCK: {' | '.join(reason)}")
            return {
                "pair": pair, "signal": "HOLD",
                "reason": f"Volatility filter: {' | '.join(reason)}",
                "volatility_passed": False,
                "secondary_indicators": secondary,
                "secondary_df": frames["secondary"],
                "layers": self._summarize_layers(analyses)
            }

        # ===== SETUP PRE-READ (needed by trend-start detection too) =====
        secondary_signal = secondary.get("smc_suggested_signal", "HOLD")
        secondary_score = secondary.get("confluence_score", 0)

        # ===== DIRECTION CHECK (Primary) =====
        primary_trend = primary.get("market_structure", "SIDEWAYS")
        primary_signal = primary.get("smc_suggested_signal", "HOLD")
        trend_direction = None
        if primary_trend in ["BULLISH", "BULLISH_PULLBACK", "WEAK_BULLISH"]:
            trend_direction = "BUY"
        elif primary_trend in ["BEARISH", "BEARISH_PULLBACK", "WEAK_BEARISH"]:
            trend_direction = "SELL"

        early_trend_start = False
        early_start_score = 0.0
        reversal_override = False

        if trend_direction is None:
            # v5.4: TREND-START catch — primary hasn't labeled a trend yet,
            # but the entry TF shows a fresh EMA cross + structure break AND
            # primary momentum has shifted. Catches the move at birth instead
            # of waiting for the 15m label (which arrives hours late).
            for try_dir in ("BUY", "SELL"):
                e_ok, e_score, e_reason = self._detect_early_trend(frames["entry"], try_dir)
                if not e_ok:
                    continue
                m_ok, m_reason = self._primary_momentum_shift(frames["primary"], try_dir)
                if not m_ok:
                    continue
                if secondary_signal not in (try_dir, "HOLD"):
                    continue  # secondary actively opposes — not a trend start
                trend_direction = try_dir
                early_trend_start = True
                early_start_score = e_score
                logger.info(f"{pair} EARLY TREND START: {try_dir} | {e_reason} | {m_reason}")
                break

            if trend_direction is None:
                return {
                    "pair": pair, "signal": "HOLD",
                    "reason": f"Primary trend unclear: {primary_trend}",
                    "volatility_passed": True,
                    "secondary_indicators": secondary,
                    "secondary_df": frames["secondary"],
                    "layers": self._summarize_layers(analyses)
                }
        else:
            # ===== v5.4.1 REVERSAL OVERRIDE =====
            # Primary (15m EMA200) still calls the OLD direction, but that
            # measure is slow by design. Check whether the entry TF shows a
            # strong, fresh reversal the other way, confirmed by primary
            # momentum (EMA21 — much faster than EMA200) and not actively
            # opposed by secondary. This is what lets the bot flip to SELL
            # while price is actually dropping instead of waiting for the
            # 15m EMA200 to catch up (the exact scenario that blew the
            # account: bot kept saying BUY while the market was selling).
            opposite_dir = "SELL" if trend_direction == "BUY" else "BUY"
            e_ok, e_score, e_reason = self._detect_early_trend(frames["entry"], opposite_dir)
            if e_ok and e_score >= self.reversal_min_score:
                m_ok, m_reason = self._primary_momentum_shift(frames["primary"], opposite_dir)
                if m_ok and secondary_signal in (opposite_dir, "HOLD"):
                    logger.info(
                        f"{pair} REVERSAL OVERRIDE: {trend_direction} -> {opposite_dir} "
                        f"| stale primary trend: {primary_trend} | {e_reason} | {m_reason}"
                    )
                    trend_direction = opposite_dir
                    early_trend_start = True
                    early_start_score = e_score
                    reversal_override = True

        # ===== SETUP CHECK (Secondary) =====
        if secondary_signal != trend_direction:
            # v5.4: neutral secondary + fresh entry structure break is NOT a
            # conflict — that's precisely the trend-start case. Anything else
            # (opposing signal) still blocks, as before.
            if not (early_trend_start and secondary_signal == "HOLD"):
                return {
                    "pair": pair, "signal": "HOLD",
                    "reason": (f"No valid setup on secondary ({secondary_signal})"
                               if secondary_signal not in ["BUY", "SELL"]
                               else f"Secondary ({secondary_signal}) vs Primary ({trend_direction}) conflict"),
                    "volatility_passed": True,
                    "secondary_indicators": secondary,
                    "secondary_df": frames["secondary"],
                    "layers": self._summarize_layers(analyses)
                }

        # ===== ENTRY CHECK (Entry timeframe) =====
        entry_signal = entry.get("smc_suggested_signal", "HOLD")
        entry_score = entry.get("confluence_score", 0)
        entry_live = entry.get("live_reversal")
        entry_cs = entry.get("candlestick_signal", "HOLD")
        entry_cs_conf = entry.get("candlestick_confidence", 0)

        entry_confirms = False
        entry_confidence = 0.0
        early_entry = False

        # v5.4.6 BUG FIX: entry.smc_suggested_signal can be "HOLD" while
        # entry.market_structure is already flatly opposite the trade (e.g.
        # a BUY with entry market_structure == "BEARISH"). The EMA9/21 cross
        # detector used below runs completely separate math from the
        # structure/confluence calc and was firing without ever checking
        # this — letting a BUY through while the entry timeframe's own
        # trend reading had already flipped bearish (confirmed case: XAUUSD
        # BUY fired with entry: HOLD (BEARISH) + a RANGE_WHIPSAW fakeout at
        # 60% risk on the same cycle — price kept falling straight through
        # both entry and stop-loss).
        entry_structure_opposes = entry.get("market_structure") in (
            ["BEARISH", "BEARISH_PULLBACK", "WEAK_BEARISH"] if trend_direction == "BUY"
            else ["BULLISH", "BULLISH_PULLBACK", "WEAK_BULLISH"]
        )

        # v5.4.7 BUG FIX: a "*_PULLBACK" primary_trend is the code's OWN
        # label for "macro trend is this direction, but the recent swing
        # structure has already turned the other way" (see
        # _find_market_structure). That is not a green light to enter NOW —
        # it's an active warning that a reversal may already be underway.
        # Confirmed case: BTCUSD fired a 99%-confidence BUY with primary
        # "HOLD (BULLISH_PULLBACK)", confirmed ONLY by a single-candle wick
        # pattern on the entry chart ("LIVE: Bullish reversal, score 0.60")
        # — price kept falling through both entry and stop-loss within the
        # hour. Same thing happened on XAUUSD the same cycle. During a
        # pullback state, only genuine entry-timeframe confluence agreement
        # (the first branch below) is accepted; the weaker single-signal
        # confirmations (a reversal wick, one candlestick pattern, or a bare
        # divergence) are disabled, since a pullback is exactly the moment
        # those are most likely to be a headfake rather than real.
        is_pullback_state = primary_trend.endswith("_PULLBACK")

        # ===== v5.5.0 SIMPLIFICATION =====
        # Prior versions had SIX independent ways to satisfy "entry
        # confirmed": full confluence, a live reversal wick, a candlestick
        # pattern, a bare RSI divergence, an EMA9/21 catch when the entry
        # layer was quiet, and a 2-layer fallback when entry didn't confirm
        # at all. Each was added to catch one specific missed trade — but
        # every extra door in is a door a weak setup can also walk through,
        # and tracking real trade outcomes over two weeks showed a clearly
        # negative track record concentrated in exactly these looser paths.
        # More gates that only ever narrow things (pullback-state,
        # entry-structure-opposition) are safe to keep layering on; more
        # gates that each independently OPEN the trade are not — complexity
        # there trades selectivity for coverage, and coverage is not what
        # was missing.
        #
        # Now there are exactly TWO ways in:
        #   1. Full confluence — entry's own suggested signal genuinely
        #      agrees, with a real confluence score. This is the only
        #      "standalone" trigger.
        #   2. Genuine early-trend-start — primary hadn't labeled a trend
        #      yet, entry broke structure, AND primary momentum (fast EMA21)
        #      independently shifted. This isn't a weaker substitute for #1;
        #      it's catching a trend at a different stage.
        # A live reversal wick, a candlestick pattern, and RSI divergence
        # are no longer standalone triggers — they're read as supporting
        # context in the reasoning text, but they do not fire a trade alone.
        # The 2-layer fallback (firing with NO entry-timeframe confirmation
        # at all) is removed outright, not just tightened further.
        # v5.5.1 BUG FIX: entry_structure_opposes was only ever wired into the
        # fallback paths removed above — the "full confluence" branch below
        # was never actually gated by it. That's a real gap: entry.smc_
        # suggested_signal (confluence/sweep/pattern math) and entry.
        # market_structure (EMA200+swing math) are two SEPARATE calculations
        # on the same entry-timeframe data and can disagree with each other.
        # Confirmed case: EURUSD fired a 92%-confidence SELL with Layer Check
        # showing "entry: SELL (WEAK_BULLISH)" — the entry timeframe's own
        # structure reading was already bullish-leaning while its confluence
        # score still said SELL. Price reversed through the stop within the
        # hour. Now a direct structure contradiction hard-blocks entry
        # confirmation regardless of how clean the confluence score looks.
        if entry_signal == trend_direction and entry_score >= 0.30 and not entry_structure_opposes:
            entry_confirms = True
            entry_confidence = entry_score
        elif early_trend_start:
            # NOT gated by entry_structure_opposes: that label is the SAME
            # kind of slower structure read this path exists to get ahead
            # of — blocking it here would defeat the point of catching a
            # reversal early.
            # Entry confirmation = the structure break that started it all
            entry_confirms = True
            entry_confidence = early_start_score * 0.75
            early_entry = True

        if not entry_confirms:
            if is_pullback_state:
                reason = (f"Pullback state ({primary_trend}) — entry timeframe didn't "
                          f"give full confluence confirmation")
            elif entry_structure_opposes:
                reason = (f"Entry market_structure ({entry.get('market_structure')}) "
                          f"opposes {trend_direction} — no confirmation path available")
            else:
                reason = (f"Entry timeframe not confirming (entry_signal={entry_signal}, "
                          f"score={entry_score:.2f}) — 2-layer and EMA-catch fallbacks "
                          f"removed in v5.5.0 simplification")
            logger.info(f"{pair} HOLD: {reason}")
            return {
                "pair": pair, "signal": "HOLD",
                "reason": reason,
                "volatility_passed": True,
                "secondary_indicators": secondary,
                "secondary_df": frames["secondary"],
                "layers": self._summarize_layers(analyses)
            }

        two_layer_mode = False  # kept as a field for downstream code/notifications; always False now

        # ===== COMBINED CONFIDENCE =====
        primary_conf = primary.get("trend_strength", 0.5)
        if primary_trend in ["BULLISH", "BEARISH"]:
            primary_conf = max(primary_conf, 0.75)
        elif primary_trend in ["BULLISH_PULLBACK", "BEARISH_PULLBACK"]:
            primary_conf = max(primary_conf, 0.60)

        combined = (primary_conf * 0.30 + secondary_score * 0.50 + entry_confidence * 0.20)

        if primary_signal == secondary_signal == entry_signal == trend_direction:
            combined = min(combined * 1.15, 0.99)

        if two_layer_mode:
            combined *= 0.85  # penalty for missing entry-layer confirmation

        if early_entry:
            combined *= 0.90  # v5.4: modest penalty for unconfirmed early entry

        if reversal_override:
            combined *= 0.90  # v5.4.1: modest penalty for overriding a stale trend read

        if secondary.get("fakeout_detected") and secondary.get("fakeout_risk", 0) > 0.3:
            combined *= 0.7

        if combined < eff_min_conf:
            return {
                "pair": pair, "signal": "HOLD",
                "reason": f"Confidence {combined:.2f} < min {eff_min_conf}"
                            + (" (morning boost)" if morning else ""),
                "volatility_passed": True,
                "secondary_indicators": secondary,
                "secondary_df": frames["secondary"],
                "layers": self._summarize_layers(analyses)
            }

        # ===== DUPLICATE SETUP CHECK =====
        setup_hash = self.signal_tracker._make_setup_hash(
            secondary.get("active_support"),
            secondary.get("active_resistance"),
            primary_trend,
            trend_direction
        )

        is_dup, dup_reason = self.signal_tracker.is_duplicate(
            pair, trend_direction, setup_hash, current_price
        )

        if is_dup:
            return {
                "pair": pair, "signal": "HOLD",
                "reason": f"Duplicate prevention: {dup_reason}",
                "volatility_passed": True,
                "secondary_indicators": secondary,
                "secondary_df": frames["secondary"],
                "layers": self._summarize_layers(analyses)
            }

        # ===== BUILD FINAL SIGNAL =====
        entry_price = current_price
        stop_loss = secondary.get("suggested_stop_loss")
        take_profit = secondary.get("suggested_take_profit")

        if stop_loss is None:
            atr = secondary.get("atr", entry_price * 0.01)
            if trend_direction == "BUY":
                stop_loss = entry_price - (atr * 1.5)
                take_profit = entry_price + (atr * 3.0)
            else:
                stop_loss = entry_price + (atr * 1.5)
                take_profit = entry_price - (atr * 3.0)

        # ===== v5.5.5: FIXED-DOLLAR-RISK SL/TP =====
        # User request: stop sizing SL/TP off ATR/support-resistance (which
        # varies trade to trade) and instead fix them to a known dollar
        # risk/reward at a known lot size — e.g. always $50 risk / $100
        # reward at 0.5 lots. The 1:2 reward:risk the bot already enforces
        # above is UNCHANGED — this only replaces the SL *distance*, and TP
        # still follows at exactly 2x that distance, so $50 risk naturally
        # becomes $100 reward without needing two separate numbers.
        #
        # SAFETY NOTE: this depends on each instrument's CONTRACT SIZE
        # (units per 1.0 lot), which varies by broker/symbol and is NOT
        # guaranteed to match the defaults below — verify in MT5 via
        # Market Watch → right-click symbol → Specification → "Contract
        # size" before trusting these numbers on a real account. Defaults
        # here are common industry conventions, not confirmed Exness values
        # for the "m"-suffixed (BTCUSDm/EURUSDm/XAUUSDm) symbol group.
        if os.getenv("FIXED_DOLLAR_RISK_MODE", "true").lower() == "true":
            target_risk_usd = float(os.getenv("TARGET_RISK_USD", "50"))
            target_lot_size = float(os.getenv("TARGET_LOT_SIZE", "0.5"))
            contract_sizes = {
                "BTC": float(os.getenv("CONTRACT_SIZE_BTCUSD", "1")),        # 1 BTC/lot — VERIFY
                "EUR": float(os.getenv("CONTRACT_SIZE_EURUSD", "100000")),   # 100k units/lot — VERIFY
                "XAU": float(os.getenv("CONTRACT_SIZE_XAUUSD", "100")),      # 100 oz/lot — VERIFY
            }
            contract_size = None
            for key, size in contract_sizes.items():
                if key in pair.upper():
                    contract_size = size
                    break

            if contract_size and target_lot_size > 0:
                risk_distance = target_risk_usd / (contract_size * target_lot_size)
                if trend_direction == "BUY":
                    stop_loss = entry_price - risk_distance
                    take_profit = entry_price + (risk_distance * 2)
                else:
                    stop_loss = entry_price + risk_distance
                    take_profit = entry_price - (risk_distance * 2)
                logger.info(
                    f"{pair} FIXED-$ SIZING: risk ${target_risk_usd:.0f} / reward "
                    f"${target_risk_usd*2:.0f} @ {target_lot_size} lot (contract={contract_size:g}) "
                    f"-> SL distance {risk_distance:.5f}"
                )
            else:
                logger.warning(f"{pair} FIXED-$ SIZING skipped — no contract size match, using ATR-based levels")

        self.signal_tracker.record_signal(pair, trend_direction, setup_hash, entry_price)

        # v5.4.3: decimals must scale with the price's own magnitude — a flat
        # round(x, 2) is fine for BTCUSD (~84,796.10) but collapses EURUSD
        # (~1.14235) so entry/SL/TP all display as "1.14". This is what's
        # stored in Redis/the API response, independent of how Telegram
        # formats it (see notifications.py's matching fix).
        decimals = 5 if entry_price < 50 else 2

        return {
            "pair": pair,
            "signal": trend_direction,
            "confidence": round(combined, 3),
            "entry_price": round(entry_price, decimals),
            "stop_loss": round(stop_loss, decimals),
            "take_profit": round(take_profit, decimals),
            "reason": self._build_reason(analyses, trend_direction, combined,
                                         two_layer_mode=two_layer_mode, morning=morning,
                                         early_trend_start=early_trend_start,
                                         early_entry=early_entry,
                                         reversal_override=reversal_override,
                                         primary_trend_raw=primary_trend),
            "volatility_passed": True,
            "primary_trend": primary_trend,
            "secondary_score": round(secondary_score, 3),
            "entry_confidence": round(entry_confidence, 3),
            "atr_pct": round(atr_pct, 3),
            "adx": round(adx, 1),
            "secondary_indicators": secondary,
            "secondary_df": frames["secondary"],
            "layers": self._summarize_layers(analyses),
            "is_early": combined < 0.80 or two_layer_mode or early_entry or early_trend_start or reversal_override,
            "two_layer_mode": two_layer_mode,
            "early_trend_start": early_trend_start,
            "early_entry": early_entry,
            "reversal_override": reversal_override,
            "morning_boost": morning,
            "setup_hash": setup_hash
        }

    def _summarize_layers(self, analyses: dict) -> dict:
        return {
            layer: {
                "signal": a.get("smc_suggested_signal", "HOLD"),
                "score": round(a.get("confluence_score", 0), 3),
                "trend": a.get("market_structure", "UNKNOWN"),
                "rsi": round(a.get("rsi", 50), 1)
            }
            for layer, a in analyses.items()
        }

    def _build_reason(self, analyses: dict, signal: str, confidence: float,
                      two_layer_mode: bool = False, morning: bool = False,
                      early_trend_start: bool = False, early_entry: bool = False,
                      reversal_override: bool = False, primary_trend_raw: str = "") -> str:
        p = analyses["primary"]
        s = analyses["secondary"]
        e = analyses["entry"]

        mode = "2-LAYER" if two_layer_mode else "3-LAYER"
        tag = "🌅 MORNING BOOST | " if morning else ""
        parts = [f"{tag}{mode} ALIGNMENT ({signal}) | Confidence: {min(confidence, 1.0) * 100:.1f}%"]
        parts.append(f"Direction: {p.get('market_structure')} (strength: {p.get('trend_strength', 0):.2f})")
        parts.append(f"Setup: {s.get('smc_reasoning', 'N/A')[:80]}")

        if two_layer_mode:
            parts.append("Entry layer quiet — fired on primary+setup agreement (penalty applied)")
        elif e.get("live_reversal"):
            parts.append(f"Entry: {e.get('live_reversal')}")
        elif e.get("candlestick_signal") != "HOLD":
            parts.append(f"Entry pattern: {e.get('candlestick_signal')}")

        if reversal_override:
            parts.append(f"🔄 REVERSAL OVERRIDE: stale primary read ({primary_trend_raw}) overridden by "
                         f"confirmed entry-TF reversal + EMA21 momentum shift")
        elif early_trend_start:
            parts.append("⚡ TREND-START catch: primary still neutral, entry structure broken + EMA cross")
        elif early_entry:
            parts.append("⚡ EARLY entry: EMA cross + structure break (pattern not yet confirmed)")

        if s.get("fakeout_detected"):
            parts.append(f"Fakeout risk: {s.get('fakeout_risk', 0) * 100:.0f}%")

        return " | ".join(parts)