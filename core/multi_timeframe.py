# core/multi_timeframe.py
# Multi-timeframe alignment engine
# v5.1: Morning-bird boost + 2-layer fallback + bug fix (primary_signal NameError)
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

    v5.1 changes:
    - Morning-bird window: temporarily relaxed thresholds to catch early session moves
    - 2-layer fallback: primary + secondary agreement can fire without entry
      confirmation (with confidence penalty) instead of dying at the entry layer
    - Fixed NameError bug: 'primary_signal' was never defined (would crash on
      a full 3-layer alignment)
    """

    def __init__(self):
        self.analyzer = TechnicalAnalyzer()
        self.signal_tracker = SignalTracker(
            cooldown_minutes=int(os.getenv("SIGNAL_COOLDOWN_MINUTES", "10")),
            price_tolerance_pct=0.005
        )

        # ===== STANDARD THRESHOLDS (relaxed vs v5.0 defaults) =====
        self.min_confidence = float(os.getenv("MIN_SIGNAL_CONFIDENCE", "0.60"))
        self.min_adx = float(os.getenv("MIN_ADX", "18"))
        self.min_atr_pct = float(os.getenv("MIN_ATR_PCT", "0.02"))

        # ===== 2-LAYER FALLBACK =====
        # If entry layer won't confirm but direction + setup agree strongly,
        # fire anyway with a confidence penalty instead of going HOLD.
        self.allow_two_layer = os.getenv("ALLOW_TWO_LAYER_FALLBACK", "true").lower() == "true"
        self.two_layer_min_score = float(os.getenv("TWO_LAYER_MIN_SCORE", "0.45"))

        # ===== MORNING-BIRD BOOST =====
        # Render servers run on UTC. Morning window defaults to 06:00-10:00 UTC
        # = 07:00-11:00 for UTC+1. Adjust via .env to match YOUR morning.
        self.morning_start_utc = int(os.getenv("MORNING_BOOST_START_UTC", "6"))
        self.morning_end_utc = int(os.getenv("MORNING_BOOST_END_UTC", "10"))
        self.morning_min_confidence = float(os.getenv("MORNING_MIN_CONFIDENCE", "0.55"))
        self.morning_min_adx = float(os.getenv("MORNING_MIN_ADX", "15"))
        self.morning_min_atr_pct = float(os.getenv("MORNING_MIN_ATR_PCT", "0.015"))
        self.morning_two_layer_min_score = float(os.getenv("MORNING_TWO_LAYER_MIN_SCORE", "0.35"))

        logger.info(
            f"MultiTimeframeEngine v5.1 | Min confidence: {self.min_confidence} | "
            f"Min ADX: {self.min_adx} | Min ATR%: {self.min_atr_pct}% | "
            f"2-layer fallback: {self.allow_two_layer} (score>={self.two_layer_min_score}) | "
            f"Morning boost: {self.morning_start_utc}:00-{self.morning_end_utc}:00 UTC"
        )

    def _morning_boost_active(self) -> bool:
        """True during the configured morning window (UTC)."""
        hour = datetime.utcnow().hour
        if self.morning_start_utc <= self.morning_end_utc:
            return self.morning_start_utc <= hour < self.morning_end_utc
        # Window wraps midnight (e.g. 22 -> 6)
        return hour >= self.morning_start_utc or hour < self.morning_end_utc

    async def analyze_asset(self, pair: str, source: str, timeframes: dict,
                            data_feed) -> Optional[Dict]:
        """
        timeframes = {
            "primary":   {"tf": "15m", "deriv_granularity": 900},
            "secondary": {"tf": "5m",  "deriv_granularity": 300},
            "entry":     {"tf": "1m",  "deriv_granularity": 60}
        }
        """
        self.signal_tracker.clear_old()
        morning = self._morning_boost_active()

        # Effective thresholds (morning window uses relaxed values)
        eff_min_conf = self.morning_min_confidence if morning else self.min_confidence
        eff_min_adx = self.morning_min_adx if morning else self.min_adx
        eff_min_atr = self.morning_min_atr_pct if morning else self.min_atr_pct
        eff_two_layer_score = self.morning_two_layer_min_score if morning else self.two_layer_min_score

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

        # ===== DIRECTION CHECK (Primary) =====
        primary_trend = primary.get("market_structure", "SIDEWAYS")
        primary_signal = primary.get("smc_suggested_signal", "HOLD")  # FIX: was never defined (NameError on full alignment)
        trend_direction = None
        if primary_trend in ["BULLISH", "BULLISH_PULLBACK", "WEAK_BULLISH"]:
            trend_direction = "BUY"
        elif primary_trend in ["BEARISH", "BEARISH_PULLBACK", "WEAK_BEARISH"]:
            trend_direction = "SELL"

        if trend_direction is None:
            return {
                "pair": pair, "signal": "HOLD",
                "reason": f"Primary trend unclear: {primary_trend}",
                "volatility_passed": True,
                "secondary_indicators": secondary,
                "secondary_df": frames["secondary"],
                "layers": self._summarize_layers(analyses)
            }

        # ===== SETUP CHECK (Secondary) =====
        secondary_signal = secondary.get("smc_suggested_signal", "HOLD")
        secondary_score = secondary.get("confluence_score", 0)

        if secondary_signal not in ["BUY", "SELL"]:
            return {
                "pair": pair, "signal": "HOLD",
                "reason": f"No valid setup on secondary ({secondary_signal})",
                "volatility_passed": True,
                "secondary_indicators": secondary,
                "secondary_df": frames["secondary"],
                "layers": self._summarize_layers(analyses)
            }

        if secondary_signal != trend_direction:
            return {
                "pair": pair, "signal": "HOLD",
                "reason": f"Secondary ({secondary_signal}) vs Primary ({trend_direction}) conflict",
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

        if entry_signal == trend_direction and entry_score >= 0.30:
            entry_confirms = True
            entry_confidence = entry_score
        elif entry_live == f"{trend_direction}_EARLY":
            entry_confirms = True
            entry_confidence = entry.get("live_reversal_score", 0.5)
        elif entry_cs == trend_direction and entry_cs_conf >= 0.50:
            entry_confirms = True
            entry_confidence = entry_cs_conf
        elif entry_score >= 0.20:
            div = entry.get("rsi_divergence", "NONE")
            expected_div = "BULLISH_DIV" if trend_direction == "BUY" else "BEARISH_DIV"
            if div == expected_div:
                entry_confirms = True
                entry_confidence = 0.40

        # ===== v5.1: 2-LAYER FALLBACK =====
        # Direction + setup agree but entry layer is quiet (common on fast
        # morning moves — the 1m layer lags the impulse). Fire with a penalty
        # instead of missing the move entirely.
        two_layer_mode = False
        if not entry_confirms:
            if (self.allow_two_layer
                    and secondary_score >= eff_two_layer_score
                    and secondary.get("fakeout_risk", 0) < 0.40):
                entry_confirms = True
                entry_confidence = 0.25
                two_layer_mode = True
                logger.info(f"{pair} 2-LAYER MODE: entry layer quiet, direction+setup agree "
                            f"(score {secondary_score:.2f}) — firing with penalty")
            else:
                return {
                    "pair": pair, "signal": "HOLD",
                    "reason": "Entry timeframe not confirming",
                    "volatility_passed": True,
                    "secondary_indicators": secondary,
                    "secondary_df": frames["secondary"],
                    "layers": self._summarize_layers(analyses)
                }

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

        self.signal_tracker.record_signal(pair, trend_direction, setup_hash, entry_price)

        return {
            "pair": pair,
            "signal": trend_direction,
            "confidence": round(combined, 3),
            "entry_price": round(entry_price, 2),
            "stop_loss": round(stop_loss, 2),
            "take_profit": round(take_profit, 2),
            "reason": self._build_reason(analyses, trend_direction, combined,
                                         two_layer_mode=two_layer_mode, morning=morning),
            "volatility_passed": True,
            "primary_trend": primary_trend,
            "secondary_score": round(secondary_score, 3),
            "entry_confidence": round(entry_confidence, 3),
            "atr_pct": round(atr_pct, 3),
            "adx": round(adx, 1),
            "secondary_indicators": secondary,
            "secondary_df": frames["secondary"],
            "layers": self._summarize_layers(analyses),
            "is_early": combined < 0.80 or two_layer_mode,
            "two_layer_mode": two_layer_mode,
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
                      two_layer_mode: bool = False, morning: bool = False) -> str:
        p = analyses["primary"]
        s = analyses["secondary"]
        e = analyses["entry"]

        mode = "2-LAYER" if two_layer_mode else "3-LAYER"
        tag = "🌅 MORNING BOOST | " if morning else ""
        parts = [f"{tag}{mode} ALIGNMENT ({signal}) | Confidence: {confidence * 100:.1f}%"]
        parts.append(f"Direction: {p.get('market_structure')} (strength: {p.get('trend_strength', 0):.2f})")
        parts.append(f"Setup: {s.get('smc_reasoning', 'N/A')[:80]}")

        if two_layer_mode:
            parts.append("Entry layer quiet — fired on direction+setup (penalty applied)")
        elif e.get("live_reversal"):
            parts.append(f"Entry: {e.get('live_reversal')}")
        elif e.get("candlestick_signal") != "HOLD":
            parts.append(f"Entry pattern: {e.get('candlestick_signal')}")

        if s.get("fakeout_detected"):
            parts.append(f"Fakeout risk: {s.get('fakeout_risk', 0) * 100:.0f}%")

        return " | ".join(parts)