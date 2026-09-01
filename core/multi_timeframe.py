# core/multi_timeframe.py
# Multi-timeframe alignment engine — ALL 3 layers must agree
# BTCUSD: 15m → 5m → 1m  |  V10(1s): 5m → 1m → 10s

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
    ALL 3 must align + volatility filter + confidence threshold.
    """

    def __init__(self):
        self.analyzer = TechnicalAnalyzer()
        self.signal_tracker = SignalTracker(
            cooldown_minutes=int(os.getenv("SIGNAL_COOLDOWN_MINUTES", "10")),
            price_tolerance_pct=0.005
        )
        self.min_confidence = float(os.getenv("MIN_SIGNAL_CONFIDENCE", "0.65"))
        self.min_adx = float(os.getenv("MIN_ADX", "20"))
        self.min_atr_pct = float(os.getenv("MIN_ATR_PCT", "0.03"))

        logger.info(
            f"MultiTimeframeEngine | Min confidence: {self.min_confidence} | "
            f"Min ADX: {self.min_adx} | Min ATR%: {self.min_atr_pct}%"
        )

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

        if atr_pct < self.min_atr_pct or adx < self.min_adx:
            reason = []
            if atr_pct < self.min_atr_pct:
                reason.append(f"ATR {atr_pct:.3f}% < {self.min_atr_pct}%")
            if adx < self.min_adx:
                reason.append(f"ADX {adx:.1f} < {self.min_adx}")
            logger.info(f"{pair} VOLATILITY BLOCK: {' | '.join(reason)}")
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

        if not entry_confirms:
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

        if secondary.get("fakeout_detected") and secondary.get("fakeout_risk", 0) > 0.3:
            combined *= 0.7

        if combined < self.min_confidence:
            return {
                "pair": pair, "signal": "HOLD",
                "reason": f"Confidence {combined:.2f} < min {self.min_confidence}",
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
            "reason": self._build_reason(analyses, trend_direction, combined),
            "volatility_passed": True,
            "primary_trend": primary_trend,
            "secondary_score": round(secondary_score, 3),
            "entry_confidence": round(entry_confidence, 3),
            "atr_pct": round(atr_pct, 3),
            "adx": round(adx, 1),
            "secondary_indicators": secondary,
            "secondary_df": frames["secondary"],
            "layers": self._summarize_layers(analyses),
            "is_early": combined < 0.80,
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

    def _build_reason(self, analyses: dict, signal: str, confidence: float) -> str:
        p = analyses["primary"]
        s = analyses["secondary"]
        e = analyses["entry"]

        parts = [f"3-LAYER ALIGNMENT ({signal}) | Confidence: {confidence * 100:.1f}%"]
        parts.append(f"Direction: {p.get('market_structure')} (strength: {p.get('trend_strength', 0):.2f})")
        parts.append(f"Setup: {s.get('smc_reasoning', 'N/A')[:80]}")

        if e.get("live_reversal"):
            parts.append(f"Entry: {e.get('live_reversal')}")
        elif e.get("candlestick_signal") != "HOLD":
            parts.append(f"Entry pattern: {e.get('candlestick_signal')}")

        if s.get("fakeout_detected"):
            parts.append(f"Fakeout risk: {s.get('fakeout_risk', 0) * 100:.0f}%")

        return " | ".join(parts)