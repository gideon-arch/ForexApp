# core/fakeout_detector.py
# Fakeout (False Breakout) Detection Engine

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)


class FakeoutDetector:
    """Detects false breakouts to avoid liquidity traps."""

    def __init__(self):
        self.fakeout_history = []
        logger.info("FakeoutDetector initialized")

    def analyze(self, df: pd.DataFrame, support: Optional[float], resistance: Optional[float],
                market_structure: str = "SIDEWAYS") -> Dict:
        if len(df) < 10 or (support is None and resistance is None):
            return {
                "fakeout_detected": False,
                "fakeout_direction": None,
                "fakeout_risk": 0.0,
                "trap_type": None,
                "reasoning": "Insufficient data or no levels",
                "confidence": 0.0
            }

        current_price = df["close"].iloc[-1]
        current_open = df["open"].iloc[-1]
        current_high = df["high"].iloc[-1]
        current_low = df["low"].iloc[-1]

        results = []

        if resistance:
            res = self._check_resistance_fakeout(df, resistance, current_price, current_high, current_open)
            if res["detected"]:
                results.append(res)

        if support:
            sup = self._check_support_fakeout(df, support, current_price, current_low, current_open)
            if sup["detected"]:
                results.append(sup)

        if support and resistance:
            rng = self._check_range_fakeout(df, support, resistance, current_price)
            if rng["detected"]:
                results.append(rng)

        wick = self._check_wick_fakeout(df, support, resistance, current_price)
        if wick["detected"]:
            results.append(wick)

        mom = self._check_momentum_fakeout(df, support, resistance, current_price)
        if mom["detected"]:
            results.append(mom)

        if not results:
            return {
                "fakeout_detected": False,
                "fakeout_direction": None,
                "fakeout_risk": 0.0,
                "trap_type": None,
                "reasoning": "No fakeout detected",
                "confidence": 0.0
            }

        results.sort(key=lambda x: x["confidence"], reverse=True)
        best = results[0]
        fakeout_risk = min(sum(r["confidence"] for r in results) / len(results) + 0.1, 1.0)

        return {
            "fakeout_detected": True,
            "fakeout_direction": best["direction"],
            "fakeout_risk": fakeout_risk,
            "trap_type": best["trap_type"],
            "reasoning": best["reasoning"],
            "confidence": best["confidence"],
            "all_signals": results
        }

    def _check_resistance_fakeout(self, df, resistance, current_price, current_high, current_open):
        if len(df) < 5:
            return {"detected": False, "confidence": 0.0}

        recent = df.tail(5)
        broke_above = False
        break_idx = None

        for i, (_, c) in enumerate(recent.iterrows()):
            if c["high"] > resistance and c["close"] > resistance:
                broke_above = True
                break_idx = i
                break

        if not broke_above or break_idx is None:
            return {"detected": False, "confidence": 0.0}

        if current_price < resistance:
            bc = recent.iloc[break_idx]
            break_extent = (bc["high"] - resistance) / resistance
            reversal_extent = (resistance - current_price) / resistance

            if break_extent > 0.001 and reversal_extent > break_extent * 0.5:
                conf = min(0.50 + break_extent * 100 + reversal_extent * 100, 0.95)
                body = abs(current_price - current_open)
                upper_wick = current_high - max(current_price, current_open)
                if upper_wick > body * 2:
                    conf = min(conf + 0.15, 0.95)
                return {
                    "detected": True,
                    "direction": "BEARISH",
                    "trap_type": "BULL_TRAP",
                    "confidence": conf,
                    "reasoning": f"Bull trap: broke R {resistance:.2f} by {break_extent*100:.2f}%, reversed {reversal_extent*100:.2f}%"
                }
        return {"detected": False, "confidence": 0.0}

    def _check_support_fakeout(self, df, support, current_price, current_low, current_open):
        if len(df) < 5:
            return {"detected": False, "confidence": 0.0}

        recent = df.tail(5)
        broke_below = False
        break_idx = None

        for i, (_, c) in enumerate(recent.iterrows()):
            if c["low"] < support and c["close"] < support:
                broke_below = True
                break_idx = i
                break

        if not broke_below or break_idx is None:
            return {"detected": False, "confidence": 0.0}

        if current_price > support:
            bc = recent.iloc[break_idx]
            break_extent = (support - bc["low"]) / support
            reversal_extent = (current_price - support) / support

            if break_extent > 0.001 and reversal_extent > break_extent * 0.5:
                conf = min(0.50 + break_extent * 100 + reversal_extent * 100, 0.95)
                body = abs(current_price - current_open)
                lower_wick = min(current_price, current_open) - current_low
                if lower_wick > body * 2:
                    conf = min(conf + 0.15, 0.95)
                return {
                    "detected": True,
                    "direction": "BULLISH",
                    "trap_type": "BEAR_TRAP",
                    "confidence": conf,
                    "reasoning": f"Bear trap: broke S {support:.2f} by {break_extent*100:.2f}%, reversed {reversal_extent*100:.2f}%"
                }
        return {"detected": False, "confidence": 0.0}

    def _check_range_fakeout(self, df, support, resistance, current_price):
        range_size = resistance - support
        if range_size <= 0:
            return {"detected": False, "confidence": 0.0}

        mid = (support + resistance) / 2
        dist = abs(current_price - mid) / range_size
        if dist > 0.4:
            return {"detected": False, "confidence": 0.0}

        highs = df["high"].tail(5).values
        lows = df["low"].tail(5).values
        touched_r = any(h > resistance * 0.998 for h in highs)
        touched_s = any(l < support * 1.002 for l in lows)

        if touched_r and touched_s:
            return {
                "detected": True,
                "direction": "NEUTRAL",
                "trap_type": "RANGE_WHIPSAW",
                "confidence": 0.60,
                "reasoning": f"Whipsaw between {support:.2f} and {resistance:.2f}"
            }
        return {"detected": False, "confidence": 0.0}

    def _check_wick_fakeout(self, df, support, resistance, current_price):
        if len(df) < 3:
            return {"detected": False, "confidence": 0.0}

        c = df.iloc[-1]
        body = abs(c["close"] - c["open"])
        total_range = c["high"] - c["low"]

        if total_range == 0 or body == 0:
            return {"detected": False, "confidence": 0.0}

        if resistance and c["high"] > resistance:
            upper_wick = c["high"] - max(c["close"], c["open"])
            if upper_wick > body * 1.5 and c["close"] < resistance:
                return {
                    "detected": True,
                    "direction": "BEARISH",
                    "trap_type": "WICK_REJECTION_RESISTANCE",
                    "confidence": 0.70,
                    "reasoning": f"Upper wick pierced R {resistance:.2f} but closed below"
                }

        if support and c["low"] < support:
            lower_wick = min(c["close"], c["open"]) - c["low"]
            if lower_wick > body * 1.5 and c["close"] > support:
                return {
                    "detected": True,
                    "direction": "BULLISH",
                    "trap_type": "WICK_REJECTION_SUPPORT",
                    "confidence": 0.70,
                    "reasoning": f"Lower wick pierced S {support:.2f} but closed above"
                }

        return {"detected": False, "confidence": 0.0}

    def _check_momentum_fakeout(self, df, support, resistance, current_price):
        if len(df) < 10 or "rsi" not in df.columns:
            return {"detected": False, "confidence": 0.0}

        rsi_vals = df["rsi"].tail(10).values
        highs = df["high"].tail(10).values
        lows = df["low"].tail(10).values

        if resistance and current_price > resistance * 0.995:
            if len(highs) >= 5 and len(rsi_vals) >= 5:
                if highs[-1] > highs[-5] and rsi_vals[-1] < rsi_vals[-5]:
                    return {
                        "detected": True,
                        "direction": "BEARISH",
                        "trap_type": "MOMENTUM_FAKEOUT_BULL",
                        "confidence": 0.65,
                        "reasoning": f"Price broke toward R but RSI divergence ({rsi_vals[-1]:.1f} vs {rsi_vals[-5]:.1f})"
                    }

        if support and current_price < support * 1.005:
            if len(lows) >= 5 and len(rsi_vals) >= 5:
                if lows[-1] < lows[-5] and rsi_vals[-1] > rsi_vals[-5]:
                    return {
                        "detected": True,
                        "direction": "BULLISH",
                        "trap_type": "MOMENTUM_FAKEOUT_BEAR",
                        "confidence": 0.65,
                        "reasoning": f"Price broke toward S but RSI divergence ({rsi_vals[-1]:.1f} vs {rsi_vals[-5]:.1f})"
                    }

        return {"detected": False, "confidence": 0.0}

    def should_block_signal(self, fakeout_result: Dict, proposed_signal: str) -> Tuple[bool, str]:
        if not fakeout_result["fakeout_detected"]:
            return False, "No fakeout detected"

        direction = fakeout_result["fakeout_direction"]
        risk = fakeout_result["fakeout_risk"]
        trap = fakeout_result["trap_type"]

        if risk >= 0.75:
            if proposed_signal == "BUY" and direction == "BEARISH":
                return True, f"BLOCKED: {trap} ({risk*100:.0f}% risk). Real move DOWN."
            if proposed_signal == "SELL" and direction == "BULLISH":
                return True, f"BLOCKED: {trap} ({risk*100:.0f}% risk). Real move UP."

        if risk >= 0.50:
            if proposed_signal == "BUY" and direction == "BEARISH":
                return True, f"BLOCKED: {trap} ({risk*100:.0f}% risk). Wait for confirmation."
            if proposed_signal == "SELL" and direction == "BULLISH":
                return True, f"BLOCKED: {trap} ({risk*100:.0f}% risk). Wait for confirmation."

        if risk >= 0.30:
            if proposed_signal == "BUY" and direction == "BEARISH":
                return False, f"WARNING: {trap} possible. Use tight stop."
            if proposed_signal == "SELL" and direction == "BULLISH":
                return False, f"WARNING: {trap} possible. Use tight stop."

        return False, "Fakeout risk low"

