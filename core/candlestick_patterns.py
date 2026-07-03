# core/candlestick_patterns.py
# Real-time candlestick pattern recognition for EARLY reversal detection
# v2: Added triangle consolidation detection

import numpy as np
import pandas as pd
from typing import Dict, Optional, List
import logging

logger = logging.getLogger(__name__)


class CandlestickPatternRecognizer:
    """Detects classic reversal patterns as they form."""

    def __init__(self):
        self.patterns_detected = {}
        logger.info("CandlestickPatternRecognizer initialized")

    def analyze(self, df: pd.DataFrame) -> Dict:
        if len(df) < 5:
            return {"signal": "HOLD", "confidence": 0.0, "patterns": [], "triangle": None}

        patterns = []
        single = self._single_candle_patterns(df)
        if single:
            patterns.extend(single)
        double = self._double_candle_patterns(df)
        if double:
            patterns.extend(double)
        triple = self._triple_candle_patterns(df)
        if triple:
            patterns.extend(triple)
        triangle = self._detect_triangle(df)
        if triangle and triangle.get("breakout_signal"):
            patterns.append(triangle["breakout_signal"])

        bullish_score = sum(p['strength'] for p in patterns if p['direction'] == 'BULLISH')
        bearish_score = sum(p['strength'] for p in patterns if p['direction'] == 'BEARISH')

        signal = "HOLD"
        confidence = 0.0
        if bullish_score > bearish_score and bullish_score >= 0.5:
            signal = "BUY"
            confidence = min(bullish_score, 0.95)
        elif bearish_score > bullish_score and bearish_score >= 0.5:
            signal = "SELL"
            confidence = min(bearish_score, 0.95)

        return {
            "signal": signal,
            "confidence": confidence,
            "patterns": patterns,
            "bullish_score": bullish_score,
            "bearish_score": bearish_score,
            "triangle": triangle
        }

    def _single_candle_patterns(self, df: pd.DataFrame) -> List[Dict]:
        patterns = []
        if len(df) < 3:
            return patterns
        c = df.iloc[-1]
        body = abs(c['close'] - c['open'])
        total_range = c['high'] - c['low'] if c['high'] != c['low'] else 0.0001
        lower_wick = min(c['open'], c['close']) - c['low']
        upper_wick = c['high'] - max(c['open'], c['close'])
        is_bull = c['close'] > c['open']
        is_bear = c['close'] < c['open']

        # HAMMER
        if is_bull and body > 0 and lower_wick >= body * 2 and upper_wick <= body * 0.1:
            if c['low'] <= np.percentile(df['low'].tail(10).values, 20):
                patterns.append({"name": "HAMMER", "direction": "BULLISH", "strength": 0.70,
                                 "description": f"Hammer at {c['low']:.2f}"})
        # INVERTED HAMMER
        if is_bull and body > 0 and upper_wick >= body * 2 and lower_wick <= body * 0.1:
            if c['low'] <= np.percentile(df['low'].tail(10).values, 25):
                patterns.append({"name": "INVERTED_HAMMER", "direction": "BULLISH", "strength": 0.55,
                                 "description": f"Inverted hammer at {c['low']:.2f}"})
        # SHOOTING STAR
        if is_bear and body > 0 and upper_wick >= body * 2 and lower_wick <= body * 0.1:
            if c['high'] >= np.percentile(df['high'].tail(10).values, 80):
                patterns.append({"name": "SHOOTING_STAR", "direction": "BEARISH", "strength": 0.70,
                                 "description": f"Shooting star at {c['high']:.2f}"})
        # HANGING MAN
        if is_bear and body > 0 and lower_wick >= body * 2 and upper_wick <= body * 0.1:
            if c['high'] >= np.percentile(df['high'].tail(10).values, 75):
                patterns.append({"name": "HANGING_MAN", "direction": "BEARISH", "strength": 0.65,
                                 "description": f"Hanging man at {c['high']:.2f}"})
        # DOJI variants
        if body <= total_range * 0.05 and total_range > 0:
            if lower_wick >= total_range * 0.6 and c['low'] <= np.percentile(df['low'].tail(10).values, 20):
                patterns.append({"name": "DRAGONFLY_DOJI", "direction": "BULLISH", "strength": 0.60,
                                 "description": f"Dragonfly doji at support {c['low']:.2f}"})
            elif upper_wick >= total_range * 0.6 and c['high'] >= np.percentile(df['high'].tail(10).values, 80):
                patterns.append({"name": "GRAVESTONE_DOJI", "direction": "BEARISH", "strength": 0.60,
                                 "description": f"Gravestone doji at resistance {c['high']:.2f}"})
        # LONG LEGGED DOJI
        if body <= total_range * 0.05 and lower_wick >= total_range * 0.3 and upper_wick >= total_range * 0.3:
            if c['low'] <= np.percentile(df['low'].tail(10).values, 15):
                patterns.append({"name": "LONG_LEGGED_DOJI_BULL", "direction": "BULLISH", "strength": 0.50,
                                 "description": f"Long-legged doji at bottom {c['low']:.2f}"})
            elif c['high'] >= np.percentile(df['high'].tail(10).values, 85):
                patterns.append({"name": "LONG_LEGGED_DOJI_BEAR", "direction": "BEARISH", "strength": 0.50,
                                 "description": f"Long-legged doji at top {c['high']:.2f}"})
        # SPINNING TOP
        if body <= total_range * 0.3 and body > total_range * 0.05 and lower_wick >= body and upper_wick >= body:
            if c['low'] <= np.percentile(df['low'].tail(10).values, 20):
                patterns.append({"name": "SPINNING_TOP_BULL", "direction": "BULLISH", "strength": 0.35,
                                 "description": "Spinning top near low"})
            elif c['high'] >= np.percentile(df['high'].tail(10).values, 80):
                patterns.append({"name": "SPINNING_TOP_BEAR", "direction": "BEARISH", "strength": 0.35,
                                 "description": "Spinning top near high"})
        return patterns

    def _double_candle_patterns(self, df: pd.DataFrame) -> List[Dict]:
        patterns = []
        if len(df) < 3:
            return patterns
        c1 = df.iloc[-2]
        c2 = df.iloc[-1]
        # BULLISH ENGULFING
        if (c1['close'] < c1['open'] and c2['close'] > c2['open'] and
            c2['open'] <= c1['close'] and c2['close'] >= c1['open']):
            c1_body = c1['open'] - c1['close']
            c2_body = c2['close'] - c2['open']
            strength = 0.75 if c2_body >= c1_body * 1.5 else 0.65
            patterns.append({"name": "BULLISH_ENGULFING", "direction": "BULLISH", "strength": strength,
                             "description": f"Bullish engulfing: {c1['close']:.2f} -> {c2['close']:.2f}"})
        # BEARISH ENGULFING
        if (c1['close'] > c1['open'] and c2['close'] < c2['open'] and
            c2['open'] >= c1['close'] and c2['close'] <= c1['open']):
            c1_body = c1['close'] - c1['open']
            c2_body = c2['open'] - c2['close']
            strength = 0.75 if c2_body >= c1_body * 1.5 else 0.65
            patterns.append({"name": "BEARISH_ENGULFING", "direction": "BEARISH", "strength": strength,
                             "description": f"Bearish engulfing: {c1['close']:.2f} -> {c2['close']:.2f}"})
        # PIERCING PATTERN
        c1_body = c1['open'] - c1['close']
        c1_range = c1['high'] - c1['low']
        if (c1['close'] < c1['open'] and c2['close'] > c2['open'] and
            c2['open'] < c1['close'] and c2['close'] > c1['open'] - c1_body * 0.5 and
            c1_body > c1_range * 0.5):
            patterns.append({"name": "PIERCING_PATTERN", "direction": "BULLISH", "strength": 0.60,
                             "description": f"Piercing pattern at {c2['close']:.2f}"})
        # DARK CLOUD COVER
        c1_body = c1['close'] - c1['open']
        if (c1['close'] > c1['open'] and c2['close'] < c2['open'] and
            c2['open'] > c1['close'] and c2['close'] < c1['open'] + c1_body * 0.5 and
            c1_body > (c1['high'] - c1['low']) * 0.5):
            patterns.append({"name": "DARK_CLOUD_COVER", "direction": "BEARISH", "strength": 0.60,
                             "description": f"Dark cloud cover at {c2['close']:.2f}"})
        # TWEEZER BOTTOM
        if abs(c1['low'] - c2['low']) / c1['low'] < 0.001 and c1['close'] < c1['open'] and c2['close'] > c2['open']:
            patterns.append({"name": "TWEEZER_BOTTOM", "direction": "BULLISH", "strength": 0.55,
                             "description": f"Tweezer bottom at {c1['low']:.2f}"})
        # TWEEZER TOP
        if abs(c1['high'] - c2['high']) / c1['high'] < 0.001 and c1['close'] > c1['open'] and c2['close'] < c2['open']:
            patterns.append({"name": "TWEEZER_TOP", "direction": "BEARISH", "strength": 0.55,
                             "description": f"Tweezer top at {c1['high']:.2f}"})
        return patterns

    def _triple_candle_patterns(self, df: pd.DataFrame) -> List[Dict]:
        patterns = []
        if len(df) < 4:
            return patterns
        c1 = df.iloc[-3]
        c2 = df.iloc[-2]
        c3 = df.iloc[-1]
        # MORNING STAR
        c1_body = c1['open'] - c1['close']
        c2_body = abs(c2['close'] - c2['open'])
        c1_range = c1['high'] - c1['low']
        if (c1['close'] < c1['open'] and c3['close'] > c3['open'] and
            c1_body > c1_range * 0.5 and c2_body < c1_body * 0.3 and
            c3['close'] > (c1['open'] + c1['close']) / 2):
            patterns.append({"name": "MORNING_STAR", "direction": "BULLISH", "strength": 0.85,
                             "description": f"Morning star: {c1['close']:.2f} -> {c3['close']:.2f}"})
        # EVENING STAR
        c1_body = c1['close'] - c1['open']
        c2_body = abs(c2['close'] - c2['open'])
        if (c1['close'] > c1['open'] and c3['close'] < c3['open'] and
            c1_body > (c1['high'] - c1['low']) * 0.5 and c2_body < c1_body * 0.3 and
            c3['close'] < (c1['open'] + c1['close']) / 2):
            patterns.append({"name": "EVENING_STAR", "direction": "BEARISH", "strength": 0.85,
                             "description": f"Evening star: {c1['close']:.2f} -> {c3['close']:.2f}"})
        # THREE WHITE SOLDIERS
        if (c1['close'] > c1['open'] and c2['close'] > c2['open'] and c3['close'] > c3['open'] and
            c2['open'] > c1['open'] and c2['open'] < c1['close'] and
            c3['open'] > c2['open'] and c3['open'] < c2['close'] and
            (c3['close'] - c3['open']) > (c1['close'] - c1['open']) * 0.5):
            patterns.append({"name": "THREE_WHITE_SOLDIERS", "direction": "BULLISH", "strength": 0.80,
                             "description": f"Three white soldiers: {c1['close']:.2f} -> {c3['close']:.2f}"})
        # THREE BLACK CROWS
        if (c1['close'] < c1['open'] and c2['close'] < c2['open'] and c3['close'] < c3['open'] and
            c2['open'] < c1['open'] and c2['open'] > c1['close'] and
            c3['open'] < c2['open'] and c3['open'] > c2['close']):
            patterns.append({"name": "THREE_BLACK_CROWS", "direction": "BEARISH", "strength": 0.80,
                             "description": f"Three black crows: {c1['close']:.2f} -> {c3['close']:.2f}"})
        # THREE INSIDE UP
        if (c1['close'] < c1['open'] and c2['close'] > c2['open'] and
            c2['open'] > c1['close'] and c2['close'] < c1['open'] and
            c3['close'] > c3['open'] and c3['close'] > c1['open']):
            patterns.append({"name": "THREE_INSIDE_UP", "direction": "BULLISH", "strength": 0.70,
                             "description": f"Three inside up: {c1['close']:.2f} -> {c3['close']:.2f}"})
        # THREE INSIDE DOWN
        if (c1['close'] > c1['open'] and c2['close'] < c2['open'] and
            c2['open'] < c1['close'] and c2['close'] > c1['open'] and
            c3['close'] < c3['open'] and c3['close'] < c1['open']):
            patterns.append({"name": "THREE_INSIDE_DOWN", "direction": "BEARISH", "strength": 0.70,
                             "description": f"Three inside down: {c1['close']:.2f} -> {c3['close']:.2f}"})
        return patterns

    def _detect_triangle(self, df: pd.DataFrame, lookback: int = 20) -> Optional[Dict]:
        if len(df) < lookback + 2:
            return None
        window = df.tail(lookback).reset_index(drop=True)
        swing_high_idx, swing_high_val = [], []
        swing_low_idx, swing_low_val = [], []
        for i in range(1, len(window) - 1):
            if (window['high'].iloc[i] > window['high'].iloc[i - 1] and
                window['high'].iloc[i] > window['high'].iloc[i + 1]):
                swing_high_idx.append(i)
                swing_high_val.append(window['high'].iloc[i])
            if (window['low'].iloc[i] < window['low'].iloc[i - 1] and
                window['low'].iloc[i] < window['low'].iloc[i + 1]):
                swing_low_idx.append(i)
                swing_low_val.append(window['low'].iloc[i])
        if len(swing_high_idx) < 2 or len(swing_low_idx) < 2:
            return None
        high_slope, high_intercept = np.polyfit(swing_high_idx, swing_high_val, 1)
        low_slope, low_intercept = np.polyfit(swing_low_idx, swing_low_val, 1)
        avg_price = window['close'].mean()
        flat_threshold = avg_price * 0.0003
        highs_falling = high_slope < -flat_threshold
        highs_flat = abs(high_slope) <= flat_threshold
        lows_rising = low_slope > flat_threshold
        lows_flat = abs(low_slope) <= flat_threshold
        triangle_type = None
        if highs_falling and lows_rising:
            triangle_type = "SYMMETRICAL"
        elif highs_flat and lows_rising:
            triangle_type = "ASCENDING"
        elif highs_falling and lows_flat:
            triangle_type = "DESCENDING"
        else:
            return None
        last_x = len(window) - 1
        upper_now = high_slope * last_x + high_intercept
        lower_now = low_slope * last_x + low_intercept
        if upper_now <= lower_now:
            return None
        range_start = high_intercept - low_intercept
        range_now = upper_now - lower_now
        if range_now >= range_start * 0.85:
            return None
        apex_distance = None
        if high_slope != low_slope:
            apex_x = (low_intercept - high_intercept) / (high_slope - low_slope)
            apex_distance = max(int(apex_x - last_x), 0)
        result = {
            "type": triangle_type, "armed": True, "apex_distance": apex_distance,
            "upper_level": float(upper_now), "lower_level": float(lower_now), "breakout_signal": None
        }
        last_candle = df.iloc[-1]
        breakout_buffer = avg_price * 0.0005
        if last_candle['close'] > upper_now + breakout_buffer:
            result["breakout_signal"] = {
                "name": f"TRIANGLE_BREAKOUT_{triangle_type}", "direction": "BULLISH",
                "strength": 0.75 if triangle_type != "DESCENDING" else 0.55,
                "description": f"{triangle_type} triangle breakout above {upper_now:.2f}"
            }
        elif last_candle['close'] < lower_now - breakout_buffer:
            result["breakout_signal"] = {
                "name": f"TRIANGLE_BREAKOUT_{triangle_type}", "direction": "BEARISH",
                "strength": 0.75 if triangle_type != "ASCENDING" else 0.55,
                "description": f"{triangle_type} triangle breakdown below {lower_now:.2f}"
            }
        return result

    def get_pattern_summary(self, patterns: List[Dict]) -> str:
        if not patterns:
            return "No clear pattern detected"
        lines = []
        for p in patterns:
            emoji = "B" if p['direction'] == 'BULLISH' else "S"
            lines.append(f"{emoji}:{p['name']}({p['strength']*100:.0f}%)")
        return " | ".join(lines)

