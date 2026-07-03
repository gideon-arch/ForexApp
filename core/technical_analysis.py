# core/technical_analysis.py
# PREDICTIVE SMC Strategy — Signals BEFORE the move, not after
# VERSION 4.0 — CONFLUENCE SCORING (No LSTM)
# Candlestick patterns + Live candle + RSI divergence + Fakeout protection
# All signals ADD UP instead of hard priority exits

import os
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
import ta
import logging

from .candlestick_patterns import CandlestickPatternRecognizer
from .fakeout_detector import FakeoutDetector

logger = logging.getLogger(__name__)


class TechnicalAnalyzer:
    def __init__(self):
        self.rsi_buy_low = float(os.getenv("RSI_BUY_LOW", "40"))
        self.rsi_sell_high = float(os.getenv("RSI_SELL_HIGH", "60"))
        self.rsi_midpoint = (self.rsi_buy_low + self.rsi_sell_high) / 2
        self.rsi_range_half = max(self.rsi_sell_high - self.rsi_midpoint, 1.0)
        self.pattern_recognizer = CandlestickPatternRecognizer()
        self.fakeout_detector = FakeoutDetector()
        logger.info(
            f"TechnicalAnalyzer v4.0 | RSI: {self.rsi_buy_low}-{self.rsi_sell_high} | "
            f"CONFLUENCE SCORING — Candlestick + Live + Divergence + Fakeout"
        )

    def calculate_indicators(self, df: pd.DataFrame) -> Dict:
        df = df.copy()
        min_candles = 50
        if len(df) < min_candles:
            raise ValueError(f"Need {min_candles}+ candles, got {len(df)}")

        try:
            df["sma_20"] = ta.trend.sma_indicator(df["close"], window=min(20, len(df)//3))
            df["sma_50"] = ta.trend.sma_indicator(df["close"], window=min(50, len(df)//2))
            df["ema_12"] = ta.trend.ema_indicator(df["close"], window=12)
            df["ema_26"] = ta.trend.ema_indicator(df["close"], window=26)
            df["ema_200"] = ta.trend.ema_indicator(df["close"], window=min(200, len(df)-10))

            df["rsi"] = ta.momentum.rsi(df["close"], window=14)

            macd = ta.trend.MACD(df["close"])
            df["macd"] = macd.macd()
            df["macd_signal"] = macd.macd_signal()
            df["macd_hist"] = macd.macd_diff()

            bb = ta.volatility.BollingerBands(df["close"])
            df["bb_high"] = bb.bollinger_hband()
            df["bb_low"] = bb.bollinger_lband()
            df["bb_middle"] = bb.bollinger_mavg()
            df["atr"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"])
            df["adx"] = ta.trend.adx(df["high"], df["low"], df["close"])

            if "volume" in df.columns and df["volume"].sum() > 0:
                df["obv"] = ta.volume.on_balance_volume(df["close"], df["volume"])
            else:
                df["obv"] = 0.0

            df["returns"] = df["close"].pct_change()
            df["volatility"] = df["returns"].rolling(window=20, min_periods=1).std().fillna(0)
            df.dropna(inplace=True)

        except Exception as e:
            logger.error(f"Indicator error: {e}")
            raise

        trend, swing_highs, swing_lows = self._find_market_structure(df)
        support, resistance = self._mark_support_resistance(df, swing_highs, swing_lows)
        sweep_type, sweep_quality = self._detect_liquidity_sweep(df, support, resistance)

        pattern_result = self.pattern_recognizer.analyze(df)
        live_reversal, live_score = self._detect_live_reversal(df)
        div_type, div_score = self._detect_rsi_divergence(df)
        fakeout_result = self.fakeout_detector.analyze(df, support, resistance, trend)

        current_rsi = df["rsi"].iloc[-1]
        rsi_zone = "NEUTRAL"
        if current_rsi <= 25.0: rsi_zone = "EXTREME_OVERSOLD"
        elif current_rsi <= self.rsi_buy_low: rsi_zone = "OVERSOLD"
        elif current_rsi >= 70.0: rsi_zone = "EXTREME_OVERBOUGHT"
        elif current_rsi >= self.rsi_sell_high: rsi_zone = "OVERBOUGHT"

        rsi_momentum = self._calculate_rsi_momentum(df)
        price_momentum = self._calculate_price_momentum(df)

        confluence_score, target_signal, reasoning = self._calculate_confluence(
            trend=trend,
            sweep_type=sweep_type,
            sweep_quality=sweep_quality,
            rsi=current_rsi,
            rsi_momentum=rsi_momentum,
            price_momentum=price_momentum,
            support=support,
            resistance=resistance,
            df=df,
            live_reversal=live_reversal,
            live_score=live_score,
            div_type=div_type,
            div_score=div_score,
            pattern_result=pattern_result,
            fakeout_result=fakeout_result
        )

        entry, stop, target = self._calculate_levels(target_signal, df, support, resistance)

        return {
            "current_price": float(df["close"].iloc[-1]),
            "sma_20": float(df["sma_20"].iloc[-1]) if "sma_20" in df else 0.0,
            "sma_50": float(df["sma_50"].iloc[-1]) if "sma_50" in df else 0.0,
            "ema_200": float(df["ema_200"].iloc[-1]) if "ema_200" in df else 0.0,
            "rsi": float(current_rsi),
            "rsi_zone": rsi_zone,
            "rsi_momentum": float(rsi_momentum),
            "price_momentum": float(price_momentum),
            "macd": float(df["macd"].iloc[-1]),
            "macd_signal": float(df["macd_signal"].iloc[-1]),
            "macd_hist": float(df["macd_hist"].iloc[-1]),
            "bb_high": float(df["bb_high"].iloc[-1]),
            "bb_low": float(df["bb_low"].iloc[-1]),
            "bb_middle": float(df["bb_middle"].iloc[-1]),
            "bb_position": float(self._calculate_bb_position(df)),
            "atr": float(df["atr"].iloc[-1]),
            "adx": float(df["adx"].iloc[-1]),
            "obv": float(df["obv"].iloc[-1]),
            "trend_strength": float(self._calculate_trend_strength(df)),
            "market_structure": trend,
            "active_support": float(support) if support else None,
            "active_resistance": float(resistance) if resistance else None,
            "liquidity_sweep": sweep_type,
            "sweep_quality": float(sweep_quality),
            "confluence_score": float(confluence_score),
            "smc_suggested_signal": target_signal,
            "suggested_entry": float(entry) if entry else None,
            "suggested_stop_loss": float(stop) if stop else None,
            "suggested_take_profit": float(target) if target else None,
            "smc_reasoning": reasoning,
            "live_reversal": live_reversal,
            "live_reversal_score": float(live_score),
            "rsi_divergence": div_type,
            "rsi_divergence_score": float(div_score),
            "candlestick_patterns": pattern_result.get("patterns", []),
            "candlestick_signal": pattern_result.get("signal", "HOLD"),
            "candlestick_confidence": pattern_result.get("confidence", 0.0),
            "candlestick_bullish_score": pattern_result.get("bullish_score", 0.0),
            "candlestick_bearish_score": pattern_result.get("bearish_score", 0.0),
            "triangle": pattern_result.get("triangle"),
            "fakeout_detected": fakeout_result.get("fakeout_detected", False),
            "fakeout_direction": fakeout_result.get("fakeout_direction"),
            "fakeout_risk": fakeout_result.get("fakeout_risk", 0.0),
            "fakeout_trap_type": fakeout_result.get("trap_type"),
            "fakeout_reasoning": fakeout_result.get("reasoning", ""),
            "fakeout_confidence": fakeout_result.get("confidence", 0.0),
            "dataframe": df,
        }

    def _find_market_structure(self, df: pd.DataFrame, window: int = 5) -> Tuple[str, list, list]:
        df["swing_high"] = df["high"] == df["high"].rolling(window=window*2+1, center=True).max()
        df["swing_low"] = df["low"] == df["low"].rolling(window=window*2+1, center=True).min()

        recent_highs = df[df["swing_high"]]["high"].tail(8).tolist()
        recent_lows = df[df["swing_low"]]["low"].tail(8).tolist()

        trend = "SIDEWAYS"
        if len(recent_highs) >= 2 and len(recent_lows) >= 2:
            hh = recent_highs[-1] > recent_highs[-2]
            hl = recent_lows[-1] > recent_lows[-2]
            lh = recent_highs[-1] < recent_highs[-2]
            ll = recent_lows[-1] < recent_lows[-2]

            if hh or hl:
                trend = "BULLISH" if (hh and hl) else "WEAK_BULLISH"
            elif lh or ll:
                trend = "BEARISH" if (lh and ll) else "WEAK_BEARISH"

        return trend, recent_highs, recent_lows

    def _mark_support_resistance(self, df: pd.DataFrame, swing_highs: list, swing_lows: list) -> Tuple[Optional[float], Optional[float]]:
        current_price = df["close"].iloc[-1]
        support = max([l for l in swing_lows if l < current_price]) if [l for l in swing_lows if l < current_price] else None
        resistance = min([h for h in swing_highs if h > current_price]) if [h for h in swing_highs if h > current_price] else None
        return support, resistance

    def _detect_liquidity_sweep(self, df: pd.DataFrame, support: Optional[float], resistance: Optional[float]) -> Tuple[Optional[str], float]:
        if len(df) < 5:
            return None, 0.0
        sweep_type = None
        quality = 0.0
        current_price = df["close"].iloc[-1]
        tolerance = current_price * 0.0005

        for i in range(-4, 0):
            candle = df.iloc[i]
            if support and candle["low"] < (support + tolerance) and candle["close"] > (support - tolerance):
                if df["close"].iloc[-1] > df["open"].iloc[-1]:
                    sweep_type = "BUY_SWEEP"
                    wick_depth = (support - candle["low"]) / support if support > 0 else 0
                    denom = candle["high"] - candle["low"]
                    recovery = (candle["close"] - candle["low"]) / denom if denom > 0 else 0.0001
                    quality = min(wick_depth * 100 + recovery, 1.0)
                    break

            if resistance and candle["high"] > (resistance - tolerance) and candle["close"] < (resistance + tolerance):
                if df["close"].iloc[-1] < df["open"].iloc[-1]:
                    sweep_type = "SELL_SWEEP"
                    wick_height = (candle["high"] - resistance) / resistance if resistance > 0 else 0
                    denom = candle["high"] - candle["low"]
                    recovery = (candle["high"] - candle["close"]) / denom if denom > 0 else 0.0001
                    quality = min(wick_height * 100 + recovery, 1.0)
                    break

        return sweep_type, quality

    def _detect_live_reversal(self, df: pd.DataFrame) -> Tuple[Optional[str], float]:
        if len(df) < 10:
            return None, 0.0

        current = df.iloc[-1]
        prev = df.iloc[-2]
        body = abs(current['close'] - current['open'])
        total_range = current['high'] - current['low']
        if total_range == 0:
            return None, 0.0

        lower_wick = min(current['open'], current['close']) - current['low']
        upper_wick = current['high'] - max(current['open'], current['close'])

        if lower_wick > body * 1.5 and current['close'] > current['open']:
            recent_low = df['low'].tail(10).min()
            if current['low'] <= recent_low * 1.002:
                rsi = df['rsi'].iloc[-1] if 'rsi' in df else 50
                if rsi <= 45:
                    return "BUY_EARLY", 0.65
                return "BUY_EARLY", 0.50

        if upper_wick > body * 1.5 and current['close'] < current['open']:
            recent_high = df['high'].tail(10).max()
            if current['high'] >= recent_high * 0.998:
                rsi = df['rsi'].iloc[-1] if 'rsi' in df else 50
                if rsi >= 55:
                    return "SELL_EARLY", 0.65
                return "SELL_EARLY", 0.50

        if current['close'] > current['open'] and prev['close'] < prev['open']:
            if current['open'] <= prev['close'] and current['close'] >= prev['open']:
                return "BUY_EARLY", 0.60

        if current['close'] < current['open'] and prev['close'] > prev['open']:
            if current['open'] >= prev['close'] and current['close'] <= prev['open']:
                return "SELL_EARLY", 0.60

        return None, 0.0

    def _detect_rsi_divergence(self, df: pd.DataFrame) -> Tuple[str, float]:
        if len(df) < 20 or 'rsi' not in df.columns:
            return "NONE", 0.0

        price_lows = df['low'].tail(15).values
        rsi_lows = df['rsi'].tail(15).values

        for i in range(5, len(price_lows)):
            if price_lows[-1] < price_lows[-i] and rsi_lows[-1] > rsi_lows[-i]:
                if price_lows[-1] < price_lows[-3]:
                    return "BULLISH_DIV", 0.70

        price_highs = df['high'].tail(15).values
        rsi_highs = df['rsi'].tail(15).values

        for i in range(5, len(price_highs)):
            if price_highs[-1] > price_highs[-i] and rsi_highs[-1] < rsi_highs[-i]:
                if price_highs[-1] > price_highs[-3]:
                    return "BEARISH_DIV", 0.70

        return "NONE", 0.0

    def _calculate_rsi_momentum(self, df: pd.DataFrame) -> float:
        if len(df) < 5:
            return 0.0
        rsi_values = df["rsi"].tail(5).values
        if len(rsi_values) < 3:
            return 0.0
        x = np.array([0, 1, 2])
        y = rsi_values[-3:]
        slope = np.polyfit(x, y, 1)[0]
        return float(slope)

    def _calculate_price_momentum(self, df: pd.DataFrame) -> float:
        if len(df) < 5:
            return 0.0
        lows = df["low"].tail(5).values
        highs = df["high"].tail(5).values

        hl = lows[-1] > lows[-2] > lows[-3] if len(lows) >= 3 else False
        lh = highs[-1] < highs[-2] < highs[-3] if len(highs) >= 3 else False

        if hl:
            return 1.0
        elif lh:
            return -1.0
        return 0.0

    def _calculate_confluence(self, trend: str, sweep_type: Optional[str], sweep_quality: float,
                             rsi: float, rsi_momentum: float, price_momentum: float,
                             support: Optional[float], resistance: Optional[float],
                             df: pd.DataFrame, live_reversal: Optional[str], live_score: float,
                             div_type: str, div_score: float, pattern_result: Dict,
                             fakeout_result: Dict) -> Tuple[float, str, str]:
        reasons = []
        score = 0.0
        current_price = df["close"].iloc[-1]
        current_ema200 = df["ema_200"].iloc[-1] if "ema_200" in df else current_price
        is_macro_uptrend = current_price > current_ema200

        # STEP 1: FAKEOUT CHECK
        fakeout_blocks = False
        fakeout_boost = 0.0
        fakeout_risk = fakeout_result.get("fakeout_risk", 0.0)
        fakeout_direction = fakeout_result.get("fakeout_direction")

        if fakeout_result.get("fakeout_detected", False):
            trap_type = fakeout_result.get("trap_type", "UNKNOWN")
            reasons.append(f"FAKEOUT: {trap_type} (risk: {fakeout_risk*100:.0f}%, real: {fakeout_direction})")
            if fakeout_risk >= 0.50:
                fakeout_blocks = True
                if fakeout_direction == "BULLISH":
                    fakeout_boost = fakeout_risk * 0.25
                elif fakeout_direction == "BEARISH":
                    fakeout_boost = -fakeout_risk * 0.25

        # STEP 2: CANDLESTICK PATTERNS
        cs_bullish = pattern_result.get("bullish_score", 0.0)
        cs_bearish = pattern_result.get("bearish_score", 0.0)
        cs_patterns = pattern_result.get("patterns", [])

        if cs_patterns:
            pattern_names = [p['name'] for p in cs_patterns]
            reasons.append(f"PATTERNS: {', '.join(pattern_names[:3])}")

        score += cs_bullish
        score -= cs_bearish

        # STEP 3: LIVE CANDLE REVERSAL
        if live_reversal == "BUY_EARLY":
            reasons.append(f"LIVE: Bullish reversal forming (score: {live_score:.2f})")
            score += live_score
        elif live_reversal == "SELL_EARLY":
            reasons.append(f"LIVE: Bearish reversal forming (score: {live_score:.2f})")
            score -= live_score

        # STEP 4: RSI DIVERGENCE
        if div_type == "BULLISH_DIV":
            reasons.append(f"DIVERGENCE: Bullish (score: {div_score:.2f})")
            score += div_score
        elif div_type == "BEARISH_DIV":
            reasons.append(f"DIVERGENCE: Bearish (score: {div_score:.2f})")
            score -= div_score

        # STEP 5: LIQUIDITY SWEEP (SMC CORE)
        if sweep_type == "BUY_SWEEP":
            if not is_macro_uptrend:
                reasons.append(f"WARN: Price {current_price:.2f} < 200 EMA {current_ema200:.2f}")
                score *= 0.7
            else:
                reasons.append("OK: Price above 200 EMA")

            reasons.append(f"SWEEP: Buy at Support {support:.2f} (quality: {sweep_quality:.2f})")
            score += 0.4 * sweep_quality

            if rsi <= self.rsi_buy_low and rsi_momentum > 0:
                reasons.append(f"RSI turning up from {rsi:.1f} (mom: +{rsi_momentum:.2f})")
                score += 0.35
            elif rsi <= self.rsi_buy_low:
                reasons.append(f"RSI at {rsi:.1f} but not turning")
                score += 0.15

            if price_momentum > 0:
                reasons.append("Momentum: higher lows")
                score += 0.2

            if trend in ["BULLISH", "WEAK_BULLISH", "SIDEWAYS"]:
                reasons.append(f"Trend {trend} aligns")
                score += 0.1

        elif sweep_type == "SELL_SWEEP":
            if is_macro_uptrend:
                reasons.append(f"WARN: Price {current_price:.2f} > 200 EMA {current_ema200:.2f}")
                score *= 0.7
            else:
                reasons.append("OK: Price below 200 EMA")

            reasons.append(f"SWEEP: Sell at Resistance {resistance:.2f} (quality: {sweep_quality:.2f})")
            score -= 0.4 * sweep_quality

            if rsi >= self.rsi_sell_high and rsi_momentum < 0:
                reasons.append(f"RSI turning down from {rsi:.1f} (mom: {rsi_momentum:.2f})")
                score -= 0.35
            elif rsi >= self.rsi_sell_high:
                reasons.append(f"RSI at {rsi:.1f} but not turning")
                score -= 0.15

            if price_momentum < 0:
                reasons.append("Momentum: lower highs")
                score -= 0.2

            if trend in ["BEARISH", "WEAK_BEARISH", "SIDEWAYS"]:
                reasons.append(f"Trend {trend} aligns")
                score -= 0.1
        else:
            if score == 0:
                reasons.append("Monitoring: No clear setup")

        # STEP 6: APPLY FAKEOUT BOOST
        score += fakeout_boost

        # STEP 7: DETERMINE SIGNAL
        signal = "HOLD"

        if fakeout_blocks and fakeout_risk >= 0.50:
            if fakeout_direction == "BULLISH" and score < 0:
                reasons.append("BLOCKED: Fakeout override (BULLISH)")
                signal = "HOLD"
                score = max(score, 0.0)
            elif fakeout_direction == "BEARISH" and score > 0:
                reasons.append("BLOCKED: Fakeout override (BEARISH)")
                signal = "HOLD"
                score = min(score, 0.0)

        if signal == "HOLD":
            if score >= 0.45:
                signal = "BUY"
            elif score <= -0.45:
                signal = "SELL"

        return abs(score), signal, "\n".join(reasons)

    def _calculate_levels(self, signal: str, df: pd.DataFrame, support: Optional[float], resistance: Optional[float]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        current_price = df["close"].iloc[-1]
        atr = df["atr"].iloc[-1] if "atr" in df else current_price * 0.01

        if signal == "BUY":
            stop = support - (atr * 0.5) if support else current_price - (atr * 2)
            risk = current_price - stop
            return current_price, stop, current_price + (risk * 2)
        elif signal == "SELL":
            stop = resistance + (atr * 0.5) if resistance else current_price + (atr * 2)
            risk = stop - current_price
            return current_price, stop, current_price - (risk * 2)

        return None, None, None

    def _calculate_bb_position(self, df: pd.DataFrame) -> float:
        last = df.iloc[-1]
        range_val = last["bb_high"] - last["bb_low"] if "bb_high" in last and "bb_low" in last else 1
        if range_val == 0:
            return 0.5
        return (last["close"] - last["bb_low"]) / range_val if "bb_low" in last else 0.5

    def _calculate_trend_strength(self, df: pd.DataFrame) -> float:
        last = df.iloc[-1]
        score = 0
        checks = 0

        if "sma_20" in last and "sma_50" in last:
            if last["sma_20"] > last["sma_50"]:
                score += 1
            checks += 1
        if "close" in last and "sma_20" in last:
            if last["close"] > last["sma_20"]:
                score += 1
            checks += 1
        if "macd" in last and "macd_signal" in last:
            if last["macd"] > last["macd_signal"]:
                score += 1
            checks += 1
        if "adx" in last:
            if last["adx"] > 20:
                score += 1
            checks += 1

        return score / checks if checks > 0 else 0.5

    def generate_technical_score(self, indicators: Dict) -> float:
        score = 0.0
        weights = {"rsi": 0.25, "macd": 0.25, "trend": 0.30, "bb": 0.20}

        rsi = indicators.get("rsi", 50.0)
        rsi_momentum = indicators.get("rsi_momentum", 0.0)

        if rsi < self.rsi_buy_low and rsi_momentum > 0:
            score += weights["rsi"] * 1.0
        elif rsi < self.rsi_buy_low:
            score += weights["rsi"] * 0.5
        elif rsi > self.rsi_sell_high and rsi_momentum < 0:
            score += weights["rsi"] * -1.0
        elif rsi > self.rsi_sell_high:
            score += weights["rsi"] * -0.5
        else:
            score += weights["rsi"] * ((self.rsi_midpoint - rsi) / self.rsi_range_half) * 0.3

        current_price = indicators.get("current_price", 1.0)
        macd_diff = indicators.get("macd", 0) - indicators.get("macd_signal", 0)
        macd_diff_pct = macd_diff / current_price if current_price > 0 else 0
        score += weights["macd"] * np.tanh(macd_diff_pct * 1000)

        trend_strength = indicators.get("trend_strength", 0.5)
        score += weights["trend"] * (trend_strength * 2 - 1)

        bb_pos = indicators.get("bb_position", 0.5)
        if bb_pos < 0.1:
            score += weights["bb"] * 1.0
        elif bb_pos > 0.9:
            score += weights["bb"] * -1.0
        else:
            score += weights["bb"] * (0.5 - bb_pos) * 2

        return float(np.clip(score, -1, 1))

