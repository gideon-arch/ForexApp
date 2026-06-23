# core/technical_analysis.py
# SMC Strategy: Structure -> S/R -> Liquidity Sweep -> RSI Confirm -> Entry

import os
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
import ta
import logging

logger = logging.getLogger(__name__)

class TechnicalAnalyzer:
    def __init__(self):
        self.rsi_buy_low = float(os.getenv("RSI_BUY_LOW", 30))
        self.rsi_sell_high = float(os.getenv("RSI_SELL_HIGH", 65))
        self.rsi_midpoint = (self.rsi_buy_low + self.rsi_sell_high) / 2
        self.rsi_range_half = max(self.rsi_sell_high - self.rsi_midpoint, 1.0)
        logger.info(f"TechnicalAnalyzer | RSI: {self.rsi_buy_low}-{self.rsi_sell_high}")

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
        
        current_rsi = df["rsi"].iloc[-1]
        rsi_zone = "NEUTRAL"
        if current_rsi <= 25.0: rsi_zone = "EXTREME_OVERSOLD"
        elif current_rsi <= self.rsi_buy_low: rsi_zone = "OVERSOLD"
        elif current_rsi >= 70.0: rsi_zone = "EXTREME_OVERBOUGHT"
        elif current_rsi >= self.rsi_sell_high: rsi_zone = "OVERBOUGHT"
        
        confluence_score, target_signal, reasoning = self._calculate_confluence(
            trend, sweep_type, sweep_quality, current_rsi, support, resistance, df
        )
        entry, stop, target = self._calculate_levels(target_signal, df, support, resistance)
        
        return {
            "current_price": float(df["close"].iloc[-1]),
            "sma_20": float(df["sma_20"].iloc[-1]) if "sma_20" in df else 0.0,
            "sma_50": float(df["sma_50"].iloc[-1]) if "sma_50" in df else 0.0,
            "ema_200": float(df["ema_200"].iloc[-1]) if "ema_200" in df else 0.0,
            "rsi": float(current_rsi),
            "rsi_zone": rsi_zone,
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
        if len(df) < 5: return None, 0.0
        sweep_type = None
        quality = 0.0
        current_price = df["close"].iloc[-1]
        tolerance = current_price * 0.0005
        
        for i in range(-4, -1):
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

    def _calculate_confluence(self, trend: str, sweep_type: Optional[str], sweep_quality: float,
                             rsi: float, support: Optional[float], resistance: Optional[float],
                             df: pd.DataFrame) -> Tuple[float, str, str]:
        reasons = []
        score = 0.0
        signal = "HOLD"
        current_price = df["close"].iloc[-1]
        current_ema200 = df["ema_200"].iloc[-1] if "ema_200" in df else current_price
        is_macro_uptrend = current_price > current_ema200
        
        if sweep_type == "BUY_SWEEP":
            if not is_macro_uptrend:
                reasons.append(f"BUY blocked: Price {current_price:.2f} < 200 EMA {current_ema200:.2f}")
                return 0.0, "HOLD", "\n".join(reasons)
            
            reasons.append(f"Liquidity Hunt at Support {support:.2f}")
            score += 0.4 * sweep_quality
            
            if rsi <= self.rsi_buy_low:
                reasons.append(f"RSI exhausted at {rsi:.1f}")
                score += 0.3
            if trend in ["BULLISH", "WEAK_BULLISH", "SIDEWAYS"]:
                reasons.append(f"Trend {trend} aligns")
                score += 0.2
            if df["close"].iloc[-1] > df["open"].iloc[-1]:
                score += 0.1
            
            signal = "BUY" if score >= 0.45 else "HOLD"
            
        elif sweep_type == "SELL_SWEEP":
            if is_macro_uptrend:
                reasons.append(f"SELL blocked: Price {current_price:.2f} > 200 EMA {current_ema200:.2f}")
                return 0.0, "HOLD", "\n".join(reasons)
            
            reasons.append(f"Liquidity Hunt at Resistance {resistance:.2f}")
            score += 0.4 * sweep_quality
            
            if rsi >= self.rsi_sell_high:
                reasons.append(f"RSI exhausted at {rsi:.1f}")
                score += 0.3
            if trend in ["BEARISH", "WEAK_BEARISH", "SIDEWAYS"]:
                reasons.append(f"Trend {trend} aligns")
                score += 0.2
            if df["close"].iloc[-1] < df["open"].iloc[-1]:
                score += 0.1
            
            signal = "SELL" if score >= 0.45 else "HOLD"
        else:
            reasons.append("Monitoring: Searching for liquidity setups")
        
        return score, signal, "\n".join(reasons)

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
        if range_val == 0: return 0.5
        return (last["close"] - last["bb_low"]) / range_val if "bb_low" in last else 0.5

    def _calculate_trend_strength(self, df: pd.DataFrame) -> float:
        last = df.iloc[-1]
        score = 0
        checks = 0
        
        if "sma_20" in last and "sma_50" in last:
            if last["sma_20"] > last["sma_50"]: score += 1
            checks += 1
        if "close" in last and "sma_20" in last:
            if last["close"] > last["sma_20"]: score += 1
            checks += 1
        if "macd" in last and "macd_signal" in last:
            if last["macd"] > last["macd_signal"]: score += 1
            checks += 1
        if "adx" in last:
            if last["adx"] > 20: score += 1
            checks += 1
        
        return score / checks if checks > 0 else 0.5

    def generate_technical_score(self, indicators: Dict) -> float:
        score = 0.0
        weights = {"rsi": 0.25, "macd": 0.25, "trend": 0.30, "bb": 0.20}
        
        rsi = indicators.get("rsi", 50.0)
        if rsi < self.rsi_buy_low: score += weights["rsi"] * 1.0
        elif rsi > self.rsi_sell_high: score += weights["rsi"] * -1.0
        else: score += weights["rsi"] * ((self.rsi_midpoint - rsi) / self.rsi_range_half) * 0.5
        
        current_price = indicators.get("current_price", 1.0)
        macd_diff = indicators.get("macd", 0) - indicators.get("macd_signal", 0)
        macd_diff_pct = macd_diff / current_price if current_price > 0 else 0
        score += weights["macd"] * np.tanh(macd_diff_pct * 1000)
        
        trend_strength = indicators.get("trend_strength", 0.5)
        score += weights["trend"] * (trend_strength * 2 - 1)
        
        bb_pos = indicators.get("bb_position", 0.5)
        if bb_pos < 0.1: score += weights["bb"] * 1.0
        elif bb_pos > 0.9: score += weights["bb"] * -1.0
        else: score += weights["bb"] * (0.5 - bb_pos) * 2
        
        return float(np.clip(score, -1, 1))