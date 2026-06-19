# core/technical_analysis.py
# SMC Strategy: Structure → S/R → Liquidity Sweep → RSI Confirm → Entry (Safe Ensemble Edition)

import os
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
import ta


class TechnicalAnalyzer:
    """SMC-Based Technical Analyzer (Optimized to prevent app crashes and lookback blindspots)

    Strategy: Structure → S/R → Liquidity Sweep → RSI Confirm → Entry
    """

    def __init__(self):
        # DYNAMIC LINKING: Synced with your updated crypto thresholds (25/30 and 65/70)
        self.rsi_buy_low = float(os.getenv("RSI_BUY_LOW", 30))  # Long alert zone boundary
        self.rsi_sell_high = float(os.getenv("RSI_SELL_HIGH", 65))  # Short alert zone boundary

        # Calculate dynamic midpoint for accurate scaling math
        self.rsi_midpoint = (self.rsi_buy_low + self.rsi_sell_high) / 2
        self.rsi_range_half = (
            self.rsi_sell_high - self.rsi_midpoint
            if (self.rsi_sell_high - self.rsi_midpoint) > 0
            else 1.0
        )

        print(
            f"📊 Advanced SMC Analyzer Loaded | Bounds: {self.rsi_buy_low} to {self.rsi_sell_high} | Midpoint: {self.rsi_midpoint}"
        )

    def calculate_indicators(self, df: pd.DataFrame) -> Dict:
        """Full SMC analysis pipeline configured to safely return a traditional

        dictionary structure for direct compatibility with LSTM models and filters.
        """
        df = df.copy()

        # Guard clause against insufficient historical data streams
        if len(df) < 200:
            raise ValueError(
                f"Insufficient candles provided for indicators. Needed: 200, Got: {len(df)}"
            )

        # Step 1: Calculate base indicators safely
        df["sma_20"] = ta.trend.sma_indicator(df["close"], window=20)
        df["sma_50"] = ta.trend.sma_indicator(df["close"], window=50)
        df["ema_12"] = ta.trend.ema_indicator(df["close"], window=12)
        df["ema_26"] = ta.trend.ema_indicator(df["close"], window=26)

        # TREND FILTER (OPTION B IMPLEMENTATION): Calculate 200 EMA
        df["ema_200"] = ta.trend.ema_indicator(df["close"], window=200)

        df["rsi"] = ta.momentum.rsi(df["close"], window=14)

        macd = ta.trend.MACD(df["close"])
        df["macd"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()
        df["macd_hist"] = macd.macd_diff()

        df["bb_high"] = ta.volatility.bollinger_hband(df["close"])
        df["bb_low"] = ta.volatility.bollinger_lband(df["close"])
        df["atr"] = ta.volatility.average_true_range(
            df["high"], df["low"], df["close"]
        )
        df["adx"] = ta.trend.adx(df["high"], df["low"], df["close"])

        if "volume" in df.columns:
            df["obv"] = ta.volume.on_balance_volume(df["close"], df["volume"])
        else:
            df["obv"] = 0.0

        df.dropna(inplace=True)

        # Step 2: Find Market Structure (Trend) - FIXED: Flexible window parameters
        trend, swing_highs, swing_lows = self._find_market_structure(df)

        # Step 3: Mark Support/Resistance
        support, resistance = self._mark_support_resistance(
            df, swing_highs, swing_lows
        )

        # Step 4: Detect Liquidity Sweeps - FIXED: Multi-candle structural trace lookback
        sweep_type, sweep_quality = self._detect_liquidity_sweep(
            df, support, resistance
        )

        # Step 5: FIXED: Reordered sequential descending/ascending conditional checks
        current_rsi = df["rsi"].iloc[-1]
        rsi_zone = "NEUTRAL"
        if current_rsi <= 25.0:
            rsi_zone = "EXTREME_OVERSOLD"
        elif current_rsi <= self.rsi_buy_low:
            rsi_zone = "OVERSOLD"
        elif current_rsi >= 70.0:
            rsi_zone = "EXTREME_OVERBOUGHT"
        elif current_rsi >= self.rsi_sell_high:
            rsi_zone = "OVERBOUGHT"

        # Step 6: Generate Confluence Metrics & Target Levels
        confluence_score, target_signal, reasoning = self._calculate_confluence(
            trend, sweep_type, sweep_quality, current_rsi, support, resistance, df
        )
        entry, stop, target = self._calculate_levels(
            target_signal, df, support, resistance
        )

        # Return traditional dict format & fixed local scope 'quality' bug
        return {
            "current_price": float(df["close"].iloc[-1]),
            "sma_20": float(df["sma_20"].iloc[-1]),
            "sma_50": float(df["sma_50"].iloc[-1]),
            "ema_200": float(df["ema_200"].iloc[-1]),
            "rsi": float(current_rsi),
            "rsi_zone": rsi_zone,
            "macd": float(df["macd"].iloc[-1]),
            "macd_signal": float(df["macd_signal"].iloc[-1]),
            "macd_hist": float(df["macd_hist"].iloc[-1]),
            "bb_high": float(df["bb_high"].iloc[-1]),
            "bb_low": float(df["bb_low"].iloc[-1]),
            "bb_position": float(self._calculate_bb_position(df)),
            "atr": float(df["atr"].iloc[-1]),
            "adx": float(df["adx"].iloc[-1]),
            "obv": float(df["obv"].iloc[-1]),
            "trend_strength": float(self._calculate_trend_strength(df)),
            # SMC parameters cleanly packed into indicators map
            "market_structure": trend,
            "active_support": float(support) if support is not None else None,
            "active_resistance": (
                float(resistance) if resistance is not None else None
            ),
            "liquidity_sweep": sweep_type,
            "sweep_quality": float(sweep_quality),
            "confluence_score": float(confluence_score),
            "smc_suggested_signal": target_signal,
            "suggested_entry": float(entry) if entry is not None else None,
            "suggested_stop_loss": float(stop) if stop is not None else None,
            "suggested_take_profit": float(target) if target is not None else None,
            "smc_reasoning": reasoning,
            "dataframe": df,
        }

    def _find_market_structure(
        self, df: pd.DataFrame, window: int = 5
    ) -> Tuple[str, list, list]:
        """Tracks swing structure with dynamic trend thresholds to avoid falling into

        permanent 'SIDEWAYS' traps when structure pullbacks get temporarily volatile.
        """
        df["swing_high"] = (
            df["high"] == df["high"].rolling(window=window * 2 + 1, center=True).max()
        )
        df["swing_low"] = (
            df["low"] == df["low"].rolling(window=window * 2 + 1, center=True).min()
        )

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

    def _mark_support_resistance(
        self, df: pd.DataFrame, swing_highs: list, swing_lows: list
    ) -> Tuple[Optional[float], Optional[float]]:
        current_price = df["close"].iloc[-1]

        support = None
        valid_supports = [l for l in swing_lows if l < current_price]
        if valid_supports:
            support = max(valid_supports)

        resistance = None
        valid_resistances = [h for h in swing_highs if h > current_price]
        if valid_resistances:
            resistance = min(valid_resistances)

        return support, resistance

    def _detect_liquidity_sweep(
        self, df: pd.DataFrame, support: Optional[float], resistance: Optional[float]
    ) -> Tuple[Optional[str], float]:
        """Scans across a 4-candle evaluation window to track multi-candle accumulation sequence.

        Also uses a relaxed proximity tolerance (0.05% or 5 pips) for crypto asset pricing metrics.
        """
        if len(df) < 5:
            return None, 0.0

        sweep_type = None
        quality = 0.0

        current_price = df["close"].iloc[-1]
        tolerance = current_price * 0.0005

        for i in range(-4, -1):
            candle = df.iloc[i]

            # BUY SWEEP CHECK
            if (
                support
                and candle["low"] < (support + tolerance)
                and candle["close"] > (support - tolerance)
            ):
                if (
                    df["close"].iloc[-1] > df["open"].iloc[-1]
                ):  # Confirmed current recovery
                    sweep_type = "BUY_SWEEP"
                    wick_depth = (
                        (support - candle["low"]) / support if support > 0 else 0
                    )
                    denom = candle["high"] - candle["low"]
                    recovery = (
                        (candle["close"] - candle["low"]) / denom
                        if denom > 0
                        else 0.0001
                    )
                    quality = min(wick_depth * 100 + recovery, 1.0)
                    break

            # SELL SWEEP CHECK
            if (
                resistance
                and candle["high"] > (resistance - tolerance)
                and candle["close"] < (resistance + tolerance)
            ):
                if (
                    df["close"].iloc[-1] < df["open"].iloc[-1]
                ):  # Confirmed current drop
                    sweep_type = "SELL_SWEEP"
                    wick_height = (
                        (candle["high"] - resistance) / resistance
                        if resistance > 0
                        else 0
                    )
                    denom = candle["high"] - candle["low"]
                    recovery = (
                        (candle["high"] - candle["close"]) / denom
                        if denom > 0
                        else 0.0001
                    )
                    quality = min(wick_height * 100 + recovery, 1.0)
                    break

        return sweep_type, quality

    def _calculate_confluence(
        self,
        trend: str,
        sweep_type: Optional[str],
        sweep_quality: float,
        rsi: float,
        support: Optional[float],
        resistance: Optional[float],
        df: pd.DataFrame,
    ) -> Tuple[float, str, str]:
        reasons = []
        score = 0.0
        signal = "HOLD"

        current_price = df["close"].iloc[-1]
        current_ema200 = df["ema_200"].iloc[-1]

        is_macro_uptrend = current_price > current_ema200

        if sweep_type == "BUY_SWEEP":
            # OPTION B TREND SHIELD: Block buy trades if the price is stuck in a macro downtrend
            if not is_macro_uptrend:
                reasons.append(
                    f"⏸️ BUY Sweep detected, but dropped because price ({current_price:.2f}) is below the 200 EMA ({current_ema200:.2f}) [Macro Downtrend]."
                )
                return 0.0, "HOLD", "\n".join(reasons)

            reasons.append(f"✅ SMC Liquidity Hunt at Support ({support:.2f})")
            score += 0.4 * sweep_quality

            # FIXED: Synced with your dynamic runtime thresholds
            if rsi <= self.rsi_buy_low:
                reasons.append(f"✅ RSI Momentum exhausted at {rsi:.1f}")
                score += 0.3
            if trend in ["BULLISH", "WEAK_BULLISH", "SIDEWAYS"]:
                reasons.append(f"✅ Market structure trend ({trend}) aligns")
                score += 0.2
            if df["close"].iloc[-1] > df["open"].iloc[-1]:
                score += 0.1

            signal = "BUY" if score >= 0.45 else "HOLD"

        elif sweep_type == "SELL_SWEEP":
            # OPTION B TREND SHIELD FIX: Block sell trades if price is above 200 EMA (Macro Uptrend)
            if is_macro_uptrend:
                reasons.append(
                    f"⏸️ SELL Sweep detected, but dropped because price ({current_price:.2f}) is above the 200 EMA ({current_ema200:.2f}) [Macro Uptrend]."
                )
                return 0.0, "HOLD", "\n".join(reasons)

            reasons.append(f"✅ SMC Liquidity Hunt at Resistance ({resistance:.2f})")
            score += 0.4 * sweep_quality

            # FIXED: Synced with your dynamic runtime thresholds
            if rsi >= self.rsi_sell_high:
                reasons.append(f"✅ RSI Momentum exhausted at {rsi:.1f}")
                score += 0.3
            if trend in ["BEARISH", "WEAK_BEARISH", "SIDEWAYS"]:
                reasons.append(f"✅ Market structure trend ({trend}) aligns")
                score += 0.2
            if df["close"].iloc[-1] < df["open"].iloc[-1]:
                score += 0.1

            signal = "SELL" if score >= 0.45 else "HOLD"

        else:
            reasons.append(
                "⏸️ Structure monitoring: Searching for valid liquidity hunt setups near active key zones."
            )

        return score, signal, "\n".join(reasons)

    def _calculate_levels(
        self,
        signal: str,
        df: pd.DataFrame,
        support: Optional[float],
        resistance: Optional[float],
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        current_price = df["close"].iloc[-1]
        atr = df["atr"].iloc[-1]

        if signal == "BUY":
            stop = support - (atr * 0.5) if support else current_price - (atr * 2)
            risk = current_price - stop
            return current_price, stop, current_price + (risk * 2)
        elif signal == "SELL":
            stop = (
                resistance + (atr * 0.5)
                if resistance
                else current_price + (atr * 2)
            )
            risk = stop - current_price
            return current_price, stop, current_price - (risk * 2)

        return None, None, None

    def _calculate_bb_position(self, df: pd.DataFrame) -> float:
        last = df.iloc[-1]
        range_val = (
            last["bb_high"] - last["bb_low"] if "bb_high" in last else 1
        )
        if range_val == 0:
            return 0.5
        return (
            (last["close"] - last["bb_low"]) / range_val
            if "bb_low" in last
            else 0.5
        )

    def _calculate_trend_strength(self, df: pd.DataFrame) -> float:
        last = df.iloc[-1]
        score = 0
        if last["sma_20"] > last["sma_50"]:
            score += 1
        if last["close"] > last["sma_20"]:
            score += 1
        if last["macd"] > last["macd_signal"]:
            score += 1
        if last["adx"] > 20:
            score += 1
        return score / 4.0

    def generate_technical_score(self, indicators: Dict) -> float:
        score = 0
        weights = {"rsi": 0.25, "macd": 0.25, "trend": 0.30, "bb": 0.20}

        rsi = indicators["rsi"]
        if rsi < self.rsi_buy_low:
            score += weights["rsi"] * 1
        elif rsi > self.rsi_sell_high:
            score += weights["rsi"] * -1
        else:
            score += (
                weights["rsi"]
                * ((self.rsi_midpoint - rsi) / self.rsi_range_half)
                * 0.5
            )

        current_price = indicators["current_price"]
        macd_diff_pct = (
            (indicators["macd"] - indicators["macd_signal"]) / current_price
            if current_price > 0
            else 0
        )

        score += weights["macd"] * np.tanh(macd_diff_pct * 1000)
        score += weights["trend"] * (indicators["trend_strength"] * 2 - 1)

        bb_pos = indicators["bb_position"]
        if bb_pos < 0.1:
            score += weights["bb"] * 1
        elif bb_pos > 0.9:
            score += weights["bb"] * -1
        else:
            score += weights["bb"] * (0.5 - bb_pos) * 2

        return float(np.clip(score, -1, 1))