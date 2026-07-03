# core/filters.py
# EARLY DETECTION VERSION — 2 of 4 filters for early, 3/4 for confirmed
# Accepts pre-calculated indicators to avoid double computation

import os
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class FilterManager:
    def __init__(self):
        self.sr_tolerance = float(os.getenv("SR_ZONE_TOLERANCE", "0.003"))
        self.min_interval = timedelta(minutes=int(os.getenv("SIGNAL_COOLDOWN_MINUTES", "30")))
        self.early_interval = timedelta(minutes=15)
        self.last_signals: Dict[str, datetime] = {}
        self.daily_counts: Dict[str, int] = {}
        self.last_reset = datetime.now().date()
        logger.info(f"FilterManager v4.0 | Early: 15m | Confirmed: 30m | 2/4 or 3/4 filters")

    def validate_signal(self, df: pd.DataFrame, signal_type: str, current_price: float,
                        indicators: Optional[Dict] = None, pair: str = "", is_early: bool = False) -> Tuple[bool, Dict]:
        now = datetime.now()
        if now.date() != self.last_reset:
            self.daily_counts = {}
            self.last_reset = now.date()

        filters_report = {}
        cooldown_key = f"{pair}_{signal_type}"

        effective_interval = self.early_interval if is_early else self.min_interval

        last_sent_time = self.last_signals.get(cooldown_key)
        if last_sent_time and (now - last_sent_time) < effective_interval:
            filters_report["cooldown_check"] = {
                "passed": False,
                "reason": f"Cooldown active until {(last_sent_time + effective_interval).strftime('%H:%M:%S')}"
            }
            return False, {"filters": filters_report, "price": current_price, "is_early": is_early}

        # Use provided indicators or compute
        if indicators is None:
            from .technical_analysis import TechnicalAnalyzer
            try:
                analyzer = TechnicalAnalyzer()
                indicators = analyzer.calculate_indicators(df)
            except Exception as e:
                logger.error(f"Filter indicator error: {e}")
                return False, {"filters": {"error": str(e)}, "price": current_price, "is_early": is_early}

        market_structure = indicators.get("market_structure", "SIDEWAYS")
        support = indicators.get("active_support")
        resistance = indicators.get("active_resistance")
        sweep = indicators.get("liquidity_sweep")
        rsi = indicators.get("rsi", 50.0)
        rsi_momentum = indicators.get("rsi_momentum", 0.0)
        price_momentum = indicators.get("price_momentum", 0.0)
        bb_pos = indicators.get("bb_position", 0.5)
        live_reversal = indicators.get("live_reversal")
        div_type = indicators.get("rsi_divergence", "NONE")
        cs_signal = indicators.get("candlestick_signal", "HOLD")
        cs_confidence = indicators.get("candlestick_confidence", 0.0)
        fakeout_detected = indicators.get("fakeout_detected", False)
        fakeout_direction = indicators.get("fakeout_direction")

        passed_filters = 0
        total_filters = 0

        if signal_type == "BUY":
            # Filter 1: Market structure
            structure_ok = market_structure not in ["BEARISH"]
            filters_report["market_structure"] = {
                "passed": structure_ok,
                "reason": f"Structure: {market_structure} {'OK' if structure_ok else 'BLOCKED'}"
            }
            if structure_ok:
                passed_filters += 1
            total_filters += 1

            # Filter 2: SMC level / sweep / oversold / early signals
            near_support = False
            in_oversold_zone = bb_pos < 0.2
            if support:
                low_bound = support * (1 - self.sr_tolerance * 2)
                high_bound = support * (1 + self.sr_tolerance * 2)
                near_support = low_bound <= current_price <= high_bound

            has_early_signal = (live_reversal in ["BUY_EARLY"] or
                               div_type == "BULLISH_DIV" or
                               (cs_signal == "BUY" and cs_confidence >= 0.50))
            sweep_or_level = (sweep == "BUY_SWEEP") or near_support or in_oversold_zone or has_early_signal
            filters_report["smc_level"] = {
                "passed": sweep_or_level,
                "reason": f"Sweep: {sweep} | Near S: {near_support} | BB Low: {in_oversold_zone} | Early: {has_early_signal}"
            }
            if sweep_or_level:
                passed_filters += 1
            total_filters += 1

            # Filter 3: RSI
            rsi_predictive = (rsi <= 45) and (rsi_momentum > 0)
            rsi_sweet_spot = 30 <= rsi <= 45
            rsi_early = (rsi <= 50) and (live_reversal == "BUY_EARLY" or div_type == "BULLISH_DIV")

            rsi_ok = rsi_predictive or rsi_sweet_spot or rsi_early
            filters_report["rsi"] = {
                "passed": rsi_ok,
                "reason": f"RSI: {rsi:.1f} | Mom: {rsi_momentum:+.2f} | Pred: {rsi_predictive} | Sweet: {rsi_sweet_spot} | Early: {rsi_early}"
            }
            if rsi_ok:
                passed_filters += 1
            total_filters += 1

            # Filter 4: Price momentum / fakeout check
            momentum_ok = price_momentum >= 0 or live_reversal == "BUY_EARLY" or div_type == "BULLISH_DIV"
            # Fakeout override: if fakeout says BEARISH, block BUY
            if fakeout_detected and fakeout_direction == "BEARISH":
                momentum_ok = False
                filters_report["fakeout_block"] = {"passed": False, "reason": "Fakeout detected — BEARISH override"}
            else:
                filters_report["fakeout_block"] = {"passed": True, "reason": "No fakeout block"}

            filters_report["price_momentum"] = {
                "passed": momentum_ok,
                "reason": f"Mom: {price_momentum:+.1f} | Early override: {live_reversal == 'BUY_EARLY' or div_type == 'BULLISH_DIV'}"
            }
            if momentum_ok:
                passed_filters += 1
            total_filters += 1

        elif signal_type == "SELL":
            structure_ok = market_structure not in ["BULLISH"]
            filters_report["market_structure"] = {
                "passed": structure_ok,
                "reason": f"Structure: {market_structure} {'OK' if structure_ok else 'BLOCKED'}"
            }
            if structure_ok:
                passed_filters += 1
            total_filters += 1

            near_resistance = False
            in_overbought_zone = bb_pos > 0.8
            if resistance:
                low_bound = resistance * (1 - self.sr_tolerance * 2)
                high_bound = resistance * (1 + self.sr_tolerance * 2)
                near_resistance = low_bound <= current_price <= high_bound

            has_early_signal = (live_reversal in ["SELL_EARLY"] or
                               div_type == "BEARISH_DIV" or
                               (cs_signal == "SELL" and cs_confidence >= 0.50))
            sweep_or_level = (sweep == "SELL_SWEEP") or near_resistance or in_overbought_zone or has_early_signal
            filters_report["smc_level"] = {
                "passed": sweep_or_level,
                "reason": f"Sweep: {sweep} | Near R: {near_resistance} | BB High: {in_overbought_zone} | Early: {has_early_signal}"
            }
            if sweep_or_level:
                passed_filters += 1
            total_filters += 1

            rsi_predictive = (rsi >= 55) and (rsi_momentum < 0)
            rsi_sweet_spot = 55 <= rsi <= 70
            rsi_early = (rsi >= 50) and (live_reversal == "SELL_EARLY" or div_type == "BEARISH_DIV")

            rsi_ok = rsi_predictive or rsi_sweet_spot or rsi_early
            filters_report["rsi"] = {
                "passed": rsi_ok,
                "reason": f"RSI: {rsi:.1f} | Mom: {rsi_momentum:+.2f} | Pred: {rsi_predictive} | Sweet: {rsi_sweet_spot} | Early: {rsi_early}"
            }
            if rsi_ok:
                passed_filters += 1
            total_filters += 1

            momentum_ok = price_momentum <= 0 or live_reversal == "SELL_EARLY" or div_type == "BEARISH_DIV"
            if fakeout_detected and fakeout_direction == "BULLISH":
                momentum_ok = False
                filters_report["fakeout_block"] = {"passed": False, "reason": "Fakeout detected — BULLISH override"}
            else:
                filters_report["fakeout_block"] = {"passed": True, "reason": "No fakeout block"}

            filters_report["price_momentum"] = {
                "passed": momentum_ok,
                "reason": f"Mom: {price_momentum:+.1f} | Early override: {live_reversal == 'SELL_EARLY' or div_type == 'BEARISH_DIV'}"
            }
            if momentum_ok:
                passed_filters += 1
            total_filters += 1

        else:
            filters_report["hold"] = {"passed": False, "reason": "Signal is HOLD"}
            return False, {"filters": filters_report, "price": current_price, "is_early": is_early}

        # 2/4 for early, 3/4 for confirmed
        if is_early:
            overall_passed = passed_filters >= 2
            confidence_boost = 0.75 if passed_filters >= 3 else 0.60
        else:
            overall_passed = passed_filters >= 3
            confidence_boost = 1.0

        if overall_passed:
            self.last_signals[cooldown_key] = now
            self.daily_counts[cooldown_key] = self.daily_counts.get(cooldown_key, 0) + 1
            label = "EARLY" if is_early else "CONFIRMED"
            filters_report["cooldown_check"] = {
                "passed": True,
                "reason": f"{label} Signal #{self.daily_counts[cooldown_key]} today ({passed_filters}/{total_filters} filters)"
            }

        return overall_passed, {"filters": filters_report, "price": current_price,
                                "is_early": is_early, "confidence_boost": confidence_boost}

    def get_status(self) -> Dict:
        return {
            "last_signals": {k: v.isoformat() for k, v in self.last_signals.items()},
            "daily_counts": self.daily_counts,
            "cooldown_minutes": self.min_interval.total_seconds() / 60,
            "early_cooldown_minutes": self.early_interval.total_seconds() / 60
        }

