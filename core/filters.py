# core/filters.py
# PREDICTIVE Validation — Signals BEFORE the move, not after

import os
from datetime import datetime, timedelta
from typing import Dict, Tuple
import pandas as pd
import logging

from .technical_analysis import TechnicalAnalyzer

logger = logging.getLogger(__name__)

class FilterManager:
    def __init__(self):
        self.sr_tolerance = float(os.getenv("SR_ZONE_TOLERANCE", "0.003"))
        self.min_interval = timedelta(minutes=int(os.getenv("SIGNAL_COOLDOWN_MINUTES", "45")))
        self.last_signals: Dict[str, datetime] = {}
        self.daily_counts: Dict[str, int] = {}
        self.last_reset = datetime.now().date()
        self.analyzer = TechnicalAnalyzer()
        logger.info(f"FilterManager | Cooldown: {self.min_interval.total_seconds()/60:.0f}m | S/R Tolerance: {self.sr_tolerance*100:.2f}%")

    def validate_signal(self, df: pd.DataFrame, signal_type: str, current_price: float, pair: str = "") -> Tuple[bool, Dict]:
        now = datetime.now()
        if now.date() != self.last_reset:
            self.daily_counts = {}
            self.last_reset = now.date()
        
        filters_report = {}
        cooldown_key = f"{pair}_{signal_type}"
        
        last_sent_time = self.last_signals.get(cooldown_key)
        if last_sent_time and (now - last_sent_time) < self.min_interval:
            filters_report["cooldown_check"] = {
                "passed": False,
                "reason": f"Cooldown active until {(last_sent_time + self.min_interval).strftime('%H:%M:%S')}"
            }
            return False, {"filters": filters_report, "price": current_price}
        
        try:
            indicators = self.analyzer.calculate_indicators(df)
        except Exception as e:
            logger.error(f"Filter indicator error: {e}")
            return False, {"filters": {"error": str(e)}, "price": current_price}
        
        market_structure = indicators.get("market_structure", "SIDEWAYS")
        support = indicators.get("active_support")
        resistance = indicators.get("active_resistance")
        sweep = indicators.get("liquidity_sweep")
        rsi = indicators.get("rsi", 50.0)
        rsi_momentum = indicators.get("rsi_momentum", 0.0)
        price_momentum = indicators.get("price_momentum", 0.0)
        bb_pos = indicators.get("bb_position", 0.5)
        adx = indicators.get("adx", 20.0)
        
        passed_filters = 0
        total_filters = 0
        
        if signal_type == "BUY":
            # FILTER 1: Structure must allow bounce (not strong bearish)
            structure_ok = market_structure not in ["BEARISH"]
            filters_report["market_structure"] = {
                "passed": structure_ok,
                "reason": f"Structure: {market_structure} {'OK' if structure_ok else 'BLOCKED'}"
            }
            if structure_ok: passed_filters += 1
            total_filters += 1
            
            # FILTER 2: Price at key level OR liquidity sweep
            near_support = False
            if support:
                low_bound = support * (1 - self.sr_tolerance)
                high_bound = support * (1 + self.sr_tolerance)
                near_support = low_bound <= current_price <= high_bound
            
            sweep_or_level = (sweep == "BUY_SWEEP") or near_support
            filters_report["smc_level"] = {
                "passed": sweep_or_level,
                "reason": f"Sweep: {sweep} | Near Support: {near_support}"
            }
            if sweep_or_level: passed_filters += 1
            total_filters += 1
            
            # FILTER 3: PREDICTIVE — RSI oversold AND turning up (not just oversold)
            # This is the KEY difference — we want RSI 35-45 that's RISING, not 20 that's still falling
            rsi_predictive = (rsi <= 45) and (rsi_momentum > 0)
            # OR: RSI is in sweet spot 35-45 (about to bounce, not already crashed to 20)
            rsi_sweet_spot = 35 <= rsi <= 45
            
            rsi_ok = rsi_predictive or rsi_sweet_spot
            filters_report["rsi"] = {
                "passed": rsi_ok,
                "reason": f"RSI: {rsi:.1f} | Momentum: {rsi_momentum:+.2f} | Predictive: {rsi_predictive} | SweetSpot: {rsi_sweet_spot}"
            }
            if rsi_ok: passed_filters += 1
            total_filters += 1
            
            # FILTER 4: PREDICTIVE — Price momentum building (higher lows)
            momentum_ok = price_momentum >= 0  # Allow neutral or positive
            filters_report["price_momentum"] = {
                "passed": momentum_ok,
                "reason": f"Price momentum: {price_momentum:+.1f}"
            }
            if momentum_ok: passed_filters += 1
            total_filters += 1
            
        elif signal_type == "SELL":
            structure_ok = market_structure not in ["BULLISH"]
            filters_report["market_structure"] = {
                "passed": structure_ok,
                "reason": f"Structure: {market_structure} {'OK' if structure_ok else 'BLOCKED'}"
            }
            if structure_ok: passed_filters += 1
            total_filters += 1
            
            near_resistance = False
            if resistance:
                low_bound = resistance * (1 - self.sr_tolerance)
                high_bound = resistance * (1 + self.sr_tolerance)
                near_resistance = low_bound <= current_price <= high_bound
            
            sweep_or_level = (sweep == "SELL_SWEEP") or near_resistance
            filters_report["smc_level"] = {
                "passed": sweep_or_level,
                "reason": f"Sweep: {sweep} | Near Resistance: {near_resistance}"
            }
            if sweep_or_level: passed_filters += 1
            total_filters += 1
            
            # PREDICTIVE: RSI overbought AND turning down, OR in sweet spot 55-65
            rsi_predictive = (rsi >= 55) and (rsi_momentum < 0)
            rsi_sweet_spot = 55 <= rsi <= 65
            
            rsi_ok = rsi_predictive or rsi_sweet_spot
            filters_report["rsi"] = {
                "passed": rsi_ok,
                "reason": f"RSI: {rsi:.1f} | Momentum: {rsi_momentum:+.2f} | Predictive: {rsi_predictive} | SweetSpot: {rsi_sweet_spot}"
            }
            if rsi_ok: passed_filters += 1
            total_filters += 1
            
            # PREDICTIVE — Price momentum building (lower highs)
            momentum_ok = price_momentum <= 0
            filters_report["price_momentum"] = {
                "passed": momentum_ok,
                "reason": f"Price momentum: {price_momentum:+.1f}"
            }
            if momentum_ok: passed_filters += 1
            total_filters += 1
            
        else:
            filters_report["hold"] = {"passed": False, "reason": "Signal is HOLD"}
            return False, {"filters": filters_report, "price": current_price}
        
        # ALL 4 filters must pass for predictive accuracy
        # But if 3 of 4 pass with high confidence, allow it
        overall_passed = (passed_filters >= 4) or (passed_filters == 3 and adx > 25)
        
        if overall_passed:
            self.last_signals[cooldown_key] = now
            self.daily_counts[cooldown_key] = self.daily_counts.get(cooldown_key, 0) + 1
            filters_report["cooldown_check"] = {
                "passed": True,
                "reason": f"Signal #{self.daily_counts[cooldown_key]} today ({passed_filters}/{total_filters} filters)"
            }
        
        return overall_passed, {"filters": filters_report, "price": current_price}

    def get_status(self) -> Dict:
        return {
            "last_signals": {k: v.isoformat() for k, v in self.last_signals.items()},
            "daily_counts": self.daily_counts,
            "cooldown_minutes": self.min_interval.total_seconds() / 60
        }