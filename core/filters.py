# core/filters.py
# High-Confluence SMC Validation + Anti-Spam Signal Cooldown

import os
from datetime import datetime, timedelta
from typing import Dict, Tuple
import pandas as pd
import logging

from .technical_analysis import TechnicalAnalyzer

logger = logging.getLogger(__name__)

class FilterManager:
    def __init__(self):
        self.sr_tolerance = float(os.getenv("SR_ZONE_TOLERANCE", 0.0015))
        self.min_interval = timedelta(minutes=int(os.getenv("SIGNAL_COOLDOWN_MINUTES", 60)))
        self.last_signals: Dict[str, datetime] = {}
        self.daily_counts: Dict[str, int] = {}
        self.last_reset = datetime.now().date()
        self.analyzer = TechnicalAnalyzer()
        self.rsi_buy_tiers = [30.0, 25.0]
        self.rsi_sell_tiers = [65.0, 70.0]
        logger.info(f"FilterManager | Cooldown: {self.min_interval.total_seconds()/60:.0f}m")

    def validate_signal(self, df: pd.DataFrame, signal_type: str, current_price: float) -> Tuple[bool, Dict]:
        now = datetime.now()
        if now.date() != self.last_reset:
            self.daily_counts = {}
            self.last_reset = now.date()
        
        filters_report = {}
        cooldown_key = f"{signal_type}"
        
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
        
        overall_passed = True
        
        if signal_type == "BUY":
            structure_ok = market_structure not in ["BEARISH", "WEAK_BEARISH"]
            filters_report["market_structure"] = {
                "passed": structure_ok,
                "reason": f"Structure: {market_structure} {'OK' if structure_ok else 'BLOCKED'}"
            }
            if not structure_ok: overall_passed = False
            
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
            if not sweep_or_level: overall_passed = False
            
            rsi_ok = any(rsi <= tier for tier in self.rsi_buy_tiers)
            filters_report["rsi"] = {
                "passed": rsi_ok,
                "reason": f"RSI: {rsi:.1f} {'OK' if rsi_ok else 'BLOCKED'}"
            }
            if not rsi_ok: overall_passed = False
            
        elif signal_type == "SELL":
            structure_ok = market_structure not in ["BULLISH", "WEAK_BULLISH"]
            filters_report["market_structure"] = {
                "passed": structure_ok,
                "reason": f"Structure: {market_structure} {'OK' if structure_ok else 'BLOCKED'}"
            }
            if not structure_ok: overall_passed = False
            
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
            if not sweep_or_level: overall_passed = False
            
            rsi_ok = any(rsi >= tier for tier in self.rsi_sell_tiers)
            filters_report["rsi"] = {
                "passed": rsi_ok,
                "reason": f"RSI: {rsi:.1f} {'OK' if rsi_ok else 'BLOCKED'}"
            }
            if not rsi_ok: overall_passed = False
            
        else:
            overall_passed = False
            filters_report["hold"] = {"passed": False, "reason": "Signal is HOLD"}
        
        if overall_passed:
            self.last_signals[cooldown_key] = now
            self.daily_counts[cooldown_key] = self.daily_counts.get(cooldown_key, 0) + 1
            filters_report["cooldown_check"] = {
                "passed": True,
                "reason": f"Signal #{self.daily_counts[cooldown_key]} today"
            }
        
        return overall_passed, {"filters": filters_report, "price": current_price}

    def get_status(self) -> Dict:
        return {
            "last_signals": {k: v.isoformat() for k, v in self.last_signals.items()},
            "daily_counts": self.daily_counts,
            "cooldown_minutes": self.min_interval.total_seconds() / 60
        }