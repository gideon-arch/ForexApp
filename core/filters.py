# core/filters.py
# High-Confluence SMC Validation + Anti-Spam Signal Cooldown

import os
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd

# FIXED: Swapped to a relative package import to guarantee cross-environment module execution
from .technical_analysis import TechnicalAnalyzer


class FilterManager:

    def __init__(self):
        # S&R Zone tolerance parameter (0.15% threshold for buffer alignment)
        self.sr_tolerance = float(os.getenv("SR_ZONE_TOLERANCE", 0.0015))

        # Anti-Spam Time Tracking Properties
        self.min_interval = timedelta(
            minutes=int(os.getenv("SIGNAL_COOLDOWN_MINUTES", 60))
        )
        self.last_signals: Dict[str, datetime] = {}
        self.daily_counts: Dict[str, int] = {}
        self.last_reset = datetime.now().date()

        # FIXED: Instantiate technical analysis instance once to keep memory references warm
        self.analyzer = TechnicalAnalyzer()

        # FIXED: Pulled dual-tier milestone target values from operational parameters
        self.rsi_buy_tiers = [30.0, 25.0]
        self.rsi_sell_tiers = [65.0, 70.0]

        print(
            f"🛡️ SMC Filter Manager Active | Cooldown: {self.min_interval.total_seconds() / 60}m | S&R Tolerance: {self.sr_tolerance * 100}%"
        )

    def validate_signal(
        self, df: pd.DataFrame, signal_type: str, current_price: float
    ) -> Tuple[bool, Dict]:
        """Executes a high-confluence Smart Money Concepts rule confirmation pass.

        Validates trend alignment, S&R bounds proximity, liquidity sweeps, and
        time cooldowns.
        """
        # Extract target ticker asset name from dataframe properties, default to UNIVERSAL
        pair = getattr(df, "pair_name", "UNIVERSAL").upper()

        # Check active daily clock tracking variables for counter resets
        now = datetime.now()
        if now.date() != self.last_reset:
            self.daily_counts = {}
            self.last_reset = now.date()

        filters_report = {}
        cooldown_key = f"{pair}_{signal_type}"

        # ⏱️ PHASE 1: TIME COOLDOWN SPAM FILTER CHECK
        last_sent_time = self.last_signals.get(cooldown_key)
        if last_sent_time and (now - last_sent_time) < self.min_interval:
            filters_report["cooldown_check"] = {
                "passed": False,
                "reason": f"⏱️ Signal blocked to prevent duplicate spam. Cooldown active until {(last_sent_time + self.min_interval).strftime('%H:%M:%S')}",
            }
            return False, {"filters": filters_report, "price": current_price}

        # Calculate indicators only after verifying the signal is not throttled by cooldowns
        indicators = self.analyzer.calculate_indicators(df)

        market_structure = indicators.get("market_structure", "SIDEWAYS")
        support = indicators.get("active_support")
        resistance = indicators.get("active_resistance")
        sweep = indicators.get("liquidity_sweep")
        rsi = indicators.get("rsi", 50.0)

        overall_passed = True

        # 🟢 PHASE 2: CONFLUENCE CHECK FOR BUY ALERTS
        if signal_type == "BUY":
            # 1. Trend Filter Guard: Block buys if market structure is strictly bearish
            structure_ok = market_structure != "BEARISH"
            filters_report["market_structure"] = {
                "passed": structure_ok,
                "reason": f"Structure is {market_structure}. (Locked if BEARISH)",
            }
            if not structure_ok:
                overall_passed = False

            # 2. Support Proximity & Liquidity Sweep Check
            near_support = False
            if support:
                low_bound = support * (1 - self.sr_tolerance)
                high_bound = support * (1 + self.sr_tolerance)
                near_support = low_bound <= current_price <= high_bound

            sweep_or_level = sweep == "BUY_SWEEP" or near_support
            filters_report["smc_level_validation"] = {
                "passed": sweep_or_level,
                "reason": f"Sweep: {sweep} | Near Support: {near_support} (Floor: {support})",
            }
            if not sweep_or_level:
                overall_passed = False

            # 3. FIXED: Multi-Tiered RSI Milestone Optimization
            # Checks if the indicator value explicitly confirms inside the target ranges
            rsi_ok = any(rsi <= tier for tier in self.rsi_buy_tiers)
            filters_report["rsi_confirmation"] = {
                "passed": rsi_ok,
                "reason": f"RSI is {rsi:.2f}. Must fall within active tier brackets ({self.rsi_buy_tiers}) for macro validation.",
            }
            if not rsi_ok:
                overall_passed = False

        # 🔴 PHASE 3: CONFLUENCE CHECK FOR SELL ALERTS
        elif signal_type == "SELL":
            # 1. Trend Filter Guard: Block sells if market structure is strictly bullish
            structure_ok = market_structure != "BULLISH"
            filters_report["market_structure"] = {
                "passed": structure_ok,
                "reason": f"Structure is {market_structure}. (Locked if BULLISH)",
            }
            if not structure_ok:
                overall_passed = False

            # 2. Resistance Proximity & Liquidity Sweep Check
            near_resistance = False
            if resistance:
                low_bound = resistance * (1 - self.sr_tolerance)
                high_bound = resistance * (1 + self.sr_tolerance)
                near_resistance = low_bound <= current_price <= high_bound

            sweep_or_level = sweep == "SELL_SWEEP" or near_resistance
            filters_report["smc_level_validation"] = {
                "passed": sweep_or_level,
                "reason": f"Sweep: {sweep} | Near Resistance: {near_resistance} (Ceiling: {resistance})",
            }
            if not sweep_or_level:
                overall_passed = False

            # 3. FIXED: Multi-Tiered RSI Milestone Optimization
            rsi_ok = any(rsi >= tier for tier in self.rsi_sell_tiers)
            filters_report["rsi_confirmation"] = {
                "passed": rsi_ok,
                "reason": f"RSI is {rsi:.2f}. Must break within active tier brackets ({self.rsi_sell_tiers}) for macro validation.",
            }
            if not rsi_ok:
                overall_passed = False

        else:
            overall_passed = False
            filters_report["hold_state"] = {
                "passed": False,
                "reason": "Strategy output is set to HOLD.",
            }

        # If the trade clears all structural filters, log the execution timestamp to state tracking maps
        if overall_passed:
            self.last_signals[cooldown_key] = now
            self.daily_counts[cooldown_key] = (
                self.daily_counts.get(cooldown_key, 0) + 1
            )
            filters_report["cooldown_check"] = {
                "passed": True,
                "reason": f"Pass updated. Daily signal index count: {self.daily_counts[cooldown_key]}",
            }

        return overall_passed, {
            "filters": filters_report,
            "price": current_price,
        }

    def get_status(self) -> Dict:
        """Get cooldown tracking state logs for operational diagnostics"""
        return {
            "last_signals": {
                k: v.isoformat() for k, v in self.last_signals.items()
            },
            "daily_counts": self.daily_counts,
        }