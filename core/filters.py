#core/filters.py
# v5.0 — 10min cooldown, 4-filter validation, multi-timeframe ready
# v5.1: Trend-entry RSI acceptance — early trend-start entries naturally
#       happen with RSI in the 45-62 mid-range, but the old gate only
#       accepted RSI <= 45 (BUY) / >= 55 (SELL) with momentum. That vetoed
#       every early structure-break entry the v5.4 engine produces. Now a
#       rising RSI 45-62 with bullish momentum (and mirror for SELL) passes.

import os
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class FilterManager:
    def __init__(self):
        self.sr_tolerance = float(os.getenv("SR_ZONE_TOLERANCE", "0.003"))
        self.min_interval = timedelta(minutes=int(os.getenv("SIGNAL_COOLDOWN_MINUTES", "10")))
        self.early_interval = timedelta(minutes=5)
        self.last_signals: Dict[str, datetime] = {}
        self.daily_counts: Dict[str, int] = {}
        self.last_reset = datetime.now().date()
        logger.info(
            f"FilterManager v5.1 | Cooldown: {self.min_interval.total_seconds() / 60:.0f}m | "
            f"Early: {self.early_interval.total_seconds() / 60:.0f}m | 2/4 early, 3/4 confirmed | "
            f"Trend-entry RSI band: ON"
        )

    def validate_signal(self, df: Optional[pd.DataFrame], signal_type: str, current_price: float,
                        indicators: Optional[Dict] = None, pair: str = "", is_early: bool = False,
                        is_reversal: bool = False) -> Tuple[bool, Dict]:
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

        fib_zones = indicators.get("fib_zones", {})
        in_golden_zone = fib_zones.get("in_golden_zone", False) if fib_zones else False
        in_premium_zone = fib_zones.get("in_premium_zone", False) if fib_zones else False
        fib_position = fib_zones.get("position", 50) if fib_zones else 50

        passed_filters = 0
        total_filters = 0

        # ===== HARD TREND BLOCK =====
        # v5.4.1: this block uses the SECONDARY timeframe's market_structure,
        # which is EMA200/swing based and lags a real-time reversal by
        # several candles — exactly like the primary-timeframe lag in
        # multi_timeframe.py. When the multi-timeframe engine has already
        # confirmed a reversal (fresh EMA9/21 cross + structure break on the
        # entry TF + primary momentum shift + secondary not opposing), it
        # sets is_reversal=True and this stale-trend veto is bypassed —
        # otherwise a confirmed reversal signal would be vetoed right back
        # into a HOLD by the very structure reading it just overrode.
        trend_blocked = False
        if is_reversal:
            filters_report["hard_trend_block"] = {
                "passed": True,
                "reason": f"Reversal-confirmed {signal_type} — stale-trend block bypassed (multi-layer confirmation)"
            }
        else:
            if signal_type == "BUY":
                if market_structure in ["BEARISH", "BEARISH_PULLBACK", "WEAK_BEARISH"]:
                    trend_blocked = True
                    filters_report["hard_trend_block"] = {
                        "passed": False,
                        "reason": f"HARD BLOCK: BUY in {market_structure}"
                    }
            elif signal_type == "SELL":
                if market_structure in ["BULLISH", "BULLISH_PULLBACK", "WEAK_BULLISH"]:
                    trend_blocked = True
                    filters_report["hard_trend_block"] = {
                        "passed": False,
                        "reason": f"HARD BLOCK: SELL in {market_structure}"
                    }

            if not trend_blocked:
                filters_report["hard_trend_block"] = {
                    "passed": True,
                    "reason": f"Trend OK: {signal_type} in {market_structure}"
                }

        if signal_type == "BUY":
            structure_ok = market_structure not in ["BEARISH", "BEARISH_PULLBACK"]
            filters_report["market_structure"] = {
                "passed": structure_ok,
                "reason": f"Structure: {market_structure} {'OK' if structure_ok else 'BLOCKED'}"
            }
            if structure_ok:
                passed_filters += 1
            total_filters += 1

            near_support = False
            in_oversold_zone = bb_pos < 0.2
            if support:
                low_bound = support * (1 - self.sr_tolerance * 2)
                high_bound = support * (1 + self.sr_tolerance * 2)
                near_support = low_bound <= current_price <= high_bound

            fib_buy_zone = in_golden_zone or fib_position >= 55
            has_early = (live_reversal in ["BUY_EARLY"] or
                        div_type == "BULLISH_DIV" or
                        (cs_signal == "BUY" and cs_confidence >= 0.50))
            sweep_or_level = (sweep == "BUY_SWEEP") or near_support or in_oversold_zone or has_early or fib_buy_zone
            filters_report["smc_level"] = {
                "passed": sweep_or_level,
                "reason": f"Sweep:{sweep} | NearS:{near_support} | BBlow:{in_oversold_zone} | Fib:{fib_buy_zone} | Early:{has_early}"
            }
            if sweep_or_level:
                passed_filters += 1
            total_filters += 1

            if fib_buy_zone and rsi <= 50:
                rsi_ok = True
                filters_report["rsi"] = {"passed": True, "reason": f"RSI:{rsi:.1f} | Fib golden relaxes"}
            else:
                rsi_predictive = (rsi <= 45) and (rsi_momentum > 0)
                rsi_sweet = 30 <= rsi <= 45
                rsi_early = (rsi <= 50) and (live_reversal == "BUY_EARLY" or div_type == "BULLISH_DIV")
                # v5.1: trend-start entries happen mid-range by nature — the
                # structure break IS the confirmation. Rising RSI 45-62 with
                # bullish momentum passes (was: only <=45, which vetoed every
                # early structure-break entry the v5.4 engine produces).
                rsi_trend = (45 <= rsi <= 62) and (rsi_momentum > 0) and (price_momentum >= 0)
                rsi_ok = rsi_predictive or rsi_sweet or rsi_early or rsi_trend
                filters_report["rsi"] = {
                    "passed": rsi_ok,
                    "reason": f"RSI:{rsi:.1f} | Mom:{rsi_momentum:+.2f} | Pred:{rsi_predictive} | Sweet:{rsi_sweet} | Trend:{rsi_trend}"
                }
            if rsi_ok:
                passed_filters += 1
            total_filters += 1

            momentum_ok = price_momentum >= 0 or live_reversal == "BUY_EARLY" or div_type == "BULLISH_DIV"
            if fakeout_detected and fakeout_direction == "BEARISH":
                momentum_ok = False
                filters_report["fakeout_block"] = {"passed": False, "reason": "Fakeout BEARISH override"}
            else:
                filters_report["fakeout_block"] = {"passed": True, "reason": "No fakeout block"}

            filters_report["price_momentum"] = {
                "passed": momentum_ok,
                "reason": f"Mom:{price_momentum:+.1f} | Early:{live_reversal == 'BUY_EARLY' or div_type == 'BULLISH_DIV'}"
            }
            if momentum_ok:
                passed_filters += 1
            total_filters += 1

        elif signal_type == "SELL":
            structure_ok = market_structure not in ["BULLISH", "BULLISH_PULLBACK"]
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

            fib_sell_zone = (in_premium_zone and fib_position <= 25) or fib_position <= 30
            has_early = (live_reversal in ["SELL_EARLY"] or
                        div_type == "BEARISH_DIV" or
                        (cs_signal == "SELL" and cs_confidence >= 0.50))
            sweep_or_level = (sweep == "SELL_SWEEP") or near_resistance or in_overbought_zone or has_early or fib_sell_zone
            filters_report["smc_level"] = {
                "passed": sweep_or_level,
                "reason": f"Sweep:{sweep} | NearR:{near_resistance} | BBhigh:{in_overbought_zone} | Fib:{fib_sell_zone} | Early:{has_early}"
            }
            if sweep_or_level:
                passed_filters += 1
            total_filters += 1

            if fib_sell_zone and rsi >= 50:
                rsi_ok = True
                filters_report["rsi"] = {"passed": True, "reason": f"RSI:{rsi:.1f} | Fib premium relaxes"}
            else:
                rsi_predictive = (rsi >= 55) and (rsi_momentum < 0)
                rsi_sweet = 55 <= rsi <= 70
                rsi_early = (rsi >= 50) and (live_reversal == "SELL_EARLY" or div_type == "BEARISH_DIV")
                # v5.1: mirror of the BUY trend band — falling RSI 38-55 with
                # bearish momentum passes for early trend-start SELLs.
                rsi_trend = (38 <= rsi <= 55) and (rsi_momentum < 0) and (price_momentum <= 0)
                rsi_ok = rsi_predictive or rsi_sweet or rsi_early or rsi_trend
                filters_report["rsi"] = {
                    "passed": rsi_ok,
                    "reason": f"RSI:{rsi:.1f} | Mom:{rsi_momentum:+.2f} | Pred:{rsi_predictive} | Sweet:{rsi_sweet} | Trend:{rsi_trend}"
                }
            if rsi_ok:
                passed_filters += 1
            total_filters += 1

            momentum_ok = price_momentum <= 0 or live_reversal == "SELL_EARLY" or div_type == "BEARISH_DIV"
            if fakeout_detected and fakeout_direction == "BULLISH":
                momentum_ok = False
                filters_report["fakeout_block"] = {"passed": False, "reason": "Fakeout BULLISH override"}
            else:
                filters_report["fakeout_block"] = {"passed": True, "reason": "No fakeout block"}

            filters_report["price_momentum"] = {
                "passed": momentum_ok,
                "reason": f"Mom:{price_momentum:+.1f} | Early:{live_reversal == 'SELL_EARLY' or div_type == 'BEARISH_DIV'}"
            }
            if momentum_ok:
                passed_filters += 1
            total_filters += 1

        else:
            filters_report["hold"] = {"passed": False, "reason": "Signal is HOLD"}
            return False, {"filters": filters_report, "price": current_price, "is_early": is_early}

        if trend_blocked:
            filters_report["cooldown_check"] = {
                "passed": False,
                "reason": f"HARD TREND BLOCK: {signal_type} blocked in {market_structure}"
            }
            return False, {"filters": filters_report, "price": current_price,
                          "is_early": is_early, "confidence_boost": 0.0}

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