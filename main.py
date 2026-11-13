# main.py - AI Crypto Signal Bot v5.5.3
# Multi-Asset | Multi-Timeframe | /analyze endpoint for cron-job.org
# BTCUSD (Kraken, Binance fallback): 15m -> 5m -> 1m
# EURUSD (Kraken primary, Deriv fallback): 15m -> 5m -> 1m
# v5.1: 8h Telegram summary + morning-bird checks + per-cycle log heartbeat
# v5.2: PER-ASSET THRESHOLDS — EURUSD forex-calibrated gates
# v5.3: Summary send result logged properly + version bumps (2-layer quality
#       gate lives in multi_timeframe.py)
# v5.4.1: Reversal override (fixes stale-trend BUY/SELL lock) + Gold added
# v5.5.3: EURUSD moved to Kraken-primary (was flaking on Deriv HTTP 520s;
#         Deriv is now just its fallback) + Deriv retry/backoff + Kraken
#         PAXG/USD proxy fallback for gold when Deriv is unreachable

from dotenv import load_dotenv
load_dotenv()

import os
import sys
import asyncio
import redis.asyncio as redis
import json
import logging
import aiohttp
from datetime import datetime
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from core.data_feed import DataFeedManager
from core.multi_timeframe import MultiTimeframeEngine
from core.filters import FilterManager
from core.notifications import NotificationManager, SignalAlert
from core.outcome_tracker import OutcomeTracker
from core.news_gate import NewsGate

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# v5.5.4: single source of truth for the version string. Every user-visible
# version display (Telegram startup/signal/outcome messages, /health, /)
# reads from this constant instead of its own hardcoded copy — that
# duplication is exactly what let the Telegram footer drift several
# versions behind the rest of the bot, repeatedly.
APP_VERSION = "5.9.0"

redis_url = os.getenv('REDIS_PUBLIC_URL', os.getenv('REDIS_URL', 'redis://localhost:6379'))
try:
    redis_client = redis.Redis.from_url(redis_url, decode_responses=True)
    logger.info("Redis connected")
except Exception as e:
    logger.warning(f"Redis failed: {e}")
    redis_client = None

app_state = {'is_running': False, 'engine': None, 'last_signals': {}}


class CloudForexEngine:
    def __init__(self):
        self.assets_config = {
            "BTCUSD": {
                "source": "kraken",
                "market_24_7": True,  # crypto never closes
                "timeframes": {
                    "primary":   {"tf": "15m", "deriv_granularity": 900},
                    "secondary": {"tf": "5m",  "deriv_granularity": 300},
                    "entry":     {"tf": "1m",  "deriv_granularity": 60}
                },
                "check_interval": 300,
                "thresholds": None  # uses engine global defaults (crypto-calibrated)
            },
            "EURUSD": {
                "source": "kraken",
                "market_24_7": False,  # standard forex hours — see _is_forex_market_closed
                "timeframes": {
                    "primary":   {"tf": "15m", "deriv_granularity": 900},
                    "secondary": {"tf": "5m",  "deriv_granularity": 300},
                    "entry":     {"tf": "1m",  "deriv_granularity": 60}
                },
                "check_interval": 300,
                # FOREX-CALIBRATED THRESHOLDS (tune via .env EURUSD_* variables)
                "thresholds": {
                    "min_atr_pct": float(os.getenv("EURUSD_MIN_ATR_PCT", "0.008")),
                    "min_adx": float(os.getenv("EURUSD_MIN_ADX", "12")),
                    "min_confidence": float(os.getenv("EURUSD_MIN_CONFIDENCE", "0.55")),
                    "two_layer_min_score": float(os.getenv("EURUSD_TWO_LAYER_MIN_SCORE", "0.40")),
                }
            },
            "frxXAUUSD": {
                "source": "deriv",
                "display_name": "XAUUSD",
                "market_24_7": False,  # standard metals hours — see _is_forex_market_closed
                "timeframes": {
                    "primary":   {"tf": "15m", "deriv_granularity": 900},
                    "secondary": {"tf": "5m",  "deriv_granularity": 300},
                    "entry":     {"tf": "1m",  "deriv_granularity": 60}
                },
                "check_interval": 300,
                # GOLD-CALIBRATED THRESHOLDS — gold moves more than forex but
                # less than crypto per candle. Tune via .env XAUUSD_* vars.
                "thresholds": {
                    "min_atr_pct": float(os.getenv("XAUUSD_MIN_ATR_PCT", "0.012")),
                    "min_adx": float(os.getenv("XAUUSD_MIN_ADX", "14")),
                    "min_confidence": float(os.getenv("XAUUSD_MIN_CONFIDENCE", "0.55")),
                    "two_layer_min_score": float(os.getenv("XAUUSD_TWO_LAYER_MIN_SCORE", "0.40")),
                }
            }
        }

        self.tasks = {}
        self.health = {}
        self.data_feed = None
        self.notification_manager = None
        self.filter_managers = {}
        self.mtf_engine = MultiTimeframeEngine()
        self.outcome_tracker = OutcomeTracker(redis_client)
        self.news_gate = NewsGate(redis_client)

        # ===== MORNING-BIRD: faster checks during the morning window =====
        self.morning_start_utc = int(os.getenv("MORNING_BOOST_START_UTC", "6"))
        self.morning_end_utc = int(os.getenv("MORNING_BOOST_END_UTC", "10"))
        self.morning_check_interval = int(os.getenv("MORNING_CHECK_INTERVAL", "180"))

        # ===== Summary heartbeat interval (hours) =====
        self.summary_interval_hours = float(os.getenv("SUMMARY_INTERVAL_HOURS", "8"))

        # ===== v5.5.3: FOREX/METALS MARKET HOURS =====
        self.forex_close_hour_utc = int(os.getenv("FOREX_CLOSE_HOUR_UTC", "21"))  # Friday
        self.forex_open_hour_utc = int(os.getenv("FOREX_OPEN_HOUR_UTC", "21"))   # Sunday

        # ===== v5.5.3: DIRECTION-REPEAT SUPPRESSION =====
        # User feedback: getting a fresh BUY alert every few minutes for the
        # same pair while the trend hasn't actually changed is noise, not
        # new information — and risks someone re-entering a trade they're
        # already in. Once a direction is alerted for a pair, the SAME
        # direction is suppressed until either it flips (always allowed
        # through immediately — that's genuinely new information) or
        # DIRECTION_HOLD_HOURS passes (a long safety refresh so a pair
        # doesn't go silent for a full trading day if it never flips).
        self.direction_hold_hours = float(os.getenv("DIRECTION_HOLD_HOURS", "2"))
        self.last_alert_direction: dict = {}  # {asset: {"direction": "BUY", "timestamp": datetime}}

        logger.info("CloudForexEngine v5.6.5 | BTCUSD + EURUSD + XAUUSD (all analyzed) | Telegram alerts: BTC only | Per-asset thresholds")

    def _is_forex_market_closed(self) -> bool:
        """
        v5.5.3: EURUSD and XAUUSD aren't 24/7 like crypto — standard forex/
        metals hours are Sunday ~21:00 UTC open through Friday ~21:00 UTC
        close (brokers vary by an hour or so; tune via .env). Outside that
        window the feed is stale/thin, which is exactly how a nonsense
        signal got generated AND how the outcome tracker tried to resolve
        a trade against flat weekend candles. BTCUSD is exempt — crypto
        trades every day.
        """
        now = datetime.utcnow()
        wd = now.weekday()  # Monday=0 ... Saturday=5, Sunday=6
        if wd == 5:  # all day Saturday
            return True
        if wd == 4 and now.hour >= self.forex_close_hour_utc:  # Friday after close
            return True
        if wd == 6 and now.hour < self.forex_open_hour_utc:  # Sunday before open
            return True
        return False

    def _morning_active(self) -> bool:
        hour = datetime.utcnow().hour
        if self.morning_start_utc <= self.morning_end_utc:
            return self.morning_start_utc <= hour < self.morning_end_utc
        return hour >= self.morning_start_utc or hour < self.morning_end_utc

    async def initialize(self):
        logger.info("Initializing Engine v5.5.3...")
        try:
            self.data_feed = DataFeedManager()
            self.notification_manager = NotificationManager(version=APP_VERSION)
            try:
                await self.news_gate.load_cached()
                await self.news_gate.refresh(force=True)
            except Exception as e:
                logger.error(f"News gate startup load failed (non-fatal): {e}")

            await self.notification_manager.send_test_message(
                "🤖 AI Crypto Bot v" + APP_VERSION + " is ONLINE!\n"
                "━━━━━━━━━━━━━━━━━━━\n"
                "✅ Assets: BTCUSD + EURUSD + XAUUSD (all tracked) | 📵 Telegram alerts: BTC only\n"
                "📐 15m → 5m → 1m for all\n"
                "🎯 Exact Entry + TP + SL\n"
                "🌅 Morning-bird boost active\n"
                "💓 Status summary every "
                f"{self.summary_interval_hours:g}h\n"
                "🆕 v5.5.3: Reversal override fix (stale-trend BUY/SELL lock) "
                "+ EURUSD/Gold removed — BTC-only\n"
                f"⚙️ Version: {APP_VERSION}"
            )
            logger.info("Startup message sent")

            for asset in self.assets_config:
                self.filter_managers[asset] = FilterManager()
                self.health[asset] = {
                    'status': 'initializing',
                    'last_check': None,
                    'signals_today': 0,
                    'last_signal': None,
                    'last_hold_reason': None
                }

            app_state['is_running'] = True
            logger.info("Engine v5.5.3 ready!")

        except Exception as e:
            logger.error(f"Engine init failed: {e}")
            raise

    async def _keep_alive_ping(self):
        """
        v5.5.3 BUG FIX: this used to ping http://localhost:8000/health, which
        never leaves the container — Render's free-tier spin-down timer only
        resets on traffic hitting the PUBLIC URL through their edge/proxy, so
        a pure loopback call was invisible to it and did nothing to prevent
        the spin-down (confirmed by logs: a graceful "Shutting down" with no
        error, right after Render's own inactivity banner).
        Render auto-injects RENDER_EXTERNAL_URL with the public https:// URL
        for every web service — use that if present.
        """
        await asyncio.sleep(60)
        external_url = os.getenv("RENDER_EXTERNAL_URL", "").rstrip("/")
        ping_url = f"{external_url}/health" if external_url else "http://localhost:8000/health"
        if external_url:
            logger.info(f"Keep-alive pinging PUBLIC url: {ping_url}")
        else:
            logger.warning(
                "RENDER_EXTERNAL_URL not set — falling back to localhost, which "
                "will NOT prevent Render free-tier spin-down. Set up an external "
                "pinger (e.g. cron-job.org hitting your /health URL every 10 min) "
                "as the reliable fix — see chat notes."
            )
        while app_state['is_running']:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(ping_url, timeout=10) as resp:
                        logger.debug(f"Keep-alive: {resp.status}")
                await asyncio.sleep(300)
            except Exception as e:
                logger.debug(f"Keep-alive failed: {e}")
                await asyncio.sleep(60)

    # ===== SUMMARY HEARTBEAT (Telegram, every N hours) =====
    async def _news_loop(self):
        """v5.9.0: keeps the calendar fresh and sends each news notice exactly once."""
        await asyncio.sleep(20)
        warned_at = None
        while app_state['is_running']:
            try:
                await self.news_gate.refresh()
                now = datetime.utcnow()
                st = self.news_gate.status()
                if st["enabled"] and not st["data_ok"]:
                    if warned_at is None or (now - warned_at).total_seconds() > 6 * 3600:
                        warned_at = now
                        await self.notification_manager.send_test_message("⚠️ NEWS CALENDAR UNAVAILABLE\n" + st["reason"])
                else:
                    warned_at = None
                for text in await self.news_gate.pop_due_notices():
                    await self.notification_manager.send_test_message(text)
            except Exception as e:
                logger.error(f"News loop error: {e}")
            await asyncio.sleep(30)

    async def _summary_heartbeat_loop(self):
        """Sends a status summary to Telegram every SUMMARY_INTERVAL_HOURS."""
        await asyncio.sleep(120)  # first summary ~2 min after startup msg
        while app_state['is_running']:
            try:
                await self._send_summary()
            except Exception as e:
                logger.error(f"Summary heartbeat error: {e}")
            await asyncio.sleep(self.summary_interval_hours * 3600)

    async def _send_summary(self):
        now = datetime.utcnow()
        morning = self._morning_active()
        lines = [
            f"💓 BOT STATUS — {now.strftime('%Y-%m-%d %H:%M UTC')}",
            "━━━━━━━━━━━━━━━━━━━",
        ]
        for asset, info in self.health.items():
            display = self.assets_config[asset].get("display_name", asset)
            status_emoji = ("🟢" if info['status'] == 'healthy'
                           else "🌙" if info['status'] == 'market_closed'
                           else "🟡" if info['status'] == 'initializing' else "🔴")
            lines.append(f"{status_emoji} {display}: {info['status']} | Signals today: {info['signals_today']}")

            last_sig = info.get('last_signal')
            if last_sig:
                lines.append(
                    f"   Last signal: {last_sig['signal']} @ {last_sig['entry_price']:,.5f} "
                    f"({last_sig['timestamp'][11:16]} UTC)"
                )
            else:
                last_hold = info.get('last_hold_reason')
                if last_hold:
                    lines.append(f"   Watching: HOLD ({last_hold[:60]})")
                else:
                    lines.append("   Watching: no data yet")

        lines.append("━━━━━━━━━━━━━━━━━━━")
        lines.append(self.news_gate.summary_line())
        lines.append(f"🌅 Morning boost: {'ACTIVE' if morning else 'off'} "
                     f"({self.morning_start_utc}:00-{self.morning_end_utc}:00 UTC)")
        lines.append(f"⏱️ Check cycle: {self._current_check_interval()}s | Summary every {self.summary_interval_hours:g}h")
        lines.append(f"🤖 AI Bot v{APP_VERSION}")

        # v5.3: log the ACTUAL result instead of claiming success
        success = await self.notification_manager.send_test_message("\n".join(lines))
        if success:
            logger.info("Summary heartbeat sent")
        else:
            logger.error("Summary heartbeat FAILED to send — check Telegram config/logs above")

    def _current_check_interval(self) -> int:
        if self._morning_active():
            return self.morning_check_interval
        return 300

    async def analyze_asset_once(self, asset: str) -> dict:
        cfg = self.assets_config[asset]
        start_time = asyncio.get_event_loop().time()

        # ===== v5.5.3: MARKET HOURS GATE =====
        # Skip BOTH new-signal analysis AND outcome-checking while a non-24/7
        # market is closed. Deliberately skipping outcome checks too: a trade
        # genuinely can't resolve over the weekend, so checking it against
        # stale/frozen closed-market candles was producing bogus resolutions
        # (confirmed case: a EURUSD trade got marked resolved while the
        # market was shut). Checking resumes cleanly once the market reopens
        # and real candles start flowing again.
        if not cfg.get("market_24_7", True) and self._is_forex_market_closed():
            self.health[asset]['status'] = 'market_closed'
            self.health[asset]['last_check'] = asyncio.get_event_loop().time()
            self.health[asset]['last_hold_reason'] = "Market closed (forex/metals weekend)"
            logger.info(f"HEARTBEAT [{asset}] cycle complete | MARKET CLOSED — skipping analysis + outcome checks")
            return {"asset": asset, "signal": "HOLD", "reason": "Market closed (forex/metals weekend)"}

        try:
            result = await self.mtf_engine.analyze_asset(
                pair=asset,
                source=cfg["source"],
                timeframes=cfg["timeframes"],
                data_feed=self.data_feed,
                threshold_overrides=cfg.get("thresholds")
            )

            if result is None:
                self.health[asset]['status'] = 'no_data'
                self.health[asset]['last_check'] = asyncio.get_event_loop().time()
                logger.info(f"HEARTBEAT [{asset}] cycle complete | NO_DATA (fetch failed)")
                return {"asset": asset, "signal": "NO_DATA", "reason": "Failed to fetch or analyze"}

            # ===== v5.5.3: OUTCOME TRACKING =====
            # Runs every cycle regardless of whether a NEW signal fires this
            # time — this is what resolves signals sent on PRIOR cycles.
            resolved = await self.outcome_tracker.check_outcomes(asset, result.get("secondary_df"))
            for record in resolved:
                await self._send_outcome_notification(record)

            if result.get("signal") == "HOLD":
                self.health[asset]['status'] = 'healthy'
                self.health[asset]['last_check'] = asyncio.get_event_loop().time()
                self.health[asset]['last_hold_reason'] = result.get("reason", "No signal")
                # Per-cycle log heartbeat — proves the loop is alive in Render logs
                logger.info(f"HEARTBEAT [{asset}] cycle complete | HOLD: {result.get('reason', 'No signal')}")
                return {
                    "asset": asset,
                    "signal": "HOLD",
                    "reason": result.get("reason", "No signal"),
                    "volatility_passed": result.get("volatility_passed", False),
                    "layers": result.get("layers", {})
                }

            # v5.9.0: NEWS BLACKOUT — no new signal while a USD red-folder event is
            # near or active (prop-firm rule). Analysis and outcome tracking carry on.
            news = self.news_gate.status()
            if news["no_new_trades"]:
                self.health[asset]['status'] = 'healthy'
                self.health[asset]['last_check'] = asyncio.get_event_loop().time()
                self.health[asset]['last_hold_reason'] = news["reason"][:120]
                logger.info(f"HEARTBEAT [{asset}] cycle complete | HOLD (news gate): {news['reason']}")
                return {"asset": asset, "signal": "HOLD", "reason": news["reason"]}

            signal_type = result["signal"]
            current_price = result["entry_price"]
            is_early = result.get("is_early", False)
            is_reversal = result.get("reversal_override", False)

            filter_passed, filter_report = self.filter_managers[asset].validate_signal(
                df=result.get("secondary_df"),
                signal_type=signal_type,
                current_price=current_price,
                indicators=result.get("secondary_indicators"),
                pair=asset,
                is_early=is_early,
                is_reversal=is_reversal
            )

            if not filter_passed:
                reason = filter_report.get('filters', {}).get('cooldown_check', {}).get('reason', 'Filter blocked')
                for fname, fres in filter_report.get('filters', {}).items():
                    if not fres.get('passed', True):
                        logger.info(f"FILTER BLOCK [{asset}]: {fname}: {fres.get('reason', '')}")
                self.health[asset]['status'] = 'healthy'
                self.health[asset]['last_check'] = asyncio.get_event_loop().time()
                self.health[asset]['last_hold_reason'] = f"Filter: {reason}"
                logger.info(f"HEARTBEAT [{asset}] cycle complete | HOLD (filter): {reason}")
                return {
                    "asset": asset,
                    "signal": "HOLD",
                    "reason": f"Filter blocked: {reason}",
                    "layers": result.get("layers", {})
                }

            confidence_boost = filter_report.get('confidence_boost', 1.0)
            adjusted_confidence = min(result["confidence"] * confidence_boost, 0.99)

            if adjusted_confidence < 0.50:
                logger.info(f"LOW CONFIDENCE SKIP [{asset}]: {adjusted_confidence:.2%}")
                self.health[asset]['status'] = 'healthy'
                self.health[asset]['last_check'] = asyncio.get_event_loop().time()
                self.health[asset]['last_hold_reason'] = f"Low confidence {adjusted_confidence:.2%}"
                logger.info(f"HEARTBEAT [{asset}] cycle complete | HOLD (low confidence)")
                return {"asset": asset, "signal": "HOLD", "reason": f"Confidence {adjusted_confidence:.2%} too low"}

            # ===== v5.5.3: DIRECTION-REPEAT GATE =====
            # A flip to the opposite direction always goes through — that's
            # real new information. The SAME direction repeating is only
            # suppressed for direction_hold_hours; a flip resets the clock.
            last = self.last_alert_direction.get(asset)
            now_dt = datetime.utcnow()
            if last and last["direction"] == signal_type:
                hours_since = (now_dt - last["timestamp"]).total_seconds() / 3600
                if hours_since < self.direction_hold_hours:
                    reason = (f"Same-direction repeat suppressed: {signal_type} already "
                              f"alerted {hours_since*60:.0f}m ago (holds for "
                              f"{self.direction_hold_hours:g}h or until it flips)")
                    logger.info(f"DIRECTION-REPEAT SUPPRESSED [{asset}]: {reason}")
                    self.health[asset]['status'] = 'healthy'
                    self.health[asset]['last_check'] = asyncio.get_event_loop().time()
                    self.health[asset]['last_hold_reason'] = reason
                    logger.info(f"HEARTBEAT [{asset}] cycle complete | HOLD (direction-repeat): {reason}")
                    return {"asset": asset, "signal": "HOLD", "reason": reason}
            self.last_alert_direction[asset] = {"direction": signal_type, "timestamp": now_dt}

            urgency = 'high' if adjusted_confidence > 0.85 else 'normal' if adjusted_confidence > 0.60 else 'low'

            display_name = cfg.get("display_name", asset)

            # v5.6.0: ONE shared timestamp for both the alert and the outcome
            # record, so the signal_id the Accept/Reject button references
            # (asset + this exact timestamp) is guaranteed to match the one
            # outcome_tracker.record_signal derives below — two separate
            # datetime.utcnow() calls could differ by microseconds and break
            # the button's lookup.
            signal_timestamp = datetime.utcnow()
            signal_timestamp_iso = signal_timestamp.isoformat()
            signal_id = f"{asset}_{signal_timestamp_iso}"

            alert = SignalAlert(
                pair=display_name,
                signal=signal_type,
                entry_price=result["entry_price"],
                confidence=adjusted_confidence,
                timestamp=signal_timestamp,
                reasoning=result["reason"],
                stop_loss=result["stop_loss"],
                take_profit=result["take_profit"],
                urgency=urgency,
                is_early=is_early,
                confluence_score=result["confidence"],
                layers=result.get("layers", {}),
                signal_id=signal_id,
                extra_lines=self._context_lines(result.get("context", {}), signal_type)
            )

            await self.notification_manager.send_telegram_alert(alert)

            signal_data = {
                "asset": asset,
                "display_name": display_name,
                "signal": signal_type,
                "entry_price": result["entry_price"],
                "stop_loss": result["stop_loss"],
                "take_profit": result["take_profit"],
                "confidence": adjusted_confidence,
                "timestamp": signal_timestamp_iso,
                "reason": result["reason"],
                "is_early": is_early,
                "reversal_override": is_reversal,
                "two_layer_mode": result.get("two_layer_mode", False),
                "morning_boost": result.get("morning_boost", False),
                "atr_pct": result.get("atr_pct"),
                "adx": result.get("adx"),
                "layers": result.get("layers", {}),
                "setup_hash": result.get("setup_hash"),
                "context": result.get("context", {})
            }

            if redis_client:
                await redis_client.setex(f"signal:{asset}:current", 3600, json.dumps(signal_data))

            await self.outcome_tracker.record_signal(signal_data)

            self.health[asset]['signals_today'] += 1
            self.health[asset]['last_signal'] = signal_data
            self.health[asset]['status'] = 'healthy'
            self.health[asset]['last_check'] = asyncio.get_event_loop().time()

            label = "🔄 REVERSAL" if is_reversal else ("🚨 EARLY" if is_early else "🚀 CONFIRMED")
            mode_tag = " [2-LAYER]" if result.get("two_layer_mode") else ""
            morning_tag = " 🌅" if result.get("morning_boost") else ""
            logger.info(
                f"{label}{mode_tag}{morning_tag} [{asset}] {signal_type} @ {result['entry_price']:.5f} | "
                f"SL:{result['stop_loss']:.5f} TP:{result['take_profit']:.5f} | "
                f"Conf:{adjusted_confidence:.2%}"
            )

            return signal_data

        except Exception as e:
            logger.error(f"Error analyzing {asset}: {e}")
            self.health[asset]['status'] = 'error'
            self.health[asset]['last_check'] = asyncio.get_event_loop().time()
            return {"asset": asset, "signal": "ERROR", "reason": str(e)}

    def _context_lines(self, ctx: dict, signal_type: str) -> list:
        """v5.8.0: info/warning lines shown on the Telegram alert, built from the signal's recorded context."""
        lines = []
        dist, ratio = ctx.get("sl_distance"), ctx.get("sl_atr_ratio")
        if dist and ratio is not None:
            dist_txt = f"{dist:,.0f}" if dist >= 10 else f"{dist:.5f}"
            lines.append(f"📏 Stop {dist_txt} pts = {ratio:.1f}x the 5m ATR (under 1.0x = noise can hit it)")
        if ctx.get("primary_opposes"):
            lines.append(f"⚠️ 15m layer's own signal is {ctx.get('primary_signal')} — AGAINST this {signal_type}")
        if ctx.get("range_chop"):
            lines.append("⚠️ Range chop (whipsaw) flagged — trend signals fail more often here")
        lines.extend(ctx.get("smc_lines") or [])
        return lines

    async def _send_outcome_notification(self, record: dict):
        """Plain-text follow-up when a tracked signal resolves WIN/LOSS/EXPIRED."""
        # v5.6.6 BUG FIX: send_telegram_alert() (the NEW-signal path) was
        # already BTC-only filtered in notifications.py, but this RESOLVED-
        # outcome path calls send_test_message() directly — a separate,
        # unfiltered route — so EUR/Gold win/loss messages kept slipping
        # through even though new EUR/Gold signals were correctly silenced.
        # Same filter, applied here too. EUR/Gold still track normally in
        # the background either way — check /api/stats any time.
        if "BTC" not in record.get("asset", "").upper():
            logger.info(f"Outcome notification suppressed (BTC-only mode): {record.get('asset')} {record.get('status')}")
            return
        try:
            status = record["status"]
            if status == "EXPIRED":
                emoji = "⌛"
                result_line = f"EXPIRED — still open after {self.outcome_tracker.timeout_hours:g}h, no TP/SL hit"
            elif status == "WIN":
                emoji = "✅"
                result_line = f"WIN — TP hit @ {record['close_price']:,.5f} ({record['r_multiple']:+.2f}R)"
            else:
                emoji = "❌"
                result_line = f"LOSS — SL hit @ {record['close_price']:,.5f} ({record['r_multiple']:+.2f}R)"

            msg = (
                f"{emoji} SIGNAL RESOLVED\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"📊 {record['display_name']} {record['direction']}\n"
                f"Entry: {record['entry_price']:,.5f}\n"
                f"{result_line}\n"
                f"Opened: {record['opened_at'][11:16]} UTC | "
                f"Closed: {record['closed_at'][11:16]} UTC\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"🤖 AI Bot v{APP_VERSION} | Outcome Tracker"
            )
            await self.notification_manager.send_test_message(msg)
        except Exception as e:
            logger.error(f"Outcome notification error: {e}")

    async def monitor_asset(self, asset: str):
        cfg = self.assets_config[asset]
        logger.info(f"Monitor started: {asset} ({cfg['source']}) | interval: {cfg['check_interval']}s")

        while app_state['is_running']:
            start = asyncio.get_event_loop().time()
            await self.analyze_asset_once(asset)
            elapsed = asyncio.get_event_loop().time() - start
            effective_interval = self._current_check_interval()
            sleep_time = max(5, effective_interval - elapsed)
            await asyncio.sleep(sleep_time)

    async def start(self):
        await self.initialize()
        external_url = os.getenv("RENDER_EXTERNAL_URL", "").rstrip("/")
        if external_url and self.notification_manager.enabled:
            await self.notification_manager.set_webhook(f"{external_url}/telegram/webhook")
        else:
            logger.warning(
                "Skipping Telegram webhook registration (no RENDER_EXTERNAL_URL or "
                "Telegram not configured) — Accept/Reject buttons won't work until this is set."
            )
        asyncio.create_task(self._keep_alive_ping())
        asyncio.create_task(self._summary_heartbeat_loop())
        asyncio.create_task(self._news_loop())
        for asset in self.assets_config:
            self.tasks[asset] = asyncio.create_task(self.monitor_asset(asset))
            await asyncio.sleep(2)

    async def stop(self):
        app_state['is_running'] = False
        for task in self.tasks.values():
            task.cancel()
        if self.notification_manager:
            await self.notification_manager.close()
        if redis_client:
            await redis_client.close()
        logger.info("Engine v5.5.3 stopped")


@asynccontextmanager
async def lifespan(app: FastAPI):
    engine = CloudForexEngine()
    app_state['engine'] = engine
    await engine.start()
    yield
    await engine.stop()


app = FastAPI(
    title="AI Crypto Signal Bot",
    description="Multi-Asset Multi-Timeframe | BTCUSD + EURUSD + XAUUSD | Telegram: BTC only",
    version=APP_VERSION,
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


@app.get("/")
async def root():
    engine = app_state.get('engine')
    assets = list(engine.assets_config.keys()) if engine else []
    return {
        "status": "running" if app_state['is_running'] else "down",
        "assets": assets,
        "version": APP_VERSION,
        "mode": "Multi-Timeframe | Per-Asset Thresholds | Morning Boost | Reversal Override | Quality-Gated 2-Layer"
    }


@app.get("/health")
async def health_check():
    engine = app_state.get('engine')
    if not engine:
        return {"status": "down", "error": "Engine not initialized"}
    return {
        "status": "healthy" if app_state['is_running'] else "down",
        "version": APP_VERSION,
        "morning_boost_active": engine._morning_active(),
        "check_interval_seconds": engine._current_check_interval(),
        "assets": {
            asset: {
                "status": info['status'],
                "signals_today": info['signals_today'],
                "last_signal": info.get('last_signal')
            }
            for asset, info in engine.health.items()
        }
    }


@app.get("/analyze")
async def trigger_analysis():
    engine = app_state.get('engine')
    if not engine or not app_state['is_running']:
        raise HTTPException(status_code=503, detail="Engine not running")

    results = {}
    for asset in engine.assets_config:
        try:
            result = await engine.analyze_asset_once(asset)
            results[asset] = result
        except Exception as e:
            results[asset] = {"asset": asset, "signal": "ERROR", "reason": str(e)}

    return {
        "triggered_at": datetime.utcnow().isoformat(),
        "results": results
    }


@app.get("/api/signal/{asset}")
async def get_signal(asset: str):
    if not redis_client:
        raise HTTPException(status_code=503, detail="Redis not available")
    data = await redis_client.get(f"signal:{asset.upper()}:current")
    if not data:
        raise HTTPException(status_code=404, detail=f"No active signal for {asset}")
    return json.loads(data)


@app.get("/api/signals")
async def get_all_signals():
    if not redis_client:
        raise HTTPException(status_code=503, detail="Redis not available")
    signals = {}
    for asset in ["BTCUSD", "EURUSD", "frxXAUUSD"]:
        data = await redis_client.get(f"signal:{asset}:current")
        if data:
            signals[asset] = json.loads(data)
    return signals


# ===== v5.6.0: EXECUTION — Accept/Reject buttons + EA polling interface =====
# Flow: signal fires -> Telegram message with Accept(Demo)/Reject buttons ->
# tap hits /telegram/webhook -> on Accept, the trade is pushed onto a Redis
# queue ("exec:queue:demo") -> your MQL5 EA polls GET /api/pending_trade via
# WebRequest, pops the next one, places the order, then calls
# POST /api/trade_executed to report back (you get a Telegram confirmation).
# Nothing executes without an explicit Accept tap — Reject or silence both
# mean no trade.

@app.post("/telegram/webhook")
async def telegram_webhook(request: Request):
    engine = app_state.get('engine')
    if not engine:
        return {"ok": True}  # ack anyway — nothing we can do, avoid Telegram retry storms

    try:
        update = await request.json()
    except Exception:
        return {"ok": True}

    callback = update.get("callback_query")
    if not callback:
        return {"ok": True}  # not a button tap — ignore (e.g. a plain text message)

    callback_id = callback.get("id", "")
    data = callback.get("data", "")
    message = callback.get("message", {})
    chat_id = message.get("chat", {}).get("id")
    message_id = message.get("message_id")

    try:
        action, signal_id = data.split(":", 1)
    except ValueError:
        await engine.notification_manager.answer_callback(callback_id, "Unrecognized button")
        return {"ok": True}

    if action == "rej":
        await engine.notification_manager.answer_callback(callback_id, "Rejected")
        if chat_id and message_id:
            await engine.notification_manager.edit_message_reply_markup(
                chat_id, message_id, text=f"❌ Rejected — no trade placed.\n\n(signal: {signal_id})"
            )
        return {"ok": True}

    if action == "acc_demo":
        if not redis_client:
            await engine.notification_manager.answer_callback(callback_id, "Redis unavailable — can't queue trade")
            return {"ok": True}

        raw = await redis_client.get(f"outcome:{signal_id}")
        if not raw:
            await engine.notification_manager.answer_callback(callback_id, "Signal expired or not found")
            return {"ok": True}

        record = json.loads(raw)
        trade_payload = {
            "signal_id": signal_id,
            "asset": record["asset"],
            "direction": record["direction"],
            "entry_price": record["entry_price"],
            "stop_loss": record["stop_loss"],
            "take_profit": record["take_profit"],
            "lot_size": float(os.getenv("TARGET_LOT_SIZE", "0.5")),
            "account": "demo",
            "queued_at": datetime.utcnow().isoformat(),
        }
        await redis_client.lpush("exec:queue:demo", json.dumps(trade_payload))
        await engine.notification_manager.answer_callback(callback_id, "Accepted — queued for your EA")
        if chat_id and message_id:
            await engine.notification_manager.edit_message_reply_markup(
                chat_id, message_id,
                text=f"✅ Accepted (Demo) — queued for execution.\n\n(signal: {signal_id})"
            )
        logger.info(f"TRADE QUEUED [demo]: {signal_id}")
        return {"ok": True}

    if action == "acc_live":
        # v5.7.0: HARD GATE — defaults to false. This is the ONLY thing
        # standing between a button tap and a real-money order, so it must
        # be a deliberate, separate action on your end (set
        # LIVE_TRADING_ENABLED=true in Render's environment variables) —
        # never something a code deploy alone can turn on.
        if os.getenv("LIVE_TRADING_ENABLED", "false").lower() != "true":
            await engine.notification_manager.answer_callback(
                callback_id, "Live trading is NOT enabled on the server yet", show_alert=True
            )
            if chat_id and message_id:
                await engine.notification_manager.edit_message_reply_markup(
                    chat_id, message_id,
                    text=f"⚠️ LIVE trading is not enabled on the server — nothing was queued.\n"
                         f"Set LIVE_TRADING_ENABLED=true on Render when you're ready.\n\n(signal: {signal_id})"
                )
            logger.warning(f"LIVE ACCEPT BLOCKED (LIVE_TRADING_ENABLED off): {signal_id}")
            return {"ok": True}

        news = engine.news_gate.status()
        if news["no_new_trades"]:
            await engine.notification_manager.answer_callback(
                callback_id, "News blackout — LIVE trade NOT queued", show_alert=True)
            if chat_id and message_id:
                await engine.notification_manager.edit_message_reply_markup(
                    chat_id, message_id, text=f"⏸ NOT queued — {news['reason']}\n\n(signal: {signal_id})")
            logger.warning(f"LIVE ACCEPT BLOCKED (news): {signal_id} | {news['reason']}")
            return {"ok": True}

        if not redis_client:
            await engine.notification_manager.answer_callback(callback_id, "Redis unavailable — can't queue trade")
            return {"ok": True}

        raw = await redis_client.get(f"outcome:{signal_id}")
        if not raw:
            await engine.notification_manager.answer_callback(callback_id, "Signal expired or not found")
            return {"ok": True}

        record = json.loads(raw)
        trade_payload = {
            "signal_id": signal_id,
            "asset": record["asset"],
            "direction": record["direction"],
            "entry_price": record["entry_price"],
            "stop_loss": record["stop_loss"],
            "take_profit": record["take_profit"],
            "lot_size": float(os.getenv("TARGET_LOT_SIZE", "0.5")),  # EA recalculates this for real anyway
            "account": "live",
            "queued_at": datetime.utcnow().isoformat(),
        }
        await redis_client.lpush("exec:queue:live", json.dumps(trade_payload))
        await engine.notification_manager.answer_callback(callback_id, "🔴 LIVE — accepted and queued", show_alert=True)
        if chat_id and message_id:
            await engine.notification_manager.edit_message_reply_markup(
                chat_id, message_id,
                text=f"🔴 Accepted (LIVE — real money) — queued for execution.\n\n(signal: {signal_id})"
            )
        logger.warning(f"TRADE QUEUED [LIVE]: {signal_id}")  # warning level — live queues should stand out in logs
        return {"ok": True}

    await engine.notification_manager.answer_callback(callback_id, "Unknown action")
    return {"ok": True}


@app.get("/api/pending_trade")
async def get_pending_trade(account: str = "demo"):
    """
    Polled by the MQL5 EA (via WebRequest) every few seconds. Pops and
    returns the oldest queued, accepted trade for this account — or an
    empty body if none. Popping (not just reading) means each trade is
    handed out exactly once, so the EA doesn't need to track what it's
    already seen.

    v5.6.1 SAFEGUARD: an Accept tap queues SL/TP calculated around the
    price AT THAT MOMENT. If the EA (laptop/MT5) isn't running to pick it
    up right away — e.g. the laptop was off overnight — the market can
    move far enough that those levels no longer make sense by the time it
    finally executes. Confirmed case: an Oct 14 accepted trade sat queued
    until the EA started running days later and fired against Oct 15's
    price. Now anything older than EXEC_QUEUE_TIMEOUT_MINUTES (default 15)
    is discarded instead of executed, with a Telegram notice so it's never
    silent about why a trade you accepted never happened. Loops past
    multiple stale entries in case several piled up while nothing was
    polling.
    """
    if not redis_client:
        raise HTTPException(status_code=503, detail="Redis not available")

    # v5.9.0: during/just before news, hand nothing out. The queued trade is left
    # untouched — if it's still waiting once the news is over it will be older than
    # the expiry limit and get discarded, never executed into post-news chaos.
    _eng = app_state.get('engine')
    if account == "live" and _eng and _eng.news_gate.status()["no_new_trades"]:
        return {}

    timeout_minutes = float(os.getenv("EXEC_QUEUE_TIMEOUT_MINUTES", "15"))
    engine = app_state.get('engine')

    while True:
        raw = await redis_client.rpop(f"exec:queue:{account}")
        if not raw:
            return {}

        trade = json.loads(raw)
        queued_at = datetime.fromisoformat(trade["queued_at"])
        age_minutes = (datetime.utcnow() - queued_at).total_seconds() / 60

        if age_minutes <= timeout_minutes:
            return trade

        logger.warning(
            f"EXPIRED TRADE DISCARDED [{account}]: {trade.get('signal_id')} "
            f"was queued {age_minutes:.1f}m ago (limit {timeout_minutes:g}m) — not executing"
        )
        if engine and engine.notification_manager:
            await engine.notification_manager.send_test_message(
                f"⌛ ACCEPTED TRADE EXPIRED — NOT EXECUTED\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"📊 {trade.get('asset', '?')} {trade.get('direction', '?')} | Account: {account}\n"
                f"Queued {age_minutes:.0f} min ago — too stale to execute safely "
                f"(limit {timeout_minutes:g} min). Market has likely moved.\n"
                f"(signal: {trade.get('signal_id', '?')})\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"Tip: only tap Accept when your laptop + MT5 + EA are on and "
                f"about to pick it up — not for later."
            )
        # loop again — keep discarding stale entries until a fresh one or empty


@app.post("/api/trade_executed")
async def trade_executed(request: Request):
    """
    Called by the EA right after it places (or fails to place) an order, so
    you get a Telegram confirmation instead of having to check MT5 yourself.
    Expected JSON body: {signal_id, asset, direction, account, status
    ("success"/"failed"), ticket (optional), lot_size (optional),
    error (optional, if status=="failed")}
    """
    engine = app_state.get('engine')
    body = await request.json()

    signal_id = body.get("signal_id", "unknown")
    status = body.get("status", "unknown")
    asset = body.get("asset", "?")
    direction = body.get("direction", "?")
    account = body.get("account", "?")

    risk_usd = body.get("risk_usd")
    balance = body.get("balance")
    risk_line = ""
    if risk_usd and balance:
        risk_pct = (float(risk_usd) / float(balance) * 100) if float(balance) > 0 else 0
        risk_line = f"💵 Risked: ${float(risk_usd):.2f} ({risk_pct:.1f}% of ${float(balance):.2f} balance)\n"

    if status == "success":
        msg = (f"🤝 TRADE EXECUTED\n━━━━━━━━━━━━━━━━━━━━━━━\n"
               f"📊 {asset} {direction} | Account: {account}\n"
               f"🎫 Ticket: {body.get('ticket', '?')} | Lot: {body.get('lot_size', '?')}\n"
               f"{risk_line}"
               f"(signal: {signal_id})")
    else:
        msg = (f"⚠️ TRADE {'SKIPPED' if 'Skipped' in str(body.get('error', '')) else 'EXECUTION FAILED'}\n"
               f"━━━━━━━━━━━━━━━━━━━━━━━\n"
               f"📊 {asset} {direction} | Account: {account}\n"
               f"{risk_line}"
               f"Reason: {body.get('error', 'unknown')}\n"
               f"(signal: {signal_id})")

    logger.info(f"EA REPORTED [{account}] {signal_id}: {status}")
    if engine and engine.notification_manager:
        await engine.notification_manager.send_test_message(msg)
    return {"ok": True}


@app.get("/api/news_status")
async def news_status():
    """Polled by the EA (about every 20 s). Also handy to open in a browser to see what the bot thinks."""
    engine = app_state.get('engine')
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    return engine.news_gate.status()


@app.get("/api/stats")
async def get_stats(asset: str = None, since: str = None, breakdown: bool = False):
    """
    Real win-rate / R-multiple stats from the outcome tracker — win_rate_pct,
    avg_r_multiple, wins/losses/expired, overall and broken down per asset.
    Pass ?asset=BTCUSD to scope to one pair.
    Pass ?since=2026-10-04 (or a full ISO timestamp) to only count signals
    OPENED at or after that point — use this to exclude trades from before
    the bot stabilized rather than letting them drag the win rate down
    forever. Combine both: ?asset=EURUSD&since=2026-10-04
    """
    engine = app_state.get('engine')
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    if not redis_client:
        raise HTTPException(status_code=503, detail="Redis not available — stats require Redis")
    return await engine.outcome_tracker.get_stats(asset=asset, since=since, breakdown=breakdown)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=False,
        workers=1,
        loop="asyncio"
    )