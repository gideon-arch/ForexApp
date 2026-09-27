# main.py - AI Crypto Signal Bot v5.4.2
# Multi-Asset | Multi-Timeframe | /analyze endpoint for cron-job.org
# BTCUSD (Kraken, Binance fallback): 15m -> 5m -> 1m
# EURUSD (Kraken primary, Deriv fallback): 15m -> 5m -> 1m
# XAUUSD (Deriv primary w/ retry, Kraken PAXG/USD proxy fallback): 15m -> 5m -> 1m
# v5.1: 8h Telegram summary + morning-bird checks + per-cycle log heartbeat
# v5.2: PER-ASSET THRESHOLDS — EURUSD forex-calibrated gates
# v5.3: Summary send result logged properly + version bumps (2-layer quality
#       gate lives in multi_timeframe.py)
# v5.4.1: Reversal override (fixes stale-trend BUY/SELL lock) + Gold added
# v5.4.2: EURUSD moved to Kraken-primary (was flaking on Deriv HTTP 520s;
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
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from core.data_feed import DataFeedManager
from core.multi_timeframe import MultiTimeframeEngine
from core.filters import FilterManager
from core.notifications import NotificationManager, SignalAlert

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

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

        # ===== MORNING-BIRD: faster checks during the morning window =====
        self.morning_start_utc = int(os.getenv("MORNING_BOOST_START_UTC", "6"))
        self.morning_end_utc = int(os.getenv("MORNING_BOOST_END_UTC", "10"))
        self.morning_check_interval = int(os.getenv("MORNING_CHECK_INTERVAL", "180"))

        # ===== Summary heartbeat interval (hours) =====
        self.summary_interval_hours = float(os.getenv("SUMMARY_INTERVAL_HOURS", "8"))

        logger.info("CloudForexEngine v5.4.2 | BTCUSD (Kraken) + EURUSD + XAUUSD (Deriv/Kraken fallback) | Per-asset thresholds")

    def _morning_active(self) -> bool:
        hour = datetime.utcnow().hour
        if self.morning_start_utc <= self.morning_end_utc:
            return self.morning_start_utc <= hour < self.morning_end_utc
        return hour >= self.morning_start_utc or hour < self.morning_end_utc

    async def initialize(self):
        logger.info("Initializing Engine v5.4.2...")
        try:
            self.data_feed = DataFeedManager()
            self.notification_manager = NotificationManager()

            await self.notification_manager.send_test_message(
                "🤖 AI Crypto Bot v5.4.2 is ONLINE!\n"
                "━━━━━━━━━━━━━━━━━━━\n"
                "✅ Assets: BTCUSD (Kraken) + EURUSD + XAUUSD (Deriv/Kraken)\n"
                "📐 15m → 5m → 1m for all\n"
                "🎯 Exact Entry + TP + SL\n"
                "🌅 Morning-bird boost active\n"
                "💓 Status summary every "
                f"{self.summary_interval_hours:g}h\n"
                "🆕 v5.4.2: Reversal override fix (stale-trend BUY/SELL lock) "
                "+ EURUSD Kraken-fallback fix + Gold added\n"
                "⚙️ Version: 5.4.2"
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
            logger.info("Engine v5.4.2 ready!")

        except Exception as e:
            logger.error(f"Engine init failed: {e}")
            raise

    async def _keep_alive_ping(self):
        await asyncio.sleep(60)
        while app_state['is_running']:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get('http://localhost:8000/health', timeout=10) as resp:
                        logger.debug(f"Keep-alive: {resp.status}")
                await asyncio.sleep(300)
            except Exception as e:
                logger.debug(f"Keep-alive failed: {e}")
                await asyncio.sleep(60)

    # ===== SUMMARY HEARTBEAT (Telegram, every N hours) =====
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
            status_emoji = "🟢" if info['status'] == 'healthy' else "🟡" if info['status'] == 'initializing' else "🔴"
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
        lines.append(f"🌅 Morning boost: {'ACTIVE' if morning else 'off'} "
                     f"({self.morning_start_utc}:00-{self.morning_end_utc}:00 UTC)")
        lines.append(f"⏱️ Check cycle: {self._current_check_interval()}s | Summary every {self.summary_interval_hours:g}h")
        lines.append("🤖 AI Bot v5.4.2")

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

            urgency = 'high' if adjusted_confidence > 0.85 else 'normal' if adjusted_confidence > 0.60 else 'low'

            display_name = cfg.get("display_name", asset)
            alert = SignalAlert(
                pair=display_name,
                signal=signal_type,
                entry_price=result["entry_price"],
                confidence=adjusted_confidence,
                timestamp=datetime.utcnow(),
                reasoning=result["reason"],
                stop_loss=result["stop_loss"],
                take_profit=result["take_profit"],
                urgency=urgency,
                is_early=is_early,
                confluence_score=result["confidence"],
                layers=result.get("layers", {})
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
                "timestamp": datetime.utcnow().isoformat(),
                "reason": result["reason"],
                "is_early": is_early,
                "reversal_override": is_reversal,
                "two_layer_mode": result.get("two_layer_mode", False),
                "morning_boost": result.get("morning_boost", False),
                "atr_pct": result.get("atr_pct"),
                "adx": result.get("adx"),
                "layers": result.get("layers", {}),
                "setup_hash": result.get("setup_hash")
            }

            if redis_client:
                await redis_client.setex(f"signal:{asset}:current", 3600, json.dumps(signal_data))

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
        asyncio.create_task(self._keep_alive_ping())
        asyncio.create_task(self._summary_heartbeat_loop())
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
        logger.info("Engine v5.4.2 stopped")


@asynccontextmanager
async def lifespan(app: FastAPI):
    engine = CloudForexEngine()
    app_state['engine'] = engine
    await engine.start()
    yield
    await engine.stop()


app = FastAPI(
    title="AI Crypto Signal Bot",
    description="Multi-Asset Multi-Timeframe | BTCUSD + EURUSD + XAUUSD",
    version="5.4.2",
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
        "version": "5.4.2",
        "mode": "Multi-Timeframe | Per-Asset Thresholds | Morning Boost | Reversal Override | Quality-Gated 2-Layer"
    }


@app.get("/health")
async def health_check():
    engine = app_state.get('engine')
    if not engine:
        return {"status": "down", "error": "Engine not initialized"}
    return {
        "status": "healthy" if app_state['is_running'] else "down",
        "version": "5.4.2",
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