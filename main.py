# main.py - AI Crypto Signal Bot v4.1 (Production Ready)
# FAST VERSION — 5m candles, 5min checks, pure price action
# NO LSTM — Candlestick + Live Reversal + RSI Divergence + Fakeout + SMC + Fibonacci

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

# Local imports
from core.data_feed import DataFeedManager
from core.technical_analysis import TechnicalAnalyzer
from core.filters import FilterManager
from core.notifications import NotificationManager, SignalAlert

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Redis
redis_url = os.getenv('REDIS_PUBLIC_URL', os.getenv('REDIS_URL', 'redis://localhost:6379'))
try:
    redis_client = redis.Redis.from_url(redis_url, decode_responses=True)
    logger.info("Redis connected")
except Exception as e:
    logger.warning(f"Redis connection failed: {e}")
    redis_client = None

app_state = {'is_running': False, 'engine': None, 'last_signals': {}}


class CloudForexEngine:
    def __init__(self):
        # v4.1: 5m timeframe, 5min check interval (was 10s — too noisy!)
        self.pairs_config = {
            "BTCUSD": {"source": "kraken", "tf": "5m", "check_interval": 300}
        }
        self.tasks = {}
        self.health = {}
        self.last_sent_signal = {pair: None for pair in self.pairs_config}
        self.data_feed = None
        self.notification_manager = None
        self.filter_managers = {}
        self.analyzer = TechnicalAnalyzer()
        logger.info("CloudForexEngine v4.1 initialized | 5m candles, 5min checks | BTCUSD only | Fibonacci + Trend Filter")

    async def initialize(self):
        logger.info("Initializing Engine v4.1...")
        try:
            self.data_feed = DataFeedManager()
            self.notification_manager = NotificationManager()

            # Send startup test message to Telegram
            try:
                await self.notification_manager.send_test_message(
                    "🤖 AI Crypto Bot v4.1 is ONLINE!\n"
                    "━━━━━━━━━━━━━━━━━━━\n"
                    "✅ Monitoring: BTCUSD only\n"
                    "📡 Source: Kraken\n"
                    "⚡ 5m candles / 5min checks\n"
                    "🎯 Signals: Candlestick + Live Reversal + RSI Div + Fakeout + SMC + Fibonacci\n"
                    "🔒 HARD TREND FILTER: No counter-trend signals\n"
                    "🧠 NO LSTM — Pure Price Action\n"
                    "⚙️ Version: 4.1"
                )
                logger.info("Startup test message sent to Telegram")
            except Exception as e:
                logger.warning(f"Failed to send startup message: {e}")

            for pair in self.pairs_config:
                self.filter_managers[pair] = FilterManager()
                self.health[pair] = {
                    'status': 'initializing',
                    'last_check': None,
                    'signals_today': 0,
                    'last_signal': None
                }
            app_state['is_running'] = True
            logger.info("Engine v4.1 initialization complete!")
        except Exception as e:
            logger.error(f"Engine initialization failed: {e}")
            raise

    async def _keep_alive_ping(self):
        await asyncio.sleep(60)
        while app_state['is_running']:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get('http://localhost:8000/health', timeout=10) as resp:
                        logger.debug(f"Keep-alive ping: {resp.status}")
                await asyncio.sleep(300)
            except Exception as e:
                logger.debug(f"Keep-alive ping failed: {e}")
                await asyncio.sleep(60)

    async def monitor_pair(self, pair: str):
        config = self.pairs_config[pair]
        logger.info(f"Starting monitor for {pair} ({config['source']}) | {config['tf']} candles every {config['check_interval']}s")
        
        while app_state['is_running']:
            try:
                start_time = asyncio.get_event_loop().time()
                df = await asyncio.wait_for(
                    self.data_feed.fetch_latest(pair, timeframe=config['tf']),
                    timeout=20.0
                )
                if df is None or df.empty:
                    logger.warning(f"No data for {pair}")
                    await asyncio.sleep(10)
                    continue
                if len(df) < 200:
                    logger.warning(f"Insufficient data for {pair}: {len(df)} candles")
                    await asyncio.sleep(10)
                    continue

                try:
                    indicators = self.analyzer.calculate_indicators(df)
                except Exception as e:
                    logger.warning(f"Technical analysis failed for {pair}: {e}")
                    await asyncio.sleep(10)
                    continue

                current_price = indicators['current_price']
                smc_signal = indicators.get('smc_suggested_signal', 'HOLD')
                smc_score = indicators.get('confluence_score', 0.0)
                rsi = indicators.get('rsi', 50.0)
                rsi_momentum = indicators.get('rsi_momentum', 0.0)
                live_reversal = indicators.get('live_reversal')
                div_type = indicators.get('rsi_divergence', 'NONE')
                cs_signal = indicators.get('candlestick_signal', 'HOLD')
                cs_confidence = indicators.get('candlestick_confidence', 0.0)
                fakeout_detected = indicators.get('fakeout_detected', False)
                fakeout_direction = indicators.get('fakeout_direction')
                fakeout_risk = indicators.get('fakeout_risk', 0.0)
                market_structure = indicators.get('market_structure', 'SIDEWAYS')
                
                # FIBONACCI info for logging
                fib_zones = indicators.get('fib_zones', {})
                nearest_fib = fib_zones.get('nearest_level', 'N/A') if fib_zones else 'N/A'
                fib_position = fib_zones.get('position', 50) if fib_zones else 50

                # EARLY WARNING: Setup forming but not confirmed yet
                if smc_score >= 0.20 and smc_score < 0.60:
                    if rsi <= 45 and rsi_momentum > 0 and smc_signal == 'BUY':
                        logger.info(f"⚠️ EARLY WARNING: {pair} BUY setup forming (score: {smc_score:.2f}, RSI: {rsi:.1f}, Fib: {nearest_fib}%)")
                    elif rsi >= 55 and rsi_momentum < 0 and smc_signal == 'SELL':
                        logger.info(f"⚠️ EARLY WARNING: {pair} SELL setup forming (score: {smc_score:.2f}, RSI: {rsi:.1f}, Fib: {nearest_fib}%)")

                # Determine if this is an early signal (0.60-0.75) or confirmed (0.75+)
                is_early = False
                final_signal = "HOLD"
                final_confidence = 0.0
                prediction_source = "CONFLUENCE"

                if smc_signal in ['BUY', 'SELL'] and smc_score >= 0.60:
                    final_signal = smc_signal
                    final_confidence = smc_score

                    if smc_score >= 0.60 and smc_score < 0.75:
                        is_early = True
                        prediction_source = "CONFLUENCE_EARLY"
                    else:
                        is_early = False
                        prediction_source = "CONFLUENCE_CONFIRMED"

                if final_signal in ['BUY', 'SELL']:
                    filter_passed, filter_report = self.filter_managers[pair].validate_signal(
                        df, final_signal, current_price,
                        indicators=indicators, pair=pair, is_early=is_early
                    )

                    if filter_passed:
                        # Check fakeout blocking
                        if fakeout_detected and fakeout_risk >= 0.40:
                            should_block, block_reason = self.analyzer.fakeout_detector.should_block_signal(
                                {"fakeout_detected": fakeout_detected, "fakeout_direction": fakeout_direction,
                                 "fakeout_risk": fakeout_risk, "trap_type": indicators.get('fakeout_trap_type')},
                                final_signal
                            )
                            if should_block:
                                logger.info(f"FAKEOUT BLOCK: {pair} {final_signal} blocked. {block_reason}")
                                self.health[pair]['status'] = 'healthy'
                                self.health[pair]['last_check'] = asyncio.get_event_loop().time()
                                elapsed = asyncio.get_event_loop().time() - start_time
                                sleep_time = max(5, config['check_interval'] - elapsed)
                                await asyncio.sleep(sleep_time)
                                continue

                        # Build alert
                        confidence_boost = filter_report.get('confidence_boost', 1.0)
                        adjusted_confidence = min(final_confidence * confidence_boost, 0.99)

                        urgency = 'high' if adjusted_confidence > 0.85 else 'normal' if adjusted_confidence > 0.60 else 'low'
                        if is_early:
                            urgency = 'normal'

                        alert = SignalAlert(
                            pair=pair,
                            signal=final_signal,
                            price=current_price,
                            confidence=adjusted_confidence,
                            timestamp=datetime.utcnow(),
                            reasoning=indicators.get('smc_reasoning', f'{prediction_source} Pattern Recognition'),
                            target=indicators.get('suggested_take_profit'),
                            stop_loss=indicators.get('suggested_stop_loss'),
                            urgency=urgency,
                            is_early=is_early,
                            confluence_score=smc_score
                        )
                        await self.notification_manager.send_telegram_alert(alert)

                        signal_data = {
                            'pair': pair,
                            'signal': final_signal,
                            'price': current_price,
                            'confidence': adjusted_confidence,
                            'timestamp': datetime.utcnow().isoformat(),
                            'target': indicators.get('suggested_take_profit'),
                            'stop_loss': indicators.get('suggested_stop_loss'),
                            'reasoning': indicators.get('smc_reasoning', ''),
                            'source': prediction_source,
                            'is_early': is_early,
                            'confluence_score': smc_score,
                            'market_structure': market_structure,
                            'fib_zone': nearest_fib,
                            'fib_position': fib_position,
                            'live_reversal': live_reversal,
                            'rsi_divergence': div_type,
                            'candlestick_signal': cs_signal,
                            'candlestick_confidence': cs_confidence,
                            'fakeout_detected': fakeout_detected,
                            'fakeout_direction': fakeout_direction,
                            'fakeout_risk': fakeout_risk
                        }
                        if redis_client:
                            await redis_client.setex(f"signal:{pair}:current", 3600, json.dumps(signal_data))

                        self.last_sent_signal[pair] = final_signal
                        self.health[pair]['signals_today'] += 1
                        self.health[pair]['last_signal'] = signal_data
                        label = "🚨 EARLY" if is_early else "🚀 CONFIRMED"
                        logger.info(f"{label} SIGNAL SENT: {pair} {final_signal} @ {current_price:.2f} (conf: {adjusted_confidence:.2%}) [Source: {prediction_source}] [Fib: {nearest_fib}%]")
                    else:
                        reason = filter_report.get('filters', {}).get('cooldown_check', {}).get('reason', 'Filter blocked')
                        # Log which filter blocked it
                        for filter_name, filter_result in filter_report.get('filters', {}).items():
                            if not filter_result.get('passed', True):
                                logger.info(f"FILTER BLOCK: {pair} {final_signal} — {filter_name}: {filter_result.get('reason', 'unknown')}")
                else:
                    logger.debug(f"FILTER DEBUG: {pair} no valid signal. SMC: {smc_signal}({smc_score:.2f}) | CS: {cs_signal}({cs_confidence:.2f}) | Live: {live_reversal} | Div: {div_type} | Fib: {nearest_fib}%")

                self.health[pair]['status'] = 'healthy'
                self.health[pair]['last_check'] = asyncio.get_event_loop().time()
                elapsed = asyncio.get_event_loop().time() - start_time
                sleep_time = max(5, config['check_interval'] - elapsed)
                await asyncio.sleep(sleep_time)

            except asyncio.TimeoutError:
                logger.error(f"Timeout fetching {pair}. Retrying in 15s...")
                await asyncio.sleep(15)
            except Exception as e:
                logger.error(f"Error in {pair} loop: {e}")
                await asyncio.sleep(10)

    async def start(self):
        await self.initialize()
        asyncio.create_task(self._keep_alive_ping())
        for pair in self.pairs_config:
            self.tasks[pair] = asyncio.create_task(self.monitor_pair(pair))
            await asyncio.sleep(2)

    async def stop(self):
        app_state['is_running'] = False
        for task in self.tasks.values():
            task.cancel()
        if self.notification_manager:
            await self.notification_manager.close()
        if redis_client:
            await redis_client.close()
        logger.info("Engine v4.1 stopped")


@asynccontextmanager
async def lifespan(app: FastAPI):
    engine = CloudForexEngine()
    app_state['engine'] = engine
    await engine.start()
    yield
    await engine.stop()


app = FastAPI(
    title="AI Crypto Signal Bot",
    description="BTCUSD pattern prediction with Telegram alerts — v4.1 Fibonacci + Trend Filter",
    version="4.1",
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
    return {
        "status": "running" if app_state['is_running'] else "down",
        "pairs": list(app_state.get('engine', {}).pairs_config.keys() if app_state.get('engine') else []),
        "version": "4.1",
        "mode": "5m candles — Pure Price Action + Fibonacci + Trend Filter (No LSTM)"
    }


@app.get("/health")
async def health_check():
    engine = app_state.get('engine')
    if not engine:
        return {"status": "down", "error": "Engine not initialized"}
    return {
        "status": "healthy" if app_state['is_running'] else "down",
        "version": "4.1",
        "pairs": {
            pair: {
                "status": info['status'],
                "signals_today": info['signals_today'],
                "last_signal": info.get('last_signal')
            }
            for pair, info in engine.health.items()
        }
    }


@app.get("/api/signal/{pair}")
async def get_signal(pair: str):
    if not redis_client:
        raise HTTPException(status_code=503, detail="Redis not available")
    data = await redis_client.get(f"signal:{pair.upper()}:current")
    if not data:
        raise HTTPException(status_code=404, detail=f"No active signal for {pair}")
    return json.loads(data)


@app.get("/api/signals")
async def get_all_signals():
    if not redis_client:
        raise HTTPException(status_code=503, detail="Redis not available")
    signals = {}
    for pair in ["BTCUSD"]:
        data = await redis_client.get(f"signal:{pair}:current")
        if data:
            signals[pair] = json.loads(data)
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

