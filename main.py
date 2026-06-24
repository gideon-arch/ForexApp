# main.py - AI Crypto Signal Bot (Production Ready)
# FINAL VERSION — With keep-alive ping + startup Telegram message

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
from core.lstm_model import EnsemblePredictor
from core.notifications import NotificationManager, SignalAlert

# TensorFlow Environment Guard
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

try:
    import tensorflow as tf
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
except Exception:
    pass

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

def check_candlestick_reversal(df_candles, signal_type: str, pair: str) -> bool:
    try:
        if df_candles is None or len(df_candles) < 4:
            return True
        
        df = df_candles.copy()
        df.columns = [col.lower() for col in df.columns]
        c1 = df.iloc[-2]
        open_1 = float(c1['open'])
        high_1 = float(c1['high'])
        low_1 = float(c1['low'])
        close_1 = float(c1['close'])
        body_1 = abs(close_1 - open_1)
        total_range = high_1 - low_1 if (high_1 - low_1) > 0 else 0.0001
        is_green_1 = close_1 > open_1
        is_red_1 = open_1 > close_1
        
        if signal_type == "BUY":
            if is_green_1: 
                return True
            if body_1 < total_range * 0.3:
                return True
            return False
            
        elif signal_type == "SELL":
            if is_red_1:
                return True
            if body_1 < total_range * 0.3:
                return True
            return False
            
        return True
    except Exception as e:
        logger.warning(f"Candlestick check error: {e}")
        return True

class CloudForexEngine:
    def __init__(self):
        self.pairs_config = {
            "BTCUSD": {"source": "deriv", "tf": "5m", "check_interval": 60},
            "ETHUSD": {"source": "kraken", "tf": "5m", "check_interval": 60}
        }
        self.tasks = {}
        self.health = {}
        self.last_sent_signal = {pair: None for pair in self.pairs_config}
        self.threshold = float(os.getenv('LSTM_THRESHOLD_PERCENT', '0.05')) / 100
        self.data_feed = None
        self.notification_manager = None
        self.filter_managers = {}
        self.predictors = {}
        self.analyzer = TechnicalAnalyzer()
        self.lstm_available = True
        logger.info("CloudForexEngine initialized | Deriv for BTCUSD, Kraken for ETHUSD")

    async def initialize(self):
        logger.info("Initializing Engine...")
        try:
            self.data_feed = DataFeedManager()
            self.notification_manager = NotificationManager()
            
            # Send startup test message to Telegram
            try:
                await self.notification_manager.send_test_message(
                    "🤖 AI Crypto Bot is ONLINE!\n"
                    "━━━━━━━━━━━━━━━━━━━\n"
                    "✅ Monitoring: BTCUSD & ETHUSD\n"
                    "⏱️ Timeframe: 5m\n"
                    "🎯 Signals: Predictive (before the move)\n"
                    "⚙️ Version: 2.1.0"
                )
                logger.info("Startup test message sent to Telegram")
            except Exception as e:
                logger.warning(f"Failed to send startup message: {e}")
            
            for pair in self.pairs_config:
                self.filter_managers[pair] = FilterManager()
                
                try:
                    self.predictors[pair] = EnsemblePredictor()
                    model_loaded = self.predictors[pair].lstm.load_model(pair)
                    if not model_loaded:
                        logger.warning(f"No model found for {pair}. LSTM will use TA fallback.")
                    else:
                        logger.info(f"Model loaded for {pair}")
                except Exception as e:
                    logger.warning(f"LSTM init failed for {pair}: {e}. Using TA-only mode.")
                    self.lstm_available = False
                    self.predictors[pair] = None
                
                self.health[pair] = {
                    'status': 'initializing',
                    'last_check': None,
                    'signals_today': 0,
                    'last_signal': None
                }
            app_state['is_running'] = True
            logger.info("Engine initialization complete!")
        except Exception as e:
            logger.error(f"Engine initialization failed: {e}")
            raise

    async def _auto_train_model(self, pair: str):
        try:
            logger.info(f"Auto-training {pair} model...")
            config = self.pairs_config[pair]
            df = await self.data_feed.fetch_historical(pair, timeframe=config['tf'], source=config['source'], limit=1000)
            if df is None or len(df) < 300:
                logger.error(f"Not enough data to train {pair}")
                return False
            result = self.predictors[pair].lstm.train(df, pair, epochs=20)
            logger.info(f"{pair} model trained! Accuracy: {result.get('accuracy', 'N/A')}")
            return True
        except Exception as e:
            logger.error(f"Auto-training failed for {pair}: {e}")
            return False

    async def _keep_alive_ping(self):
        """Ping our own health endpoint to keep Render free tier alive"""
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
        logger.info(f"Starting monitor for {pair} ({config['source']})")
        while app_state['is_running']:
            try:
                start_time = asyncio.get_event_loop().time()
                df = await asyncio.wait_for(
                    self.data_feed.fetch_latest(pair, timeframe=config['tf'], source=config['source']),
                    timeout=20.0
                )
                if df is None or df.empty:
                    logger.warning(f"No data for {pair}")
                    await asyncio.sleep(15)
                    continue
                if len(df) < 200:
                    logger.warning(f"Insufficient data for {pair}: {len(df)} candles")
                    await asyncio.sleep(15)
                    continue
                
                try:
                    indicators = self.analyzer.calculate_indicators(df)
                except Exception as e:
                    logger.warning(f"Technical analysis failed for {pair}: {e}")
                    await asyncio.sleep(10)
                    continue
                
                current_price = indicators['current_price']
                tech_score = self.analyzer.generate_technical_score(indicators)
                smc_signal = indicators.get('smc_suggested_signal', 'HOLD')
                smc_score = indicators.get('confluence_score', 0.0)
                
                lstm_signal = "HOLD"
                lstm_confidence = 0.0
                prediction_source = "TA_ONLY"
                
                if self.lstm_available and self.predictors[pair] is not None:
                    try:
                        prediction = self.predictors[pair].predict(
                            df, tech_score=tech_score, fund_score=0.0, pair=pair
                        )
                        lstm_signal = prediction['signal']
                        lstm_confidence = prediction['confidence']
                        prediction_source = "LSTM"
                    except Exception as e:
                        logger.warning(f"LSTM prediction failed for {pair}: {e}. Using TA fallback.")
                
                final_signal = "HOLD"
                final_confidence = 0.0
                
                if lstm_signal in ['BUY', 'SELL'] and lstm_confidence >= 0.4:
                    final_signal = lstm_signal
                    final_confidence = lstm_confidence
                elif smc_signal in ['BUY', 'SELL'] and smc_score >= 0.40:
                    final_signal = smc_signal
                    final_confidence = min(smc_score + 0.2, 0.85)
                    prediction_source = "SMC_FALLBACK"
                
                if final_signal in ['BUY', 'SELL']:
                    filter_passed, filter_report = self.filter_managers[pair].validate_signal(
                        df, final_signal, current_price, pair=pair
                    )
                    if filter_passed:
                        candle_ok = check_candlestick_reversal(df, final_signal, pair)
                        if candle_ok:
                            alert = SignalAlert(
                                pair=pair,
                                signal=final_signal,
                                price=current_price,
                                confidence=final_confidence,
                                timestamp=datetime.utcnow(),
                                reasoning=indicators.get('smc_reasoning', f'{prediction_source} Pattern Recognition'),
                                target=indicators.get('suggested_take_profit'),
                                stop_loss=indicators.get('suggested_stop_loss'),
                                urgency='high' if final_confidence > 0.85 else 'normal'
                            )
                            await self.notification_manager.send_telegram_alert(alert)
                            
                            signal_data = {
                                'pair': pair,
                                'signal': final_signal,
                                'price': current_price,
                                'confidence': final_confidence,
                                'timestamp': datetime.utcnow().isoformat(),
                                'target': indicators.get('suggested_take_profit'),
                                'stop_loss': indicators.get('suggested_stop_loss'),
                                'reasoning': indicators.get('smc_reasoning', ''),
                                'source': prediction_source
                            }
                            if redis_client:
                                await redis_client.setex(f"signal:{pair}:current", 3600, json.dumps(signal_data))
                            
                            self.last_sent_signal[pair] = final_signal
                            self.health[pair]['signals_today'] += 1
                            self.health[pair]['last_signal'] = signal_data
                            logger.info(f"SIGNAL SENT: {pair} {final_signal} @ {current_price:.2f} (conf: {final_confidence:.2%}) [Source: {prediction_source}]")
                        else:
                            logger.info(f"FILTER DEBUG: {pair} {final_signal} rejected by Candle Reversal Check.")
                    else:
                        reason = filter_report.get('filters', {}).get('cooldown_check', {}).get('reason', 'Filter blocked')
                        logger.info(f"FILTER DEBUG: {pair} {final_signal} rejected. Reason: {reason}")
                else:
                    logger.info(f"FILTER DEBUG: {pair} no valid signal. LSTM: {lstm_signal}({lstm_confidence:.2f}) | SMC: {smc_signal}({smc_score:.2f})")
                
                self.health[pair]['status'] = 'healthy'
                self.health[pair]['last_check'] = asyncio.get_event_loop().time()
                elapsed = asyncio.get_event_loop().time() - start_time
                sleep_time = max(5, config['check_interval'] - elapsed)
                await asyncio.sleep(sleep_time)
                
            except asyncio.TimeoutError:
                logger.error(f"Timeout fetching {pair}. Retrying in 20s...")
                await asyncio.sleep(20)
            except Exception as e:
                logger.error(f"Error in {pair} loop: {e}")
                await asyncio.sleep(15)

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
        if redis_client:
            await redis_client.close()
        logger.info("Engine stopped")

@asynccontextmanager
async def lifespan(app: FastAPI):
    engine = CloudForexEngine()
    app_state['engine'] = engine
    await engine.start()
    yield
    await engine.stop()

app = FastAPI(
    title="AI Crypto Signal Bot",
    description="BTCUSD & ETHUSD pattern prediction with Telegram alerts",
    version="2.1.0",
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
        "version": "2.1.0"
    }

@app.get("/health")
async def health_check():
    engine = app_state.get('engine')
    if not engine:
        return {"status": "down", "error": "Engine not initialized"}
    return {
        "status": "healthy" if app_state['is_running'] else "down",
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
    for pair in ["BTCUSD", "ETHUSD"]:
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