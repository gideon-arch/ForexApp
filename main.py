# main.py - AI Crypto Signal Bot (Production Ready)
from dotenv import load_dotenv
load_dotenv()

import os
import sys
import asyncio
import redis.asyncio as redis
import json
import logging
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
        c2 = df.iloc[-3]
        open_1 = float(c1['open'])
        high_1 = float(c1['high'])
        low_1 = float(c1['low'])
        close_1 = float(c1['close'])
        open_2 = float(c2['open'])
        high_2 = float(c2['high'])
        low_2 = float(c2['low'])
        close_2 = float(c2['close'])
        body_1 = abs(close_1 - open_1)
        range_1 = high_1 - low_1 if (high_1 - low_1) > 0 else 0.0001
        is_green_1 = close_1 > open_1
        is_red_1 = open_1 > close_1
        is_red_2 = open_2 > close_2
        is_green_2 = close_2 > open_2
        
        if signal_type == "BUY":
            if is_red_1: return False
            if is_green_1 and is_red_2 and (abs(close_2 - open_2) > (body_1 * 2)): return False
            lower_wick_1 = min(open_1, close_1) - low_1
            if lower_wick_1 >= (2 * body_1): return True
            if is_green_1 and is_red_2 and close_1 >= open_2 and open_1 <= close_2: return True
        elif signal_type == "SELL":
            if is_green_1: return False
            upper_wick_1 = high_1 - max(open_1, close_1)
            if upper_wick_1 >= (2 * body_1): return True
            if is_red_1 and is_green_2 and close_1 <= open_2 and open_1 >= close_2: return True
        return True
    except Exception as e:
        logger.warning(f"Candlestick check error: {e}")
        return True

class CloudForexEngine:
    def __init__(self):
        # Original config: Deriv for BTCUSD, Kraken for ETHUSD
        self.pairs_config = {
            "BTCUSD": {"source": "deriv", "tf": "5m", "check_interval": 60},
            "ETHUSD": {"source": "kraken", "tf": "5m", "check_interval": 60}
        }
        self.tasks = {}
        self.health = {}
        self.last_sent_signal = {pair: None for pair in self.pairs_config}
        self.threshold = float(os.getenv('LSTM_THRESHOLD_PERCENT', '0.1')) / 100
        self.data_feed = None
        self.notification_manager = None
        self.filter_managers = {}
        self.predictors = {}
        self.analyzer = TechnicalAnalyzer()
        logger.info("CloudForexEngine initialized | Deriv for BTCUSD, Kraken for ETHUSD")

    async def initialize(self):
        logger.info("Initializing Engine...")
        try:
            self.data_feed = DataFeedManager()
            self.notification_manager = NotificationManager()
            for pair in self.pairs_config:
                self.filter_managers[pair] = FilterManager()
                self.predictors[pair] = EnsemblePredictor()
                model_loaded = self.predictors[pair].lstm.load_model(pair)
                if not model_loaded:
                    logger.warning(f"No model found for {pair}. Will train on live data...")
                    await self._auto_train_model(pair)
                else:
                    logger.info(f"Model loaded for {pair}")
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
                
                try:
                    prediction = self.predictors[pair].predict(
                        df, tech_score=tech_score, fund_score=0.0, pair=pair
                    )
                except Exception as e:
                    logger.warning(f"LSTM prediction failed for {pair}: {e}")
                    prediction = {"signal": "HOLD", "confidence": 0.0}
                
                lstm_signal = prediction['signal']
                lstm_confidence = prediction['confidence']
                
                if lstm_signal in ['BUY', 'SELL'] and lstm_confidence >= 0.6:
                    filter_passed, filter_report = self.filter_managers[pair].validate_signal(
                        df, lstm_signal, current_price
                    )
                    if filter_passed:
                        candle_ok = check_candlestick_reversal(df, lstm_signal, pair)
                        if candle_ok:
                            alert = SignalAlert(
                                pair=pair,
                                signal=lstm_signal,
                                price=current_price,
                                confidence=lstm_confidence,
                                timestamp=datetime.utcnow(),
                                reasoning=indicators.get('smc_reasoning', 'AI Pattern Recognition'),
                                target=indicators.get('suggested_take_profit'),
                                stop_loss=indicators.get('suggested_stop_loss'),
                                urgency='high' if lstm_confidence > 0.85 else 'normal'
                            )
                            await self.notification_manager.send_telegram_alert(alert)
                            
                            signal_data = {
                                'pair': pair,
                                'signal': lstm_signal,
                                'price': current_price,
                                'confidence': lstm_confidence,
                                'timestamp': datetime.utcnow().isoformat(),
                                'target': indicators.get('suggested_take_profit'),
                                'stop_loss': indicators.get('suggested_stop_loss'),
                                'reasoning': indicators.get('smc_reasoning', '')
                            }
                            if redis_client:
                                await redis_client.setex(f"signal:{pair}:current", 3600, json.dumps(signal_data))
                            
                            self.last_sent_signal[pair] = lstm_signal
                            self.health[pair]['signals_today'] += 1
                            self.health[pair]['last_signal'] = signal_data
                            logger.info(f"SIGNAL SENT: {pair} {lstm_signal} @ {current_price:.2f} (conf: {lstm_confidence:.2%})")
                        else:
                            logger.info(f"Candlestick rejected {lstm_signal} for {pair}")
                    else:
                        reason = filter_report.get('filters', {}).get('cooldown_check', {}).get('reason', 'Filter blocked')
                        logger.info(f"Filter blocked {lstm_signal} for {pair}: {reason}")
                else:
                    if self.health[pair].get('last_check') is None or \
                       (asyncio.get_event_loop().time() - self.health[pair]['last_check']) > 300:
                        logger.info(f"{pair}: HOLD (LSTM conf: {lstm_confidence:.2%}, Tech score: {tech_score:.3f})")
                
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
    version="2.0.0",
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
        "version": "2.0.0"
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

@app.get("/test-telegram")
async def test_telegram():
    engine = app_state.get('engine')
    if not engine or not engine.notification_manager:
        raise HTTPException(status_code=500, detail="Engine or NotificationManager not initialized")
    
    try:
        from core.notifications import SignalAlert
        from datetime import datetime
        
        test_alert = SignalAlert(
            pair="TEST",
            signal="TEST_SIGNAL",
            price=0.0,
            confidence=1.0,
            timestamp=datetime.utcnow(),
            reasoning="Connection test from FastAPI /test-telegram endpoint."
        )
        await engine.notification_manager.send_telegram_alert(test_alert)
        return {"status": "success", "message": "Test message sent to Telegram"}
    except Exception as e:
        logger.error(f"Test message failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

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