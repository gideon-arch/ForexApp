# main.py - FULL PRODUCTION UPDATE
from dotenv import load_dotenv
load_dotenv()
import os
import sys
import asyncio
import redis.asyncio as redis
import json
import logging
import gc
import pandas as pd
import httpx
import time
from datetime import datetime
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

# Local imports
from core.data_feed import DataFeedManager
from core.technical_analysis import TechnicalAnalyzer
from core.filters import FilterManager
from core.lstm_model import EnsemblePredictor
from core.notifications import NotificationManager, SignalAlert

# =====================================================================
# 🧠 TENSORFLOW ENVIRONMENT GUARD
# =====================================================================
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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

redis_url = os.getenv('REDIS_PUBLIC_URL', os.getenv('REDIS_URL', 'redis://localhost:6379'))
redis_client = redis.Redis.from_url(redis_url, decode_responses=True)

app_state = {'is_running': False, 'engine': None}

# =====================================================================
# 🕯️ GEOMETRIC CANDLESTICK REVERSAL PATTERN SCANNER
# =====================================================================
def check_candlestick_reversal(df_candles: pd.DataFrame, signal_type: str, pair: str, is_scalp_track: bool = False) -> bool:
    try:
        if df_candles is None or len(df_candles) < 4: return True
        df = df_candles.copy()
        df.columns = [col.lower() for col in df.columns]
        c1 = df.iloc[-2]; c2 = df.iloc[-3]
        open_1, high_1, low_1, close_1 = float(c1['open']), float(c1['high']), float(c1['low']), float(c1['close'])
        open_2, high_2, low_2, close_2 = float(c2['open']), float(c2['high']), float(c2['low']), float(c2['close'])
        body_1 = abs(close_1 - open_1); range_1 = high_1 - low_1 if (high_1 - low_1) > 0 else 0.0001
        is_green_1 = close_1 > open_1; is_red_1 = open_1 > close_1; is_red_2 = open_2 > close_2; is_green_2 = close_2 > open_2
        
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
    except: return True

# =====================================================================
# 🛸 CLOUD FOREX ENGINE
# =====================================================================
class CloudForexEngine:
    def __init__(self):
        self.pairs_config = {
            "ETHUSD": {"source": "kraken", "tf": "5m", "check_interval": 60},
            "BTCUSD": {"source": "deriv", "tf": "5m", "check_interval": 60}
        }
        self.tasks = {}
        self.health = {}
        self.last_sent_signal = {pair: None for pair in self.pairs_config}
        self.threshold = float(os.getenv('LSTM_THRESHOLD_PERCENT', 0.1)) / 100
        self.data_feed = None
        self.notification_manager = None
        self.filter_managers = {}
        self.predictors = {}
        self.analyzer = TechnicalAnalyzer()
    
    async def initialize(self):
        logger.info("🚀 Initializing Engine with Source-Specific Routing...")
        try:
            self.data_feed = DataFeedManager()
            self.notification_manager = NotificationManager({
                'gmail_user': os.getenv('FROM_EMAIL'), 'gmail_password': os.getenv('GMAIL_APP_PASSWORD'),
                'twilio_sid': os.getenv('TWILIO_SID'), 'twilio_token': os.getenv('TWILIO_TOKEN'),
                'twilio_from_number': os.getenv('TWILIO_FROM'), 'from_email': os.getenv('FROM_EMAIL')
            })
            for pair in self.pairs_config:
                self.filter_managers[pair] = FilterManager()
                self.predictors[pair] = EnsemblePredictor()
                # Loading model with explicit error reporting
                try:
                    self.predictors[pair].lstm.load_model(pair)
                    logger.info(f"✅ Loaded model for {pair}")
                except Exception as e:
                    logger.error(f"❌ Critical: Could not load model for {pair}: {e}")
                self.health[pair] = {'status': 'initializing', 'last_check': None, 'signals_today': 0}
            app_state['is_running'] = True
        except Exception as e:
            logger.error(f"❌ Failed to initialize engine: {e}")
            raise e
    
    async def monitor_pair(self, pair: str):
        config = self.pairs_config[pair]
        while app_state['is_running']:
            try:
                # Adding timeout to network requests to prevent hanging
                df = await asyncio.wait_for(
                    self.data_feed.fetch_latest(pair, timeframe=config['tf'], source=config['source']), 
                    timeout=15.0
                )
                if df is None or df.empty:
                    logger.warning(f"⚠️ No data received for {pair}")
                    await asyncio.sleep(10); continue
                
                indicators = self.analyzer.calculate_indicators(df)
                prediction = self.predictors[pair].predict(df, self.analyzer.generate_technical_score(indicators), 0.0, pair)
                
                self.health[pair]['status'] = 'healthy'
                await asyncio.sleep(5) 
            except asyncio.TimeoutError:
                logger.error(f"❌ Connection Timeout for {pair}. Retrying...")
                await asyncio.sleep(10)
            except Exception as e:
                logger.error(f"❌ Error in {pair} loop: {e}")
                await asyncio.sleep(10)
    
    async def start(self):
        await self.initialize()
        for pair in self.pairs_config:
            self.tasks[pair] = asyncio.create_task(self.monitor_pair(pair))
    
    async def stop(self):
        app_state['is_running'] = False
        for task in self.tasks.values(): task.cancel()
        await redis_client.close()

@asynccontextmanager
async def lifespan(app: FastAPI):
    engine = CloudForexEngine()
    app_state['engine'] = engine
    await engine.start()
    yield
    await engine.stop()

app = FastAPI(title="AI Crypto Pro", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/health")
async def health_check():
    return {"status": "healthy" if app_state['is_running'] else "down"}

@app.get("/api/signal/{pair}")
async def get_signal(pair: str):
    data = await redis_client.get(f"signal:{pair.upper()}:current")
    if not data: raise HTTPException(status_code=404, detail="No signal")
    return json.loads(data)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=False)