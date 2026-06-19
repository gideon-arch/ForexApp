# main.py
# Main FastAPI - FAST SCALPING 5M/1M PRODUCTION TIMEFRAMES

import os
import sys

# =====================================================================
# 🧠 TENSORFLOW ENVIRONMENT GUARD (Prevents Render 512MB RAM Crashes)
# =====================================================================
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'      # Force TensorFlow to run strictly on CPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'        # Suppress heavy warning allocation strings
os.environ['OMP_NUM_THREADS'] = '1'             # Restrict OpenMP layout to 1 core thread
os.environ['MKL_NUM_THREADS'] = '1'             # Restrict MKL engine execution layout

try:
    import tensorflow as tf
    # Limit runtime CPU threads to protect platform system memory structures
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    
    # Enable dynamic memory growth tracking to keep memory allocations completely tiny
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
except Exception:
    pass

# Standard system modules can now be imported safely
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import asyncio
import redis.asyncio as redis
import json
from datetime import datetime
import logging
import gc  # Garbage Collector interface to prevent memory build-up
import pandas as pd
import httpx  # Added for lightweight internal self-pings

from core.data_feed import DataFeedManager
from core.technical_analysis import TechnicalAnalyzer
from core.filters import FilterManager
from core.lstm_model import EnsemblePredictor
from core.notifications import NotificationManager, SignalAlert

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

redis_url = os.getenv('REDIS_PUBLIC_URL', os.getenv('REDIS_URL', 'redis://localhost:6379'))
logger.info(f"🔗 Attempting Redis initialization with target string: {redis_url.split('@')[-1] if '@' in redis_url else redis_url}")
redis_client = redis.Redis.from_url(redis_url, decode_responses=True)

# 🔄 TRANSITIONED ENTIRELY TO HIGH-LIQUIDITY CRYPTO
app_state = {'is_running': False, 'pairs': ['ETHUSD', 'BTCUSD'], 'engine': None}


# =====================================================================
# 🕯️ GEOMETRIC CANDLESTICK REVERSAL PATTERN SCANNER
# =====================================================================
def check_candlestick_reversal(df_candles: pd.DataFrame, signal_type: str, pair: str, is_scalp_track: bool = False) -> bool:
    try:
        if df_candles is None or len(df_candles) < 4:
            return True  # Fallback
            
        df = df_candles.copy()
        df.columns = [col.lower() for col in df.columns]
        
        c1 = df.iloc[-2]  # Most recently finished closed candle
        c2 = df.iloc[-3]  # Previous closed candle
        
        open_1, high_1, low_1, close_1 = float(c1['open']), float(c1['high']), float(c1['low']), float(c1['close'])
        open_2, high_2, low_2, close_2 = float(c2['open']), float(c2['high']), float(c2['low']), float(c2['close'])
        
        body_1 = abs(close_1 - open_1)
        range_1 = high_1 - low_1 if (high_1 - low_1) > 0 else 0.0001
        
        is_green_1 = close_1 > open_1
        is_red_1 = open_1 > close_1
        is_red_2 = open_2 > close_2
        is_green_2 = close_2 > open_2
        
        if signal_type == "BUY":
            if is_red_1:
                logger.info(f"🛑 [CANDLE GUARD] ({pair}) Blocked BUY: Market has begun dropping. Last candle closed RED.")
                return False
                
            if is_green_1 and is_red_2:
                body_2 = abs(close_2 - open_2)
                if body_2 > (body_1 * 2):
                    logger.info(f"🛑 [CANDLE GUARD] ({pair}) Blocked BUY: Weak bounce against heavy selling pressure.")
                    return False

            lower_wick_1 = min(open_1, close_1) - low_1
            upper_wick_1 = high_1 - max(open_1, close_1)
            
            if lower_wick_1 >= (2 * body_1) and upper_wick_1 <= (0.2 * body_1):
                logger.info(f"🔍 Candle Guard ({pair}): BULLISH HAMMER detected! Reversal verified.")
                return True
                
            if is_green_1 and is_red_2:
                if close_1 >= open_2 and open_1 <= close_2:
                    logger.info(f"🔍 Candle Guard ({pair}): BULLISH ENGULFING detected! Reversal verified.")
                    return True
                    
            if is_green_1 and body_1 > (range_1 * 0.25):
                return True
                
            if is_scalp_track:
                logger.info(f"⚠️ Candle Guard ({pair}): Scalper logic passed base structural requirement.")
                return True
                
            logger.info(f"⚠️ Candle Guard ({pair}): AI wants to BUY, but market candles are flat. BLOCKED.")
            return False

        elif signal_type == "SELL":
            if is_green_1:
                logger.info(f"🛑 [CANDLE GUARD] ({pair}) Blocked SELL: Market is spiking up. Last candle closed GREEN.")
                return False

            upper_wick_1 = high_1 - max(open_1, close_1)
            lower_wick_1 = min(open_1, close_1) - low_1
            
            if upper_wick_1 >= (2 * body_1) and lower_wick_1 <= (0.2 * body_1):
                logger.info(f"🔍 Candle Guard ({pair}): SHOOTING STAR detected! Reversal verified.")
                return True
                
            if is_red_1 and is_green_2:
                if close_1 <= open_2 and open_1 >= close_2:
                    logger.info(f"🔍 Candle Guard ({pair}): BEARISH ENGULFING detected! Reversal verified.")
                    return True
                    
            if is_red_1 and body_1 > (range_1 * 0.25):
                return True
                
            if is_scalp_track:
                logger.info(f"⚠️ Candle Guard ({pair}): Scalper logic passed base structural requirement.")
                return True
                
            logger.info(f"⚠️ Candle Guard ({pair}): AI wants to SELL, but market candles are flat. BLOCKED.")
            return False
            
        return True
    except Exception as e:
        logger.error(f"⚠️ Error in Candlestick Reversal Layer: {e}")
        return True  # Fallback


class CloudForexEngine:
    def __init__(self):
        # FIXED: Cleans up formatting characters like slashes, dashes, or spaces automatically to protect pair tracking paths
        pairs_env = os.getenv('TRACKED_PAIRS', 'ETHUSD,BTCUSD')
        self.active_pairs = [p.strip().upper().replace("/", "").replace("-", "") for p in pairs_env.split(',') if p.strip()]
        
        self.pairs = {pair: {'check_interval': 60, 'tf': '5m'} for pair in self.active_pairs}
        self.tasks = {}
        self.health = {}
        
        self.last_sent_signal = {pair: None for pair in self.active_pairs}
        self.threshold = float(os.getenv('LSTM_THRESHOLD_PERCENT', 0.1)) / 100
        self.data_feed = None
        self.notification_manager = None
        self.filter_managers = {}
        self.predictors = {}
        self.analyzer = TechnicalAnalyzer()
    
    async def initialize(self):
        logger.info("🚀 Initializing Cloud Forex Engine with Memory Safeguards...")
        self.data_feed = DataFeedManager()
        
        self.notification_manager = NotificationManager({
            'gmail_user': os.getenv('FROM_EMAIL'),
            'gmail_password': os.getenv('GMAIL_APP_PASSWORD'),
            'twilio_sid': os.getenv('TWILIO_SID'),
            'twilio_token': os.getenv('TWILIO_TOKEN'),
            'twilio_from_number': os.getenv('TWILIO_FROM'),
            'from_email': os.getenv('FROM_EMAIL')
        })
        
        for pair in self.pairs:
            self.filter_managers[pair] = FilterManager()  
            self.predictors[pair] = EnsemblePredictor()
            
            predictor = self.predictors[pair]
            # FIXED: Blocked live network fallback training loops on weak cloud boxes
            if not predictor.lstm.load_model(pair):
                logger.error(f"❌ Missing pre-trained model file for {pair} inside storage context!")
                logger.warning("⚠️ CRITICAL SAFEGUARD: Server training blocked to prevent Out-Of-Memory crashes.")
                logger.info("💡 Solution: Train your model locally, save the .keras file, and push it via Git.")
                
            self.health[pair] = {
                'status': 'initializing',
                'last_check': None,
                'signals_today': 0
            }
        
        app_state['is_running'] = True
        logger.info("✅ Engine initialized - RUNNING ACTIVE EVALUATIONS")
    
    async def monitor_pair(self, pair: str):
        config = self.pairs[pair]
        base_sleep = config.get('check_interval', 60)
        tf_string = config.get('tf', '5m')
        
        while app_state['is_running']:
            try:
                start = datetime.now()
                logger.info(f"📊 [HEARTBEAT] Scanning fast {tf_string} charts for {pair}...")
                
                try:
                    df = await self.data_feed.fetch_latest(pair, tf_string)
                except TypeError:
                    df = await self.data_feed.fetch_latest(pair=pair, timeframe=tf_string)
                
                if df is None or df.empty:
                    logger.warning(f"⚠️ Empty dataset returned for {pair}. Retrying in next loop cycle.")
                    await asyncio.sleep(base_sleep)
                    continue
                
                df.pair_name = pair
                
                indicators = self.analyzer.calculate_indicators(df)
                tech_score = self.analyzer.generate_technical_score(indicators)
                
                fund_score = 0.0
                prediction = self.predictors[pair].predict(df, tech_score, fund_score, pair)
                current_price = indicators['current_price']
                current_rsi = float(indicators['rsi'])
                
                filter_mgr = self.filter_managers[pair]
                can_trade, filter_results = filter_mgr.validate_signal(df, prediction['signal'], current_price)
                
                if can_trade and prediction['signal'] != 'HOLD':
                    candles_confirmed = check_candlestick_reversal(df, prediction['signal'], pair, is_scalp_track=False)
                    if not candles_confirmed:
                        can_trade = False
                
                await self._cache_signal(pair, {
                    'pair': pair,
                    'price': current_price,
                    'signal': prediction['signal'],
                    'confidence': prediction['confidence'],
                    'threshold': self.threshold,
                    'filters': filter_results,
                    'indicators': {
                        'rsi': current_rsi,
                        'macd': indicators['macd'],
                        'adx': indicators['adx'],
                        'atr': indicators['atr']
                    },
                    'timestamp': datetime.now().isoformat()
                })
                
                triggered_signal = None
                summary_reasoning = ""
                signal_urgency = 'normal'
                
                atr_pad = indicators.get('atr', current_price * 0.001) * 1.5
                raw_tp = indicators.get('suggested_take_profit') or prediction.get('lstm_details', {}).get('predicted_target')
                raw_sl = indicators.get('suggested_stop_loss') or prediction.get('lstm_details', {}).get('suggested_stop_loss')
                
                if can_trade and prediction['signal'] != 'HOLD':
                    triggered_signal = prediction['signal']
                    reasons = [v['reason'] for k, v in filter_results.get('filters', {}).items() if v.get('passed')]
                    summary_reasoning = f"🚨🚨 [SMC SCALPER ALERT] High Confluence Target Match. Status: {', '.join(reasons)}. Structure: {indicators.get('market_structure')}. RSI: {current_rsi:.2f}. Setup confirmed."
                    signal_urgency = 'high' if prediction['confidence'] > 0.85 else 'normal'
                
                elif prediction['signal'] == 'HOLD' or not can_trade:
                    # Sync targets: 65 and 70 thresholds
                    if (63.5 <= current_rsi <= 66.5) or (68.5 <= current_rsi <= 72.0):
                        candles_confirmed = check_candlestick_reversal(df, "SELL", pair, is_scalp_track=True)
                        if candles_confirmed:
                            triggered_signal = "SELL"
                            summary_reasoning = f"📊 [FAST SCALPER TRACK] {pair} running into an overbought extremity boundary at RSI {current_rsi:.2f}. Local reversal candles verified."
                            signal_urgency = 'normal'
                            
                    # Sync targets: 25 and 30 thresholds
                    elif (28.5 <= current_rsi <= 31.5) or (23.5 <= current_rsi <= 26.5):
                        candles_confirmed = check_candlestick_reversal(df, "BUY", pair, is_scalp_track=True)
                        if candles_confirmed:
                            triggered_signal = "BUY"
                            summary_reasoning = f"📊 [FAST SCALPER TRACK] {pair} flattening into an oversold extremity boundary at RSI {current_rsi:.2f}. Local stabilization candles verified."
                            signal_urgency = 'normal'
                
                if triggered_signal is not None:
                    if self.last_sent_signal.get(pair) != triggered_signal:
                        
                        if triggered_signal == "SELL":
                            if raw_tp and raw_tp > current_price:
                                final_tp = current_price - abs(current_price - raw_sl) if raw_sl else current_price - atr_pad
                                final_sl = current_price + abs(current_price - raw_tp)
                            else:
                                final_tp = raw_tp if raw_tp else current_price - atr_pad
                                final_sl = raw_sl if raw_sl else current_price + atr_pad
                                
                            if final_tp >= current_price: final_tp = current_price - atr_pad
                            if final_sl <= current_price: final_sl = current_price + atr_pad
                        else:
                            if raw_tp and raw_tp < current_price:
                                final_tp = current_price + abs(current_price - raw_sl) if raw_sl else current_price + atr_pad
                                final_sl = current_price - abs(current_price - raw_tp)
                            else:
                                final_tp = raw_tp if raw_tp else current_price + atr_pad
                                final_sl = raw_sl if raw_sl else current_price - atr_pad
                                
                            if final_tp <= current_price: final_tp = current_price + atr_pad
                            if final_sl >= current_price: final_sl = current_price - atr_pad
                        
                        alert = SignalAlert(
                            pair=pair,
                            signal=triggered_signal,
                            price=current_price,
                            confidence=prediction['confidence'] if triggered_signal == prediction['signal'] else 0.70,
                            timestamp=datetime.now(),
                            reasoning=summary_reasoning,
                            target=round(float(final_tp), 2),  
                            stop_loss=round(float(final_sl), 2),
                            urgency=signal_urgency
                        )
                        
                        prefs = await self._get_user_preferences(pair)
                        await self.notification_manager.send_all(alert, prefs)
                        
                        self.last_sent_signal[pair] = triggered_signal
                        self.health[pair]['signals_today'] += 1
                        logger.info(f"📢 [{signal_urgency.upper()} SIGNAL DISPATCHED]: {pair} {triggered_signal} @ {current_price} | RSI: {current_rsi:.2f}")
                    else:
                        logger.info(f"😴 [SPAM GUARD ACTIVE] {pair} execution state still holds {triggered_signal} at RSI {current_rsi:.2f}. Duplicate message suppressed.")
                else:
                    if self.last_sent_signal.get(pair) is not None:
                        logger.info(f"🔄 [SPAM GUARD RESET] {pair} market has returned to neutral zone. Rearming triggers.")
                    self.last_sent_signal[pair] = None
                
                self.health[pair]['last_check'] = datetime.now().isoformat()
                self.health[pair]['status'] = 'healthy'
                
                del df
                gc.collect()
                
                elapsed = (datetime.now() - start).total_seconds()
                sleep_time = max(1, base_sleep - elapsed)
                await asyncio.sleep(sleep_time)
                
            except asyncio.CancelledError:
                logger.info(f"🛑 Monitoring task for {pair} was explicitly cancelled.")
                break
            except Exception as e:
                logger.error(f"❌ CRITICAL ERROR monitoring {pair}: {e}", exc_info=True)
                if pair in self.health:
                    self.health[pair]['status'] = 'error'
                await asyncio.sleep(10)
    
    def _sanitize_data(self, data):
        if isinstance(data, dict):
            return {k: self._sanitize_data(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._sanitize_data(v) for v in data]
        elif hasattr(data, 'item'):  
            return data.item()
        elif isinstance(data, (bool, int, float, str)) or data is None:
            return data
        else:
            return str(data)

    async def _cache_signal(self, pair: str, data: dict):
        try:
            clean_data = self._sanitize_data(data)
            key = f"signal:{pair}:current"
            await redis_client.setex(key, 300, json.dumps(clean_data))
            
            history_key = f"signal:{pair}:history"
            await redis_client.lpush(history_key, json.dumps(clean_data))
            await redis_client.ltrim(history_key, 0, 999)
        except Exception as e:
            logger.error(f"⚠️ Redis Caching Failure: {e}")
    
    async def _get_user_preferences(self, pair: str) -> dict:
        return {
            'emails': [os.getenv('ALERT_EMAIL')] if os.getenv('ALERT_EMAIL') else [],
            'phone_numbers': [os.getenv('ALERT_PHONE')] if os.getenv('ALERT_PHONE') else [],
            'min_confidence': 0.5  
        }
    
    async def start(self):
        await self.initialize()
        for pair in self.pairs:
            self.tasks[pair] = asyncio.create_task(self.monitor_pair(pair))
        asyncio.create_task(self._health_reporter())
    
    async def stop(self):
        logger.info("🛑 Stopping engine and canceling active analysis pipelines...")
        app_state['is_running'] = False
        for pair, task in self.tasks.items():
            logger.info(f"Canceling tracking thread for {pair}")
            task.cancel()
        
        if self.tasks:
            await asyncio.gather(*self.tasks.values(), return_exceptions=True)
        await redis_client.close()
        logger.info("👋 Redis connection closed cleanly.")
    
    async def _health_reporter(self):
        while app_state['is_running']:
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                break


# =====================================================================
# 🛸 INTERNAL ANTI-SLEEP SELF-KEEP-ALIVE ENGINE
# =====================================================================
async def keep_alive_loop():
    await asyncio.sleep(30)
    # FIXED: Replaced hardcoded URL with dynamic lookup to prevent cross-domain request loops
    render_service_name = os.getenv("RENDER_SERVICE_NAME", "forexapp-j9w1")
    app_url = f"https://{render_service_name}.onrender.com/health"
    logger.info(f"🛰️ [KEEP-ALIVE] Internal keep-alive loop initialized targeting: {app_url}")
    
    async with httpx.AsyncClient() as client:
        while app_state.get('is_running', True):
            try:
                headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AI-MatrixCore/3.1.1"}
                response = await client.get(app_url, headers=headers, timeout=15.0)
                logger.info(f"🤖 [SELF-KEEP-ALIVE] Ping status confirmed: {response.status_code} - Container Active.")
            except Exception as e:
                logger.warning(f"⚠️ [SELF-KEEP-ALIVE] Internal loop checkpoint failure: {e}")
            
            await asyncio.sleep(600)


@asynccontextmanager
async def lifespan(app: FastAPI):
    engine = CloudForexEngine()
    app_state['engine'] = engine
    await engine.start()
    
    keep_alive_task = asyncio.create_task(keep_alive_loop())
    
    yield
    
    keep_alive_task.cancel()
    await engine.stop()


app = FastAPI(
    title="AI Crypto Pro",
    description="24/7 Trading Signals | Kraken Streams | Dynamic Parameters",
    version="3.1.1",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


@app.get("/health")
async def health_check():
    engine = app_state.get('engine')
    thresh = engine.threshold if engine else (float(os.getenv('LSTM_THRESHOLD_PERCENT', 0.1)) / 100)
    
    pairs_health = {}
    if engine and engine.health:
        for pair, metric in engine.health.items():
            pairs_health[pair] = {
                "status": str(metric.get('status', 'unknown')),
                "last_check": metric.get('last_check'),
                "signals_today": int(metric.get('signals_today', 0))
            }
            
    return {
        "status": "healthy" if app_state['is_running'] else "down",
        "threshold_percent": f"{thresh * 100}%",
        "pairs": pairs_health,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/api/signal/{pair}")
async def get_signal(pair: str):
    pair = pair.upper()
    if not app_state.get('engine') or pair not in app_state['engine'].pairs:
        raise HTTPException(status_code=400, detail="Invalid pair")
    
    key = f"signal:{pair}:current"
    data = await redis_client.get(key)
    
    if not data:
        raise HTTPException(status_code=404, detail="No signal found")
    
    return json.loads(data)


@app.get("/api/signals/{pair}/history")
async def get_history(pair: str, limit: int = 50):
    pair = pair.upper()
    key = f"signal:{pair}:history"
    data = await redis_client.lrange(key, 0, limit - 1)
    return [json.loads(d) for d in data]


@app.get("/test-signal")
async def test_signal():
    engine = app_state.get('engine')
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not ready")
    
    prefs = {
        'emails': [os.getenv('ALERT_EMAIL')] if os.getenv('ALERT_EMAIL') else [],
        'phone_numbers': [os.getenv('ALERT_PHONE')] if os.getenv('ALERT_PHONE') else [],
        'min_confidence': 0.1  
    }
    
    eth_alert = SignalAlert(
        pair="ETHUSD",
        signal="BUY",
        price=1850.50,
        confidence=0.85,
        timestamp=datetime.now(),
        reasoning="TEST ALERT - Manual verification check for Ethereum! Candle confirmation metrics and live RSI pipelines are active.",
        target=1890.00,  
        stop_loss=1820.00,
        urgency="high"
    )
    await engine.notification_manager.send_all(eth_alert, prefs, force=True)
    
    await asyncio.sleep(1.5)
    
    btc_alert = SignalAlert(
        pair="BTCUSD",
        signal="SELL",
        price=67240.00,
        confidence=0.91,
        timestamp=datetime.now(),
        reasoning="TEST ALERT - Manual verification check for Bitcoin! Kraken stream channels and internal matrix queues are fully active.",
        target=66100.00,  
        stop_loss=67850.00,
        urgency="high"
    )
    await engine.notification_manager.send_all(btc_alert, prefs, force=True)
    
    return {
        "message": "🚨 TWO-WAY MATRIX SYNC COMPLETE: Ethereum & Bitcoin test alerts successfully routed directly to Telegram!",
        "channels": ["ETHUSD", "BTCUSD"],
        "bypass_filters": True,
        "timestamp": datetime.now().isoformat()
    }


@app.websocket("/ws/live")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    engine = app_state.get('engine')
    if engine:
        engine.notification_manager.register_ws(websocket)
    try:
        while True:
            msg = await websocket.receive_text()
            data = json.loads(msg)
            if data.get('action') == 'subscribe':
                pair = data.get('pair')
                signal_data = await redis_client.get(f"signal:{pair}:current")
                if signal_data:
                    await websocket.send_json({
                        'type': 'initial',
                        'data': json.loads(signal_data)
                    })
    except WebSocketDisconnect:
        if engine:
            engine.notification_manager.unregister_ws(websocket)


@app.post("/api/retrain/{pair}")
async def retrain(pair: str, background_tasks: BackgroundTasks):
    pair = pair.upper()
    if not app_state.get('engine') or pair not in app_state['engine'].pairs:
        raise HTTPException(status_code=400, detail="Invalid pair")
    
    async def do_retrain():
        engine = app_state['engine']
        df = await engine.data_feed.fetch_historical(pair, days=90)
        engine.predictors[pair].lstm.train(df, pair)
        del df
        gc.collect()
    
    background_tasks.add_task(do_retrain)
    return {"message": f"Retraining {pair}", "status": "processing"}


# =====================================================================
# 🚀 PRODUCTION ENTRYPOINT
# =====================================================================
if __name__ == "__main__":
    import uvicorn
    port_env = os.getenv("PORT", "8000")
    port = int(port_env) if port_env.isdigit() else 8000
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)