# manual_test.py
# Test the signal pipeline with REAL market data

import asyncio
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_pipeline():
    """Test the complete signal pipeline."""
    from core.data_feed import DataFeedManager
    from core.technical_analysis import TechnicalAnalyzer
    from core.lstm_model import EnsemblePredictor
    from core.filters import FilterManager
    from core.notifications import NotificationManager, SignalAlert
    
    logger.info("🧪 Testing AI Crypto Signal Pipeline")
    logger.info("="*50)
    
    # 1. Test Data Feed
    logger.info("\n1️⃣ Testing Data Feed...")
    feed = DataFeedManager()
    
    for pair, source in [("BTCUSD", "deriv"), ("ETHUSD", "kraken")]:
        df = await feed.fetch_latest(pair, timeframe="5m", source=source)
        if df is not None:
            logger.info(f"✅ {pair}: {len(df)} candles from {source}")
            logger.info(f"   Latest: O={df['open'].iloc[-1]:.2f} H={df['high'].iloc[-1]:.2f} L={df['low'].iloc[-1]:.2f} C={df['close'].iloc[-1]:.2f}")
        else:
            logger.error(f"❌ {pair}: Failed to fetch")
    
    # 2. Test Technical Analysis
    logger.info("\n2️⃣ Testing Technical Analysis...")
    analyzer = TechnicalAnalyzer()
    
    if df is not None and len(df) >= 50:
        try:
            indicators = analyzer.calculate_indicators(df)
            logger.info(f"✅ Indicators calculated:")
            logger.info(f"   RSI: {indicators['rsi']:.2f}")
            logger.info(f"   Trend: {indicators['market_structure']}")
            logger.info(f"   Signal: {indicators['smc_suggested_signal']}")
            logger.info(f"   Confluence: {indicators['confluence_score']:.3f}")
            
            tech_score = analyzer.generate_technical_score(indicators)
            logger.info(f"   Tech Score: {tech_score:.3f}")
        except Exception as e:
            logger.error(f"❌ Technical analysis error: {e}")
    
    # 3. Test LSTM Prediction
    logger.info("\n3️⃣ Testing LSTM Prediction...")
    predictor = EnsemblePredictor()
    
    # Try to load model
    model_loaded = predictor.lstm.load_model("ETHUSD")
    if model_loaded:
        logger.info("✅ ETHUSD model loaded")
        if df is not None and len(df) >= 60:
            pred = predictor.predict(df, tech_score=tech_score, fund_score=0.0, pair="ETHUSD")
            logger.info(f"✅ Prediction: {pred['signal']} (conf: {pred['confidence']:.2%})")
            logger.info(f"   Ensemble Score: {pred['ensemble_score']:.3f}")
    else:
        logger.warning("⚠️ No model found. Run train_model.py first!")
    
    # 4. Test Filters
    logger.info("\n4️⃣ Testing Filters...")
    filter_mgr = FilterManager()
    
    if df is not None:
        passed, report = filter_mgr.validate_signal(df, "BUY", df['close'].iloc[-1])
        logger.info(f"✅ Filter test: {'PASSED' if passed else 'BLOCKED'}")
        for key, val in report.get('filters', {}).items():
            logger.info(f"   {key}: {val.get('reason', 'N/A')}")
    
    # 5. Test Telegram
    logger.info("\n5️⃣ Testing Telegram...")
    notifier = NotificationManager()
    
    if notifier.telegram_enabled:
        await notifier.send_test_message("🧪 Test message from AI Crypto Bot!")
    else:
        logger.warning("⚠️ Telegram not configured. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID")
    
    logger.info("\n✅ Pipeline test complete!")

if __name__ == "__main__":
    asyncio.run(test_pipeline())