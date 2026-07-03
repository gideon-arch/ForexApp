# manual_test.py
# Test the signal pipeline v4.0 with REAL market data
# NO LSTM — Pure Price Action

import asyncio
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def test_pipeline():
    """Test the complete signal pipeline v4.0."""
    from core.data_feed import DataFeedManager
    from core.technical_analysis import TechnicalAnalyzer
    from core.filters import FilterManager
    from core.notifications import NotificationManager, SignalAlert

    logger.info("🧪 Testing AI Crypto Signal Pipeline v4.0")
    logger.info("="*50)

    # 1. Test Data Feed — Kraken only, BTCUSD only
    logger.info("\n1️⃣ Testing Data Feed...")
    feed = DataFeedManager()

    df = await feed.fetch_latest("BTCUSD", timeframe="3m")
    if df is not None:
        logger.info(f"✅ BTCUSD: {len(df)} candles from Kraken")
        logger.info(f"   Latest: O={df['open'].iloc[-1]:.2f} H={df['high'].iloc[-1]:.2f} L={df['low'].iloc[-1]:.2f} C={df['close'].iloc[-1]:.2f}")
    else:
        logger.error("❌ BTCUSD: Failed to fetch from Kraken")
        return

    # 2. Test Technical Analysis
    logger.info("\n2️⃣ Testing Technical Analysis...")
    analyzer = TechnicalAnalyzer()

    if len(df) >= 50:
        try:
            indicators = analyzer.calculate_indicators(df)
            logger.info(f"✅ Indicators calculated:")
            logger.info(f"   RSI: {indicators['rsi']:.2f} (zone: {indicators['rsi_zone']})")
            logger.info(f"   Trend: {indicators['market_structure']}")
            logger.info(f"   Signal: {indicators['smc_suggested_signal']}")
            logger.info(f"   Confluence: {indicators['confluence_score']:.3f}")
            logger.info(f"   Live Reversal: {indicators['live_reversal']}")
            logger.info(f"   RSI Divergence: {indicators['rsi_divergence']}")
            logger.info(f"   Candlestick: {indicators['candlestick_signal']} ({indicators['candlestick_confidence']:.2f})")
            logger.info(f"   Fakeout: {indicators['fakeout_detected']} (risk: {indicators['fakeout_risk']:.2f})")

            # Pattern summary
            patterns = indicators.get('candlestick_patterns', [])
            if patterns:
                pattern_names = [p['name'] for p in patterns]
                logger.info(f"   Patterns: {', '.join(pattern_names)}")

            # Triangle detection
            triangle = indicators.get('triangle')
            if triangle and triangle.get('armed'):
                logger.info(f"   Triangle: {triangle['type']} (apex in {triangle['apex_distance']} candles)")

            tech_score = analyzer.generate_technical_score(indicators)
            logger.info(f"   Tech Score: {tech_score:.3f}")
        except Exception as e:
            logger.error(f"❌ Technical analysis error: {e}")
            return
    else:
        logger.error(f"❌ Not enough data: {len(df)} candles")
        return

    # 3. Test Filters — pass pre-calculated indicators
    logger.info("\n3️⃣ Testing Filters...")
    filter_mgr = FilterManager()

    signal_type = indicators['smc_suggested_signal']
    if signal_type in ['BUY', 'SELL']:
        # Test early signal (2/4 filters)
        passed_early, report_early = filter_mgr.validate_signal(
            df, signal_type, df['close'].iloc[-1],
            indicators=indicators, pair="BTCUSD", is_early=True
        )
        logger.info(f"✅ Early filter test ({signal_type}): {'PASSED' if passed_early else 'BLOCKED'}")
        for key, val in report_early.get('filters', {}).items():
            logger.info(f"   {key}: {val.get('reason', 'N/A')}")

        # Test confirmed signal (3/4 filters)
        passed_conf, report_conf = filter_mgr.validate_signal(
            df, signal_type, df['close'].iloc[-1],
            indicators=indicators, pair="BTCUSD", is_early=False
        )
        logger.info(f"✅ Confirmed filter test ({signal_type}): {'PASSED' if passed_conf else 'BLOCKED'}")
    else:
        logger.info("ℹ️ Signal is HOLD — skipping filter test")

    # 4. Test Fakeout Detector directly
    logger.info("\n4️⃣ Testing Fakeout Detector...")
    from core.fakeout_detector import FakeoutDetector
    fakeout = FakeoutDetector()
    fakeout_result = fakeout.analyze(
        df,
        indicators.get('active_support'),
        indicators.get('active_resistance'),
        indicators.get('market_structure', 'SIDEWAYS')
    )
    if fakeout_result['fakeout_detected']:
        logger.info(f"⚠️ Fakeout detected: {fakeout_result['trap_type']}")
        logger.info(f"   Direction: {fakeout_result['fakeout_direction']}")
        logger.info(f"   Risk: {fakeout_result['fakeout_risk']:.2f}")
        logger.info(f"   Reasoning: {fakeout_result['reasoning']}")
    else:
        logger.info("✅ No fakeout detected")

    # 5. Test Telegram
    logger.info("\n5️⃣ Testing Telegram...")
    notifier = NotificationManager()

    if notifier.enabled:
        await notifier.send_test_message("🧪 Test message from AI Crypto Bot v4.0! No LSTM — Pure Price Action")
        logger.info("✅ Telegram test message sent")
    else:
        logger.warning("⚠️ Telegram not configured. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID")

    # 6. Test Candlestick Patterns directly
    logger.info("\n6️⃣ Testing Candlestick Patterns...")
    from core.candlestick_patterns import CandlestickPatternRecognizer
    cs = CandlestickPatternRecognizer()
    cs_result = cs.analyze(df)
    logger.info(f"   Signal: {cs_result['signal']} (confidence: {cs_result['confidence']:.2f})")
    logger.info(f"   Bullish score: {cs_result['bullish_score']:.2f}")
    logger.info(f"   Bearish score: {cs_result['bearish_score']:.2f}")
    if cs_result['patterns']:
        logger.info(f"   Patterns: {cs.get_pattern_summary(cs_result['patterns'])}")

    logger.info("\n✅ Pipeline test v4.0 complete!")
    logger.info("="*50)


if __name__ == "__main__":
    asyncio.run(test_pipeline())
'''

with open(output_dir + "manual_test.py", "w") as f:
    f.write(manual_test_py)

# ============================================================
# FIXED: test_api.py
# ============================================================
test_api_py = r'''# test_api.py
# Test API connectivity and endpoints

import aiohttp
import asyncio


async def test_kraken():
    print("\nTesting Kraken...")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("https://api.kraken.com/0/public/Time", timeout=10) as response:
                print(f"Kraken Status: {response.status}")
                data = await response.json()
                print(f"Kraken Server Time: {data.get('result', {}).get('unixtime', 'N/A')}")
                return True
    except Exception as e:
        print(f"Kraken failed: {e}")
        return False


async def test_local_api():
    print("\nTesting Local API Endpoints...")
    try:
        async with aiohttp.ClientSession() as session:
            # Test root
            async with session.get("http://localhost:8000/", timeout=5) as response:
                print(f"Root Status: {response.status}")
                data = await response.json()
                print(f"Root Response: {data}")

            # Test health
            async with session.get("http://localhost:8000/health", timeout=5) as response:
                print(f"Health Status: {response.status}")
                data = await response.json()
                print(f"Health Response: {data}")

            # Test signals
            async with session.get("http://localhost:8000/api/signals", timeout=5) as response:
                print(f"Signals Status: {response.status}")
                data = await response.json()
                print(f"Signals Response: {data}")

            return True
    except Exception as e:
        print(f"Local API test failed (is the server running?): {e}")
        return False


async def main():
    print("="*50)
    print("API Connectivity Test")
    print("="*50)

    kraken_ok = await test_kraken()
    api_ok = await test_local_api()

    print("\n" + "="*50)
    print("Results:")
    print(f"  Kraken API: {'✅ OK' if kraken_ok else '❌ FAILED'}")
    print(f"  Local API:  {'✅ OK' if api_ok else '❌ FAILED (server may not be running)'}")
    print("="*50)


asyncio.run(main())

