# manual_test.py
# Test the multi-asset multi-timeframe pipeline v5.0

import asyncio
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def test_pipeline():
    from core.data_feed import DataFeedManager
    from core.multi_timeframe import MultiTimeframeEngine
    from core.notifications import NotificationManager

    logger.info("🧪 Testing AI Signal Pipeline v5.0")
    logger.info("=" * 50)

    feed = DataFeedManager()
    mtf = MultiTimeframeEngine()
    notifier = NotificationManager()

    assets = {
        "BTCUSD": {
            "source": "kraken",
            "timeframes": {
                "primary": {"tf": "15m", "deriv_granularity": 900},
                "secondary": {"tf": "5m", "deriv_granularity": 300},
                "entry": {"tf": "1m", "deriv_granularity": 60}
            }
        },
        "frxEURUSD": {
            "source": "deriv",
            "timeframes": {
                "primary": {"tf": "15m", "deriv_granularity": 900},
                "secondary": {"tf": "5m", "deriv_granularity": 300},
                "entry": {"tf": "1m", "deriv_granularity": 60}
            }
        }
    }

    for asset, cfg in assets.items():
        logger.info(f"\n{'='*50}")
        logger.info(f"Testing {asset} ({cfg['source']})")

        result = await mtf.analyze_asset(
            pair=asset,
            source=cfg["source"],
            timeframes=cfg["timeframes"],
            data_feed=feed
        )

        if result is None:
            logger.error(f"❌ {asset}: Analysis failed")
            continue

        if result.get("signal") == "HOLD":
            logger.info(f"ℹ️ {asset}: HOLD — {result.get('reason', 'No signal')}")
            if result.get("layers"):
                for layer, data in result["layers"].items():
                    logger.info(f"   {layer}: {data['signal']} ({data['trend']})")
            continue

        logger.info(f"✅ {asset}: {result['signal']} @ {result['entry_price']:.5f}")
        logger.info(f"   SL: {result['stop_loss']:.5f} | TP: {result['take_profit']:.5f}")
        logger.info(f"   Confidence: {result['confidence']*100:.1f}%")
        logger.info(f"   ATR%: {result.get('atr_pct', 0):.3f}% | ADX: {result.get('adx', 0):.1f}")
        logger.info(f"   Reason: {result['reason'][:100]}")

        if notifier.enabled:
            from core.notifications import SignalAlert
            alert = SignalAlert(
                pair=asset,
                signal=result["signal"],
                entry_price=result["entry_price"],
                confidence=result["confidence"],
                timestamp=datetime.utcnow(),
                reasoning=result["reason"],
                stop_loss=result["stop_loss"],
                take_profit=result["take_profit"],
                layers=result.get("layers", {})
            )
            await notifier.send_telegram_alert(alert)
            logger.info("   📨 Telegram alert sent")

    logger.info("\n✅ Pipeline test v5.0 complete!")


if __name__ == "__main__":
    asyncio.run(test_pipeline())