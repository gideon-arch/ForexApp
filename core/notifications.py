# core/notifications.py
# Telegram Signal Alerts

import asyncio
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass
import os
import logging
import html

logger = logging.getLogger(__name__)

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    logging.warning("requests not available. Telegram disabled.")

@dataclass
class SignalAlert:
    pair: str
    signal: str
    price: float
    confidence: float
    timestamp: datetime
    reasoning: str
    target: Optional[float] = None
    stop_loss: Optional[float] = None
    urgency: str = "normal"

class NotificationManager:
    def __init__(self):
        self.telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')
        self.telegram_enabled = False
        self.alert_history = []
        self.rate_limits: Dict[str, datetime] = {}
        self.min_interval = 30
        
        if REQUESTS_AVAILABLE and self.telegram_token and self.telegram_chat_id:
            self.telegram_enabled = True
            logger.info("Telegram notifications enabled")
        else:
            if not REQUESTS_AVAILABLE: logger.warning("Telegram disabled: requests not installed")
            elif not self.telegram_token: logger.warning("Telegram disabled: TELEGRAM_BOT_TOKEN not set")
            elif not self.telegram_chat_id: logger.warning("Telegram disabled: TELEGRAM_CHAT_ID not set")

    def _check_rate_limit(self, pair: str) -> bool:
        now = datetime.now()
        rate_key = pair.upper()
        last = self.rate_limits.get(rate_key)
        if last is None or (now - last).total_seconds() >= self.min_interval:
            self.rate_limits[rate_key] = now
            return True
        logger.info(f"Rate limit active for {pair}: {(now - last).seconds}s < {self.min_interval}s")
        return False

    async def send_telegram_alert(self, alert: SignalAlert, force: bool = False):
        if not self.telegram_enabled:
            return
        
        if not force and not self._check_rate_limit(alert.pair):
            return
        
        emojis = {'BUY': '🟢 🚀', 'SELL': '🔴 📉', 'HOLD': '🟡 ⏸️'}
        status_emoji = emojis.get(alert.signal, '⚪')
        
        clean_pair = html.escape(alert.pair)
        clean_signal = html.escape(alert.signal)
        clean_confidence = html.escape(f"{alert.confidence:.1%}")
        clean_price = html.escape(f"{alert.price:.2f}")
        clean_target = html.escape(f"{alert.target:.2f}") if alert.target else "N/A"
        clean_sl = html.escape(f"{alert.stop_loss:.2f}") if alert.stop_loss else "N/A"
        clean_time = html.escape(alert.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC'))
        clean_reason = html.escape(alert.reasoning[:500]) if alert.reasoning else "AI Pattern Recognition"
        
        message = (
            f"{status_emoji} <b>AI SIGNAL: {clean_signal}</b> {clean_pair}\n"
            f"━━━━━━━━━━━━━━━━━━━\n"
            f"📊 <b>Confidence:</b> <code>{clean_confidence}</code>\n"
            f"💰 <b>Price:</b> <code>{clean_price}</code>\n"
            f"🎯 <b>Take Profit:</b> <code>{clean_target}</code>\n"
            f"🛑 <b>Stop Loss:</b> <code>{clean_sl}</code>\n"
            f"⏰ <b>Time:</b> <code>{clean_time}</code>\n\n"
            f"💡 <b>Analysis:</b>\n<i>{clean_reason}</i>\n\n"
            f"⚙️ <code>AI Crypto Bot v2.1</code>"
        )
        
        try:
            await asyncio.to_thread(self._send_telegram_request, message)
            self.alert_history.append({'time': datetime.now(), 'alert': alert})
        except Exception as e:
            logger.error(f"Telegram send failed: {e}")

    def _send_telegram_request(self, message: str):
        if not REQUESTS_AVAILABLE:
            return
        
        url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
        payload = {
            "chat_id": self.telegram_chat_id,
            "text": message,
            "parse_mode": "HTML",
            "disable_web_page_preview": True
        }
        
        response = requests.post(url, json=payload, timeout=15)
        if response.status_code == 200:
            logger.info("Telegram alert sent")
        elif response.status_code == 429:
            logger.warning("Telegram rate limit hit")
        else:
            logger.error(f"Telegram API error {response.status_code}: {response.text[:200]}")

    async def send_test_message(self, text: str = "🤖 AI Crypto Bot is online!"):
        if not self.telegram_enabled:
            logger.info("Telegram not configured, skipping test")
            return
        
        try:
            await asyncio.to_thread(self._send_telegram_request, text)
            logger.info("Test message sent")
        except Exception as e:
            logger.error(f"Test message failed: {e}")

    def get_stats(self) -> Dict:
        return {
            "enabled": self.telegram_enabled,
            "total_alerts": len(self.alert_history),
            "last_alerts": [
                {"time": a["time"].isoformat(), "pair": a["alert"].pair, "signal": a["alert"].signal}
                for a in self.alert_history[-10:]
            ]
        }