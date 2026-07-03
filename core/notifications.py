# core/notifications.py
# Telegram notification system with urgency levels

import os
import asyncio
import aiohttp
from datetime import datetime
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)


class SignalAlert:
    """Represents a trading signal alert"""

    def __init__(self, pair: str, signal: str, price: float, confidence: float,
                 timestamp: datetime, reasoning: str, target: Optional[float] = None,
                 stop_loss: Optional[float] = None, urgency: str = "normal",
                 is_early: bool = False, confluence_score: float = 0.0):
        self.pair = pair
        self.signal = signal
        self.price = price
        self.confidence = confidence
        self.timestamp = timestamp
        self.reasoning = reasoning
        self.target = target
        self.stop_loss = stop_loss
        self.urgency = urgency
        self.is_early = is_early
        self.confluence_score = confluence_score

    def to_dict(self) -> Dict:
        return {
            "pair": self.pair,
            "signal": self.signal,
            "price": self.price,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat(),
            "reasoning": self.reasoning,
            "target": self.target,
            "stop_loss": self.stop_loss,
            "urgency": self.urgency,
            "is_early": self.is_early,
            "confluence_score": self.confluence_score
        }


class NotificationManager:
    """Manages Telegram alerts for trading signals"""

    def __init__(self):
        self.bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
        self.enabled = bool(self.bot_token and self.chat_id)
        self.session: Optional[aiohttp.ClientSession] = None
        self.last_alert_time: Optional[datetime] = None
        self.min_alert_interval = 30  # seconds between alerts
        logger.info(f"NotificationManager initialized | Enabled: {self.enabled}")

    async def _get_session(self) -> aiohttp.ClientSession:
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
        return self.session

    async def send_test_message(self, message: str) -> bool:
        if not self.enabled:
            logger.warning("Telegram not configured — skipping test message")
            return False
        return await self._send_telegram_message(message)

    async def send_telegram_alert(self, alert: SignalAlert) -> bool:
        if not self.enabled:
            logger.warning("Telegram not configured — alert not sent")
            return False

        # Rate limiting
        if self.last_alert_time:
            elapsed = (datetime.utcnow() - self.last_alert_time).total_seconds()
            if elapsed < self.min_alert_interval:
                logger.info(f"Rate limited: {elapsed:.0f}s since last alert")
                return False

        message = self._format_alert_message(alert)
        success = await self._send_telegram_message(message)
        if success:
            self.last_alert_time = datetime.utcnow()
        return success

    async def send_early_warning(self, pair: str, signal: str, price: float,
                                  confluence_score: float, reasoning: str) -> bool:
        if not self.enabled:
            return False

        alert = SignalAlert(
            pair=pair,
            signal=signal,
            price=price,
            confidence=confluence_score,
            timestamp=datetime.utcnow(),
            reasoning=reasoning,
            urgency="low",
            is_early=True,
            confluence_score=confluence_score
        )
        return await self.send_telegram_alert(alert)

    def _format_alert_message(self, alert: SignalAlert) -> str:
        emoji_signal = "🟢 BUY" if alert.signal == "BUY" else "🔴 SELL" if alert.signal == "SELL" else "⚪ HOLD"
        emoji_urgency = "🔥" if alert.urgency == "high" else "⚡" if alert.urgency == "normal" else "📡"
        early_label = " [EARLY]" if alert.is_early else ""

        lines = [
            f"{emoji_urgency} AI CRYPTO SIGNAL{early_label}",
            f"━━━━━━━━━━━━━━━━━━━━━━━",
            f"📊 Pair: {alert.pair}",
            f"{emoji_signal}",
            f"💰 Price: {alert.price:,.2f}",
            f"🎯 Confidence: {alert.confidence*100:.1f}%",
            f"📈 Confluence Score: {alert.confluence_score:.2f}",
            f"⏰ Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}",
        ]

        if alert.target:
            lines.append(f"🎯 Target: {alert.target:,.2f}")
        if alert.stop_loss:
            lines.append(f"🛑 Stop Loss: {alert.stop_loss:,.2f}")

        lines.extend([
            f"━━━━━━━━━━━━━━━━━━━━━━━",
            f"🧠 Analysis:",
            f"{alert.reasoning}",
            f"━━━━━━━━━━━━━━━━━━━━━━━",
            f"🤖 AI Crypto Bot v4.1 | Fibonacci + Trend Filter | Pure Price Action"
        ])

        return "\\n".join(lines)

    async def _send_telegram_message(self, message: str) -> bool:
        if not self.bot_token or not self.chat_id:
            return False

        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": message,
            "parse_mode": "HTML",
            "disable_web_page_preview": True
        }

        try:
            session = await self._get_session()
            async with session.post(url, json=payload, timeout=10) as response:
                if response.status == 200:
                    logger.info("Telegram message sent successfully")
                    return True
                else:
                    text = await response.text()
                    logger.error(f"Telegram API error {response.status}: {text[:200]}")
                    return False
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")
            return False

    async def close(self):
        if self.session and not self.session.closed:
            await self.session.close()
            logger.info("NotificationManager session closed")

