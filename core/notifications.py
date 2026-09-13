# core/notifications.py
# Telegram notification system — multi-asset, exact entry/TP/SL
# v5.3: Removed parse_mode="HTML" (was crashing on '<' in hold reasons, e.g.
#       "Confidence 0.58 < min 0.60" -> Telegram 400 "can't parse entities").
#       All messages are plain text + emojis, so no formatting is lost.

import os
import asyncio
import aiohttp
from datetime import datetime
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)


class SignalAlert:
    def __init__(self, pair: str, signal: str, entry_price: float, confidence: float,
                 timestamp: datetime, reasoning: str, stop_loss: float,
                 take_profit: float, urgency: str = "normal",
                 is_early: bool = False, confluence_score: float = 0.0,
                 layers: Optional[Dict] = None):
        self.pair = pair
        self.signal = signal
        self.entry_price = entry_price
        self.confidence = confidence
        self.timestamp = timestamp
        self.reasoning = reasoning
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.urgency = urgency
        self.is_early = is_early
        self.confluence_score = confluence_score
        self.layers = layers or {}

    def to_dict(self) -> Dict:
        return {
            "pair": self.pair,
            "signal": self.signal,
            "entry_price": self.entry_price,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat(),
            "reasoning": self.reasoning,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "urgency": self.urgency,
            "is_early": self.is_early,
            "confluence_score": self.confluence_score,
            "layers": self.layers
        }


class NotificationManager:
    def __init__(self):
        self.bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
        self.enabled = bool(self.bot_token and self.chat_id)
        self.session: Optional[aiohttp.ClientSession] = None
        self.last_alert_time: Optional[datetime] = None
        self.min_alert_interval = 15
        logger.info(f"NotificationManager v5.3 | Enabled: {self.enabled}")

    async def _get_session(self) -> aiohttp.ClientSession:
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
        return self.session

    async def send_test_message(self, message: str) -> bool:
        if not self.enabled:
            logger.warning("Telegram not configured")
            return False
        return await self._send_telegram_message(message)

    async def send_telegram_alert(self, alert: SignalAlert) -> bool:
        if not self.enabled:
            logger.warning("Telegram not configured")
            return False

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

    def _format_alert_message(self, alert: SignalAlert) -> str:
        emoji_signal = "🟢 BUY" if alert.signal == "BUY" else "🔴 SELL"
        emoji_urgency = "🔥" if alert.urgency == "high" else "⚡" if alert.urgency == "normal" else "📡"
        early_label = " [EARLY]" if alert.is_early else ""

        # Asset-specific emoji
        asset_emoji = "₿" if "BTC" in alert.pair.upper() else "📊"

        lines = [
            f"{emoji_urgency} {asset_emoji} AI SIGNAL{early_label}",
            f"━━━━━━━━━━━━━━━━━━━━━━━",
            f"📊 Asset: {alert.pair}",
            f"{emoji_signal}",
            f"",
            f"🎯 ENTRY:  {alert.entry_price:,.2f}",
            f"🛑 SL:     {alert.stop_loss:,.2f}",
            f"💰 TP:     {alert.take_profit:,.2f}",
            f"",
            f"Confidence: {min(alert.confidence, 1.0) * 100:.1f}%",
            f"Confluence: {alert.confluence_score:.2f}",
            f"⏰ Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}",
            f"━━━━━━━━━━━━━━━━━━━━━━━",
            f"🧠 Analysis:",
            f"{alert.reasoning}",
        ]

        if alert.layers:
            lines.append(f"━━━━━━━━━━━━━━━━━━━━━━━")
            lines.append(f"📐 Layer Check:")
            for layer, data in alert.layers.items():
                sig = data.get("signal", "HOLD")
                trend = data.get("trend", "?")
                lines.append(f"  {layer}: {sig} ({trend})")

        lines.append(f"━━━━━━━━━━━━━━━━━━━━━━━")
        lines.append(f"🤖 AI Bot v5.3 | Multi-Timeframe | Morning Boost")

        return "\n".join(lines)

    async def _send_telegram_message(self, message: str) -> bool:
        if not self.bot_token or not self.chat_id:
            return False

        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        # v5.3: NO parse_mode — plain text. HTML mode crashed on '<' characters
        # in hold reasons (e.g. "Confidence 0.58 < min 0.60").
        payload = {
            "chat_id": self.chat_id,
            "text": message,
            "disable_web_page_preview": True
        }

        try:
            session = await self._get_session()
            async with session.post(url, json=payload, timeout=10) as response:
                if response.status == 200:
                    logger.info("Telegram sent successfully")
                    return True
                else:
                    text = await response.text()
                    logger.error(f"Telegram API error {response.status}: {text[:200]}")
                    return False
        except Exception as e:
            logger.error(f"Failed to send Telegram: {e}")
            return False

    async def close(self):
        if self.session and not self.session.closed:
            await self.session.close()
            logger.info("NotificationManager closed")