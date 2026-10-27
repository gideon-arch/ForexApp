# core/notifications.py
# Telegram notification system — multi-asset, exact entry/TP/SL
# v5.3: Removed parse_mode="HTML" (was crashing on '<' in hold reasons, e.g.
#       "Confidence 0.58 < min 0.60" -> Telegram 400 "can't parse entities").
#       All messages are plain text + emojis, so no formatting is lost.
# v5.4.3: Price formatting was hardcoded to 2 decimal places (:,.2f) for
#         EVERY asset. That's fine for BTCUSD (~84,796.10) but destroys
#         precision for EURUSD (~1.14235) — entry/SL/TP all rounded to
#         "1.14" and became visually identical in the Telegram message,
#         even though the underlying numbers were distinct. Now the decimal
#         count adapts to the price's magnitude (5dp under 50, 2dp above).

import os
import asyncio
import aiohttp
import html as html_lib
from datetime import datetime
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)


class SignalAlert:
    def __init__(self, pair: str, signal: str, entry_price: float, confidence: float,
                 timestamp: datetime, reasoning: str, stop_loss: float,
                 take_profit: float, urgency: str = "normal",
                 is_early: bool = False, confluence_score: float = 0.0,
                 layers: Optional[Dict] = None, signal_id: Optional[str] = None):
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
        # v5.6.0: same ID the outcome tracker uses for this signal — lets the
        # Accept/Reject button on this message reference the exact trade.
        self.signal_id = signal_id

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
    def __init__(self, version: str = "unknown"):
        """
        v5.5.4 BUG FIX: the version shown in the Telegram footer was a
        separate hardcoded string in THIS file, independent of the one in
        main.py — so every version bump needed editing two files, and it
        kept getting missed (confirmed repeatedly: footer showed v5.4.5,
        then v5.5.1, while the rest of the bot was already several versions
        ahead). `version` is now passed in from main.py's single
        APP_VERSION constant at startup — there is nothing left to forget
        to update here.
        """
        self.bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
        self.enabled = bool(self.bot_token and self.chat_id)
        self.session: Optional[aiohttp.ClientSession] = None
        self.last_alert_time: Optional[datetime] = None
        self.min_alert_interval = 15
        self.version = version
        logger.info(f"NotificationManager v{self.version} | Enabled: {self.enabled}")

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
        # v5.6.5: Telegram alerts limited to BTC only, per request — EUR and
        # Gold signal quality hasn't earned trust yet. The engine itself is
        # UNCHANGED: it still analyzes and the outcome tracker still records
        # all three pairs exactly as before, so EUR/Gold stats keep building
        # in the background (check /api/stats any time) — they just don't
        # reach your phone anymore.
        if "BTC" not in alert.pair.upper():
            logger.info(f"Telegram alert suppressed (BTC-only mode): {alert.pair} {alert.signal}")
            return False

        if not self.enabled:
            logger.warning("Telegram not configured")
            return False

        if self.last_alert_time:
            elapsed = (datetime.utcnow() - self.last_alert_time).total_seconds()
            if elapsed < self.min_alert_interval:
                logger.info(f"Rate limited: {elapsed:.0f}s since last alert")
                return False

        message = self._format_alert_message(alert)

        # v5.6.0: Accept/Reject buttons — only attached when the caller gave
        # us a signal_id to reference. callback_data is capped at 64 bytes by
        # Telegram; "acc_demo:" / "rej:" + signal_id comfortably fits typical
        # IDs like "BTCUSD_2026-10-12T18:26:00.123456".
        # v5.7.0: added a genuine Accept (Live) button — on its OWN row,
        # separated from Demo/Reject, specifically so it's never the button
        # your thumb lands on by habit. The server still hard-gates this
        # behind LIVE_TRADING_ENABLED (default off) regardless of which
        # button gets tapped — see the webhook handler in main.py.
        reply_markup = None
        if alert.signal_id:
            reply_markup = {
                "inline_keyboard": [
                    [
                        {"text": "✅ Accept (Demo)", "callback_data": f"acc_demo:{alert.signal_id}"},
                        {"text": "❌ Reject", "callback_data": f"rej:{alert.signal_id}"},
                    ],
                    [
                        {"text": "🔴 Accept (LIVE — real money)", "callback_data": f"acc_live:{alert.signal_id}"},
                    ],
                ]
            }

        success = await self._send_telegram_message(message, parse_mode="HTML", reply_markup=reply_markup)
        if success:
            self.last_alert_time = datetime.utcnow()
        return success

    def _price_decimals(self, price: float) -> int:
        """
        v5.4.3: pick decimal places from the price's own magnitude instead of
        a hardcoded 2dp. Forex pairs like EURUSD (~1.xxxxx) need 5dp or the
        entry/SL/TP — which can differ by only a few hundredths of a unit —
        all round to the same displayed number. BTC/Gold-range prices (low
        thousands and up) are fine at 2dp.
        """
        return 5 if abs(price) < 50 else 2

    def _format_alert_message(self, alert: SignalAlert) -> str:
        """
        v5.4.5: Entry/SL/TP are now wrapped in HTML <code> tags. On Telegram
        mobile, tapping monospaced text copies just that value — so the user
        can tap the SL number alone and paste it straight into MT5's Stop
        Loss field, instead of having to select it out of the whole message.
        This alert (unlike send_test_message, used for the summary/startup
        messages) is safe to send as parse_mode=HTML because every piece of
        dynamic text below is HTML-escaped first — that's what avoids the
        original v5.3 crash, where a raw "<" in a hold-reason string like
        "Confidence 0.58 < min 0.60" broke Telegram's HTML parser. The
        summary/startup messages still go out as plain text for that same
        reason (they DO contain raw "<" from filter comparisons).
        """
        emoji_signal = "🟢 BUY" if alert.signal == "BUY" else "🔴 SELL"
        emoji_urgency = "🔥" if alert.urgency == "high" else "⚡" if alert.urgency == "normal" else "📡"
        early_label = " [EARLY]" if alert.is_early else ""

        # Asset-specific emoji
        asset_emoji = "₿" if "BTC" in alert.pair.upper() else "📊"

        entry_dp = self._price_decimals(alert.entry_price)
        sl_dp = self._price_decimals(alert.stop_loss)
        tp_dp = self._price_decimals(alert.take_profit)

        esc = lambda s: html_lib.escape(str(s), quote=False)

        lines = [
            f"{emoji_urgency} {asset_emoji} AI SIGNAL{early_label}",
            f"━━━━━━━━━━━━━━━━━━━━━━━",
            f"📊 Asset: {esc(alert.pair)}",
            f"{emoji_signal}",
            f"",
            f"🎯 ENTRY:  <code>{alert.entry_price:,.{entry_dp}f}</code>",
            f"🛑 SL:     <code>{alert.stop_loss:,.{sl_dp}f}</code>",
            f"💰 TP:     <code>{alert.take_profit:,.{tp_dp}f}</code>",
            f"",
            f"Confidence: {min(alert.confidence, 1.0) * 100:.1f}%",
            f"Confluence: {alert.confluence_score:.2f}",
        ]

        if os.getenv("FIXED_DOLLAR_RISK_MODE", "true").lower() == "true":
            target_risk = os.getenv("TARGET_RISK_USD", "50")
            target_lot = os.getenv("TARGET_LOT_SIZE", "0.5")
            lines.append(f"💵 Sized for: ${target_risk} risk / ${float(target_risk)*2:.0f} reward @ {target_lot} lot")

        lines += [
            f"⏰ Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}",
            f"━━━━━━━━━━━━━━━━━━━━━━━",
            f"🧠 Analysis:",
            f"{esc(alert.reasoning)}",
        ]

        if alert.layers:
            lines.append(f"━━━━━━━━━━━━━━━━━━━━━━━")
            lines.append(f"📐 Layer Check:")
            for layer, data in alert.layers.items():
                sig = data.get("signal", "HOLD")
                trend = data.get("trend", "?")
                lines.append(f"  {esc(layer)}: {esc(sig)} ({esc(trend)})")

        lines.append(f"━━━━━━━━━━━━━━━━━━━━━━━")
        lines.append(f"🤖 AI Bot v{self.version} | Multi-Timeframe | Morning Boost")
        lines.append(f"👆 Tap a price above to copy it")

        return "\n".join(lines)

    async def _send_telegram_message(self, message: str, parse_mode: Optional[str] = None,
                                      reply_markup: Optional[Dict] = None) -> bool:
        if not self.bot_token or not self.chat_id:
            return False

        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        # v5.3: summary/startup messages (send_test_message) stay plain text —
        # HTML mode crashed on a raw '<' in hold reasons like
        # "Confidence 0.58 < min 0.60". Signal alerts now opt into HTML
        # (see send_telegram_alert) since their dynamic text is escaped.
        payload = {
            "chat_id": self.chat_id,
            "text": message,
            "disable_web_page_preview": True
        }
        if parse_mode:
            payload["parse_mode"] = parse_mode
        if reply_markup:
            payload["reply_markup"] = reply_markup

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

    async def set_webhook(self, webhook_url: str) -> bool:
        """
        v5.6.0: registers webhook_url with Telegram so button taps on
        Accept/Reject get POSTed to our FastAPI app. Safe to call on every
        startup — Telegram just re-confirms the same URL if unchanged.
        """
        if not self.bot_token:
            return False
        url = f"https://api.telegram.org/bot{self.bot_token}/setWebhook"
        try:
            session = await self._get_session()
            async with session.post(url, json={"url": webhook_url}, timeout=10) as response:
                data = await response.json()
                if data.get("ok"):
                    logger.info(f"Telegram webhook registered: {webhook_url}")
                    return True
                logger.error(f"Telegram setWebhook failed: {data}")
                return False
        except Exception as e:
            logger.error(f"Telegram setWebhook error: {e}")
            return False

    async def answer_callback(self, callback_query_id: str, text: str = "", show_alert: bool = False) -> bool:
        """Acknowledges a button tap — Telegram shows a loading spinner on the
        button until this is called, and expects it within ~10s.
        show_alert=True forces a real popup dialog instead of a brief toast —
        used for live-trading accept/block so it can't be missed at a glance."""
        if not self.bot_token:
            return False
        url = f"https://api.telegram.org/bot{self.bot_token}/answerCallbackQuery"
        payload = {"callback_query_id": callback_query_id}
        if text:
            payload["text"] = text
        if show_alert:
            payload["show_alert"] = True
        try:
            session = await self._get_session()
            async with session.post(url, json=payload, timeout=10) as response:
                return response.status == 200
        except Exception as e:
            logger.error(f"Telegram answerCallbackQuery error: {e}")
            return False

    async def edit_message_reply_markup(self, chat_id, message_id: int, text: Optional[str] = None) -> bool:
        """
        After a button is tapped, replace the keyboard (and optionally the
        text) so it's visually clear the tap registered — prevents double-taps
        and confusion about whether the trade was actually accepted.
        """
        if not self.bot_token:
            return False
        try:
            session = await self._get_session()
            if text is not None:
                url = f"https://api.telegram.org/bot{self.bot_token}/editMessageText"
                payload = {"chat_id": chat_id, "message_id": message_id, "text": text,
                           "parse_mode": "HTML"}
            else:
                url = f"https://api.telegram.org/bot{self.bot_token}/editMessageReplyMarkup"
                payload = {"chat_id": chat_id, "message_id": message_id,
                           "reply_markup": {"inline_keyboard": []}}
            async with session.post(url, json=payload, timeout=10) as response:
                return response.status == 200
        except Exception as e:
            logger.error(f"Telegram edit message error: {e}")
            return False

    async def close(self):
        if self.session and not self.session.closed:
            await self.session.close()
            logger.info("NotificationManager closed")