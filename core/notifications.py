# core/notifications.py

import asyncio
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass
import os
import requests
import html  # Native sanitization for web standard elements
from twilio.rest import Client as TwilioClient


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
    def __init__(self, config: Dict):
        self.config = config
        self.enabled_channels = []
        self.alert_history = []
        
        # Rate intervals mapped with logical separation baselines
        self.channel_intervals = {
            'telegram': 10,  # Minimum seconds between the exact same asset action
            'sms': 300
        }
        # Dynamic rate limit tracking registry matrix
        self.rate_limits: Dict[str, datetime] = {}
        
        # Pulling your Telegram variables from environment variables
        self.telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')
        
        if self.telegram_token and self.telegram_chat_id:
            self.telegram_enabled = True
            self.enabled_channels.append('telegram')
            print("✅ Telegram Signal Bot pipeline initialized successfully!")
        else:
            self.telegram_enabled = False
            print("⚠️ Telegram missing (Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in environment)")
        
        if config.get('twilio_sid') and config.get('twilio_token'):
            self.twilio_client = TwilioClient(
                config['twilio_sid'],
                config['twilio_token']
            )
            self.twilio_from = config.get('twilio_from_number')
            self.sms_enabled = True
            self.enabled_channels.append('sms')
            print("✅ Twilio SMS initialized")
        else:
            self.sms_enabled = False
            print("⚠️ Twilio not configured")
        
        self.ws_connections = []
    
    # FIXED: Grouped tracking signature key by asset to prevent rapid alternating action signals
    def _check_rate_limit(self, channel: str, pair: str) -> bool:
        now = datetime.now()
        rate_key = f"{channel}_{pair.upper()}"
        
        last = self.rate_limits.get(rate_key)
        interval = self.channel_intervals.get(channel, 10)
        
        if last is None or (now - last).total_seconds() >= interval:
            self.rate_limits[rate_key] = now
            return True
        return False

    async def send_telegram_alert(self, alert: SignalAlert, force: bool = False):
        if not self.telegram_enabled:
            return
        # FIXED: Rate check evaluated by asset layout to keep streaming data stable
        if not force and not self._check_rate_limit('telegram', alert.pair):
            return
            
        emojis = {'BUY': '🟢 🚀', 'SELL': '🔴 📉', 'HOLD': '🟡 ⏸️'}
        status_emoji = emojis.get(alert.signal, '⚪')
        
        threshold_pct = os.getenv('LSTM_THRESHOLD_PERCENT', '0.1')
        alert_category = "CRYPTO" if "BTC" in alert.pair.upper() else "FOREX"
        raw_reasoning = alert.reasoning if alert.reasoning else "No breakdown summary provided"
        
        # HTML Entity Sanitization to protect layout strings from syntax errors
        clean_pair = html.escape(alert.pair)
        clean_signal = html.escape(alert.signal)
        clean_confidence = html.escape(f"{alert.confidence:.2%}")
        clean_price = html.escape(f"{alert.price:.5f}" if alert.price else "N/A")
        clean_target = html.escape(f"{alert.target:.5f}" if alert.target else "N/A")
        clean_sl = html.escape(f"{alert.stop_loss:.5f}" if alert.stop_loss else "N/A")
        clean_time = html.escape(alert.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC'))
        clean_reason = html.escape(raw_reasoning)
        clean_pct = html.escape(str(threshold_pct))

        message = (
            f"{status_emoji} <b>AI {alert_category} ALERT: {clean_signal}</b> {clean_pair}\n"
            f"━━━━━━━━━━━━━━━━━━━\n"
            f"📊 <b>AI Confidence:</b> <code>{clean_confidence}</code>\n"
            f"💰 <b>Execution Price:</b> <code>{clean_price}</code>\n"
            f"🎯 <b>Take Profit Target:</b> <code>{clean_target}</code>\n"
            f"🛑 <b>Risk Stop Loss:</b> <code>{clean_sl}</code>\n"
            f"⏰ <b>Time:</b> <code>{clean_time}</code>\n\n"
            f"💡 <b>Strategy Reasoning Summary:</b>\n<i>{clean_reason}</i>\n\n"
            f"⚙️ <code>System Matrix: Baseline {clean_pct}% Threshold</code>"
        )
        
        try:
            await asyncio.to_thread(self._execute_telegram_send, message)
        except Exception as e:
            print(f"❌ Telegram pipeline alert failed: {e}")

    def _execute_telegram_send(self, message: str):
        url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
        payload = {
            "chat_id": self.telegram_chat_id,
            "text": message,
            "parse_mode": "HTML"
        }
        response = requests.post(url, json=payload, timeout=10)
        if response.status_code == 200:
            print("📡 Telegram Signal broadcasted successfully!")
        else:
            print(f"❌ Telegram API Error: {response.text}")
    
    async def send_sms_alert(self, alert: SignalAlert, phone_numbers: List[str], force: bool = False):
        if not self.sms_enabled:
            return
        if not force and not self._check_rate_limit('sms', alert.pair):
            return
        if not force and alert.confidence < 0.85:
            return
        
        emojis = {'BUY': '🟢', 'SELL': '🔴', 'HOLD': '🟡'}
        message = f"{emojis.get(alert.signal, '⚪')} {alert.pair} {alert.signal}\nConfidence: {alert.confidence:.1%}\nPrice: {alert.price:.5f}"
        
        for number in phone_numbers:
            try:
                # Wrap twilio client call in a thread execution pool executor to maintain event loop velocity
                await asyncio.to_thread(
                    self.twilio_client.messages.create,
                    body=message,
                    from_=self.twilio_from,
                    to=number
                )
            except Exception as e:
                print(f"❌ SMS pipeline failed for number {number}: {e}")
    
    async def broadcast_ws(self, alert: SignalAlert):
        message = {
            'type': 'new_signal',
            'data': {
                'pair': alert.pair, 'signal': alert.signal, 'price': alert.price,
                'confidence': alert.confidence, 'timestamp': alert.timestamp.isoformat()
            }
        }
        for ws in list(self.ws_connections):
            try:
                await ws.send_json(message)
            except:
                self.ws_connections.remove(ws)
    
    async def send_all(self, alert: SignalAlert, user_prefs: Dict, force: bool = False):
        tasks = [self.broadcast_ws(alert)]
        
        if self.telegram_enabled:
            tasks.append(self.send_telegram_alert(alert, force=force))
        
        if 'sms' in self.enabled_channels and user_prefs.get('phone_numbers'):
            tasks.append(self.send_sms_alert(alert, user_prefs['phone_numbers'], force=force))
        
        await asyncio.gather(*tasks, return_exceptions=True)
        self.alert_history.append({'timestamp': datetime.now(), 'alert': alert})
    
    def register_ws(self, websocket):
        self.ws_connections.append(websocket)
    
    def unregister_ws(self, websocket):
        if websocket in self.ws_connections:
            self.ws_connections.remove(websocket)