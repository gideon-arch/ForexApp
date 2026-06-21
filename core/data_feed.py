# core/data_feed.py
import os
import aiohttp
import pandas as pd
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class DataFeedManager:
    def __init__(self):
        self.deriv_app_id = os.getenv("DERIV_APP_ID", "1089") # Ensure this is in your Render env
        self.deriv_url = f"wss://ws.binaryws.com/websockets/v3?app_id={self.deriv_app_id}"

    async def fetch_latest(self, pair: str, timeframe: str = "5m", source: str = "kraken") -> Optional[pd.DataFrame]:
        pair = pair.upper()
        if source == "kraken":
            kraken_pair = "XXBTZUSD" if pair == "BTCUSD" else "XETHZUSD" if pair == "ETHUSD" else pair
            return await self._fetch_from_kraken(kraken_pair)
        elif source == "deriv":
            return await self._fetch_from_deriv(pair)
        return None

    async def _fetch_from_kraken(self, pair: str) -> Optional[pd.DataFrame]:
        url = f"https://api.kraken.com/0/public/OHLC?pair={pair}"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=10) as response:
                    if response.status != 200: return None
                    json_data = await response.json()
                    if 'result' in json_data:
                        data_key = list(json_data['result'].keys())[0]
                        df = pd.DataFrame(json_data['result'][data_key], 
                                          columns=['timestamp', 'open', 'high', 'low', 'close', 'vwap', 'volume', 'trades'])
                        for col in ['open', 'high', 'low', 'close', 'volume']: df[col] = pd.to_numeric(df[col])
                        return df
        except Exception as e:
            logger.error(f"❌ Kraken Exception: {e}")
        return None

    async def _fetch_from_deriv(self, pair: str) -> Optional[pd.DataFrame]:
        logger.info(f"Attempting to fetch {pair} from Deriv...")
        # Deriv uses standard websocket JSON-RPC
        symbol = "frxBTCUSD" if pair == "BTCUSD" else "R_50" # Adjust symbol format
        msg = {"ticks_history": symbol, "style": "candles", "granularity": 300, "count": 100}
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(self.deriv_url) as ws:
                    await ws.send_json(msg)
                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            data = msg.json()
                            if "candles" in data:
                                df = pd.DataFrame(data["candles"])
                                # Rename Deriv columns to match your model's expected names
                                df = df.rename(columns={"epoch": "timestamp", "open": "open", "high": "high", "low": "low", "close": "close"})
                                df["volume"] = 0 # Deriv candle history often lacks volume in standard calls
                                return df
                            break
        except Exception as e:
            logger.error(f"❌ Deriv Exception: {e}")
        return None