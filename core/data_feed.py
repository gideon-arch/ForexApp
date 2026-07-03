# core/data_feed.py
# Data fetcher — Kraken only for BTCUSD

import os
import aiohttp
import pandas as pd
import logging
from typing import Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class DataFeedManager:
    def __init__(self):
        self.cache = {}
        self.cache_ttl = 30
        logger.info("DataFeedManager initialized | Kraken for BTCUSD only")

    async def fetch_latest(self, pair: str = "BTCUSD", timeframe: str = "3m") -> Optional[pd.DataFrame]:
        pair = pair.upper()
        cache_key = f"{pair}_{timeframe}"

        if cache_key in self.cache:
            cached_time, cached_df = self.cache[cache_key]
            if (datetime.now() - cached_time).seconds < self.cache_ttl:
                logger.debug(f"Cache hit for {pair}")
                return cached_df

        try:
            df = await self._fetch_from_kraken(pair, timeframe)
            if df is not None and len(df) >= 200:
                self.cache[cache_key] = (datetime.now(), df)
                logger.info(f"Fetched {len(df)} candles for {pair}")
                return df
            elif df is not None and len(df) > 0:
                logger.warning(f"Only {len(df)} candles for {pair} (need 200+)")
                return df
            else:
                return None
        except Exception as e:
            logger.error(f"Error fetching {pair}: {e}")
            return None

    async def fetch_historical(self, pair: str = "BTCUSD", timeframe: str = "3m", limit: int = 1000) -> Optional[pd.DataFrame]:
        pair = pair.upper()
        try:
            return await self._fetch_from_kraken(pair, timeframe, limit)
        except Exception as e:
            logger.error(f"Historical fetch error: {e}")
            return None

    async def _fetch_from_kraken(self, pair: str, timeframe: str = "3m", limit: int = 720) -> Optional[pd.DataFrame]:
        # Kraken uses XBT for Bitcoin, not BTC
        # Pair mapping: BTCUSD -> XXBTZUSD (Kraken's format)
        kraken_pair_map = {
            "BTCUSD": "XXBTZUSD",
            "BTC-USD": "XXBTZUSD",
            "XBTUSD": "XXBTZUSD",
        }
        kraken_pair = kraken_pair_map.get(pair, pair)

        # Kraken only supports: 1, 5, 15, 30, 60, 240, 1440, 10080, 21600
        # 3m is NOT supported by Kraken! Use 5m as fallback
        interval_map = {
            "1m": 1,
            "5m": 5,
            "15m": 15,
            "30m": 30,
            "1h": 60,
            "4h": 240,
            "1d": 1440,
            "1w": 10080,
            "2w": 21600
        }
        
        # Fallback 3m -> 5m since Kraken doesn't have 3m
        if timeframe == "3m":
            logger.warning(f"Kraken does not support 3m interval, falling back to 5m")
            interval = 5
        else:
            interval = interval_map.get(timeframe, 5)

        url = "https://api.kraken.com/0/public/OHLC"
        params = {"pair": kraken_pair, "interval": interval}

        logger.info(f"Fetching {kraken_pair} from Kraken | interval={interval} (requested: {timeframe})")

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=15) as response:
                    logger.info(f"Kraken response status: {response.status}")

                    if response.status != 200:
                        logger.error(f"Kraken HTTP {response.status}")
                        return None

                    json_data = await response.json()

                    if "error" in json_data and json_data["error"]:
                        logger.error(f"Kraken API error: {json_data['error']}")
                        return None

                    if "result" not in json_data:
                        logger.error("No 'result' in Kraken response")
                        return None

                    data_key = list(json_data["result"].keys())[0]
                    raw_data = json_data["result"][data_key][-limit:]

                    logger.info(f"Kraken returned {len(raw_data)} raw candles for {kraken_pair}")

                    df = pd.DataFrame(raw_data, columns=[
                        "timestamp", "open", "high", "low", "close",
                        "vwap", "volume", "trades"
                    ])

                    for col in ["open", "high", "low", "close", "volume", "vwap"]:
                        df[col] = pd.to_numeric(df[col], errors="coerce")

                    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
                    df.dropna(inplace=True)

                    logger.info(f"Kraken success: {len(df)} valid rows for {kraken_pair}")
                    return df

        except Exception as e:
            logger.error(f"Kraken fetch error for {kraken_pair}: {e}")
            return None
