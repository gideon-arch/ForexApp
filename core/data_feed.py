# core/data_feed.py
# Multi-source data fetcher with fallback and caching

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

    async def fetch_latest(self, pair: str, timeframe: str = "5m", source: str = "kraken") -> Optional[pd.DataFrame]:
        pair = pair.upper()
        cache_key = f"{pair}_{timeframe}_{source}"
        
        if cache_key in self.cache:
            cached_time, cached_df = self.cache[cache_key]
            if (datetime.now() - cached_time).seconds < self.cache_ttl:
                return cached_df
        
        try:
            if source.lower() == "deriv":
                df = await self._fetch_from_deriv(pair, timeframe)
            elif source.lower() == "kraken":
                kraken_pair = "XXBTZUSD" if pair == "BTCUSD" else "XETHZUSD"
                df = await self._fetch_from_kraken(kraken_pair, timeframe)
            else:
                return None
            
            if df is not None and len(df) >= 200:
                self.cache[cache_key] = (datetime.now(), df)
                return df
            return df
            
        except Exception as e:
            logger.error(f"Error fetching {pair}: {e}")
            return None

    async def fetch_historical(self, pair: str, timeframe: str = "5m", source: str = "kraken", limit: int = 1000) -> Optional[pd.DataFrame]:
        pair = pair.upper()
        try:
            if source.lower() == "deriv":
                return await self._fetch_from_deriv(pair, timeframe, limit)
            elif source.lower() == "kraken":
                kraken_pair = "XXBTZUSD" if pair == "BTCUSD" else "XETHZUSD"
                return await self._fetch_from_kraken(kraken_pair, timeframe, limit)
            return None
        except Exception as e:
            logger.error(f"Historical fetch error: {e}")
            return None

    async def _fetch_from_kraken(self, pair: str, timeframe: str = "5m", limit: int = 720) -> Optional[pd.DataFrame]:
        interval_map = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440}
        interval = interval_map.get(timeframe, 5)
        url = f"https://api.kraken.com/0/public/OHLC?pair={pair}&interval={interval}"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=15) as response:
                    if response.status != 200:
                        return None
                    json_data = await response.json()
                    if 'error' in json_data and json_data['error']:
                        return None
                    if 'result' not in json_data:
                        return None
                    
                    data_key = list(json_data['result'].keys())[0]
                    raw_data = json_data['result'][data_key][-limit:]
                    
                    df = pd.DataFrame(raw_data, columns=[
                        'timestamp', 'open', 'high', 'low', 'close', 
                        'vwap', 'volume', 'trades'
                    ])
                    
                    for col in ['open', 'high', 'low', 'close', 'volume', 'vwap']:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                    df.dropna(inplace=True)
                    return df
        except Exception as e:
            logger.error(f"Kraken error: {e}")
            return None

    async def _fetch_from_deriv(self, pair: str, timeframe: str = "5m", limit: int = 1000) -> Optional[pd.DataFrame]:
        deriv_symbol = f"{pair[:3]}/{pair[3:]}"
        granularity_map = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600, "4h": 14400, "1d": 86400}
        granularity = granularity_map.get(timeframe, 300)
        url = f"https://api.deriv.com/api/ticks_history?ticks_history={deriv_symbol}&granularity={granularity}&count={limit}&end=latest&style=candles"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=15) as response:
                    if response.status != 200:
                        kraken_pair = "XXBTZUSD" if pair == "BTCUSD" else "XETHZUSD"
                        return await self._fetch_from_kraken(kraken_pair, timeframe, limit)
                    
                    json_data = await response.json()
                    if 'candles' not in json_data:
                        kraken_pair = "XXBTZUSD" if pair == "BTCUSD" else "XETHZUSD"
                        return await self._fetch_from_kraken(kraken_pair, timeframe, limit)
                    
                    candles = json_data['candles']
                    df = pd.DataFrame(candles)
                    df.rename(columns={
                        'epoch': 'timestamp', 'open': 'open', 'high': 'high',
                        'low': 'low', 'close': 'close'
                    }, inplace=True)
                    
                    if 'volume' not in df.columns:
                        df['volume'] = 0.0
                    
                    for col in ['open', 'high', 'low', 'close', 'volume']:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                    df.dropna(inplace=True)
                    return df
        except Exception as e:
            logger.error(f"Deriv error: {e}")
            kraken_pair = "XXBTZUSD" if pair == "BTCUSD" else "XETHZUSD"
            return await self._fetch_from_kraken(kraken_pair, timeframe, limit)