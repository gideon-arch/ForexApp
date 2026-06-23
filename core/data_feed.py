# core/data_feed.py
# Data fetcher - Deriv for BTCUSD, Kraken for ETHUSD

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
        logger.info("DataFeedManager initialized | Deriv for BTCUSD, Kraken for ETHUSD")

    async def fetch_latest(self, pair: str, timeframe: str = "5m", source: str = "kraken") -> Optional[pd.DataFrame]:
        pair = pair.upper()
        cache_key = f"{pair}_{timeframe}_{source}"
        
        if cache_key in self.cache:
            cached_time, cached_df = self.cache[cache_key]
            if (datetime.now() - cached_time).seconds < self.cache_ttl:
                logger.debug(f"Cache hit for {pair}")
                return cached_df
        
        try:
            if pair == "BTCUSD":
                # Use Deriv for BTCUSD
                df = await self._fetch_from_deriv(pair, timeframe)
            elif pair == "ETHUSD":
                # Use Kraken for ETHUSD
                df = await self._fetch_from_kraken("XETHZUSD", timeframe)
            else:
                logger.error(f"Unknown pair: {pair}")
                return None
            
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

    async def fetch_historical(self, pair: str, timeframe: str = "5m", source: str = "kraken", limit: int = 1000) -> Optional[pd.DataFrame]:
        pair = pair.upper()
        try:
            if pair == "BTCUSD":
                return await self._fetch_from_deriv(pair, timeframe, limit)
            elif pair == "ETHUSD":
                return await self._fetch_from_kraken("XETHZUSD", timeframe, limit)
            else:
                return None
        except Exception as e:
            logger.error(f"Historical fetch error: {e}")
            return None

    async def _fetch_from_kraken(self, pair: str, timeframe: str = "5m", limit: int = 720) -> Optional[pd.DataFrame]:
        """Fetch from Kraken public API"""
        interval_map = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440}
        interval = interval_map.get(timeframe, 5)
        
        url = "https://api.kraken.com/0/public/OHLC"
        params = {'pair': pair, 'interval': interval}
        
        logger.info(f"Fetching {pair} from Kraken | interval={interval}")
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=15) as response:
                    logger.info(f"Kraken response status: {response.status}")
                    
                    if response.status != 200:
                        logger.error(f"Kraken HTTP {response.status}")
                        return None
                    
                    json_data = await response.json()
                    
                    if 'error' in json_data and json_data['error']:
                        logger.error(f"Kraken API error: {json_data['error']}")
                        return None
                    
                    if 'result' not in json_data:
                        logger.error("No 'result' in Kraken response")
                        return None
                    
                    data_key = list(json_data['result'].keys())[0]
                    raw_data = json_data['result'][data_key][-limit:]
                    
                    logger.info(f"Kraken returned {len(raw_data)} raw candles for {pair}")
                    
                    df = pd.DataFrame(raw_data, columns=[
                        'timestamp', 'open', 'high', 'low', 'close', 
                        'vwap', 'volume', 'trades'
                    ])
                    
                    for col in ['open', 'high', 'low', 'close', 'volume', 'vwap']:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                    df.dropna(inplace=True)
                    
                    logger.info(f"Kraken success: {len(df)} valid rows for {pair}")
                    return df
                    
        except Exception as e:
            logger.error(f"Kraken fetch error for {pair}: {e}")
            return None

    async def _fetch_from_deriv(self, pair: str, timeframe: str = "5m", limit: int = 1000) -> Optional[pd.DataFrame]:
        """Fetch from Deriv API using their WebSocket/REST endpoint"""
        logger.info(f"Fetching {pair} from Deriv")
        
        # Map timeframe to seconds
        granularity_map = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600, "4h": 14400, "1d": 86400}
        granularity = granularity_map.get(timeframe, 300)
        
        # Use Deriv's API endpoint
        # Try the newer API endpoint first
        api_url = "https://api.deriv.com/api/ticks_history"
        
        params = {
            'ticks_history': 'BTC/USD',
            'adjust_start_time': 1,
            'count': min(limit, 5000),
            'end': 'latest',
            'start': 1,
            'style': 'candles',
            'granularity': granularity
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                logger.info(f"Deriv request: {api_url} with params: {params}")
                
                async with session.get(api_url, params=params, timeout=20) as response:
                    logger.info(f"Deriv response status: {response.status}")
                    logger.info(f"Deriv content-type: {response.headers.get('content-type', 'unknown')}")
                    
                    if response.status != 200:
                        logger.error(f"Deriv HTTP {response.status}")
                        return await self._deriv_fallback_to_kraken(pair, timeframe, limit)
                    
                    # Check if response is JSON
                    content_type = response.headers.get('content-type', '')
                    if 'application/json' not in content_type:
                        text = await response.text()
                        logger.error(f"Deriv returned non-JSON: {content_type[:50]}")
                        logger.error(f"Response preview: {text[:200]}")
                        return await self._deriv_fallback_to_kraken(pair, timeframe, limit)
                    
                    json_data = await response.json()
                    logger.info(f"Deriv JSON keys: {list(json_data.keys())}")
                    
                    if 'error' in json_data and json_data['error']:
                        logger.error(f"Deriv API error: {json_data['error']}")
                        return await self._deriv_fallback_to_kraken(pair, timeframe, limit)
                    
                    if 'candles' not in json_data:
                        logger.error(f"No 'candles' in Deriv response. Keys: {list(json_data.keys())}")
                        return await self._deriv_fallback_to_kraken(pair, timeframe, limit)
                    
                    candles = json_data['candles']
                    logger.info(f"Deriv returned {len(candles)} candles")
                    
                    df = pd.DataFrame(candles)
                    
                    # Rename columns
                    column_map = {
                        'epoch': 'timestamp',
                        'open': 'open',
                        'high': 'high',
                        'low': 'low',
                        'close': 'close'
                    }
                    df.rename(columns=column_map, inplace=True)
                    
                    if 'volume' not in df.columns:
                        df['volume'] = 0.0
                    
                    for col in ['open', 'high', 'low', 'close', 'volume']:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                    df.dropna(inplace=True)
                    
                    logger.info(f"Deriv success: {len(df)} valid rows")
                    return df
                    
        except Exception as e:
            logger.error(f"Deriv error: {e}")
            return await self._deriv_fallback_to_kraken(pair, timeframe, limit)

    async def _deriv_fallback_to_kraken(self, pair: str, timeframe: str, limit: int) -> Optional[pd.DataFrame]:
        """Fallback to Kraken when Deriv fails"""
        logger.warning(f"Deriv failed for {pair}, falling back to Kraken")
        kraken_pair = "XXBTZUSD" if pair == "BTCUSD" else "XETHZUSD"
        return await self._fetch_from_kraken(kraken_pair, timeframe, limit)