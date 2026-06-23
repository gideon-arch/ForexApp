# core/data_feed.py
# Multi-source data fetcher with API authentication

import os
import aiohttp
import pandas as pd
import logging
from typing import Optional
from datetime import datetime
import time
import hmac
import hashlib
import base64

logger = logging.getLogger(__name__)

class DataFeedManager:
    def __init__(self):
        self.cache = {}
        self.cache_ttl = 30
        
        # API Keys from environment
        self.deriv_api_key = os.getenv('DERIV_API_KEY')
        self.deriv_account_id = os.getenv('DERIV_ACCOUNT_ID')
        self.kraken_api_key = os.getenv('KRAKEN_API_KEY')
        self.kraken_secret = os.getenv('KRAKEN_SECRET')
        
        logger.info("DataFeedManager initialized")

    async def fetch_latest(self, pair: str, timeframe: str = "5m", source: str = "kraken") -> Optional[pd.DataFrame]:
        pair = pair.upper()
        cache_key = f"{pair}_{timeframe}_{source}"
        
        if cache_key in self.cache:
            cached_time, cached_df = self.cache[cache_key]
            if (datetime.now() - cached_time).seconds < self.cache_ttl:
                return cached_df
        
        try:
            if source.lower() == "deriv":
                df = await self._fetch_from_deriv_authenticated(pair, timeframe)
            elif source.lower() == "kraken":
                kraken_pair = "XXBTZUSD" if pair == "BTCUSD" else "XETHZUSD"
                df = await self._fetch_from_kraken_authenticated(kraken_pair, timeframe)
            else:
                logger.error(f"Unknown source: {source}")
                return None
            
            if df is not None and len(df) >= 200:
                self.cache[cache_key] = (datetime.now(), df)
                logger.info(f"Fetched {len(df)} candles for {pair} from {source}")
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
            if source.lower() == "deriv":
                return await self._fetch_from_deriv_authenticated(pair, timeframe, limit)
            elif source.lower() == "kraken":
                kraken_pair = "XXBTZUSD" if pair == "BTCUSD" else "XETHZUSD"
                return await self._fetch_from_kraken_authenticated(kraken_pair, timeframe, limit)
            else:
                return None
        except Exception as e:
            logger.error(f"Historical fetch error: {e}")
            return None

    async def _fetch_from_kraken_authenticated(self, pair: str, timeframe: str = "5m", limit: int = 720) -> Optional[pd.DataFrame]:
        """Fetch from Kraken using public API (no auth needed for OHLC)"""
        interval_map = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440}
        interval = interval_map.get(timeframe, 5)
        
        url = "https://api.kraken.com/0/public/OHLC"
        params = {
            'pair': pair,
            'interval': interval
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=15) as response:
                    if response.status != 200:
                        logger.error(f"Kraken HTTP {response.status}")
                        return None
                    
                    json_data = await response.json()
                    
                    if 'error' in json_data and json_data['error']:
                        logger.error(f"Kraken API error: {json_data['error']}")
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
            logger.error(f"Kraken fetch error: {e}")
            return None

    async def _fetch_from_deriv_authenticated(self, pair: str, timeframe: str = "5m", limit: int = 1000) -> Optional[pd.DataFrame]:
        """Fetch from Deriv using WebSocket/REST API"""
        # Deriv uses a simpler REST API for ticks
        deriv_symbol = f"{pair[:3]}/{pair[3:]}"  # BTC/USD format
        
        # Try the ticks endpoint first
        url = "https://ticks.deriv.com/api/v1/ticks_history"
        params = {
            'ticks_history': deriv_symbol,
            'granularity': 300,
            'count': min(limit, 5000),
            'end': 'latest',
            'style': 'candles'
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                headers = {}
                if self.deriv_api_key:
                    headers['Authorization'] = f'Bearer {self.deriv_api_key}'
                
                async with session.get(url, params=params, headers=headers, timeout=15) as response:
                    if response.status != 200:
                        logger.warning(f"Deriv HTTP {response.status}, trying fallback")
                        return await self._fetch_from_deriv_fallback(pair, timeframe, limit)
                    
                    content_type = response.headers.get('content-type', '')
                    if 'application/json' not in content_type:
                        logger.warning(f"Deriv returned {content_type}, trying fallback")
                        return await self._fetch_from_deriv_fallback(pair, timeframe, limit)
                    
                    json_data = await response.json()
                    
                    if 'candles' not in json_data:
                        logger.warning("No candles in Deriv response, trying fallback")
                        return await self._fetch_from_deriv_fallback(pair, timeframe, limit)
                    
                    candles = json_data['candles']
                    df = pd.DataFrame(candles)
                    
                    df.rename(columns={
                        'epoch': 'timestamp',
                        'open': 'open',
                        'high': 'high',
                        'low': 'low',
                        'close': 'close'
                    }, inplace=True)
                    
                    if 'volume' not in df.columns:
                        df['volume'] = 0.0
                    
                    for col in ['open', 'high', 'low', 'close', 'volume']:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                    df.dropna(inplace=True)
                    
                    return df
                    
        except Exception as e:
            logger.error(f"Deriv error: {e}, trying fallback")
            return await self._fetch_from_deriv_fallback(pair, timeframe, limit)

    async def _fetch_from_deriv_fallback(self, pair: str, timeframe: str = "5m", limit: int = 1000) -> Optional[pd.DataFrame]:
        """Fallback to Kraken if Deriv fails"""
        logger.info("Using Kraken as fallback for Deriv")
        kraken_pair = "XXBTZUSD" if pair == "BTCUSD" else "XETHZUSD"
        return await self._fetch_from_kraken_authenticated(kraken_pair, timeframe, limit)