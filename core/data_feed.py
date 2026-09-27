# core/data_feed.py
# Data fetcher — Kraken + Deriv + Binance fallback

import os
import asyncio
import ssl
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
        self._ssl_context = None
        logger.info("DataFeedManager initialized | Kraken + Deriv + Binance Fallback")

    def _get_ssl_context(self):
        if self._ssl_context is None:
            try:
                import certifi
                self._ssl_context = ssl.create_default_context(cafile=certifi.where())
            except Exception:
                self._ssl_context = ssl.create_default_context()
        return self._ssl_context

    def _deriv_to_kraken_tf(self, deriv_granularity) -> str:
        mapping = {
            "60": "1m", "120": "1m", "180": "3m", "300": "5m",
            "600": "10m", "900": "15m", "1800": "30m",
            "3600": "1h", "14400": "4h", "86400": "1d"
        }
        return mapping.get(str(deriv_granularity), "5m")

    async def fetch_latest(self, pair: str = "BTCUSD", timeframe: str = "5m",
                           source: str = "kraken", deriv_symbol: str = None,
                           fallback: bool = True) -> Optional[pd.DataFrame]:
        cache_key = f"{source}_{pair}_{deriv_symbol or ''}_{timeframe}"
        if cache_key in self.cache:
            cached_time, cached_df = self.cache[cache_key]
            if (datetime.now() - cached_time).seconds < self.cache_ttl:
                logger.debug(f"Cache hit for {cache_key}")
                return cached_df

        df = None
        tried = []

        # Try primary source
        try:
            if source == "deriv":
                granularity = int(timeframe) if str(timeframe).isdigit() else 300
                df = await self._fetch_from_deriv(deriv_symbol or pair, granularity)
                tried.append("deriv")
            else:
                df = await self._fetch_from_kraken(pair, timeframe)
                tried.append("kraken")
        except Exception as e:
            logger.error(f"Primary source {source} failed for {pair}: {e}")

        # Fallback chain
        if df is None and fallback:
            # BTCUSD fallback: Kraken -> Binance
            if "BTC" in pair.upper() and "kraken" in tried:
                logger.warning(f"🔄 Kraken failed, trying Binance for {pair}...")
                try:
                    df = await self._fetch_from_binance(pair, timeframe)
                    tried.append("binance")
                    if df is not None:
                        logger.info(f"✅ Binance fallback succeeded for {pair}")
                except Exception as e:
                    logger.error(f"Binance fallback failed: {e}")

            # EURUSD fallback: Deriv -> Kraken
            # BUG FIX: this used to pass the raw Deriv symbol ("frxEURUSD")
            # straight to Kraken, which has no such pair — so the fallback
            # always silently failed. Strip the "frx" prefix first.
            if "EUR" in pair.upper() and "deriv" in tried:
                kraken_fallback_pair = pair.upper().replace("FRX", "")
                logger.warning(f"🔄 Deriv failed, trying Kraken for {kraken_fallback_pair}...")
                try:
                    kraken_tf = self._deriv_to_kraken_tf(timeframe) if str(timeframe).isdigit() else timeframe
                    df = await self._fetch_from_kraken(kraken_fallback_pair, kraken_tf)
                    tried.append("kraken")
                    if df is not None:
                        logger.info(f"✅ Kraken fallback succeeded for {kraken_fallback_pair}")
                except Exception as e:
                    logger.error(f"Kraken fallback failed: {e}")

        if df is not None and len(df) >= 50:
            self.cache[cache_key] = (datetime.now(), df)
            logger.info(f"Fetched {len(df)} candles for {pair} ({tried[-1]})")
            return df
        elif df is not None and len(df) > 0:
            logger.warning(f"Only {len(df)} candles for {pair} (need 50+)")
            return df
        else:
            return None

    async def fetch_historical(self, pair: str = "BTCUSD", timeframe: str = "5m",
                               source: str = "kraken", deriv_symbol: str = None,
                               limit: int = 1000) -> Optional[pd.DataFrame]:
        try:
            if source == "deriv":
                granularity = int(timeframe) if str(timeframe).isdigit() else 300
                return await self._fetch_from_deriv(deriv_symbol or pair, granularity, count=limit)
            else:
                return await self._fetch_from_kraken(pair, timeframe, limit)
        except Exception as e:
            logger.error(f"Historical fetch error: {e}")
            return None

    # ========== KRAKEN ==========
    async def _fetch_from_kraken(self, pair: str, timeframe: str = "5m", limit: int = 720) -> Optional[pd.DataFrame]:
        kraken_pair_map = {
            "BTCUSD": "XXBTZUSD", "BTC-USD": "XXBTZUSD", "XBTUSD": "XXBTZUSD",
            "EURUSD": "EURUSD", "EUR-USD": "EURUSD",
        }
        kraken_pair = kraken_pair_map.get(pair.upper(), pair.upper())

        interval_map = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "4h": 240, "1d": 1440}
        interval = interval_map.get(timeframe, 5)

        url = "https://api.kraken.com/0/public/OHLC"
        params = {"pair": kraken_pair, "interval": interval}

        try:
            connector = aiohttp.TCPConnector(ssl=self._get_ssl_context())
            async with aiohttp.ClientSession(connector=connector) as session:
                async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=20)) as response:
                    if response.status != 200:
                        logger.error(f"Kraken HTTP {response.status}")
                        return None
                    json_data = await response.json()
                    if "error" in json_data and json_data["error"]:
                        logger.error(f"Kraken API error: {json_data['error']}")
                        return None
                    if "result" not in json_data:
                        return None

                    data_key = list(json_data["result"].keys())[0]
                    raw_data = json_data["result"][data_key][-limit:]

                    df = pd.DataFrame(raw_data, columns=[
                        "timestamp", "open", "high", "low", "close", "vwap", "volume", "trades"
                    ])
                    for col in ["open", "high", "low", "close", "volume", "vwap"]:
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
                    df.dropna(inplace=True)
                    return df
        except Exception as e:
            logger.error(f"Kraken fetch error for {kraken_pair}: {type(e).__name__}: {e}")
            return None

    # ========== BINANCE ==========
    async def _fetch_from_binance(self, pair: str, timeframe: str = "5m", limit: int = 500) -> Optional[pd.DataFrame]:
        """Binance fallback for BTCUSD — fast and reliable"""
        binance_symbol = "BTCUSDT" if "BTC" in pair.upper() else pair.upper().replace("-", "")
        
        interval_map = {"1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m", "1h": "1h", "4h": "4h", "1d": "1d"}
        interval = interval_map.get(timeframe, "5m")

        url = "https://api.binance.com/api/v3/klines"
        params = {"symbol": binance_symbol, "interval": interval, "limit": limit}

        try:
            connector = aiohttp.TCPConnector(ssl=self._get_ssl_context())
            async with aiohttp.ClientSession(connector=connector) as session:
                async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=15)) as response:
                    if response.status != 200:
                        logger.error(f"Binance HTTP {response.status}")
                        return None
                    
                    raw_data = await response.json()
                    if not isinstance(raw_data, list):
                        logger.error(f"Binance unexpected response: {raw_data}")
                        return None

                    df = pd.DataFrame(raw_data, columns=[
                        "timestamp", "open", "high", "low", "close", "volume",
                        "close_time", "quote_volume", "trades", "taker_buy", "taker_buy_quote", "ignore"
                    ])
                    
                    for col in ["open", "high", "low", "close", "volume"]:
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                    
                    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                    df["vwap"] = df["close"]
                    df["trades"] = pd.to_numeric(df["trades"], errors="coerce").fillna(0).astype(int)
                    df = df[["timestamp", "open", "high", "low", "close", "volume", "vwap", "trades"]]
                    df.dropna(inplace=True)
                    return df
        except Exception as e:
            logger.error(f"Binance fetch error: {type(e).__name__}: {e}")
            return None

    # ========== DERIV ==========
    async def _fetch_from_deriv(self, symbol: str, granularity: int, count: int = 200) -> Optional[pd.DataFrame]:
        import websockets
        import json

        uri = "wss://ws.derivws.com/websockets/v3?app_id=1089"

        try:
            async with websockets.connect(uri, ping_interval=20, ping_timeout=10) as ws:
                req = {
                    "ticks_history": symbol,
                    "adjust_start_time": 1,
                    "count": count,
                    "end": "latest",
                    "start": 1,
                    "style": "candles",
                    "granularity": granularity
                }
                await ws.send(json.dumps(req))
                response = await asyncio.wait_for(ws.recv(), timeout=15.0)
                data = json.loads(response)

                if data.get("msg_type") != "candles":
                    err = data.get("error", {})
                    logger.error(f"Deriv API error for {symbol}: {err}")
                    return None

                candles = data.get("candles", [])
                if not candles:
                    logger.error(f"No candles from Deriv for {symbol}")
                    return None

                df = pd.DataFrame(candles)
                df["timestamp"] = pd.to_datetime(df["epoch"], unit="s")

                for col in ["open", "high", "low", "close"]:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

                df["volume"] = 0.0
                df["vwap"] = df["close"]
                df["trades"] = 0

                df = df[["timestamp", "open", "high", "low", "close", "volume", "vwap", "trades"]]
                df.dropna(inplace=True)
                logger.info(f"Deriv: {len(df)} candles for {symbol} ({granularity}s)")
                return df

        except Exception as e:
            logger.error(f"Deriv fetch error for {symbol}: {type(e).__name__}: {e}")
            return None