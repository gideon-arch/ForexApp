# core/data_feed.py
# Kraken API Engine for Pure Crypto Infrastructure (BTC/USD + ETH/USD)

import asyncio
import json
import os
from datetime import datetime, timedelta
from typing import Dict
import aiohttp
import pandas as pd


class DataFeedManager:
    """Universal data feed manager

    - Track institutional Crypto pipelines natively using Kraken API
    """

    def __init__(self):
        # Kraken API base connectivity
        self.kraken_base_url = "https://api.kraken.com"

        # FETCH RSI THRESHOLDS FROM ENVIRONMENT (Defaults to 25/30 and 65/70 zones)
        self.rsi_buy_low = float(os.getenv("RSI_BUY_LOW", 30))
        self.rsi_sell_high = float(os.getenv("RSI_SELL_HIGH", 65))

        # Base asset configurations
        self.symbols = {
            "ETHUSD": {
                "source": "kraken",
                "symbol": "XETHZUSD",  # Kraken's institutional standard ticker for Ethereum
            },
            "BTCUSD": {
                "source": "kraken",
                "symbol": "XXBTZUSD",  # Kraken's institutional standard ticker for Bitcoin
            },
        }

        # Map human-readable timeframes to Kraken specific OHLC intervals (in minutes)
        self.kraken_timeframe_map = {
            "1m": 1,
            "5m": 5,
            "15m": 15,
            "30m": 30,
            "1h": 60,
            "4h": 240,
            "1d": 1440,
        }

        print(
            f"⚙️ Loaded RSI Thresholds -> Buy Low (Oversold): {self.rsi_buy_low}, Sell High (Overbought): {self.rsi_sell_high}"
        )

    async def fetch_latest(
        self, pair: str, timeframe: str = "5m"
    ) -> pd.DataFrame:
        """Fetch latest price data matching the requested timeframe"""
        pair = pair.upper()
        if pair not in self.symbols:
            raise ValueError(
                f"Pair {pair} not supported within current Crypto schema"
            )

        config = self.symbols[pair].copy()

        if config["source"] == "kraken":
            # FIXED: Safe dictionary extraction fallback ensuring seamless signature mapping
            config["interval"] = self.kraken_timeframe_map.get(
                timeframe, 5
            )  # Default to 5m
            return await self._fetch_kraken(pair, config)
        else:
            raise ValueError(
                f"Unknown source engine configuration: {config['source']}"
            )

    async def _fetch_kraken(self, pair: str, config: Dict) -> pd.DataFrame:
        """Fetch Spot assets from Kraken API (Most recent data up to 720 bars)"""
        symbol = config["symbol"]
        interval = config["interval"]
        url = f"{self.kraken_base_url}/0/public/OHLC"

        # Calculate appropriate fallback history window based on timeframe
        lookback_hours = 500 if interval >= 60 else 48
        since = int(
            (datetime.now() - timedelta(hours=lookback_hours)).timestamp()
        )

        params = {"pair": symbol, "interval": interval, "since": since}

        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                if response.status != 200:
                    raise Exception(
                        f"Kraken API connection error: {response.status}"
                    )

                data = await response.json()
                if data.get("error"):
                    raise Exception(f"Kraken execution error: {data['error']}")

                result = data.get("result", {})

                pair_key = [k for k in result.keys() if k != "last"]
                ohlc_data = result.get(pair_key[0], []) if pair_key else []

                if not ohlc_data:
                    raise Exception(
                        f"No OHLC candle arrays returned from Kraken for asset {symbol}"
                    )

                df_data = []
                for candle in ohlc_data:
                    df_data.append(
                        {
                            "datetime": datetime.fromtimestamp(int(candle[0])),
                            "open": float(candle[1]),
                            "high": float(candle[2]),
                            "low": float(candle[3]),
                            "close": float(candle[4]),
                            "volume": float(candle[6]),
                        }
                    )

                df = pd.DataFrame(df_data)
                df.set_index("datetime", inplace=True)
                print(f"✅ Kraken Layer: Fetched {len(df)} live candles for {pair}")
                return df

    async def fetch_historical(self, pair: str, days: int = 180) -> pd.DataFrame:
        """Fetches heavy structural lookback datasets to hydrate LSTM model arrays

        Paginates execution past the 720 item hard window cap limits natively.
        """
        pair = pair.upper()
        if pair not in self.symbols:
            raise ValueError(
                f"Pair {pair} not supported within current Crypto schema"
            )

        config = self.symbols[pair].copy()
        config["interval"] = 60  # 1h bars for deep historical matrix calculations
        return await self._fetch_kraken_historical_paginated(
            pair, config, days
        )

    async def _fetch_kraken_historical_paginated(
        self, pair: str, config: Dict, days: int
    ) -> pd.DataFrame:
        """FIXED: Robust chunk-based pagination loop that forces parameters past

        Kraken's edge lookback limits using real parsed candle markers.
        """
        symbol = config["symbol"]
        interval = config["interval"]

        target_start_time = datetime.now() - timedelta(days=days)
        current_since = int(target_start_time.timestamp())
        now_ts = int(datetime.now().timestamp())

        url = f"{self.kraken_base_url}/0/public/OHLC"
        all_candles = []

        async with aiohttp.ClientSession() as session:
            while current_since < now_ts:
                params = {
                    "pair": symbol,
                    "interval": interval,
                    "since": current_since,
                }

                async with session.get(url, params=params) as response:
                    if response.status != 200:
                        print(
                            f"⚠️ Pagination chunk step failed. Status: {response.status}. Finalizing current arrays."
                        )
                        break

                    data = await response.json()
                    if data.get("error"):
                        raise Exception(
                            f"Kraken historical chunk extraction error: {data['error']}"
                        )

                    result = data.get("result", {})
                    pair_key = [k for k in result.keys() if k != "last"]
                    ohlc_data = result.get(pair_key[0], []) if pair_key else []

                    # Break if no new data array is returned to prevent infinite loops
                    if not ohlc_data:
                        break

                    chunk_candles = []
                    last_candle_ts = current_since

                    for candle in ohlc_data:
                        ts = int(candle[0])
                        chunk_candles.append(
                            {
                                "datetime": datetime.fromtimestamp(ts),
                                "open": float(candle[1]),
                                "high": float(candle[2]),
                                "low": float(candle[3]),
                                "close": float(candle[4]),
                                "volume": float(candle[6]),
                            }
                        )
                        last_candle_ts = ts

                    # If we only got back data we already have or progress stalled, terminate
                    if last_candle_ts <= current_since:
                        break

                    all_candles.extend(chunk_candles)

                    # FIXED: Advance time parameter using the absolute last candle timestamp processed + 1 second
                    # This prevents getting stuck on Kraken's internal high-precision data boundaries.
                    current_since = last_candle_ts + 1

                # Protect Kraken public API call rate limits
                await asyncio.sleep(0.3)

        if not all_candles:
            raise Exception(
                f"Historical hydration matrix build failed empty for target symbol asset: {symbol}"
            )

        df = pd.DataFrame(all_candles)
        df.set_index("datetime", inplace=True)

        # De-duplicate entries intersecting at chunk borders
        df = df[~df.index.duplicated(keep="first")]
        df.sort_index(inplace=True)

        print(
            f"✅ Kraken Historical Layer: Fully Hydrated {len(df)} continuous training blocks for {pair}"
        )
        return df

    async def get_current_price(self, pair: str) -> float:
        """Core utility function to extract current valuation wicks"""
        df = await self.fetch_latest(pair)
        return df["close"].iloc[-1]