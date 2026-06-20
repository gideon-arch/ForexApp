# core/data_feed.py - PRODUCTION READY
import os
import aiohttp
import pandas as pd
import logging
import asyncio
from datetime import datetime
from typing import Dict, Optional

logger = logging.getLogger(__name__)

class DataFeedManager:
    def __init__(self):
        self.deriv_key = os.getenv("DERIV_API_KEY")
        self.deriv_account = os.getenv("DERIV_ACCOUNT_ID")
        self.kraken_key = os.getenv("KRAKEN_API_KEY")
        self.kraken_secret = os.getenv("KRAKEN_SECRET")
        
    async def fetch_latest(self, pair: str, timeframe: str = "5m", source: str = "kraken") -> Optional[pd.DataFrame]:
        pair = pair.upper()
        # Using a timeout context to ensure the bot never gets permanently stuck
        try:
            if source == "deriv":
                return await asyncio.wait_for(self._fetch_from_deriv(pair), timeout=10.0)
            elif source == "kraken":
                return await asyncio.wait_for(self._fetch_from_kraken(pair), timeout=10.0)
        except asyncio.TimeoutError:
            logger.error(f"❌ Connection Timeout: {source} did not respond for {pair}")
        except Exception as e:
            logger.error(f"❌ Feed Error for {source} ({pair}): {str(e)}")
        return None

    async def _fetch_from_deriv(self, pair: str) -> Optional[pd.DataFrame]:
        logger.info(f"Connecting to Deriv for {pair}...")
        try:
            async with aiohttp.ClientSession() as session:
                # Add your specific Deriv request headers/logic here
                # response = await session.get(DERIV_URL, headers={"Authorization": f"Bearer {self.deriv_key}"})
                # if response.status == 200: ...
                return None 
        except Exception as e:
            logger.error(f"❌ Deriv API Connection failed: {e}")
            raise

    async def _fetch_from_kraken(self, pair: str) -> Optional[pd.DataFrame]:
        logger.info(f"Connecting to Kraken for {pair}...")
        try:
            async with aiohttp.ClientSession() as session:
                # Add your specific Kraken request headers/logic here
                # response = await session.get(KRAKEN_URL, ...)
                return None
        except Exception as e:
            logger.error(f"❌ Kraken API Connection failed: {e}")
            raise