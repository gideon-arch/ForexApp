# core/data_feed.py
import os
import aiohttp
import pandas as pd
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class DataFeedManager:
    def __init__(self):
        self.deriv_key = os.getenv("DERIV_API_KEY")
        self.kraken_key = os.getenv("KRAKEN_API_KEY")
        self.kraken_secret = os.getenv("KRAKEN_SECRET")

    async def fetch_latest(self, pair: str, timeframe: str = "5m", source: str = "kraken") -> Optional[pd.DataFrame]:
        pair = pair.upper()
        if source == "deriv":
            return await self._fetch_from_deriv(pair)
        elif source == "kraken":
            return await self._fetch_from_kraken(pair)
        return None

    async def _fetch_from_kraken(self, pair: str) -> Optional[pd.DataFrame]:
        logger.info(f"Attempting to fetch {pair} from Kraken...")
        # Replace this URL with the actual Kraken API endpoint you use
        url = f"https://api.kraken.com/0/public/OHLC?pair={pair}"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=10) as response:
                    if response.status != 200:
                        logger.error(f"❌ Kraken Error: Status {response.status} - {await response.text()}")
                        return None
                    data = await response.json()
                    # Add your parsing logic here
                    return pd.DataFrame(data)
        except Exception as e:
            logger.error(f"❌ Kraken Exception for {pair}: {str(e)}")
            return None

    async def _fetch_from_deriv(self, pair: str) -> Optional[pd.DataFrame]:
        logger.info(f"Attempting to fetch {pair} from Deriv...")
        try:
            # Replace with your actual Deriv logic
            return None
        except Exception as e:
            logger.error(f"❌ Deriv Exception for {pair}: {str(e)}")
            return None