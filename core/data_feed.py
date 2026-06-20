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
        # Note: Kraken pair names often need internal formatting (e.g., XXBTZUSD for BTC/USD)
        url = f"https://api.kraken.com/0/public/OHLC?pair={pair}"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=10) as response:
                    if response.status != 200:
                        logger.error(f"❌ Kraken Error: Status {response.status}")
                        return None
                    
                    json_data = await response.json()
                    
                    # --- FIXED DATA EXTRACTION ---
                    if 'result' in json_data:
                        # Kraken returns a dict where the key is the pair name (e.g., 'XXBTZUSD')
                        # We get the first key dynamically to ensure we get the data
                        data_key = list(json_data['result'].keys())[0]
                        raw_data = json_data['result'][data_key]
                        
                        # Define column names based on Kraken OHLCVT format
                        columns = ['timestamp', 'open', 'high', 'low', 'close', 'vwap', 'volume', 'trades']
                        
                        # Convert to DataFrame
                        df = pd.DataFrame(raw_data, columns=columns)
                        
                        # Ensure numeric types
                        for col in ['open', 'high', 'low', 'close', 'volume']:
                            df[col] = pd.to_numeric(df[col])
                            
                        return df
                    else:
                        logger.error(f"❌ Kraken Error: 'result' key missing in response: {json_data}")
                        return None
                        
        except Exception as e:
            logger.error(f"❌ Kraken Exception for {pair}: {str(e)}")
            return None

    async def _fetch_from_deriv(self, pair: str) -> Optional[pd.DataFrame]:
        logger.info(f"Attempting to fetch {pair} from Deriv...")
        try:
            # Add your actual Deriv API logic here
            return None
        except Exception as e:
            logger.error(f"❌ Deriv Exception for {pair}: {str(e)}")
            return None