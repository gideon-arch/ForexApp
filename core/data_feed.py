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
        # Ensure pairs are treated correctly
        self.tracked_pairs = os.getenv("TRACKED_PAIRS", "ETHUSD,BTCUSD").split(",")

    async def fetch_latest(self, pair: str, timeframe: str = "5m", source: str = "kraken") -> Optional[pd.DataFrame]:
        pair = pair.upper()
        # Handle pair formatting for Kraken (API often expects 'XXBTZUSD' for 'BTCUSD')
        if source == "kraken":
            # Basic mapping; if you have specific pairs, add them here
            kraken_pair = "XXBTZUSD" if pair == "BTCUSD" else "XETHZUSD" if pair == "ETHUSD" else pair
            return await self._fetch_from_kraken(kraken_pair)
        elif source == "deriv":
            logger.warning(f"Deriv fetching not yet implemented for {pair}")
            return None
        return None

    async def _fetch_from_kraken(self, pair: str) -> Optional[pd.DataFrame]:
        logger.info(f"Attempting to fetch {pair} from Kraken...")
        url = f"https://api.kraken.com/0/public/OHLC?pair={pair}"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=10) as response:
                    if response.status != 200:
                        logger.error(f"❌ Kraken Error: Status {response.status}")
                        return None
                    
                    json_data = await response.json()
                    
                    if 'result' in json_data:
                        data_key = list(json_data['result'].keys())[0]
                        raw_data = json_data['result'][data_key]
                        
                        columns = ['timestamp', 'open', 'high', 'low', 'close', 'vwap', 'volume', 'trades']
                        df = pd.DataFrame(raw_data, columns=columns)
                        
                        # Convert columns to numeric, crucial for the LSTM model
                        for col in ['open', 'high', 'low', 'close', 'volume']:
                            df[col] = pd.to_numeric(df[col])
                            
                        # Ensure 'close' exists - this fixes your KeyError
                        if 'close' in df.columns:
                            return df
                        else:
                            logger.error(f"❌ Kraken data missing 'close' column")
                            return None
                    return None
        except Exception as e:
            logger.error(f"❌ Kraken Exception for {pair}: {str(e)}")
            return None