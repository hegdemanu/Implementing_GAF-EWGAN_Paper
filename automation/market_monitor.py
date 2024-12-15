import requests
import pandas as pd
import time
from typing import Dict, Optional, List
from datetime import datetime, timedelta
import logging
from ratelimit import limits, sleep_and_retry

class AlphaVantageAPI:
    """Handler for Alpha Vantage API requests with rate limiting."""
    
    def __init__(self, api_key: str, calls_per_minute: int = 5):
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"
        self.calls_per_minute = calls_per_minute
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    @sleep_and_retry
    @limits(calls=5, period=60)  # Rate limit: 5 calls per minute
    def _make_request(self, params: Dict) -> Optional[Dict]:
        """Make rate-limited API request with error handling."""
        try:
            params['apikey'] = self.api_key
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            
            # Check for API-specific errors
            data = response.json()
            if "Error Message" in data:
                raise ValueError(f"API Error: {data['Error Message']}")
                
            return data
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Request failed: {str(e)}")
            raise
        except ValueError as e:
            self.logger.error(f"Data processing error: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error: {str(e)}")
            raise
            
    def get_daily_adjusted(self, symbol: str) -> pd.DataFrame:
        """Get daily adjusted price data for a symbol."""
        params = {
            "function": "TIME_SERIES_DAILY_ADJUSTED",
            "symbol": symbol,
            "outputsize": "full"
        }
        
        try:
            data = self._make_request(params)
            time_series = data.get("Time Series (Daily)", {})
            
            if not time_series:
                raise ValueError(f"No data returned for symbol {symbol}")
                
            # Convert to DataFrame
            df = pd.DataFrame.from_dict(time_series, orient='index')
            df.index = pd.to_datetime(df.index)
            
            # Rename columns
            df.columns = ['open', 'high', 'low', 'close', 'adjusted_close', 
                         'volume', 'dividend', 'split_coefficient']
            
            # Convert to numeric
            for col in df.columns:
                df[col] = pd.to_numeric(df[col])
                
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to fetch daily data for {symbol}: {str(e)}")
            raise

class MarketDataPipeline:
    """Pipeline for processing and storing market data."""
    
    def __init__(self, api: AlphaVantageAPI):
        self.api = api
        self.data_cache = {}
        self.logger = logging.getLogger(__name__)
        
    def fetch_symbols(self, symbols: List[str], 
                     refresh_if_older_than: timedelta = timedelta(days=1)):
        """Fetch data for multiple symbols with caching."""
        results = {}
        
        for symbol in symbols:
            try:
                # Check cache
                if symbol in self.data_cache:
                    last_update, data = self.data_cache[symbol]
                    if datetime.now() - last_update < refresh_if_older_than:
                        results[symbol] = data
                        continue
                
                # Fetch new data
                self.logger.info(f"Fetching data for {symbol}")
                data = self.api.get_daily_adjusted(symbol)
                
                # Update cache
                self.data_cache[symbol] = (datetime.now(), data)
                results[symbol] = data
                
            except Exception as e:
                self.logger.error(f"Failed to process {symbol}: {str(e)}")
                continue
                
        return results
    
    def process_market_data(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Process raw market data into features."""
        processed = {}
        
        for symbol, df in data.items():
            try:
                # Calculate returns
                df['returns'] = df['adjusted_close'].pct_change()
                
                # Calculate volatility
                df['volatility'] = df['returns'].rolling(window=20).std()
                
                # Calculate volume indicators
                df['volume_ma'] = df['volume'].rolling(window=20).mean()
                df['relative_volume'] = df['volume'] / df['volume_ma']
                
                # Calculate price momentum
                df['momentum'] = df['adjusted_close'].pct_change(periods=10)
                
                # Remove NaN values
                df = df.dropna()
                
                processed[symbol] = df
                
            except Exception as e:
                self.logger.error(f"Failed to process {symbol}: {str(e)}")
                continue
                
        return processed

def main():
    # Setup API and pipeline
    api_key = "YOUR_API_KEY"
    api = AlphaVantageAPI(api_key)
    pipeline = MarketDataPipeline(api)
    
    # Define symbols to track
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN"]
    
    try:
        # Fetch and process data
        raw_data = pipeline.fetch_symbols(symbols)
        processed_data = pipeline.process_market_data(raw_data)
        
        # Log summary statistics
        for symbol, data in processed_data.items():
            print(f"\nSummary for {symbol}:")
            print(f"Latest close: ${data['adjusted_close'].iloc[-1]:.2f}")
            print(f"Daily return: {data['returns'].iloc[-1]:.2%}")
            print(f"10-day momentum: {data['momentum'].iloc[-1]:.2%}")
            print(f"20-day volatility: {data['volatility'].iloc[-1]:.2%}")
            
    except Exception as e:
        logging.error(f"Pipeline execution failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
