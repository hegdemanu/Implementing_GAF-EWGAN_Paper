from alpha_vantage.timeseries import TimeSeries
import pandas as pd
import numpy as np
from typing import List, Dict

class StockDataLoader:
    """Load and preprocess stock data."""
    
    def __init__(self, api_key: str):
        self.ts = TimeSeries(key=api_key, output_format='pandas')
        self.ti = TechnicalIndicators()
        
    def fetch_data(self, symbol: str) -> pd.DataFrame:
        """Fetch daily stock data from Alpha Vantage."""
        data, _ = self.ts.get_daily(symbol=symbol, outputsize='full')
        data.columns = ['open', 'high', 'low', 'close', 'volume']
        return data
    
    def process_data(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Process raw data into features."""
        # Calculate indicators
        basic_indicators = self.ti.calculate_basic_indicators(df)
        advanced_indicators = self.ti.calculate_advanced_indicators(df, basic_indicators)
        
        # Combine features
        features = pd.concat([
            df,
            basic_indicators,
            advanced_indicators
        ], axis=1)
        
        # Remove NaN values
        features = features.dropna()
        
        # Convert to GAF format
        gaf_converter = GAFConverter()
        gaf_data = []
        
        # Create sequences of 60 days
        for i in range(60, len(features)):
            sequence = features.iloc[i-60:i]
            gaf = gaf_converter.transform(sequence['close'].values)
            gaf_data.append(gaf)
        
        return {
            'gaf_data': np.array(gaf_data),
            'prices': features['close'].values[60:],
            'features': features.values[60:]
        }
