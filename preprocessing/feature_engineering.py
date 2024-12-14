

import pandas as pd
import numpy as np
import talib

class TechnicalIndicators:
    """Calculate technical indicators for stock data."""
    
    @staticmethod
    def calculate_basic_indicators(df, periods=[6, 12, 21]):
        """Calculate basic technical indicators."""
        indicators = {}
        
        for period in periods:
            # Moving Averages
            indicators[f'SMA_{period}'] = talib.SMA(df['close'], timeperiod=period)
            
            # Bollinger Bands
            upper, middle, lower = talib.BBANDS(df['close'], timeperiod=period)
            indicators[f'BB_upper_{period}'] = upper
            indicators[f'BB_middle_{period}'] = middle
            indicators[f'BB_lower_{period}'] = lower
            
            # CCI
            indicators[f'CCI_{period}'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=period)
            
            # ATR
            indicators[f'ATR_{period}'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=period)
        
        return pd.DataFrame(indicators)
    
    @staticmethod
    def calculate_advanced_indicators(df, basic_indicators):
        """Calculate advanced technical indicators."""
        advanced = {}
        
        # BB Cross Signals
        advanced['BB_cross'] = np.where(
            df['close'] > basic_indicators['BB_upper_21'], 1,
            np.where(df['close'] < basic_indicators['BB_lower_21'], -1, 0)
        )
        
        # MA Slopes
        advanced['MA5_slope'] = np.gradient(basic_indicators['SMA_6'])
        advanced['MA20_slope'] = np.gradient(basic_indicators['SMA_21'])
        
        # ATR Changes
        advanced['ATR_change'] = np.gradient(basic_indicators['ATR_21'])
        
        # Price Changes
        advanced['price_change'] = df['close'].pct_change()
        
        return pd.DataFrame(advanced)
