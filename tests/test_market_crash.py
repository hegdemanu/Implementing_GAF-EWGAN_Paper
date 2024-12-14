# 
import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from preprocessing.data_loader import StockDataLoader
from models.ensemble import GAFEWGANEnsemble
from evaluation.trader import DayTrader

class TestMarketCrashScenarios(unittest.TestCase):
    def setUp(self):
        self.dates = pd.date_range(start='2020-01-01', end='2020-12-31')
        self.base_price = 100
        
        # Create synthetic crash data
        self.prices = []
        current_price = self.base_price
        
        for i in range(len(self.dates)):
            if i > 50 and i < 80:  # Simulate crash period
                current_price *= 0.95  # 5% daily decline
            elif i >= 80 and i < 110:  # Recovery period
                current_price *= 1.02  # 2% daily increase
            else:
                current_price *= (1 + np.random.normal(0, 0.01))  # Normal volatility
            self.prices.append(current_price)
        
        self.crash_data = pd.DataFrame({
            'date': self.dates,
            'close': self.prices
        })
        
        # Initialize model
        self.model = GAFEWGANEnsemble(n_models=3)
        self.trader = DayTrader(initial_balance=10000)

    def test_crash_detection(self):
        """Test if model identifies market crash periods."""
        crash_periods = self.trader.detect_crash_periods(self.crash_data['close'])
        
        # Should detect the simulated crash period
        self.assertTrue(any(crash_periods[50:80]))
        self.assertFalse(any(crash_periods[0:50]))
    
    def test_crash_performance(self):
        """Test model performance during crash periods."""
        # Train model on pre-crash data
        pre_crash_data = self.crash_data.iloc[0:50]
        self.model.train_on_period(pre_crash_data)
        
        # Test on crash period
        crash_period_data = self.crash_data.iloc[50:80]
        crash_performance = self.model.evaluate_period(crash_period_data)
        
        # Model should maintain reasonable drawdown during crash
        self.assertLess(crash_performance['max_drawdown'], 0.3)  # Max 30% drawdown
        self.assertGreater(crash_performance['sharpe_ratio'], -2.0)  # Reasonable risk-adjusted return

    def test_recovery_adaptation(self):
        """Test model adaptation during recovery periods."""
        recovery_data = self.crash_data.iloc[80:110]
        recovery_performance = self.model.evaluate_period(recovery_data)
        
        # Should capture recovery momentum
        self.assertGreater(recovery_performance['profit_factor'], 1.0)
        self.assertGreater(recovery_performance['win_rate'], 0.5)
