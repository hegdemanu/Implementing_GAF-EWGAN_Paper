import unittest
from datetime import datetime, timedelta
from evaluation.trader import DayTrader

class TestDayTrader(unittest.TestCase):
    def setUp(self):
        self.trader = DayTrader(initial_balance=10000)
        self.start_date = datetime(2023, 1, 1)
    
    def test_trade_execution(self):
        """Test trade execution logic."""
        # Test long position
        self.trader.execute_trade(110, 100, self.start_date)
        self.assertTrue(len(self.trader.trades) == 1)
        self.assertEqual(self.trader.trades[0]['position'], 'long')
        
        # Test short position
        self.trader.execute_trade(90, 100, self.start_date + timedelta(days=1))
        self.assertTrue(len(self.trader.trades) == 2)
        self.assertEqual(self.trader.trades[1]['position'], 'short')
    
    def test_performance_metrics(self):
        """Test performance metrics calculation."""
        # Execute some sample trades
        self.trader.execute_trade(110, 100, self.start_date)
        self.trader.execute_trade(90, 100, self.start_date + timedelta(days=1))
        
        metrics = self.trader.get_performance_metrics()
        
        # Check all required metrics are present
        required_metrics = ['win_loss_ratio', 'batting_average', 'avg_profit_per_trade', 
                          'annual_return', 'total_trades', 'final_balance']
        for metric in required_metrics:
            self.assertIn(metric, metrics)
