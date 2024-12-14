class DayTrader:
    """Simulates day trading based on model predictions."""
    
    def __init__(self, initial_balance=10000):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.trades = []
    
    def execute_trade(self, prediction, actual_price, timestamp):
        """Execute a trade based on prediction."""
        threshold = 0.05  # 5% threshold
        
        price_change = (prediction - actual_price) / actual_price
        
        if abs(price_change) > threshold:
            position = 'long' if price_change > 0 else 'short'
            size = self.balance  # Use full balance
            
            # Calculate profit
            profit = size * price_change if position == 'long' else -size * price_change
            
            trade = {
                'timestamp': timestamp,
                'position': position,
                'size': size,
                'profit': profit,
                'price_change': price_change
            }
            
            self.trades.append(trade)
            self.balance += profit
    
    def get_performance_metrics(self):
        """Calculate all performance metrics."""
        years = (self.trades[-1]['timestamp'] - self.trades[0]['timestamp']).days / 365
        
        return {
            'win_loss_ratio': FinancialMetrics.win_loss_ratio(self.trades),
            'batting_average': FinancialMetrics.batting_average(self.trades),
            'avg_profit_per_trade': FinancialMetrics.average_profit_per_trade(self.trades),
            'annual_return': FinancialMetrics.annual_return(self.trades, self.initial_balance, years),
            'total_trades': len(self.trades),
            'final_balance': self.balance
        }
