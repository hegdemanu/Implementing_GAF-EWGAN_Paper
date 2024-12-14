class FinancialMetrics:
    """Calculate financial performance metrics."""
    
    @staticmethod
    def win_loss_ratio(trades):
        """Calculate Win-Loss Ratio."""
        wins = sum(1 for trade in trades if trade['profit'] > 0)
        losses = sum(1 for trade in trades if trade['profit'] < 0)
        return (wins / len(trades)) / (losses / len(trades)) if losses > 0 else float('inf')
    
    @staticmethod
    def batting_average(trades):
        """Calculate Batting Average."""
        wins = sum(1 for trade in trades if trade['profit'] > 0)
        return wins / len(trades)
    
    @staticmethod
    def average_profit_per_trade(trades):
        """Calculate Average Profit per Trade."""
        return sum(trade['profit'] for trade in trades) / len(trades)
    
    @staticmethod
    def annual_return(trades, initial_balance, years):
        """Calculate Annual Return."""
        final_balance = initial_balance + sum(trade['profit'] for trade in trades)
        return (final_balance / initial_balance) ** (1/years) - 1
