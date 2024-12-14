

class PerformanceAnalyzer:
    """Analyze model and trading performance."""
    
    @staticmethod
    def calculate_risk_metrics(returns: List[float]) -> Dict[str, float]:
        """Calculate risk-adjusted performance metrics."""
        returns = np.array(returns)
        
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)  # Annualized
        sortino_ratio = np.mean(returns) / np.std(returns[returns < 0]) * np.sqrt(252)
        max_drawdown = np.min(np.minimum.accumulate(returns))
        
        return {
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'volatility': np.std(returns) * np.sqrt(252)
        }
    
    @staticmethod
    def analyze_trade_distribution(trades: List[Dict]) -> Dict[str, float]:
        """Analyze trade characteristics."""
        df = pd.DataFrame(trades)
        
        return {
            'avg_trade_duration': df['duration'].mean(),
            'profit_factor': abs(df[df['profit'] > 0]['profit'].sum() / 
                               df[df['profit'] < 0]['profit'].sum()),
            'largest_win': df['profit'].max(),
            'largest_loss': df['profit'].min(),
            'profit_std': df['profit'].std()
        }
