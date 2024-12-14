
class CrashAnalyzer:
    """Analyze model performance during market crashes."""
    
    def __init__(self, lookback_window=20):
        self.lookback_window = lookback_window
    
    def identify_crash_periods(self, prices, threshold=-0.20):
        """Identify market crash periods using drawdown analysis."""
        rolling_max = pd.Series(prices).rolling(window=self.lookback_window).max()
        drawdown = prices / rolling_max - 1
        
        # Identify crash periods (drawdown below threshold)
        crash_periods = drawdown < threshold
        
        # Find crash start and end dates
        crash_starts = crash_periods.ne(crash_periods.shift()).cumsum()[crash_periods]
        crash_ends = crash_periods.ne(crash_periods.shift(-1)).cumsum()[crash_periods]
        
        return pd.DataFrame({
            'start': crash_starts.index,
            'end': crash_ends.index,
            'drawdown': drawdown[crash_periods]
        })
    
    def calculate_crash_metrics(self, model_performance, crash_periods):
        """Calculate performance metrics during crash periods."""
        crash_metrics = {}
        
        for _, period in crash_periods.iterrows():
            period_mask = (model_performance.index >= period['start']) & \
                         (model_performance.index <= period['end'])
            period_perf = model_performance[period_mask]
            
            crash_metrics[f"crash_{period['start']}"] = {
                'duration': len(period_perf),
                'max_drawdown': period_perf['drawdown'].min(),
                'recovery_time': self._calculate_recovery_time(period_perf),
                'sharpe_ratio': self._calculate_sharpe_ratio(period_perf['returns']),
                'win_rate': (period_perf['returns'] > 0).mean()
            }
        
        return crash_metrics
    
    def _calculate_recovery_time(self, performance):
        """Calculate time to recover from drawdown peak."""
        max_drawdown_idx = performance['drawdown'].idxmin()
        recovery_mask = performance.index > max_drawdown_idx
        recovery_data = performance[recovery_mask]
        
        if not recovery_data.empty:
            recovery_point = recovery_data[recovery_data['drawdown'] >= 0].index[0] \
                if any(recovery_data['drawdown'] >= 0) else None
            return (recovery_point - max_drawdown_idx).days if recovery_point else None
        return None
    
    def _calculate_sharpe_ratio(self, returns, risk_free_rate=0.0):
        """Calculate annualized Sharpe ratio."""
        excess_returns = returns - risk_free_rate/252  # Daily risk-free rate
        return np.sqrt(252) * excess_returns.mean() / excess_returns.std()
    
    def analyze_model_adaptability(self, model, crash_periods, data):
        """Analyze how well the model adapts to crash conditions."""
        adaptability_metrics = {}
        
        for _, period in crash_periods.iterrows():
            # Pre-crash performance
            pre_crash_start = period['start'] - timedelta(days=self.lookback_window)
            pre_crash_data = data[pre_crash_start:period['start']]
            pre_crash_pred = model.predict(pre_crash_data)
            
            # During-crash performance
            crash_data = data[period['start']:period['end']]
            crash_pred = model.predict(crash_data)
            
            # Calculate adaptation metrics
            prediction_shift = np.corrcoef(pre_crash_pred, crash_pred)[0,1]
            uncertainty_increase = np.std(crash_pred) / np.std(pre_crash_pred)
            
            adaptability_metrics[f"crash_{period['start']}"] = {
                'prediction_shift': prediction_shift,
                'uncertainty_increase': uncertainty_increase,
                'recovery_speed': self._calculate_recovery_speed(model, period, data)
            }
        
        return adaptability_metrics
    
    def _calculate_recovery_speed(self, model, crash_period, data):
        """Calculate how quickly model predictions stabilize after crash."""
        post_crash_start = crash_period['end']
        post_crash_end = post_crash_start + timedelta(days=self.lookback_window)
        post_crash_data = data[post_crash_start:post_crash_end]
        
        predictions = model.predict(post_crash_data)
        prediction_volatility = pd.Series(predictions).rolling(window=5).std()
        
        return (prediction_volatility[5:] / prediction_volatility.iloc[0]).mean()
