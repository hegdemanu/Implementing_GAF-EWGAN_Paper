# 

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Optional
from datetime import datetime, timedelta

@dataclass
class RiskAlert:
    """Data class for risk alerts."""
    timestamp: datetime
    alert_type: str
    severity: str
    message: str
    metrics: Dict[str, float]

class RiskMonitor:
    """Real-time risk monitoring system."""
    
    def __init__(self, 
                 volatility_threshold: float = 0.02,
                 drawdown_threshold: float = -0.1,
                 var_threshold: float = -0.05,
                 window_size: int = 20):
        self.volatility_threshold = volatility_threshold
        self.drawdown_threshold = drawdown_threshold
        self.var_threshold = var_threshold
        self.window_size = window_size
        self.alerts: List[RiskAlert] = []
        
    def calculate_risk_metrics(self, prices: pd.Series) -> Dict[str, float]:
        """Calculate current risk metrics."""
        if len(prices) < self.window_size:
            return {}
            
        returns = prices.pct_change().dropna()
        rolling_prices = prices.rolling(window=self.window_size)
        
        metrics = {
            'volatility': returns.std() * np.sqrt(252),
            'drawdown': (prices / rolling_prices.max() - 1).iloc[-1],
            'var_95': np.percentile(returns, 5) * np.sqrt(252),
            'current_return': returns.iloc[-1]
        }
        
        return metrics
    
    def check_conditions(self, metrics: Dict[str, float]) -> List[RiskAlert]:
        """Check for alert conditions."""
        current_time = datetime.now()
        alerts = []
        
        # Volatility alert
        if metrics['volatility'] > self.volatility_threshold:
            alerts.append(RiskAlert(
                timestamp=current_time,
                alert_type='HIGH_VOLATILITY',
                severity='WARNING',
                message=f"Volatility ({metrics['volatility']:.2%}) above threshold ({self.volatility_threshold:.2%})",
                metrics=metrics
            ))
        
        # Drawdown alert
        if metrics['drawdown'] < self.drawdown_threshold:
            alerts.append(RiskAlert(
                timestamp=current_time,
                alert_type='SEVERE_DRAWDOWN',
                severity='CRITICAL',
                message=f"Drawdown ({metrics['drawdown']:.2%}) below threshold ({self.drawdown_threshold:.2%})",
                metrics=metrics
            ))
        
        # VaR alert
        if metrics['var_95'] < self.var_threshold:
            alerts.append(RiskAlert(
                timestamp=current_time,
                alert_type='VAR_BREACH',
                severity='WARNING',
                message=f"VaR ({metrics['var_95']:.2%}) below threshold ({self.var_threshold:.2%})",
                metrics=metrics
            ))
        
        return alerts
    
    def monitor_tick(self, price: float, timestamp: Optional[datetime] = None) -> List[RiskAlert]:
        """Process new price tick and generate alerts."""
        if timestamp is None:
            timestamp = datetime.now()
            
        # Update price history
        if not hasattr(self, 'price_history'):
            self.price_history = pd.Series(dtype=float)
        
        self.price_history[timestamp] = price
        
        # Calculate metrics and check conditions
        metrics = self.calculate_risk_metrics(self.price_history)
        if metrics:
            new_alerts = self.check_conditions(metrics)
            self.alerts.extend(new_alerts)
            return new_alerts
        
        return []
