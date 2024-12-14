# 

class ModelMonitor:
    """Monitor model performance and adaptation during market stress."""
    
    def __init__(self, 
                 base_rmse_threshold: float = 0.02,
                 prediction_shift_threshold: float = 0.5,
                 uncertainty_threshold: float = 2.0):
        self.base_rmse_threshold = base_rmse_threshold
        self.prediction_shift_threshold = prediction_shift_threshold
        self.uncertainty_threshold = uncertainty_threshold
        self.performance_history = []
        
    def calculate_model_metrics(self, 
                              predictions: np.ndarray, 
                              actual: np.ndarray,
                              ensemble_predictions: Optional[List[np.ndarray]] = None) -> Dict[str, float]:
        """Calculate model performance metrics."""
        metrics = {
            'rmse': np.sqrt(np.mean((predictions - actual) ** 2)),
            'prediction_bias': np.mean(predictions - actual),
            'hit_rate': np.mean(np.sign(predictions) == np.sign(actual))
        }
        
        if ensemble_predictions:
            metrics.update({
                'ensemble_disagreement': np.std([p.mean() for p in ensemble_predictions]),
                'ensemble_uncertainty': np.mean([np.std(p) for p in ensemble_predictions])
            })
            
        return metrics
    
    def check_model_health(self, metrics: Dict[str, float]) -> List[str]:
        """Check for model health issues."""
        issues = []
        
        if metrics['rmse'] > self.base_rmse_threshold:
            issues.append(f"High prediction error: {metrics['rmse']:.4f}")
            
        if abs(metrics['prediction_bias']) > self.base_rmse_threshold:
            issues.append(f"Significant prediction bias: {metrics['prediction_bias']:.4f}")
            
        if 'ensemble_disagreement' in metrics:
            if metrics['ensemble_disagreement'] > self.prediction_shift_threshold:
                issues.append("High ensemble disagreement")
                
        if 'ensemble_uncertainty' in metrics:
            if metrics['ensemble_uncertainty'] > self.uncertainty_threshold:
                issues.append("High prediction uncertainty")
                
        return issues
    
    def update_monitoring(self, 
                         predictions: np.ndarray,
                         actual: np.ndarray,
                         ensemble_predictions: Optional[List[np.ndarray]] = None,
                         timestamp: Optional[datetime] = None) -> Dict:
        """Update monitoring stats and return current status."""
        if timestamp is None:
            timestamp = datetime.now()
            
        metrics = self.calculate_model_metrics(
            predictions, actual, ensemble_predictions
        )
        
        issues = self.check_model_health(metrics)
        
        monitoring_update = {
            'timestamp': timestamp,
            'metrics': metrics,
            'issues': issues,
            'status': 'HEALTHY' if not issues else 'WARNING',
            'needs_retraining': len(issues) > 2  # Simple retraining trigger
        }
        
        self.performance_history.append(monitoring_update)
        return monitoring_update
