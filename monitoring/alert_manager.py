

class AlertManager:
    """Manage and distribute risk and model alerts."""
    
    def __init__(self, notification_handlers: Optional[Dict] = None):
        self.notification_handlers = notification_handlers or {}
        self.alert_history = []
        
    def add_handler(self, severity: str, handler: callable):
        """Add notification handler for given severity."""
        self.notification_handlers[severity] = handler
        
    def process_alert(self, alert: RiskAlert):
        """Process and distribute alert."""
        self.alert_history.append(alert)
        
        # Handle alert based on severity
        handler = self.notification_handlers.get(alert.severity)
        if handler:
            handler(alert)
            
    def get_active_alerts(self, 
                         lookback: timedelta = timedelta(hours=24)) -> List[RiskAlert]:
        """Get active alerts within lookback period."""
        cutoff_time = datetime.now() - lookback
        return [
            alert for alert in self.alert_history
            if alert.timestamp > cutoff_time
        ]
        
    def get_alert_summary(self) -> Dict:
        """Get summary of recent alerts."""
        active_alerts = self.get_active_alerts()
        
        return {
            'total_alerts': len(active_alerts),
            'by_severity': {
                severity: len([a for a in active_alerts if a.severity == severity])
                for severity in set(a.severity for a in active_alerts)
            },
            'by_type': {
                alert_type: len([a for a in active_alerts if a.alert_type == alert_type])
                for alert_type in set(a.alert_type for a in active_alerts)
            }
        }
