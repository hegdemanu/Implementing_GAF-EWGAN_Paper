
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional
import logging

@dataclass
class MarketAlert:
    timestamp: datetime
    symbol: str
    alert_type: str
    severity: str
    message: str
    metrics: Dict[str, float]

class AlertHandlers:
    @staticmethod
    def log_alert(alert: MarketAlert):
        logging.info(f"ALERT - {alert.symbol} - {alert.alert_type}: {alert.message}")
    
    @staticmethod
    def slack_alert(alert: MarketAlert):
        # Implement your Slack integration
        pass
    
    @staticmethod
    def email_alert(alert: MarketAlert):
        # Implement your email integration
        pass
