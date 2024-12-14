
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime

class CrashVisualization:
    """Visualization tools for analyzing market crash periods."""
    
    def __init__(self, figsize=(15, 10)):
        self.figsize = figsize
        plt.style.use('seaborn')
    
    def plot_crash_overview(self, prices, crash_periods, predictions=None):
        """Plot overview of crash periods with model predictions."""
        plt.figure(figsize=self.figsize)
        
        # Plot actual prices
        plt.plot(prices.index, prices, label='Actual', color='blue', alpha=0.7)
        
        # Highlight crash periods
        for _, period in crash_periods.iterrows():
            plt.axvspan(period['start'], period['end'], 
                       color='red', alpha=0.2)
        
        # Plot predictions if available
        if predictions is not None:
            plt.plot(predictions.index, predictions, 
                    label='Predicted', color='green', linestyle='--')
        
        plt.title('Market Price Movement with Crash Periods')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        
    def plot_crash_performance_comparison(self, metrics, crash_periods):
        """Compare model performance during different crash periods."""
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        
        # Prepare data
        performance_data = pd.DataFrame(metrics).T
        
        # Plot drawdown comparison
        sns.barplot(data=performance_data, y='max_drawdown', ax=axes[0,0])
        axes[0,0].set_title('Maximum Drawdown by Crash Period')
        
        # Plot recovery time comparison
        sns.barplot(data=performance_data, y='recovery_time', ax=axes[0,1])
        axes[0,1].set_title('Recovery Time (Days)')
        
        # Plot Sharpe ratio comparison
        sns.barplot(data=performance_data, y='sharpe_ratio', ax=axes[1,0])
        axes[1,0].set_title('Sharpe Ratio During Crash')
        
        # Plot win rate comparison
        sns.barplot(data=performance_data, y='win_rate', ax=axes[1,1])
        axes[1,1].set_title('Win Rate During Crash')
        
        plt.tight_layout()
    
    def plot_model_adaptation(self, adaptability_metrics, window_size=20):
        """Visualize how model adapts during crash periods."""
        fig, axes = plt.subplots(3, 1, figsize=self.figsize)
        
        data = pd.DataFrame(adaptability_metrics).T
        
        # Plot prediction shift
        data['prediction_shift'].plot(ax=axes[0], kind='bar')
        axes[0].set_title('Model Prediction Shift')
        axes[0].set_ylabel('Correlation')
        
        # Plot uncertainty increase
        data['uncertainty_increase'].plot(ax=axes[1], kind='bar')
        axes[1].set_title('Uncertainty Increase')
        axes[1].set_ylabel('Ratio')
        
        # Plot recovery speed
        data['recovery_speed'].plot(ax=axes[2], kind='bar')
        axes[2].set_title('Recovery Speed')
        axes[2].set_ylabel('Volatility Ratio')
        
        plt.tight_layout()
    
    def plot_ensemble_behavior(self, ensemble_predictions, crash_periods):
        """Visualize ensemble model behavior during crashes."""
        plt.figure(figsize=self.figsize)
        
        # Plot individual model predictions
        for i, preds in enumerate(ensemble_predictions):
            plt.plot(preds.index, preds, alpha=0.3, 
                    color='gray', label='Base Model' if i == 0 else '')
        
        # Plot ensemble prediction
        ensemble_mean = pd.DataFrame(ensemble_predictions).mean()
        plt.plot(ensemble_mean.index, ensemble_mean, 
                color='red', linewidth=2, label='Ensemble Mean')
        
        # Plot prediction uncertainty
        ensemble_std = pd.DataFrame(ensemble_predictions).std()
        plt.fill_between(ensemble_mean.index, 
                        ensemble_mean - 2*ensemble_std,
                        ensemble_mean + 2*ensemble_std,
                        color='red', alpha=0.2)
        
        # Highlight crash periods
        for _, period in crash_periods.iterrows():
            plt.axvspan(period['start'], period['end'], 
                       color='yellow', alpha=0.2)
        
        plt.title('Ensemble Model Behavior During Crashes')
        plt.xlabel('Date')
        plt.ylabel('Prediction')
        plt.legend()
        
    def plot_risk_metrics(self, risk_metrics, crash_periods):
        """Visualize risk metrics evolution during crashes."""
        fig, axes = plt.subplots(3, 1, figsize=self.figsize)
        
        # Value at Risk
        axes[0].plot(risk_metrics.index, risk_metrics['VaR'])
        axes[0].set_title('Value at Risk (def plot_risk_metrics(self, risk_metrics, crash_periods):
        """Visualize risk metrics evolution during crashes."""
        fig, axes = plt.subplots(3, 1, figsize=self.figsize)
        
        # Value at Risk
        axes[0].plot(risk_metrics.index, risk_metrics['VaR'])
        axes[0].set_title('Value at Risk (95%)')
        for _, period in crash_periods.iterrows():
            axes[0].axvspan(period['start'], period['end'], color='red', alpha=0.2)
        
        # Rolling Volatility
        axes[1].plot(risk_metrics.index, risk_metrics['volatility'])
        axes[1].set_title('Rolling Volatility')
        for _, period in crash_periods.iterrows():
            axes[1].axvspan(period['start'], period['end'], color='red', alpha=0.2)
        
        # Drawdown
        axes[2].plot(risk_metrics.index, risk_metrics['drawdown'])
        axes[2].set_title('Drawdown')
        for _, period in crash_periods.iterrows():
            axes[2].axvspan(period['start'], period['end'], color='red', alpha=0.2)
        
        plt.tight_layout()
