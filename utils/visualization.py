

import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict
import pandas as pd

class Visualizer:
    """Visualization utilities for model analysis."""
    
    @staticmethod
    def plot_training_history(history: Dict[str, List[float]], save_path: str = None):
        """Plot training metrics history."""
        plt.figure(figsize=(12, 8))
        
        # Plot losses
        plt.subplot(2, 1, 1)
        plt.plot(history['d_loss'], label='Discriminator Loss')
        plt.plot(history['g_loss'], label='Generator Loss')
        plt.title('Training Losses')
        plt.legend()
        
        # Plot validation metrics
        plt.subplot(2, 1, 2)
        plt.plot(history['val_rmse'], label='Validation RMSE')
        plt.title('Validation Metrics')
        plt.legend()
        
        if save_path:
            plt.savefig(save_path)
        plt.close()
    
    @staticmethod
    def plot_predictions(actual: List[float], predicted: List[float], save_path: str = None):
        """Plot actual vs predicted prices."""
        plt.figure(figsize=(12, 6))
        plt.plot(actual, label='Actual')
        plt.plot(predicted, label='Predicted')
        plt.title('Stock Price Prediction')
        plt.legend()
        
        if save_path:
            plt.savefig(save_path)
        plt.close()
    
    @staticmethod
    def plot_trading_performance(trader_history: List[Dict], save_path: str = None):
        """Plot trading performance metrics."""
        plt.figure(figsize=(15, 10))
        
        # Create DataFrame from trading history
        df = pd.DataFrame(trader_history)
        
        # Plot balance over time
        plt.subplot(2, 2, 1)
        plt.plot(df['timestamp'], df['balance'])
        plt.title('Account Balance Over Time')
        
        # Plot profit distribution
        plt.subplot(2, 2, 2)
        sns.histplot(df['profit'])
        plt.title('Profit Distribution')
        
        # Plot cumulative returns
        plt.subplot(2, 2, 3)
        cumulative_returns = (1 + df['profit']).cumprod()
        plt.plot(df['timestamp'], cumulative_returns)
        plt.title('Cumulative Returns')
        
        # Plot win/loss ratio over time
        plt.subplot(2, 2, 4)
        win_ratio = df['profit'].apply(lambda x: 1 if x > 0 else 0).rolling(50).mean()
        plt.plot(df['timestamp'], win_ratio)
        plt.title('Win Ratio (50-trade rolling window)')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.close()
    
    @staticmethod
    def plot_gaf_samples(gaf_data: List[np.ndarray], n_samples: int = 5, save_path: str = None):
        """Plot sample GAF visualizations."""
        fig, axes = plt.subplots(1, n_samples, figsize=(4*n_samples, 4))
        
        for i, ax in enumerate(axes):
            if i < len(gaf_data):
                im = ax.imshow(gaf_data[i], cmap='viridis')
                ax.set_title(f'GAF Sample {i+1}')
                plt.colorbar(im, ax=ax)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.close()
