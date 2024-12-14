
import torch
from torch.utils.data import DataLoader, TensorDataset
import argparse
from pathlib import Path
import yaml
import logging

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config

def prepare_data(data_loader: StockDataLoader, symbols: List[str]):
    """Prepare data for all symbols."""
    all_data = []
    
    for symbol in symbols:
        logging.info(f"Processing data for {symbol}")
        raw_data = data_loader.fetch_data(symbol)
        processed_data = data_loader.process_data(raw_data)
        all_data.append(processed_data)
    
    return all_data

def create_dataloaders(processed_data: List[Dict], batch_size: int):
    """Create train/val/test dataloaders."""
    # Combine data from all symbols
    gaf_data = np.concatenate([d['gaf_data'] for d in processed_data])
    prices = np.concatenate([d['prices'] for d in processed_data])
    
    # Split data
    train_size = int(0.7 * len(gaf_data))
    val_size = int(0.15 * len(gaf_data))
    
    # Convert to tensors
    X = torch.FloatTensor(gaf_data)
    y = torch.FloatTensor(prices).reshape(-1, 1)
    
    # Create datasets
    train_dataset = TensorDataset(
        X[:train_size], y[:train_size]
    )
    val_dataset = TensorDataset(
        X[train_size:train_size+val_size],
        y[train_size:train_size+val_size]
    )
    test_dataset = TensorDataset(
        X[train_size+val_size:],
        y[train_size+val_size:]
    )
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()
    
    # Setup
    setup_logging()
    config = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize data loader
    data_loader = StockDataLoader(config['alpha_vantage_key'])
    
    # Prepare data
    processed_data = prepare_data(data_loader, config['symbols'])
    train_loader, val_loader, test_loader = create_dataloaders(
        processed_data, config['batch_size']
    )
    
    # Initialize ensemble model
    model = GAFEWGANEnsemble(
        n_models=config['n_models'],
        device=device
    )
    
    # Train base models
    logging.info("Training base models...")
    model.train_base_models(train_loader, config['base_epochs'])
    
    # Train meta-learner
    logging.info("Training meta-learner...")
    model.train_meta_learner(val_loader, config['meta_epochs'])
    
    # Evaluate on test set
    logging.info("Evaluating model...")
    trader = DayTrader(initial_balance=config['initial_balance'])
    
    model.eval()
    with torch.no_grad():
        for data, price in test_loader:
            data = data.to(device)
            predictions = model.predict(data)
            
            for pred, actual in zip(predictions, price):
                trader.execute_trade(pred.item(), actual.item(), None)
    
    # Print performance metrics
    metrics = trader.get_performance_metrics()
    for metric, value in metrics.items():
        logging.info(f"{metric}: {value}")

if __name__ == "__main__":
    main()
