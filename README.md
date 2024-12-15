# Implementing_GAF-EWGAN_Paper
# GAF-EWGAN Stock Market Prediction

Implementation of "An enhanced Wasserstein generative adversarial network with Gramian Angular Fields for efficient stock market prediction during market crash periods"

## Overview

This project implements a novel stock market prediction model that combines:
- Gramian Angular Fields (GAF) for time series encoding
- Wasserstein Generative Adversarial Networks (WGAN)
- Ensemble learning with meta-learner
- Advanced technical indicators

The model is designed to be particularly robust during market crash periods.

## Features

- Data Processing:
  - Automated stock data fetching from Alpha Vantage
  - Technical indicator calculation
  - GAF transformation
  - Feature engineering

- Model Architecture:
  - ConvLSTM-based generator
  - Conv1D-based discriminator
  - WGAN with gradient penalty
  - Ensemble model with meta-learner

- Trading Simulation:
  - Day trading strategy implementation
  - Performance metrics calculation
  - Risk analysis
  - Visualization tools

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/gaf-ewgan.git
cd gaf-ewgan
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up configuration:
- Copy `config.yaml.example` to `config.yaml`
- Add your Alpha Vantage API key
- Adjust model parameters as needed

## Usage

1. Data Preparation:
```bash
python -m preprocessing.data_loader --config config.yaml
```

2. Model Training:
```bash
python main.py --config config.yaml
```

3. Evaluation:
```bash
python -m evaluation.trader --model-path checkpoints/best_model.pth
```

## Example Notebooks

1. `1_data_preparation.ipynb`: Demonstrates data loading and preprocessing
2. `2_model_training.ipynb`: Shows model training process
3. `3_trading_simulation.ipynb`: Illustrates trading simulation and analysis

## Project Structure

```
gaf-ewgan/
├── data/                    # Data storage
├── models/                  # Model implementations
├── preprocessing/           # Data processing
├── training/               # Training utilities
├── evaluation/             # Evaluation tools
├── utils/                  # Helper functions
├── tests/                  # Unit tests
├── example_notebooks/      # Example notebooks
├── config.yaml             # Configuration
└── requirements.txt        # Dependencies
```

## Running Tests

```bash
python -m pytest tests/
```

## Results

The model achieves:
- Annual return: 16.49%
- Win-Loss Ratio: 7.68
- Batting Average: 50.2%
- Average Profit per Trade: 0.13

## Requirements

```
# requirements.txt

torch>=1.9.0
pandas>=1.3.0
numpy>=1.19.0
matplotlib>=3.4.0
seaborn>=0.11.0
alpha_vantage>=2.3.1
talib-binary>=0.4.19
pytest>=6.2.5
jupyter>=1.0.0
pyyaml>=5.4.1
```

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a pull request

## License

MIT License - see LICENSE file for details.

## Citation

If you use this code in your research, please cite the original paper:
```
@article{ghasemieh2023enhanced,
  title={An enhanced Wasserstein generative adversarial network with Gramian Angular Fields for efficient stock market prediction during market crash periods},
  author={Ghasemieh, Alireza and Kashef, Rasha},
  journal={Applied Intelligence},
  year={2023}
}
```

## Acknowledgments

- Original paper authors for the GAF-EWGAN architecture
- Alpha Vantage for providing financial data API


#### Update
# GAF-EWGAN Stock Market Prediction System

An enhanced Wasserstein generative adversarial network with Gramian Angular Fields for efficient stock market prediction during market crash periods, including real-time monitoring capabilities.

## Features

### Market Prediction
- Automated data fetching from Alpha Vantage API
- Advanced technical indicator calculations
- GAF (Gramian Angular Fields) transformation
- WGAN-based prediction model
- Ensemble learning with meta-learner

### Real-time Monitoring
- Live market data tracking
- Customizable alert thresholds
- Multiple alert types (price, volume, volatility)
- Real-time monitoring dashboard
- Alert history tracking

### Trading Simulation
- Day trading strategy implementation
- Performance metrics calculation
- Risk analysis
- Position management

## Project Structure
```
gaf-ewgan/
├── data/                    # Data storage
│   ├── raw/                # Raw market data
│   └── processed/          # Processed data and GAF images
├── models/                 # Model implementations
│   ├── generator.py        # WGAN generator
│   ├── discriminator.py    # WGAN discriminator
│   └── ensemble.py        # Ensemble model
├── monitoring/            # Monitoring system
│   ├── __init__.py       # Package initialization
│   ├── market_monitor.py # Real-time monitoring
│   ├── alerts.py        # Alert system
│   └── state_tracker.py # Market state tracking
├── preprocessing/        # Data processing
│   ├── feature_engineering.py
│   ├── gaf.py          # GAF conversion
│   └── data_loader.py  # Data loading utilities
├── training/            # Training utilities
│   ├── trainer.py      # Training loop
│   └── optimizer.py    # Custom optimizers
├── evaluation/         # Evaluation tools
│   ├── metrics.py     # Performance metrics
│   └── trader.py      # Trading simulation
├── utils/             # Helper functions
│   ├── config.py     # Configuration handling
│   └── visualization.py # Plotting utilities
├── dashboard/         # Monitoring dashboard
│   └── app.py        # Streamlit dashboard
├── tests/            # Unit tests
├── config.yaml       # Configuration
├── requirements.txt  # Dependencies
└── main.py          # Main script
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/gaf-ewgan.git
cd gaf-ewgan
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up configuration:
- Copy `config.yaml.example` to `config.yaml`
- Add your Alpha Vantage API key
- Adjust model and monitoring parameters as needed

## Usage

### Start the System
```bash
python main.py
```

This will:
1. Start the monitoring dashboard
2. Begin real-time market monitoring
3. Train or load the prediction model
4. Run trading simulation if enabled
5. Continue monitoring until interrupted

### Access the Dashboard
Open your browser and navigate to:
```
http://localhost:8501
```

### Configuration Options

#### Monitoring Settings
```yaml
monitoring:
  update_interval: 60  # seconds
  thresholds:
    price_change: 0.02
    volume_spike: 2.0
    volatility: 0.015
    momentum: 0.05
  alert_handlers:
    - log
    - slack
    - email
```

#### Model Parameters
```yaml
model:
  train: true
  n_models: 10
  device: "cuda"
  batch_size: 64
  epochs: 100
```

## Monitoring Features

### Real-time Alerts
- Price movement alerts
- Volume spike detection
- Volatility monitoring
- Momentum tracking

### Dashboard Components
- Real-time price charts
- Technical indicators
- Alert history
- Performance metrics

### Alert Types
1. PRICE_MOVEMENT
   - Significant price changes
   - Customizable thresholds

2. VOLUME_SPIKE
   - Unusual volume activity
   - Relative volume analysis

3. HIGH_VOLATILITY
   - Volatility spikes
   - Risk monitoring

4. STRONG_MOMENTUM
   - Trend strength
   - Direction changes

## Model Performance

### Training Metrics
- Generator loss
- Discriminator loss
- Validation accuracy

### Financial Metrics
- Annual return
- Sharpe ratio
- Maximum drawdown
- Win/Loss ratio

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

```
@article{ghasemieh2023enhanced,
  title={An enhanced Wasserstein generative adversarial network with Gramian Angular Fields for efficient stock market prediction during market crash periods},
  author={Ghasemieh, Alireza and Kashef, Rasha},
  journal={Applied Intelligence},
  year={2023}
}
```

## Acknowledgments

- Original paper authors
- Alpha Vantage for market data
- Streamlit for dashboard components

## Support

For support or questions:
- Open an issue
- Contact the maintainers

Happy trading! 📈
