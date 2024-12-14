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
