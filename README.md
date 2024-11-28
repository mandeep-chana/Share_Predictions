# Stock Market Analysis Tool 🚀

## Overview

A comprehensive stock market analysis tool that combines technical analysis, machine learning predictions, and market sentiment analysis to provide detailed insights for stock trading decisions.

## Features

- **Technical Analysis**
  - Candlestick charts with interactive hover information
  - Multiple technical indicators (RSI, MACD, Bollinger Bands)
  - Moving averages (SMA, EMA)
  - Volume analysis
  - Parabolic SAR
- **Machine Learning Predictions**
  - LSTM-based price predictions
  - Prophet forecasting
  - Customizable sequence lengths and parameters
- **Trading Strategies**
  - SMA Crossover with Risk Management
  - Bollinger Band Strategy with RSI confirmation
  - Strategy optimization using grid search
  - Comprehensive backtesting
- **Market Analysis**
  - Economic indicators tracking
  - Options data analysis
  - Market regime detection
  - Sentiment analysis from news

## Installation

### Prerequisites

- Python 3.8+
- pip package manager

### Dependencies

```
pip install -r requirements.txt
```

Required packages:

- yfinance
- pandas
- numpy
- tensorflow
- prophet
- plotly
- beautifulsoup4
- scikit-learn
- ta
- backtesting
- matplotlib
- textblob

## Usage

1. Create a `tickers.txt` file with your stock symbols:

```
AAPLMSFTGOOGL
```

1. Run the main script:

```
python stock_analyzer.py
```

1. Enter date range when prompted or press Enter for default (last 2 years)

## Output Structure

```
output/├── {ticker}/│   ├── technical_analysis/│   │   ├── interactive_analysis_{timestamp}.html│   │   └── technical_analysis_plot_{timestamp}.png│   ├── lstm_analysis/│   │   ├── lstm_prediction.png│   │   └── lstm_metrics.json│   ├── forecast_plots/│   │   ├── forecast_plot.png│   │   └── forecast_components.png│   ├── strategies/│   │   ├── strategy_comparison.json│   │   └── optimization_results.csv│   └── market_analysis.json
```

## Configuration

- Adjust technical analysis parameters in the respective strategy classes
- Modify LSTM sequence length and architecture in `LSTMPredictor` class
- Configure risk parameters in `SmaCrossWithRisk` strategy

## Logging

- Logs are stored in the `logs` directory
- Uses rotating file handler (10MB per file, 5 backup files)
- Includes both file and console logging

## Error Handling

- Comprehensive error handling for data fetching
- Validation of input data
- Graceful degradation when services are unavailable

## Contributing

1. Fork the repository
1. Create your feature branch (`git checkout -b feature/AmazingFeature`)
1. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
1. Push to the branch (`git push origin feature/AmazingFeature`)
1. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details

## Disclaimer

This tool is for educational and research purposes only. Always conduct your own research and consult with financial advisors before making investment decisions.

## Acknowledgments

- Data provided by Yahoo Finance
- Technical analysis indicators from `ta` library
- Machine learning implementations using TensorFlow and Prophet

<br>
