# Stock Analysis Tool

## Project Overview
A comprehensive Python-based stock analysis tool that performs technical analysis, price forecasting, backtesting, and news sentiment analysis. The tool fetches real-time stock data, performs various analyses, and generates detailed visualizations and reports to assist in making informed investment decisions.

## Features
- Technical Analysis
  - Moving Averages (SMA, EMA)
  - RSI (Relative Strength Index)
  - MACD (Moving Average Convergence Divergence)
  - Bollinger Bands
  - LSTM-based Price Predictions
  - Automated Trading Strategy Backtesting
  - Economic Indicators Analysis
  - Options Data Analysis
  - News Sentiment Analysis
  - Multi-Exchange Support (NYSE, NASDAQ, LSE, etc.)

- **Price Forecasting**
  - Prophet model integration
  - 180-day price predictions
  - Trend and seasonality analysis
  - Confidence intervals

- **Backtesting**
  - Moving Average Crossover strategy
  - Performance metrics
  - Trade visualization
  - Equity curve analysis

- **News Sentiment Analysis**
  - Financial news aggregation
  - Sentiment scoring
  - Multiple news sources integration

## Installation
1. Clone the repository:
```bash
git clone https://github.com/yourusername/stock-analysis-tool.git
cd stock-analysis-tool
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage
1. Add stock tickers to `tickers.txt`:
```
AAPL
MSFT
GOOGL
```

2. Run the analysis:
```bash
python Share_Price_V1.py
```

## Dependencies
- pandas
- numpy
- yfinance
- prophet
- beautifulsoup4
- matplotlib
- textblob
- ta
- backtesting
- requests
- logging

## File Structure
```
stock-analysis-tool/
├── Share_Price_V1.py
├── news_sources.py
├── company_sources.py
├── tickers.txt
├── requirements.txt
├── output/
│   ├── TICKER_SYMBOL/
│   │   ├── technical_analysis/
│   │   ├── forecast_plots/
│   │   └── backtest_results/
└── logs/
    └── stock_analyzer.log
```

## Configuration
- Modify `news_sources.py` to add/remove news sources
- Adjust technical analysis parameters in `perform_technical_analysis()`
- Configure Prophet model parameters in `forecast_prices()`
- Customize backtesting strategy in `SmaCross` class

## Output
The tool generates the following outputs for each analyzed stock:

### Technical Analysis
- Technical indicators CSV file
- Summary statistics
- Visualization plots with price, indicators, and signals

### Forecasting
- Price predictions CSV
- Forecast plots with confidence intervals
- Component analysis plots (trend, seasonality)

### Backtesting
- Performance metrics
- Trade statistics
- Equity curve visualization
- Strategy performance summary

### Logs
- Detailed execution logs
- Error tracking
- Performance metrics

## Contributing
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Disclaimer
This tool is for educational and research purposes only. Always perform your own due diligence before making investment decisions. The creators are not responsible for any financial losses incurred using this tool.
