# Stock Price Forecasting and News Analysis Tool

## Overview
This Python-based tool performs stock price forecasting and news sentiment analysis for any given stock ticker. It combines historical price data analysis with current news sentiment to provide a comprehensive view of a stock's performance and future predictions.

## Features
- Historical stock data retrieval (2 years of data)
- Real-time financial news scraping from multiple sources
- News sentiment analysis
- Stock price forecasting using Facebook Prophet
- Detailed visualization of forecasts and trends
- CSV exports of all collected and analyzed data

## Requirements
python
pip install yfinance pandas numpy prophet beautifulsoup4 requests matplotlib textblob


## Installation
1. Clone the repository:
bash
git clone https://github.com/yourusername/stock-analysis-tool.git
cd stock-analysis-tool



## Output Files
The script generates several output files:
- `{ticker}_news.csv`: Recent news articles and their sentiment scores
- `{ticker}_stock_data.csv`: Historical stock price data
- `{ticker}_forecast.csv`: Detailed forecast data
- `{ticker}_forecast_plot.png`: Visual representation of the forecast
- `{ticker}_forecast_components.png`: Breakdown of forecast components

## Features in Detail

### News Analysis
- Scrapes news from Yahoo Finance and Finviz
- Performs sentiment analysis on news headlines
- Calculates average, most positive, and most negative sentiment scores
- Removes duplicate news items
- Sorts news by sentiment impact

### Stock Price Analysis
- Downloads historical stock data using yfinance
- Processes and prepares data for forecasting
- Handles missing data and anomalies
- Provides comprehensive error handling

### Forecasting
- Uses Facebook Prophet for time series forecasting
- Includes daily, weekly, and yearly seasonality
- Generates 30-day price forecasts
- Provides confidence intervals for predictions
- Creates detailed visualization of trends

## Example Output
Fetching news for AAPL...
Found 10 unique news articles

Top news articles by sentiment impact:
[News articles with sentiment scores]

Sentiment Summary:
Average sentiment: 0.123
Most positive: 0.456
Most negative: -0.789

Next 7 days forecast:
[Forecast data]

Forecast Summary:
Average predicted price: $150.25
Maximum predicted price: $155.75
Minimum predicted price: $145.50


## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer
This tool is for educational and research purposes only. Do not use it as the sole basis for investment decisions. Always conduct thorough research and consult with financial advisors before making investment decisions.

## Author
Mannie
