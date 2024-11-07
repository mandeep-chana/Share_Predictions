# Stock Forecasting and Analysis Tool

This Python script analyzes and forecasts stock prices for multiple companies based on their ticker symbols. The script reads ticker symbols from a text file (`tickers.txt`), fetches stock data and financial news for each ticker, and generates a forecast of future stock prices. Each ticker’s results are saved in separate folders for easy management and visualization.

## Features
- **Fetches Financial News**: Collects recent news articles and performs sentiment analysis.
- **Retrieves Historical Stock Data**: Downloads two years of historical stock data using Yahoo Finance.
- **Stock Price Forecasting**: Uses Facebook’s Prophet model to generate a 30-day stock price forecast.
- **Data Organization**: Creates a dedicated output folder for each ticker, containing news, stock data, and forecast plots.

## Setup and Installation

### Prerequisites
- Python 3.7+
- The following Python libraries:
  - `yfinance` (for fetching stock data)
  - `pandas` (for data manipulation)
  - `numpy` (for numerical operations)
  - `prophet` (for forecasting)
  - `beautifulsoup4` (for web scraping)
  - `requests` (for HTTP requests)
  - `matplotlib` (for plotting forecast results)
  - `textblob` (for sentiment analysis)

### Installation
1. Clone the repository or download the script.
2. Install the required libraries by running:
   ```bash
   pip install yfinance pandas numpy prophet beautifulsoup4 requests matplotlib textblob
Create a tickers.txt file in the same directory as the script. List each stock ticker on a new line. For example:
Copy code
AAPL
MSFT
TSLA
Usage
Run the script: Execute the script by running:

bash
Copy code
python stock_forecast.py
View Output: After running, you will find a folder for each ticker symbol. Each folder will contain:

<TICKER>_news.csv: Contains recent news articles and sentiment analysis.
<TICKER>_stock_data.csv: Contains historical stock data for the past two years.
<TICKER>_forecast.csv: Contains forecasted stock prices for the next 30 days.
forecast_plots/: Contains forecast and component plots for visual analysis.
Example Output Structure
For a ticker AAPL, the folder structure will look like this:

Copy code
AAPL/
├── AAPL_news.csv
├── AAPL_stock_data.csv
├── AAPL_forecast.csv
└── forecast_plots/
    ├── AAPL_forecast_plot.png
    └── AAPL_forecast_components.png
Troubleshooting
Error fetching stock data: Ensure that the ticker symbol in tickers.txt is correct. Some tickers may not have data available.
Error fetching news: News sources and HTML structure may change over time, so selectors may need updating if scraping fails.
License
This project is licensed under the MIT License. See the LICENSE file for details.

Contributions
Feel free to submit issues or pull requests for any improvements or bug fixes. Your contributions are welcome!

Author
Mannie
