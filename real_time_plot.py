import sys
import logging
from bokeh.plotting import figure, curdoc
from bokeh.models import ColumnDataSource
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime
import random
import traceback

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(console_handler)

# Optionally, create a file handler
file_handler = logging.FileHandler('app.log')
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(file_handler)

logger.debug("Logging initialized.")

# Replace with your actual Alpaca API credentials
from config import ALPACA_API_KEY, ALPACA_SECRET_KEY

# Initialize Alpaca API client
client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)

# Create a ColumnDataSource for Bokeh
source = ColumnDataSource(data=dict(time=[], price=[]))

def update_data(ticker):
  logger.debug(f"Running update_data callback for ticker: {ticker}...")
  try:
      # Test with random data first
      current_time = datetime.now()
      random_price = random.uniform(100, 200)
      test_data = dict(time=[current_time], price=[random_price])
      source.stream(test_data, rollover=200)
      logger.info(f"Test data updated: {test_data}")

      # Attempt to fetch real data from Alpaca API
      request_params = StockBarsRequest(
          symbol_or_symbols=ticker,
          timeframe=TimeFrame.Minute,
          limit=1
      )
      logger.debug(f"Requesting data from Alpaca API with params: {request_params}")
      bars = client.get_stock_bars(request_params)

      logger.debug(f"Received response from Alpaca API: {bars}")

      if not bars or ticker not in bars:
          logger.error(f"No data returned for ticker: {ticker}")
          return

      latest_price = bars[ticker][0].c
      api_data = dict(time=[current_time], price=[latest_price])
      source.stream(api_data, rollover=200)
      logger.info(f"API data updated: {api_data}")

  except Exception as e:
      logger.error(f"Error fetching data from Alpaca: {str(e)}")
      logger.debug(traceback.format_exc())

def create_real_time_plot(ticker):
  """Create a real-time plot using Bokeh."""
  p = figure(title=f"Real-time Price of {ticker}", x_axis_label='Time', y_axis_label='Price', x_axis_type='datetime')
  p.line('time', 'price', source=source, line_width=2)

  # Add a periodic callback to update the data
  curdoc().add_periodic_callback(lambda: update_data(ticker), 1000)
  logger.debug("Periodic callback added for update_data.")
  return p

if __name__ == "__main__":
  if len(sys.argv) < 2:
      logger.error("Please provide a ticker symbol.")
      sys.exit(1)

  ticker = sys.argv[1]
  plot = create_real_time_plot(ticker)
  logger.info(f"Plot object created for ticker: {ticker}")
  curdoc().add_root(plot)
  logger.info("Plot added to the document.")