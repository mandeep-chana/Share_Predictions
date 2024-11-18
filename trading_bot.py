import alpaca_trade_api as tradeapi
import logging
import time
from config import ALPACA_API_KEY, ALPACA_SECRET_KEY

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Alpaca API
api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, base_url='https://paper-api.alpaca.markets')

def read_tickers(file_path):
    """Read tickers from a file."""
    with open(file_path, 'r') as file:
        tickers = [line.strip() for line in file if line.strip()]
    logging.info(f"Read {len(tickers)} tickers from {file_path}")
    return tickers

def get_last_price(ticker):
    """Get the last price of a ticker."""
    try:
        barset = api.get_barset(ticker, 'minute', limit=1)
        last_price = barset[ticker][0].c if barset[ticker] else None
        logging.info(f"Fetched last price for {ticker}: {last_price}")
        return last_price
    except Exception as e:
        logging.error(f"Error fetching last price for {ticker}: {e}")
        return None

def place_order(ticker, qty, side, order_type='market', time_in_force='gtc'):
    """Place an order."""
    try:
        api.submit_order(
            symbol=ticker,
            qty=qty,
            side=side,
            type=order_type,
            time_in_force=time_in_force
        )
        logging.info(f"Order placed: {side} {qty} shares of {ticker}")
    except Exception as e:
        logging.error(f"Error placing order for {ticker}: {e}")

def main():
    """Main function to read tickers, fetch prices, and place orders based on trading logic."""
    tickers = read_tickers('tickers.txt')
    logging.info(f"Tickers to process: {tickers}")

    for ticker in tickers:
        last_price = get_last_price(ticker)
        if last_price is not None:
            logging.info(f"Last price of {ticker}: {last_price}")
            # Example trading logic: Buy 1 share if the price is below $100
            if last_price < 100:
                place_order(ticker, 1, 'buy')
            # Example trading logic: Sell 1 share if the price is above $200
            elif last_price > 200:
                place_order(ticker, 1, 'sell')
        else:
            logging.warning(f"No price data for {ticker}")

if __name__ == "__main__":
    while True:
        main()
        logging.info("Sleeping for 60 seconds before next iteration")
        time.sleep(60)  # Run every 60 seconds