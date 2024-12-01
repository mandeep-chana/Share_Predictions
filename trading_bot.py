import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import APIError
import logging
import time
from config import ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL
import os
from datetime import datetime
import json
from logging.handlers import RotatingFileHandler

# Enhanced logging setup
def setup_logging():
    # Create logs directory if it doesn't exist
    logs_dir = 'logs'
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    # Create separate log files for different purposes
    timestamp = datetime.now().strftime("%Y%m%d")
    main_log = os.path.join(logs_dir, f'trading_bot_main_{timestamp}.log')
    api_log = os.path.join(logs_dir, f'trading_bot_api_{timestamp}.log')
    error_log = os.path.join(logs_dir, f'trading_bot_error_{timestamp}.log')

    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s - [%(filename)s:%(lineno)d]'
    )

    # Set up main logger
    main_handler = RotatingFileHandler(main_log, maxBytes=10*1024*1024, backupCount=5)
    main_handler.setFormatter(detailed_formatter)

    # Set up API logger
    api_handler = RotatingFileHandler(api_log, maxBytes=10*1024*1024, backupCount=5)
    api_handler.setFormatter(detailed_formatter)

    # Set up error logger
    error_handler = RotatingFileHandler(error_log, maxBytes=10*1024*1024, backupCount=5)
    error_handler.setFormatter(detailed_formatter)
    error_handler.setLevel(logging.ERROR)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(detailed_formatter)

    # Configure root logger
    logging.getLogger('').setLevel(logging.INFO)
    logging.getLogger('').addHandler(main_handler)
    logging.getLogger('').addHandler(error_handler)
    logging.getLogger('').addHandler(console_handler)

    # Create API logger
    api_logger = logging.getLogger('alpaca_api')
    api_logger.addHandler(api_handler)
    api_logger.setLevel(logging.INFO)

    return logging.getLogger('trading_bot'), api_logger

# Initialize loggers
logger, api_logger = setup_logging()

# Initialize Alpaca API with enhanced logging
try:
    api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, base_url=ALPACA_BASE_URL)
    account = api.get_account()
    logger.info("Successfully initialized Alpaca API")
    logger.info(f"Account Status: {account.status}")
    logger.info(f"Portfolio Value: ${float(account.portfolio_value)}")
except Exception as e:
    logger.error(f"Failed to initialize Alpaca API: {str(e)}", exc_info=True)
    raise

def get_position(api, ticker):
    """Get current position for a ticker with enhanced logging"""
    api_logger.info(f"Requesting position information for {ticker}")
    try:
        position = api.get_position(ticker)
        api_logger.info(f"Current position for {ticker}: {position.qty} shares at avg entry ${position.avg_entry_price}")
        return position
    except APIError as e:
        if "position does not exist" in str(e):
            api_logger.info(f"No current position exists for {ticker}")
        else:
            api_logger.error(f"API Error getting position for {ticker}: {str(e)}", exc_info=True)
        return None
    except Exception as e:
        api_logger.error(f"Unexpected error getting position for {ticker}: {str(e)}", exc_info=True)
        return None

def get_last_price(api, ticker):
    """Get the last price of a ticker with enhanced logging"""
    api_logger.info(f"Requesting last price for {ticker}")
    try:
        barset = api.get_barset(ticker, 'minute', limit=1)
        if barset and barset[ticker]:
            last_price = float(barset[ticker][0].c)
            api_logger.info(f"Retrieved last price for {ticker}: ${last_price}")
            return last_price
        api_logger.warning(f"No price data available for {ticker}")
        return None
    except Exception as e:
        api_logger.error(f"Error getting last price for {ticker}: {str(e)}", exc_info=True)
        return None

def place_order(api, ticker, side, qty=1, order_type='market', time_in_force='gtc'):
    """Place an order with enhanced logging"""
    api_logger.info(f"Attempting to place {side} order for {qty} shares of {ticker}")
    try:
        order = api.submit_order(
            symbol=ticker,
            qty=qty,
            side=side,
            type=order_type,
            time_in_force=time_in_force
        )
        api_logger.info(f"Successfully placed order: {side} {qty} shares of {ticker} (Order ID: {order.id})")
        logger.info(f"Order details - Type: {order.type}, Status: {order.status}, Filled Qty: {order.filled_qty}")
        return order
    except Exception as e:
        api_logger.error(f"Failed to place {side} order for {ticker}: {str(e)}", exc_info=True)
        return None

def process_analysis_results(api, analysis):
    """Process analysis results with enhanced logging"""
    logger.info(f"Processing analysis results for {analysis['ticker']}")
    try:
        ticker = analysis['ticker']
        technical_indicators = analysis['technical_indicators']
        lstm_prediction = analysis['lstm_prediction']

        # Log analysis details
        logger.info(f"Analysis details for {ticker}:")
        logger.info(f"Technical Indicators: {json.dumps(technical_indicators, indent=2)}")
        logger.info(f"LSTM Prediction: {lstm_prediction}")

        # Get current position and price
        position = get_position(api, ticker)
        current_price = get_last_price(api, ticker)

        if current_price is None:
            logger.warning(f"Skipping trading decision for {ticker} due to missing price data")
            return

        # Trading logic with detailed logging
        if technical_indicators['rsi'] is not None and lstm_prediction is not None:
            logger.info(f"Evaluating trading conditions for {ticker}")
            logger.info(f"Current RSI: {technical_indicators['rsi']}, LSTM Prediction: {lstm_prediction}, Current Price: {current_price}")

            if technical_indicators['rsi'] < 30 and lstm_prediction > current_price:
                if not position:
                    logger.info(f"Buy signal triggered for {ticker}")
                    place_order(api, ticker, 'buy')
                else:
                    logger.info(f"Buy signal ignored - existing position for {ticker}")
            elif technical_indicators['rsi'] > 70 and lstm_prediction < current_price:
                if position:
                    logger.info(f"Sell signal triggered for {ticker}")
                    place_order(api, ticker, 'sell')
                else:
                    logger.info(f"Sell signal ignored - no position for {ticker}")
            else:
                logger.info(f"No trading signals triggered for {ticker}")

    except Exception as e:
        logger.error(f"Error processing analysis results for {ticker}: {str(e)}", exc_info=True)

def main(analysis_file=None):
    """Main function with enhanced logging"""
    logger.info("Starting trading bot execution")
    try:
        if analysis_file is None and len(sys.argv) > 1:
            analysis_file = sys.argv[1]
            logger.info(f"Using analysis file from command line: {analysis_file}")

        if analysis_file:
            logger.info(f"Loading analysis results from {analysis_file}")
            try:
                with open(analysis_file, 'r') as f:
                    analysis_results = json.load(f)
                logger.info(f"Successfully loaded analysis results for {analysis_results['ticker']}")
                process_analysis_results(api, analysis_results)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse analysis file: {str(e)}", exc_info=True)
            except Exception as e:
                logger.error(f"Error reading analysis file: {str(e)}", exc_info=True)
        else:
            logger.error("No analysis file provided")

    except Exception as e:
        logger.error(f"Fatal error in trading bot: {str(e)}", exc_info=True)
    finally:
        logger.info("Trading bot execution completed")

if __name__ == "__main__":
    main()