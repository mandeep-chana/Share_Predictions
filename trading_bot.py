import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import APIError
import logging
import time
from config import ALPACA_CONFIG
import os
from datetime import datetime, timedelta
import json
from logging.handlers import RotatingFileHandler
import websockets
import asyncio
import sys
import plotly.graph_objects as go
import pandas as pd
import webbrowser

class AlpacaStreamHandler:
    def __init__(self, api_key, secret_key, base_url, logger):
        self.api_key = api_key
        self.secret_key = secret_key
        self.logger = logger
        self.ws_url = ALPACA_CONFIG['WS_URL']
        self.websocket = None
        self.running = False
        self.candlestick_data = []
        self.current_candle = None
        self.last_candle_time = None
        self.candle_interval = 60
        self.html_template = self.load_html_template()

    def load_html_template(self):
        try:
            with open('Stream.html', 'r') as file:
                template = file.read()
                template = template.replace('YOUR_API_KEY', self.api_key)
                template = template.replace('YOUR_SECRET_KEY', self.secret_key)
                return template
        except Exception as e:
            self.logger.error(f"Error loading HTML template: {str(e)}")
            return None

    async def connect(self):
        try:
            self.websocket = await websockets.connect(self.ws_url)
            self.logger.info("WebSocket connection established")

            auth_message = {
                "action": "auth",
                "key": self.api_key,
                "secret": self.secret_key
            }
            await self.websocket.send(json.dumps(auth_message))

            response = await self.websocket.recv()
            auth_response = json.loads(response)

            if auth_response[0]["msg"] == "authenticated":
                self.logger.info("WebSocket authentication successful")
                return True
            else:
                self.logger.error("WebSocket authentication failed")
                return False

        except Exception as e:
            self.logger.error(f"WebSocket connection error: {str(e)}")
            return False

    async def subscribe_to_market_data(self, symbols):
        try:
            subscribe_message = {
                "action": "subscribe",
                "trades": symbols,
                "quotes": symbols,
                "bars": symbols
            }
            await self.websocket.send(json.dumps(subscribe_message))
            response = await self.websocket.recv()
            self.logger.info(f"Subscription response: {response}")

        except Exception as e:
            self.logger.error(f"Subscription error: {str(e)}")

    def update_candlestick(self, trade_data):
        current_time = datetime.now()

        if self.current_candle is None or (current_time - self.last_candle_time).seconds >= self.candle_interval:
            if self.current_candle is not None:
                self.candlestick_data.append(self.current_candle)

            self.current_candle = {
                'timestamp': current_time,
                'open': trade_data['price'],
                'high': trade_data['price'],
                'low': trade_data['price'],
                'close': trade_data['price'],
                'volume': trade_data['size']
            }
            self.last_candle_time = current_time
        else:
            self.current_candle['high'] = max(self.current_candle['high'], trade_data['price'])
            self.current_candle['low'] = min(self.current_candle['low'], trade_data['price'])
            self.current_candle['close'] = trade_data['price']
            self.current_candle['volume'] += trade_data['size']

    async def handle_message(self, message):
        try:
            data = json.loads(message)

            if data[0]['T'] == 'trade':
                trade_data = {
                    'price': float(data[0]['p']),
                    'size': float(data[0]['s']),
                    'timestamp': datetime.fromtimestamp(data[0]['t'] / 1e9)
                }
                self.update_candlestick(trade_data)
                await self.update_chart()

        except Exception as e:
            self.logger.error(f"Error processing message: {str(e)}")

    async def update_chart(self):
        try:
            if not self.candlestick_data:
                return

            df = pd.DataFrame(self.candlestick_data)

            fig = go.Figure(data=[go.Candlestick(
                x=df['timestamp'],
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close']
            )])

            fig.update_layout(
                title='Real-time Candlestick Chart',
                yaxis_title='Price',
                xaxis_title='Time',
                template='plotly_dark'
            )

            fig.write_html('real_time_chart.html', auto_open=False)

        except Exception as e:
            self.logger.error(f"Error updating chart: {str(e)}")

    async def reconnect(self):
        """Reconnect to the WebSocket server"""
        self.logger.info("Attempting to reconnect...")
        await self.connect()

    async def stream_listener(self):
        self.running = True
        while self.running:
            try:
                message = await self.websocket.recv()
                await self.handle_message(message)
            except websockets.exceptions.ConnectionClosed:
                self.logger.error("WebSocket connection closed")
                await self.reconnect()
            except Exception as e:
                self.logger.error(f"Stream listener error: {str(e)}")
                await asyncio.sleep(5)

    async def start(self, symbols=['AAPL']):
        if await self.connect():
            if self.html_template:
                with open('real_time_chart.html', 'w') as f:
                    f.write(self.html_template)
                webbrowser.open('real_time_chart.html')

            await self.subscribe_to_market_data(symbols)
            await self.stream_listener()

    async def stop(self):
        self.running = False
        if self.websocket:
            await self.websocket.close()

def setup_logging():
    logs_dir = 'logs'
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    timestamp = datetime.now().strftime("%Y%m%d")
    main_log = os.path.join(logs_dir, f'trading_bot_main_{timestamp}.log')
    api_log = os.path.join(logs_dir, f'trading_bot_api_{timestamp}.log')
    error_log = os.path.join(logs_dir, f'trading_bot_error_{timestamp}.log')

    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s - [%(filename)s:%(lineno)d]'
    )

    main_handler = RotatingFileHandler(main_log, maxBytes=10*1024*1024, backupCount=5)
    main_handler.setFormatter(detailed_formatter)

    api_handler = RotatingFileHandler(api_log, maxBytes=10*1024*1024, backupCount=5)
    api_handler.setFormatter(detailed_formatter)

    error_handler = RotatingFileHandler(error_log, maxBytes=10*1024*1024, backupCount=5)
    error_handler.setFormatter(detailed_formatter)
    error_handler.setLevel(logging.ERROR)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(detailed_formatter)

    logging.getLogger('').setLevel(logging.INFO)
    logging.getLogger('').addHandler(main_handler)
    logging.getLogger('').addHandler(error_handler)
    logging.getLogger('').addHandler(console_handler)

    api_logger = logging.getLogger('alpaca_api')
    api_logger.addHandler(api_handler)
    api_logger.setLevel(logging.INFO)

    return logging.getLogger('trading_bot'), api_logger

def get_position(api, ticker, api_logger):
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

def get_last_price(api, ticker, api_logger):
    api_logger.info(f"Requesting last price for {ticker}")
    try:
        bars = api.get_bars(ticker, '1Min', limit=1)
        if bars and len(bars) > 0:
            last_price = float(bars[0].c)
            api_logger.info(f"Retrieved last price for {ticker}: ${last_price}")
            return last_price
        api_logger.warning(f"No price data available for {ticker}")
        return None
    except Exception as e:
        api_logger.error(f"Error getting last price for {ticker}: {str(e)}", exc_info=True)
        return None

def place_order(api, ticker, side, logger, api_logger):
    """Place an order with the specified parameters"""
    try:
        if side == 'buy':
            api.submit_order(
                symbol=ticker,
                qty=1,  # Adjust quantity as needed
                side='buy',
                type='market',
                time_in_force='gtc'
            )
            logger.info(f"Buy order placed for {ticker}")
        elif side == 'sell':
            api.submit_order(
                symbol=ticker,
                qty=1,  # Adjust quantity as needed
                side='sell',
                type='market',
                time_in_force='gtc'
            )
            logger.info(f"Sell order placed for {ticker}")
    except Exception as e:
        logger.error(f"Error placing {side} order for {ticker}: {str(e)}")

def process_analysis_results(api, analysis, logger, api_logger):
    logger.info(f"Processing analysis results for {analysis['ticker']}")
    try:
        ticker = analysis['ticker']
        technical_indicators = analysis['technical_indicators']
        lstm_prediction = analysis['lstm_prediction']

        logger.info(f"Analysis details for {ticker}:")
        logger.info(f"Technical Indicators: {json.dumps(technical_indicators, indent=2)}")
        logger.info(f"LSTM Prediction: {lstm_prediction}")

        position = get_position(api, ticker, api_logger)
        current_price = get_last_price(api, ticker, api_logger)

        if current_price is None:
            logger.warning(f"Skipping trading decision for {ticker} due to missing price data")
            return

        if technical_indicators['rsi'] is not None and lstm_prediction is not None:
            logger.info(f"Evaluating trading conditions for {ticker}")
            logger.info(f"Current RSI: {technical_indicators['rsi']}, LSTM Prediction: {lstm_prediction}, Current Price: {current_price}")

            if technical_indicators['rsi'] < 30 and lstm_prediction > current_price:
                if not position:
                    logger.info(f"Buy signal triggered for {ticker}")
                    place_order(api, ticker, 'buy', logger, api_logger)
                else:
                    logger.info(f"Buy signal ignored - existing position for {ticker}")
            elif technical_indicators['rsi'] > 70 and lstm_prediction < current_price:
                if position:
                    logger.info(f"Sell signal triggered for {ticker}")
                    place_order(api, ticker, 'sell', logger, api_logger)
                else:
                    logger.info(f"Sell signal ignored - no position for {ticker}")
            else:
                logger.info(f"No trading signals triggered for {ticker}")

    except Exception as e:
        logger.error(f"Error processing analysis results for {ticker}: {str(e)}", exc_info=True)

async def main_async(analysis_file=None):
    logger, api_logger = setup_logging()
    logger.info("Starting trading bot execution")

    try:
        api = tradeapi.REST(
            key_id=ALPACA_CONFIG['API_KEY'],
            secret_key=ALPACA_CONFIG['SECRET_KEY'],
            base_url=ALPACA_CONFIG['BASE_URL']
        )

        stream_handler = AlpacaStreamHandler(
            ALPACA_CONFIG['API_KEY'],
            ALPACA_CONFIG['SECRET_KEY'],
            ALPACA_CONFIG['BASE_URL'],
            logger
        )

        if analysis_file:
            logger.info(f"Loading analysis results from {analysis_file}")
            try:
                with open(analysis_file, 'r') as f:
                    analysis_results = json.load(f)
                logger.info(f"Successfully loaded analysis results for {analysis_results['ticker']}")
                process_analysis_results(api, analysis_results, logger, api_logger)
            except Exception as e:
                logger.error(f"Error reading analysis file: {str(e)}", exc_info=True)

        await stream_handler.start(['AAPL'])

    except Exception as e:
        logger.error(f"Fatal error in trading bot: {str(e)}", exc_info=True)
    finally:
        logger.info("Trading bot execution completed")

def main(analysis_file=None):
    asyncio.run(main_async(analysis_file))

if __name__ == "__main__":
    main()