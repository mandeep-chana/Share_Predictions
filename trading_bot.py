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

        def __init__(self, api_key, secret_key, logger, ticker):
            self.api_key = api_key
            self.secret_key = secret_key
            self.logger = logger
            self.ticker = ticker
            self.websocket_clients = set()
            self.ws = None  # WebSocket connection to Alpaca
            self.websocket = None  # Local WebSocket server
            self.running = False
            self.candlestick_data = []

        def load_html_template(self):
            """Load and customize the HTML template"""
            html_content = '''
            <!DOCTYPE html>
            <html>
            <head>
                <title>Real-time Stock Chart</title>
                <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            </head>
            <body>
                <h2>Real-time Stock Chart for <span id="ticker"></span></h2>
                <div id="chart"></div>
                <script>
                    const ws = new WebSocket('ws://localhost:8765');
                    let trace = {
                        x: [],
                        y: [],
                        type: 'scatter',
                        mode: 'lines',
                        name: 'Price'
                    };
                    let layout = {
                        title: 'Real-time Stock Price',
                        xaxis: { title: 'Time' },
                        yaxis: { title: 'Price' }
                    };
                    Plotly.newPlot('chart', [trace], layout);

                    ws.onmessage = function(event) {
                        const data = JSON.parse(event.data);
                        if (data.symbol) {
                            document.getElementById('ticker').textContent = data.symbol;
                        }
                        if (data.price && data.timestamp) {
                            trace.x.push(new Date(data.timestamp));
                            trace.y.push(data.price);
                            Plotly.update('chart', [trace], layout);
                        }
                    };
                </script>
            </body>
            </html>
            '''
            with open('Stream.html', 'w') as f:
                f.write(html_content)
            self.logger.info("HTML template created successfully")

        async def handle_client(self, websocket, path):
            """Handle web client connections"""
            try:
                self.websocket_clients.add(websocket)
                self.logger.info("New client connected")

                # Send initial ticker information
                initial_message = {
                    "symbol": self.ticker
                }
                await websocket.send(json.dumps(initial_message))

                try:
                    async for message in websocket:
                        # Handle any messages from the client if needed
                        pass
                except websockets.exceptions.ConnectionClosed:
                    pass
                finally:
                    self.websocket_clients.remove(websocket)
                    self.logger.info("Client disconnected")
            except Exception as e:
                self.logger.error(f"Error handling client connection: {str(e)}")

        async def connect(self):
            """Connect to Alpaca WebSocket"""
            try:
                self.ws = await websockets.connect('wss://stream.data.alpaca.markets/v2/iex')
                self.logger.info("Connected to Alpaca WebSocket")
            except Exception as e:
                self.logger.error(f"Error connecting to Alpaca WebSocket: {str(e)}")
                raise

        async def authenticate(self):
            """Authenticate with Alpaca"""
            try:
                auth_message = {
                    "action": "auth",
                    "key": self.api_key,
                    "secret": self.secret_key
                }
                await self.ws.send(json.dumps(auth_message))
                response = await self.ws.recv()
                self.logger.info(f"Authentication response: {response}")
            except Exception as e:
                self.logger.error(f"Authentication error: {str(e)}")
                raise

        async def subscribe_to_market_data(self, symbols):
            """Subscribe to market data for specified symbols"""
            try:
                subscribe_message = {
                    "action": "subscribe",
                    "trades": symbols,
                    "quotes": symbols,
                    "bars": symbols
                }
                await self.ws.send(json.dumps(subscribe_message))
                response = await self.ws.recv()
                self.logger.info(f"Subscription response: {response}")
            except Exception as e:
                self.logger.error(f"Subscription error: {str(e)}")
                raise

        async def handle_message(self, message):
            """Handle incoming messages from Alpaca"""
            try:
                data = json.loads(message)
                if isinstance(data, list) and len(data) > 0:
                    for msg in data:
                        if msg.get('T') == 't':  # Trade message
                            await self.broadcast_to_clients({
                                'symbol': msg.get('S'),
                                'price': msg.get('p'),
                                'timestamp': msg.get('t')
                            })
            except Exception as e:
                self.logger.error(f"Error handling message: {str(e)}")

        async def broadcast_to_clients(self, message):
            """Broadcast message to all connected clients"""
            if self.websocket_clients:
                disconnected_clients = set()
                for client in self.websocket_clients:
                    try:
                        await client.send(json.dumps(message))
                    except websockets.exceptions.ConnectionClosed:
                        disconnected_clients.add(client)
                    except Exception as e:
                        self.logger.error(f"Error broadcasting to client: {str(e)}")
                        disconnected_clients.add(client)

                # Remove disconnected clients
                self.websocket_clients -= disconnected_clients

        async def stream_listener(self):
            """Listen for messages from Alpaca WebSocket"""
            try:
                while self.running:
                    if self.ws:
                        message = await self.ws.recv()
                        await self.handle_message(message)
                    else:
                        self.logger.error("WebSocket connection not established")
                        await asyncio.sleep(5)
                        await self.reconnect()
            except websockets.exceptions.ConnectionClosed:
                self.logger.error("WebSocket connection closed")
                await self.reconnect()
            except Exception as e:
                self.logger.error(f"Stream listener error: {str(e)}")
                await asyncio.sleep(5)

        async def reconnect(self):
            """Reconnect to Alpaca WebSocket"""
            try:
                self.logger.info("Attempting to reconnect...")
                await self.connect()
                await self.authenticate()
                await self.subscribe_to_market_data([self.ticker])
            except Exception as e:
                self.logger.error(f"Error reconnecting: {str(e)}")

        async def start(self):
            """Start the WebSocket server and connect to Alpaca"""
            try:
                self.logger.info("=== Starting AlpacaStreamHandler ===")

                # Start local WebSocket server
                self.logger.info("Starting local WebSocket server...")
                self.websocket = await websockets.serve(
                    self.handle_client,
                    'localhost',
                    8765
                )
                self.logger.info("Local WebSocket server started on ws://localhost:8765")

                # Connect to Alpaca WebSocket
                self.logger.info("Connecting to Alpaca WebSocket...")
                await self.connect()
                self.logger.info("Connected to Alpaca WebSocket")

                # Authenticate
                self.logger.info("Authenticating with Alpaca...")
                await self.authenticate()
                self.logger.info("Authentication successful")

                # Subscribe to market data
                self.logger.info(f"Subscribing to market data for {self.ticker}...")
                await self.subscribe_to_market_data([self.ticker])
                self.logger.info("Market data subscription successful")

                # Start listening for messages
                self.logger.info("Starting message listener...")
                self.running = True

                # Keep the WebSocket connection alive
                while True:
                    try:
                        await self.stream_listener()
                    except websockets.exceptions.ConnectionClosed:
                        self.logger.error("WebSocket connection closed")
                        await self.reconnect()
                    except Exception as e:
                        self.logger.error(f"Error in stream listener: {str(e)}")
                        await asyncio.sleep(5)  # Wait before retrying

            except Exception as e:
                self.logger.error(f"Error in start method: {str(e)}", exc_info=True)
                raise

def optimize_strategy(analyzer, strategy_class, parameter_ranges):
    """Optimize strategy parameters using grid search"""
    try:
        logging.info(f"Starting strategy optimization for {analyzer.ticker_symbol}")

        best_params = None
        best_sharpe = float('-inf')
        results = []

        # Generate parameter combinations
        param_combinations = [dict(zip(parameter_ranges.keys(), v))
                            for v in itertools.product(*parameter_ranges.values())]

        for params in param_combinations:
            try:
                # Create strategy with current parameters
                strategy = type('OptimizedStrategy', (strategy_class,), params)

                # Run backtest with error handling
                bt = Backtest(analyzer.stock_data, strategy, cash=10000, commission=.002)
                with np.errstate(divide='ignore', invalid='ignore'):  # Handle divide by zero
                    stats = bt.run()

                # Extract results and handle potential NaN values
                result = {
                    'parameters': params,
                    'sharpe_ratio': float(stats['Sharpe Ratio']) if not np.isnan(stats['Sharpe Ratio']) else -999,
                    'return_pct': float(stats['Return [%]']) if not np.isnan(stats['Return [%]']) else 0,
                    'max_drawdown': float(stats['Max. Drawdown [%]']) if not np.isnan(stats['Max. Drawdown [%]']) else -100,
                    'win_rate': float(stats['Win Rate [%]']) if not np.isnan(stats['Win Rate [%]']) else 0,
                    'profit_factor': float(stats.get('Profit Factor', 0)) if not np.isnan(stats.get('Profit Factor', 0)) else 0,
                    'num_trades': int(stats['# Trades'])
                }

                results.append(result)

                # Update best parameters if necessary
                if result['sharpe_ratio'] > best_sharpe and result['sharpe_ratio'] != -999:
                    best_sharpe = result['sharpe_ratio']
                    best_params = params.copy()

            except Exception as e:
                logging.warning(f"Error in parameter combination {params}: {str(e)}")
                continue

        return best_params, pd.DataFrame(results)

    except Exception as e:
        logging.error(f"Error in strategy optimization: {str(e)}")
        return None, None

def setup_logging():
    """Set up logging configuration"""
    # Create logs directory if it doesn't exist
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)

    # Create logger
    logger = logging.getLogger('trading_bot')
    logger.setLevel(logging.INFO)

    # Create log file path with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'trading_bot_{timestamp}.log')

    # Create file handler
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5
    )

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create formatters and add it to the handlers
    file_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s - [%(filename)s:%(lineno)d]'
    console_format = '%(asctime)s - %(levelname)s - %(message)s'

    file_formatter = logging.Formatter(file_format)
    console_formatter = logging.Formatter(console_format)

    file_handler.setFormatter(file_formatter)
    console_handler.setFormatter(console_formatter)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Log initial startup message
    logger.info("Logger initialized successfully")

    return logger


def get_position(api, ticker, api_logger):
    api_logger.info(f"Requesting position information for {ticker}")
    try:
        position = api.get_position(ticker)
        api_logger.info(f"Current position for {ticker}: {position.qty} shares at avg entry ${position.avg_entry_price}")
        return position
    except APIError as e:
        if "position does not exist" in str(e) or "404" in str(e):
            api_logger.info(f"No current position exists for {ticker}")
            return None
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
    """Main async function to run the trading bot"""
    try:
        # Initialize logger
        logger = setup_logging()
        logger.info("=== Trading Bot Starting ===")
        logger.info("Initializing components...")

        # Read ticker from file
        logger.info("Reading ticker from file...")
        with open('tickers.txt', 'r') as file:
            ticker = file.readline().strip()
        logger.info(f"Loaded ticker: {ticker}")

        # Initialize stream handler
        logger.info("Initializing Alpaca Stream Handler...")
        stream_handler = AlpacaStreamHandler(
            ALPACA_CONFIG['API_KEY'],
            ALPACA_CONFIG['SECRET_KEY'],
            logger,
            ticker
        )
        logger.info("Stream handler initialized successfully")

        # Generate and open HTML template
        logger.info("Generating HTML template...")
        stream_handler.load_html_template()
        logger.info("Opening HTML template in browser...")
        webbrowser.open('file://' + os.path.realpath('Stream.html'))

        # Start the WebSocket server
        logger.info("Starting WebSocket server...")
        await stream_handler.start()
        logger.info("WebSocket server started successfully")

    except Exception as e:
        logger.error(f"Fatal error in trading bot: {str(e)}", exc_info=True)
        raise
    finally:
        logger.info("=== Trading Bot Shutdown ===")

def main(analysis_file=None):
    try:
        asyncio.run(main_async(analysis_file))
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
    except Exception as e:
        print(f"Error in main: {str(e)}")

if __name__ == "__main__":
    main()