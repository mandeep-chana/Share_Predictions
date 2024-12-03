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
        self.trade_ws_url = ALPACA_CONFIG['TRADE_WS_URL']
        self.websocket = None
        self.running = False
        self.clients = set()  # Add this for WebSocket server
        self.candlestick_data = []
        self.current_candle = None
        self.last_candle_time = None
        self.candle_interval = 60
        self.html_template = '''<!DOCTYPE html>
        <html>
        <head>
            <title>Real-time Candlestick Chart</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body { background-color: #1a1a1a; color: white; }
                #chart { width: 100%; height: 800px; }
                #ticker-info {
                    font-size: 24px;
                    font-weight: bold;
                    margin: 10px 0;
                    padding: 10px;
                    background-color: #2a2a2a;
                    border-radius: 5px;
                    display: inline-block;
                }
            </style>
        </head>
        <body>
            <div id="ticker-info">Ticker: <span id="ticker-symbol">Loading...</span></div>
            <div id="chart"></div>
            <script>
                const urlParams = new URLSearchParams(window.location.search);
                const ticker = urlParams.get('symbol') || 'AAPL';
                document.getElementById('ticker-symbol').textContent = ticker;

                let candleData = {
                    timestamp: [],
                    open: [],
                    high: [],
                    low: [],
                    close: [],
                    volume: []
                };

                function updateChart() {
                    const trace = {
                        x: candleData.timestamp,
                        open: candleData.open,
                        high: candleData.high,
                        low: candleData.low,
                        close: candleData.close,
                        type: 'candlestick',
                        xaxis: 'x',
                        yaxis: 'y'
                    };

                    const layout = {
                        title: `${ticker} Real-time Candlestick Chart`,
                        yaxis: {title: 'Price'},
                        xaxis: {title: 'Time'},
                        paper_bgcolor: '#1a1a1a',
                        plot_bgcolor: '#1a1a1a',
                        font: { color: '#ffffff' }
                    };

                    Plotly.newPlot('chart', [trace], layout);
                }

                // Initialize empty chart
                updateChart();

                // WebSocket connection
                const ws = new WebSocket('wss://stream.data.alpaca.markets/v2/iex');

                ws.onopen = () => {
                    console.log('Connected to Alpaca WebSocket');
                    ws.send(JSON.stringify({
                        "action": "auth",
                        "key": "''' + self.api_key + '''",
                        "secret": "''' + self.secret_key + '''"
                    }));
                };

                ws.onmessage = (event) => {
                    const data = JSON.parse(event.data);

                    if (data.stream === 'authorization' && data.data.status === 'authorized') {
                        console.log('Authenticated, subscribing to trades');
                        ws.send(JSON.stringify({
                            "action": "subscribe",
                            "trades": [ticker],
                            "quotes": [ticker],
                            "bars": [ticker]
                        }));
                    }

                    if (data.stream === 'trade') {
                        const trade = data.data;
                        // Update candlestick data
                        const timestamp = new Date(trade.t);
                        candleData.timestamp.push(timestamp);
                        candleData.open.push(trade.p);
                        candleData.high.push(trade.p);
                        candleData.low.push(trade.p);
                        candleData.close.push(trade.p);
                        candleData.volume.push(trade.s);

                        // Update chart
                        updateChart();
                    }
                };

                ws.onerror = (error) => {
                    console.error('WebSocket error:', error);
                };

                ws.onclose = () => {
                    console.log('WebSocket connection closed');
                };
            </script>
        </body>
        </html>'''

    async def start_server(self):
        """Start the local WebSocket server."""
        try:
            server = await websockets.serve(
                self.websocket_server,
                "localhost",
                8765
            )
            self.logger.info("Local WebSocket server started on ws://localhost:8765")
            return server
        except Exception as e:
            self.logger.error(f"Error starting WebSocket server: {str(e)}")
            raise

    async def websocket_server(self, websocket, path):
        """Handle web client connections."""
        self.clients.add(websocket)
        try:
            await websocket.wait_closed()
        finally:
            self.clients.remove(websocket)

    async def broadcast(self, message):
        """Broadcast message to all connected web clients."""
        if self.clients:
            await asyncio.gather(
                *[client.send(json.dumps(message)) for client in self.clients]
            )

    def load_html_template(self):
        try:
            with open('Stream.html', 'w') as file:
                file.write(self.html_template)
            return True
        except Exception as e:
            self.logger.error(f"Error creating HTML template: {str(e)}")
            return False

    async def connect(self):
        try:
            self.websocket = await websockets.connect(self.ws_url)
            self.logger.info("Market data WebSocket connection established")

            # First connection response
            initial_response = await self.websocket.recv()
            initial_data = json.loads(initial_response)
            self.logger.info(f"Initial connection response: {initial_data}")

            # Send authentication message
            auth_message = {
                "action": "auth",
                "key": self.api_key,
                "secret": self.secret_key
            }
            await self.websocket.send(json.dumps(auth_message))

            # Get authentication response
            auth_response = await self.websocket.recv()
            auth_data = json.loads(auth_response)
            self.logger.info(f"Authentication response: {auth_data}")

            # Check for successful authentication
            if isinstance(auth_data, list) and auth_data[0].get('T') == 'success':
                self.logger.info("WebSocket authentication successful")
                return True
            elif isinstance(auth_data, dict) and auth_data.get('stream') == 'authorization' and auth_data.get('data', {}).get('status') == 'authorized':
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
            symbols = [symbol.upper() for symbol in symbols]

            # Updated subscription message for v2 API
            subscribe_message = {
                "action": "subscribe",
                "trades": symbols,
                "quotes": symbols,
                "bars": symbols
            }

            await self.websocket.send(json.dumps(subscribe_message))

            # Get subscription confirmation
            response = await self.websocket.recv()
            subscription_data = json.loads(response)
            self.logger.info(f"Subscription response: {subscription_data}")

            # Check for successful subscription (v2 API format)
            if isinstance(subscription_data, list) and subscription_data[0].get('T') == 'subscription':
                self.logger.info(f"Successfully subscribed to data for symbols: {symbols}")
                return True
            else:
                self.logger.warning(f"Unexpected subscription response: {subscription_data}")
                return False

        except Exception as e:
            self.logger.error(f"Subscription error: {str(e)}", exc_info=True)
            return False

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
            self.logger.debug(f"Received message: {data}")

            if isinstance(data, dict):
                stream = data.get('stream')

                if stream == 'trade':
                    trade_data = data.get('data', {})
                    processed_trade = {
                        'price': float(trade_data.get('p', 0)),
                        'size': float(trade_data.get('s', 0)),
                        'timestamp': datetime.fromtimestamp(trade_data.get('t', 0) / 1e9)
                    }
                    self.update_candlestick(processed_trade)
                    await self.update_chart()

                elif stream == 'error':
                    self.logger.error(f"Stream error: {data}")

                elif stream == 'listening':
                    self.logger.info(f"Listening confirmation: {data}")

        except Exception as e:
            self.logger.error(f"Error processing message: {str(e)}", exc_info=True)

    async def reconnect(self):
        """Reconnect to the WebSocket server"""
        self.logger.info("Attempting to reconnect...")
        retry_count = 0
        max_retries = 5

        while retry_count < max_retries:
            try:
                if await self.connect():
                    await self.subscribe_to_market_data(['AAPL'])  # Re-subscribe to data
                    self.logger.info("Successfully reconnected")
                    return True
                retry_count += 1
                await asyncio.sleep(5 * retry_count)  # Exponential backoff
            except Exception as e:
                self.logger.error(f"Reconnection attempt {retry_count + 1} failed: {str(e)}")

        self.logger.error("Failed to reconnect after maximum retries")
        return False

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

    async def stream_listener(self):
        self.running = True
        while self.running:
            try:
                message = await self.websocket.recv()
                await self.handle_message(message)
            except websockets.exceptions.ConnectionClosed:
                self.logger.error("WebSocket connection closed")
                if await self.reconnect():
                    continue
                else:
                    break
            except Exception as e:
                self.logger.error(f"Stream listener error: {str(e)}")
                await asyncio.sleep(5)

    async def start(self, symbols=['AAPL']):
        """Start both the local server and Alpaca connection."""
        # Start local WebSocket server
        server = await self.start_server()

        # Load and open HTML template with symbol parameter
        if self.load_html_template():
            symbol = symbols[0] if symbols else 'AAPL'
            webbrowser.open(f'file://{os.path.abspath("Stream.html")}?symbol={symbol}')

        # Connect to Alpaca
        if await self.connect():
            await self.subscribe_to_market_data(symbols)
            await self.stream_listener()

        # Keep the server running
        await asyncio.Future()  # run forever

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
    logger, api_logger = setup_logging()
    logger.info("Starting trading bot execution")

    try:
        stream_handler = AlpacaStreamHandler(
            ALPACA_CONFIG['API_KEY'],
            ALPACA_CONFIG['SECRET_KEY'],
            ALPACA_CONFIG['BASE_URL'],
            logger
        )

        # Create and open the chart first
        if stream_handler.load_html_template():
            webbrowser.open('Stream.html')

        # Then start the WebSocket connection
        await stream_handler.start(['AAPL'])

    except Exception as e:
        logger.error(f"Fatal error in trading bot: {str(e)}", exc_info=True)
    finally:
        logger.info("Trading bot execution completed")

def main(analysis_file=None):
    asyncio.run(main_async(analysis_file))

if __name__ == "__main__":
    main()