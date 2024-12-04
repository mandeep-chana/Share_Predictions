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
            """Initialize the ENTER class"""
            self.api_key = api_key
            self.secret_key = secret_key
            self.logger = logger
            self.ticker = ticker
            self.websocket_clients = set()
            self.ws = None  # WebSocket connection to Alpaca
            self.websocket = None  # Local WebSocket server
            self.running = False
            self.candlestick_data = []

        async def connect_to_alpaca(self):
            """Connect to Alpaca WebSocket stream"""
            try:
                auth_data = {
                    "action": "auth",
                    "key": self.api_key,
                    "secret": self.secret_key
                }
                self.ws = await websockets.connect('wss://stream.data.alpaca.markets/v2/iex')
                await self.ws.send(json.dumps(auth_data))
                response = await self.ws.recv()
                self.logger.info(f"Authentication response: {response}")

                # Subscribe to market data
                await self.subscribe_to_market_data([self.ticker])
            except Exception as e:
                self.logger.error(f"Connection error: {str(e)}")
                raise

        async def subscribe_to_market_data(self, symbols):
            """Subscribe to market data for specified symbols"""
            try:
                subscribe_message = {
                    "action": "subscribe",
                    "bars": symbols,
                    "trades": symbols,  # Subscribe to both bars and trades
                    "quotes": symbols  # Add quotes subscription
                }
                self.logger.debug(f"Sending subscription message: {subscribe_message}")
                await self.ws.send(json.dumps(subscribe_message))
                response = await self.ws.recv()
                self.logger.info(f"Subscription response: {response}")
            except Exception as e:
                self.logger.error(f"Subscription error: {str(e)}")
                raise

        def load_html_template(self):
            """Load and customize the HTML template"""
            html_content = '''
<!DOCTYPE html>
<html>
<head>
    <title>Professional Trading Chart</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body { 
            background-color: #131722;
            margin: 0;
            padding: 20px;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif;
        }
        #chart { 
            height: 800px;
            background-color: #131722;
        }
        .chart-container {
            padding: 20px;
            border-radius: 5px;
        }
        .header {
            color: #ffffff;
            padding: 10px;
            margin-bottom: 10px;
            display: flex;
            align-items: center;
        }
        .price-info {
            margin-left: 20px;
            font-size: 14px;
            color: #787b86;
        }
    </style>
</head>
<body>
    <div class="chart-container">
        <div class="header">
            <h2 id="symbol">Loading...</h2>
            <div class="price-info">
                <span id="price">Price: --</span>
                <span id="volume">Volume: --</span>
            </div>
        </div>
        <div id="chart"></div>
    </div>

    <script>
        const ws = new WebSocket('ws://localhost:8765');

        let candleData = {
            x: [],
            open: [],
            high: [],
            low: [],
            close: [],
            type: 'candlestick',
            increasing: {
                line: { width: 2, color: '#26a69a' },
                fillcolor: '#26a69a'
            },
            decreasing: {
                line: { width: 2, color: '#ef5350' },
                fillcolor: '#ef5350'
            },
            whiskerwidth: 1,
            width: 0.7
        };

        let volumeData = {
            x: [],
            y: [],
            type: 'bar',
            yaxis: 'y2',
            marker: {
                color: [],
                line: { width: 0 }
            },
            opacity: 0.5
        };

        let layout = {
            plot_bgcolor: '#131722',
            paper_bgcolor: '#131722',
            dragmode: 'zoom',
            margin: { r: 60, t: 25, b: 40, l: 60 },
            showlegend: false,
            xaxis: {
                rangeslider: { visible: false },
                type: 'date',
                gridcolor: '#1e222d',
                linecolor: '#1e222d',
                tickfont: { color: '#787b86' },
                tickformat: '%H:%M:%S'
            },
            yaxis: {
                title: 'Price',
                titlefont: { color: '#787b86' },
                gridcolor: '#1e222d',
                linecolor: '#1e222d',
                tickfont: { color: '#787b86' },
                domain: [0.3, 1]
            },
            yaxis2: {
                title: 'Volume',
                titlefont: { color: '#787b86' },
                gridcolor: '#1e222d',
                linecolor: '#1e222d',
                tickfont: { color: '#787b86' },
                domain: [0, 0.2]
            }
        };

        Plotly.newPlot('chart', [candleData, volumeData], layout, {
            responsive: true,
            displayModeBar: false
        });

        ws.onmessage = function(event) {
            try {
                const data = JSON.parse(event.data);

                if (data.symbol) {
                    document.getElementById('symbol').textContent = data.symbol;
                }

                if (data.candlestick) {
                    const timestamp = new Date(data.candlestick.timestamp);

                    // Update candlestick data
                    candleData.x.push(timestamp);
                    candleData.open.push(data.candlestick.open);
                    candleData.high.push(data.candlestick.high);
                    candleData.low.push(data.candlestick.low);
                    candleData.close.push(data.candlestick.close);

                    // Update volume data
                    volumeData.x.push(timestamp);
                    volumeData.y.push(data.candlestick.volume);
                    volumeData.marker.color.push(
                        data.candlestick.close >= data.candlestick.open ? 
                        'rgba(38, 166, 154, 0.5)' : 'rgba(239, 83, 80, 0.5)'
                    );

                    // Keep last 100 points
                    if (candleData.x.length > 100) {
                        candleData.x.shift();
                        candleData.open.shift();
                        candleData.high.shift();
                        candleData.low.shift();
                        candleData.close.shift();
                        volumeData.x.shift();
                        volumeData.y.shift();
                        volumeData.marker.color.shift();
                    }

                    // Update price info
                    document.getElementById('price').textContent = 
                        `Price: ${data.candlestick.close.toFixed(2)}`;
                    document.getElementById('volume').textContent = 
                        `Volume: ${data.candlestick.volume}`;

                    // Update the chart
                    Plotly.update('chart', 
                        {
                            x: [candleData.x, volumeData.x],
                            open: [candleData.open],
                            high: [candleData.high],
                            low: [candleData.low],
                            close: [candleData.close],
                            y: [, volumeData.y],
                            'marker.color': [, volumeData.marker.color]
                        },
                        {
                            'xaxis.range': [
                                new Date(Date.now() - 30 * 60 * 1000),
                                new Date(Date.now())
                            ]
                        }
                    );
                }
            } catch (error) {
                console.error('Error processing message:', error);
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
                    "symbol": self.ticker,
                    "candlestick": {
                        "timestamp": datetime.now().isoformat(),
                        "open": 0,
                        "high": 0,
                        "low": 0,
                        "close": 0,
                        "volume": 0
                    }
                }
                await websocket.send(json.dumps(initial_message))
                self.logger.info("Sent initial test message")

                try:
                    async for message in websocket:
                        self.logger.debug(f"Received client message: {message}")
                except websockets.exceptions.ConnectionClosed:
                    pass
                finally:
                    self.websocket_clients.remove(websocket)
                    self.logger.info("Client disconnected")
            except Exception as e:
                self.logger.error(f"Error handling client connection: {str(e)}")

        async def broadcast_to_clients(self, message):
            """Broadcast message to all connected clients"""
            if self.websocket_clients:
                await asyncio.gather(
                    *[client.send(json.dumps(message)) for client in self.websocket_clients.copy()]
                )

        async def handle_message(self, message):
            """Handle incoming messages from Alpaca"""
            try:
                data = json.loads(message)
                self.logger.debug(f"Received raw message: {message}")

                if isinstance(data, list) and len(data) > 0:
                    for msg in data:
                        self.logger.debug(f"Processing message: {msg}")

                        # Handle bar data
                        if msg.get('T') == 'b':  # Bar data
                            # Extract and validate data
                            if all(key in msg for key in ['t', 'o', 'h', 'l', 'c', 'v', 'S']):
                                candlestick_data = {
                                    'symbol': msg['S'],
                                    'candlestick': {
                                        'timestamp': msg['t'],
                                        'open': float(msg['o']),
                                        'high': float(msg['h']),
                                        'low': float(msg['l']),
                                        'close': float(msg['c']),
                                        'volume': float(msg['v'])
                                    }
                                }

                                self.logger.debug(f"Broadcasting candlestick data: {candlestick_data}")
                                await self.broadcast_to_clients(candlestick_data)
                                self.logger.info(f"Sent candlestick data: {candlestick_data}")
                            else:
                                self.logger.warning(f"Incomplete bar data received: {msg}")

            except json.JSONDecodeError as e:
                self.logger.error(f"JSON decode error: {str(e)}, Message: {message}")
            except Exception as e:
                self.logger.error(f"Error handling message: {str(e)}", exc_info=True)

        async def start(self):
            """Start the WebSocket server and connect to Alpaca"""
            try:
                self.logger.info("Starting local WebSocket server...")
                self.load_html_template()

                # Start local WebSocket server
                start_server = websockets.serve(
                    self.handle_client,
                    "localhost",
                    8765
                )
                await start_server

                # Connect to Alpaca
                await self.connect_to_alpaca()
                self.running = True

                # Main message loop
                while self.running:
                    try:
                        message = await self.ws.recv()
                        await self.handle_message(message)
                    except websockets.exceptions.ConnectionClosed:
                        self.logger.error("Connection to Alpaca closed unexpectedly")
                        break
                    except Exception as e:
                        self.logger.error(f"Error in message loop: {str(e)}")
                        break

            except Exception as e:
                self.logger.error(f"Error in start method: {str(e)}")
                raise

        async def stop(self):
            """Stop the WebSocket server and disconnect from Alpaca"""
            self.running = False
            if self.ws:
                await self.ws.close()
            # Close all client connections
            if self.websocket_clients:
                await asyncio.gather(*[client.close() for client in self.websocket_clients])
            self.websocket_clients.clear()

def optimize_strategy(analyzer, strategy_class, parameter_ranges):
    """Optimize strategy parameters using grid search with improved error handling"""
    try:
        logging.info(f"Starting strategy optimization for {analyzer.ticker_symbol}")

        best_params = None
        best_score = float('-inf')
        results = []

        # Generate parameter combinations
        param_combinations = [dict(zip(parameter_ranges.keys(), v))
                            for v in itertools.product(*parameter_ranges.values())]

        # Add progress tracking
        total_combinations = len(param_combinations)
        processed = 0

        for params in param_combinations:
            try:
                processed += 1
                logging.info(f"Testing combination {processed}/{total_combinations}: {params}")

                # Create strategy with current parameters
                strategy = type('OptimizedStrategy', (strategy_class,), params)

                # Run backtest with improved error handling
                bt = Backtest(analyzer.stock_data, strategy, cash=10000, commission=.002)

                # Suppress numpy warnings and handle division by zero
                with np.errstate(divide='ignore', invalid='ignore'):
                    stats = bt.run()

                    # Calculate metrics with safety checks
                    sharpe_ratio = float(stats['Sharpe Ratio']) if not np.isnan(stats['Sharpe Ratio']) and not np.isinf(stats['Sharpe Ratio']) else -999
                    return_pct = float(stats['Return [%]']) if not np.isnan(stats['Return [%]']) else 0
                    max_drawdown = float(stats['Max. Drawdown [%]']) if not np.isnan(stats['Max. Drawdown [%]']) else -100
                    win_rate = float(stats['Win Rate [%]']) if not np.isnan(stats['Win Rate [%]']) else 0
                    profit_factor = float(stats.get('Profit Factor', 0)) if not np.isnan(stats.get('Profit Factor', 0)) else 0
                    num_trades = int(stats['# Trades'])

                    # Skip invalid combinations
                    if num_trades == 0:
                        continue

                    # Calculate composite score
                    score = calculate_strategy_score(
                        sharpe_ratio=sharpe_ratio,
                        return_pct=return_pct,
                        max_drawdown=max_drawdown,
                        win_rate=win_rate,
                        profit_factor=profit_factor,
                        num_trades=num_trades
                    )

                    result = {
                        'parameters': params,
                        'score': score,
                        'sharpe_ratio': sharpe_ratio,
                        'return_pct': return_pct,
                        'max_drawdown': max_drawdown,
                        'win_rate': win_rate,
                        'profit_factor': profit_factor,
                        'num_trades': num_trades
                    }

                    results.append(result)

                    # Update best parameters if better score found
                    if score > best_score:
                        best_score = score
                        best_params = params.copy()
                        logging.info(f"New best parameters found: {best_params} (Score: {best_score:.2f})")

            except Exception as e:
                logging.warning(f"Error in parameter combination {params}: {str(e)}")
                continue

        # If no valid results found, use default parameters
        if not best_params:
            logging.warning("No optimal parameters found, using default parameters")
            best_params = {
                'bb_window': 20,
                'bb_std': 2.0,
                'rsi_window': 14,
                'rsi_upper': 70,
                'rsi_lower': 30
            }

        logging.info(f"Optimization completed. Best parameters: {best_params}")

        # Create summary of results
        if results:
            results_df = pd.DataFrame(results)
            summary = create_optimization_summary(results_df, analyzer.ticker_symbol)
            logging.info("\nOptimization Summary:\n" + summary)

        return best_params, pd.DataFrame(results) if results else None

    except Exception as e:
        logging.error(f"Error in strategy optimization: {str(e)}")
        return None, None

def calculate_strategy_score(sharpe_ratio, return_pct, max_drawdown, win_rate, profit_factor, num_trades):
    """Calculate a composite score for strategy evaluation"""
    try:
        # Normalize and weight different metrics
        normalized_sharpe = min(max(sharpe_ratio, -3), 3) / 3  # Normalize between -1 and 1
        normalized_return = min(max(return_pct, -100), 100) / 100  # Normalize between -1 and 1
        normalized_drawdown = (max_drawdown + 100) / 100  # Normalize between 0 and 1
        normalized_winrate = win_rate / 100  # Already between 0 and 1
        normalized_trades = min(num_trades / 100, 1)  # Normalize with max at 100 trades

        # Weighted sum of metrics
        score = (
            0.3 * normalized_sharpe +
            0.25 * normalized_return +
            0.2 * (1 - normalized_drawdown) +  # Invert drawdown so higher is better
            0.15 * normalized_winrate +
            0.1 * normalized_trades
        )

        return score

    except Exception as e:
        logging.error(f"Error calculating strategy score: {str(e)}")
        return float('-inf')

def create_optimization_summary(results_df, ticker):
    """Create a summary of optimization results"""
    try:
        summary = f"\nOptimization Summary for {ticker}\n"
        summary += "=" * 50 + "\n"

        # Best strategy details
        best_row = results_df.loc[results_df['score'].idxmax()]
        summary += f"Best Strategy Parameters:\n"
        for param, value in best_row['parameters'].items():
            summary += f"  {param}: {value}\n"

        summary += f"\nPerformance Metrics:\n"
        summary += f"  Sharpe Ratio: {best_row['sharpe_ratio']:.2f}\n"
        summary += f"  Return: {best_row['return_pct']:.2f}%\n"
        summary += f"  Max Drawdown: {best_row['max_drawdown']:.2f}%\n"
        summary += f"  Win Rate: {best_row['win_rate']:.2f}%\n"
        summary += f"  Number of Trades: {best_row['num_trades']}\n"

        return summary

    except Exception as e:
        logging.error(f"Error creating optimization summary: {str(e)}")
        return "Error creating optimization summary"

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