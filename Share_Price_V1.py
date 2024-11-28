import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from prophet import Prophet
from bs4 import BeautifulSoup
import requests
import matplotlib.pyplot as plt
from textblob import TextBlob
import os
import json
import ta
from news_sources import NEWS_SOURCES
from company_sources import COMPANY_SOURCES, get_clean_ticker
import logging
from logging.handlers import RotatingFileHandler
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from backtesting.test import SMA
import itertools
from bokeh.models import DatetimeTickFormatter
# ML and Deep Learning imports
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
# Additional imports for economic indicators and options data
import pandas_datareader as pdr
from fredapi import Fred
import mibian


def parse_date(date_str):
    """Parse a date string in the format YYYY-MM-DD."""
    try:
        return datetime.strptime(date_str, '%Y-%m-%d')
    except ValueError:
        raise ValueError(f"Invalid date format: {date_str}. Expected format: YYYY-MM-DD")

def get_date_input(prompt):
    while True:
        date_str = input(prompt).strip()
        if not date_str:  # If the user presses Enter
            return None
        try:
            return parse_date(date_str)  # Assuming parse_date is your date parsing function
        except ValueError as e:
            print(f"Invalid date format: {e}. Please use YYYY-MM-DD format.")

class SmaCrossWithRisk(Strategy):
    n1 = 10  # Short-term SMA
    n2 = 20  # Long-term SMA
    risk_per_trade = 0.02  # 2% of total equity
    stop_loss_atr_multiplier = 2
    trailing_stop_atr_multiplier = 1.5

    def init(self):
        # Initialize indicators
        self.sma1 = self.I(SMA, self.data.Close, self.n1)
        self.sma2 = self.I(SMA, self.data.Close, self.n2)
        # Calculate ATR manually using Pandas
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)
        close = pd.Series(self.data.Close)
        self.atr = self.I(self.calculate_atr, high, low, close)

    def calculate_atr(self, high, low, close, period=14):
        """Calculate ATR using the classic formula."""
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr

    def next(self):
        # Calculate position size based on ATR
        atr = self.atr[-1]
        position_size = self.calculate_position_size(atr)

        # Check for buy signal
        if crossover(self.sma1, self.sma2):
            self.buy(size=position_size)
            self.stop_loss_price = self.data.Close[-1] - self.stop_loss_atr_multiplier * atr
            self.trailing_stop_price = self.data.Close[-1] - self.trailing_stop_atr_multiplier * atr

        # Check for sell signal
        elif crossover(self.sma2, self.sma1):
            self.sell(size=position_size)
            self.stop_loss_price = self.data.Close[-1] + self.stop_loss_atr_multiplier * atr
            self.trailing_stop_price = self.data.Close[-1] + self.trailing_stop_atr_multiplier * atr

        # Update stop-loss and trailing stop for open positions
        if self.position.is_long:
            self.trailing_stop_price = max(self.trailing_stop_price,
                                           self.data.Close[-1] - self.trailing_stop_atr_multiplier * atr)
            if self.data.Close[-1] < self.stop_loss_price or self.data.Close[-1] < self.trailing_stop_price:
                self.position.close()

        elif self.position.is_short:
            self.trailing_stop_price = min(self.trailing_stop_price,
                                           self.data.Close[-1] + self.trailing_stop_atr_multiplier * atr)
            if self.data.Close[-1] > self.stop_loss_price or self.data.Close[-1] > self.trailing_stop_price:
                self.position.close()

    def calculate_position_size(self, atr):
        """Calculate position size based on risk per trade and ATR."""
        risk_amount = self.equity * self.risk_per_trade
        position_size = risk_amount / (atr * self.stop_loss_atr_multiplier)
        # Ensure position size is a positive whole number of units
        return max(1, int(position_size))


def setup_logging():
    """Set up logging configuration"""
    # Create logs directory if it doesn't exist
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)

    # Create log file path
    log_file = os.path.join(log_dir, 'stock_analyzer.log')

    # Configure logging format
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'

    # Set up rotating file handler (10 MB per file, keep 5 backup files)
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5
    )
    file_handler.setFormatter(logging.Formatter(log_format, date_format))

    # Set up console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(log_format, date_format))

    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Remove any existing handlers to avoid duplication
    logger.handlers = []

    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logging.info("Logging setup completed")


# Call setup_logging at the start of your script
setup_logging()


class LSTMPredictor:
    def __init__(self, sequence_length=60):
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None

        # Suppress TensorFlow warnings
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    def create_sequences(self, data):
        """Create sequences for LSTM input"""
        X, y = [], []
        data = np.array(data)
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:(i + self.sequence_length)])
            y.append(data[i + self.sequence_length])
        return np.array(X), np.array(y)

    def build_model(self, input_shape):
        """Build LSTM model architecture"""
        tf.keras.backend.clear_session()  # Clear previous session

        model = Sequential([
            LSTM(units=50, activation='relu', input_shape=input_shape,
                 return_sequences=True, kernel_initializer='glorot_uniform'),
            Dropout(0.2),
            LSTM(units=50, activation='relu', return_sequences=False),
            Dropout(0.2),
            Dense(units=25, activation='relu'),
            Dense(units=1)
        ])

        # Use Adam optimizer with reduced learning rate
        optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)

        # Compile model with Huber loss for robustness
        model.compile(optimizer=optimizer, loss='huber')

        return model

    def prepare_data(self, data):
        """Prepare data for LSTM model"""
        # Ensure data is numpy array and reshape
        data = np.array(data).reshape(-1, 1)

        # Scale the data
        scaled_data = self.scaler.fit_transform(data)

        # Create sequences
        X, y = self.create_sequences(scaled_data)

        # Reshape X to (samples, time steps, features)
        X = X.reshape((X.shape[0], X.shape[1], 1))

        # Split data
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        return X_train, X_test, y_train, y_test

    def train(self, data, epochs=50, batch_size=32):
        """Train LSTM model"""
        try:
            # Prepare data
            X_train, X_test, y_train, y_test = self.prepare_data(data)

            # Build model
            self.model = self.build_model((self.sequence_length, 1))

            # Early stopping callback
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )

            # Reduce learning rate callback
            reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=0.0001
            )

            # Train model
            history = self.model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_test, y_test),
                callbacks=[early_stopping, reduce_lr],
                verbose=1,
                shuffle=False
            )

            return history, (X_test, y_test)

        except Exception as e:
            logging.error(f"Error in LSTM training: {str(e)}")
            return None, None

    def predict(self, data):
        """Make predictions using trained model"""
        try:
            if self.model is None:
                raise ValueError("Model has not been trained yet")

            # Prepare data for prediction
            data = np.array(data).reshape(-1, 1)
            scaled_data = self.scaler.transform(data)

            # Create sequences
            X, _ = self.create_sequences(scaled_data)
            X = X.reshape((X.shape[0], X.shape[1], 1))

            # Make predictions
            scaled_predictions = self.model.predict(X, verbose=0)

            # Inverse transform predictions
            predictions = self.scaler.inverse_transform(scaled_predictions)

            return predictions

        except Exception as e:
            logging.error(f"Error in LSTM prediction: {str(e)}")
            return None


class BollingerBandStrategy(Strategy):
    """Bollinger Band breakout strategy with RSI confirmation"""

    # Define parameters that can be optimized
    bb_window = 20  # Bollinger Band period
    bb_std = 2.0  # Number of standard deviations
    rsi_window = 14  # RSI period
    rsi_upper = 70  # RSI overbought level
    rsi_lower = 30  # RSI oversold level

    def init(self):
        # Calculate Bollinger Bands
        close = self.data.Close
        self.sma = self.I(lambda x: pd.Series(x).rolling(self.bb_window).mean(), close)
        self.std = self.I(lambda x: pd.Series(x).rolling(self.bb_window).std(), close)

        self.upper = self.I(lambda: self.sma + self.bb_std * self.std)
        self.lower = self.I(lambda: self.sma - self.bb_std * self.std)

        # Calculate RSI
        self.rsi = self.I(self.calculate_rsi)

        # Store previous values for divergence calculation
        self.prev_rsi_values = []
        self.prev_price_values = []

    def calculate_rsi(self):
        """Calculate RSI"""
        close = pd.Series(self.data.Close)
        delta = close.diff()

        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_window).mean()

        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def check_divergence(self):
        """Check for RSI divergence patterns"""
        if len(self.prev_rsi_values) < 2:
            return None

        # Get last two price and RSI values
        price_change = self.data.Close[-1] - self.data.Close[-2]
        rsi_change = self.rsi[-1] - self.rsi[-2]

        # Bearish divergence: Price making higher highs, RSI making lower highs
        if price_change > 0 and rsi_change < 0 and self.rsi[-1] > self.rsi_upper:
            return 'bearish'

        # Bullish divergence: Price making lower lows, RSI making higher lows
        if price_change < 0 and rsi_change > 0 and self.rsi[-1] < self.rsi_lower:
            return 'bullish'

        return None

    def next(self):
        # Store values for divergence calculation
        self.prev_rsi_values.append(self.rsi[-1])
        self.prev_price_values.append(self.data.Close[-1])

        # Keep only last 10 values
        if len(self.prev_rsi_values) > 10:
            self.prev_rsi_values.pop(0)
            self.prev_price_values.pop(0)

        # Check for Bollinger Band breakout
        bb_breakout = None
        if self.data.Close[-1] > self.upper[-1]:
            bb_breakout = 'up'
        elif self.data.Close[-1] < self.lower[-1]:
            bb_breakout = 'down'

        # Check for RSI divergence
        divergence = self.check_divergence()

        # Trading logic
        if bb_breakout == 'up' and self.rsi[-1] < self.rsi_upper:
            # Bullish breakout with RSI confirmation
            if not self.position.is_long:
                self.position.close()
                self.buy()

        elif bb_breakout == 'down' and self.rsi[-1] > self.rsi_lower:
            # Bearish breakout with RSI confirmation
            if not self.position.is_short:
                self.position.close()
                self.sell()

        # Exit positions on divergence
        if divergence == 'bearish' and self.position.is_long:
            self.position.close()
        elif divergence == 'bullish' and self.position.is_short:
            self.position.close()


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
            # Create strategy with current parameters
            strategy = type('OptimizedStrategy', (strategy_class,), params)

            # Run backtest
            bt = Backtest(analyzer.stock_data, strategy, cash=10000, commission=.002)
            stats = bt.run()

            # Extract results and convert to native Python types
            result = {
                'parameters': params,
                'sharpe_ratio': stats['Sharpe Ratio'],
                'return_pct': stats['Return [%]'],
                'max_drawdown': stats['Max. Drawdown [%]'],
                'win_rate': stats['Win Rate [%]'],
                'profit_factor': stats.get('Profit Factor', 0),
                'num_trades': stats['# Trades']
            }

            results.append(result)

            # Update best parameters if necessary
            if result['sharpe_ratio'] > best_sharpe:
                best_sharpe = result['sharpe_ratio']
                best_params = params.copy()

        # Convert results to DataFrame
        results_df = pd.DataFrame(results)

        logging.info(f"Optimization completed. Best parameters: {best_params}")
        logging.info(f"Best Sharpe Ratio: {best_sharpe}")

        return best_params, results_df

    except Exception as e:
        logging.error(f"Error in strategy optimization: {str(e)}")
        return None, None


def perform_backtest(self):
    """Perform backtesting on the stock data using multiple strategies."""
    if self.stock_data is None or self.stock_data.empty:
        logging.warning(f"No data available for backtesting of {self.ticker_symbol}")
        return None

    logging.info(f"Starting backtest for {self.ticker_symbol}")

    try:
        # Create strategies directory
        strategies_dir = os.path.join(self.output_dir, 'strategies')
        os.makedirs(strategies_dir, exist_ok=True)

        # Test both strategies
        strategies = {
            'SMA_Cross': SmaCrossWithRisk,
            'Bollinger_RSI': BollingerBandStrategy
        }

        results = {}

        for strategy_name, strategy_class in strategies.items():
            try:
                # Run basic backtest
                bt = Backtest(self.stock_data, strategy_class, cash=10000, commission=.002)
                stats = bt.run()

                # Extract results and convert to native Python types
                strategy_results = {
                    'Return [%]': stats['Return [%]'],
                    'Sharpe Ratio': stats['Sharpe Ratio'],
                    'Max. Drawdown [%]': stats['Max. Drawdown [%]'],
                    'Win Rate [%]': stats['Win Rate [%]'],
                    'Profit Factor': stats.get('Profit Factor', 0),
                    '# Trades': stats['# Trades']
                }

                results[strategy_name] = strategy_results

                # Save individual strategy results
                strategy_file = os.path.join(strategies_dir, f'{strategy_name}_results.json')
                with open(strategy_file, 'w') as f:
                    json.dump(strategy_results, f, indent=4)

                # Create and save strategy performance plot
                self._create_strategy_plot(bt, stats, strategies_dir, strategy_name)

            except Exception as e:
                logging.error(f"Error testing strategy {strategy_name}: {str(e)}")
                continue

        # Optimize Bollinger Band strategy
        bb_params = {
            'bb_window': range(10, 31, 5),
            'bb_std': [1.5, 2.0, 2.5],
            'rsi_window': range(10, 21, 5),
            'rsi_upper': range(65, 81, 5),
            'rsi_lower': range(20, 36, 5)
        }

        best_params, optimization_results = optimize_strategy(
            self,
            BollingerBandStrategy,
            bb_params
        )

        if best_params and optimization_results is not None:
            # Save optimization results
            opt_results_path = os.path.join(strategies_dir, 'optimization_results.csv')
            optimization_results.to_csv(opt_results_path)

            # Create and test optimized strategy
            OptimizedStrategy = type('OptimizedStrategy', (BollingerBandStrategy,), best_params)
            bt = Backtest(self.stock_data, OptimizedStrategy, cash=10000, commission=.002)
            stats = bt.run()

            # Save optimized strategy results
            optimized_results = {
                'parameters': best_params,
                'performance': {
                    'Return [%]': stats['Return [%]'],
                    'Sharpe Ratio': stats['Sharpe Ratio'],
                    'Max. Drawdown [%]': stats['Max. Drawdown [%]'],
                    'Win Rate [%]': stats['Win Rate [%]'],
                    'Profit Factor': stats.get('Profit Factor', 0),
                    '# Trades': stats['# Trades']
                }
            }

            opt_stats_path = os.path.join(strategies_dir, 'optimized_strategy_results.json')
            with open(opt_stats_path, 'w') as f:
                json.dump(optimized_results, f, indent=4)

            # Create and save optimized strategy plot
            self._create_strategy_plot(bt, stats, strategies_dir, 'Optimized_Strategy')

        # Save comparison of all strategies
        comparison_file = os.path.join(strategies_dir, 'strategy_comparison.json')
        with open(comparison_file, 'w') as f:
            json.dump(results, f, indent=4)

        return results

    except Exception as e:
        logging.error(f"Error in backtesting: {str(e)}")
        return None


class StockAnalyzer:
    # Class-level constants
    EXCHANGE_MAPPINGS = {
        # Americas
        'NYSE:': '',  # New York Stock Exchange (no suffix needed)
        'NASDAQ:': '',  # NASDAQ (no suffix needed)
        'TSX:': '.TO',  # Toronto Stock Exchange
        'BVMF:': '.SA',  # Brazil Stock Exchange

        # Europe
        'LON:': '.L',  # London Stock Exchange
        'FRA:': '.F',  # Frankfurt Stock Exchange
        'PAR:': '.PA',  # Euronext Paris
        'AMS:': '.AS',  # Euronext Amsterdam
        'BRU:': '.BR',  # Euronext Brussels
        'LIS:': '.LS',  # Euronext Lisbon
        'MCE:': '.MC',  # Madrid Stock Exchange
        'MIL:': '.MI',  # Italian Stock Exchange
        'VIE:': '.VI',  # Vienna Stock Exchange
        'STO:': '.ST',  # Stockholm Stock Exchange
        'OSL:': '.OL',  # Oslo Stock Exchange
        'CPH:': '.CO',  # Copenhagen Stock Exchange
        'HEL:': '.HE',  # Helsinki Stock Exchange
        'IST:': '.IS',  # Istanbul Stock Exchange
        'WAR:': '.WA',  # Warsaw Stock Exchange
        'ATH:': '.AT',  # Athens Stock Exchange

        # Asia-Pacific
        'TYO:': '.T',  # Tokyo Stock Exchange
        'HKG:': '.HK',  # Hong Kong Stock Exchange
        'SHA:': '.SS',  # Shanghai Stock Exchange
        'SHE:': '.SZ',  # Shenzhen Stock Exchange
        'KRX:': '.KS',  # Korea Exchange
        'TPE:': '.TW',  # Taiwan Stock Exchange
        'BOM:': '.BO',  # Bombay Stock Exchange
        'NSE:': '.NS',  # National Stock Exchange of India
        'ASX:': '.AX',  # Australian Securities Exchange
        'NZE:': '.NZ',  # New Zealand Exchange
        'SGX:': '.SI',  # Singapore Exchange

        # Other regions
        'JSE:': '.JO',  # Johannesburg Stock Exchange
        'TASE:': '.TA',  # Tel Aviv Stock Exchange
    }

    def __init__(self, ticker_symbol, start_date=None, end_date=None):
        """Initialize the StockAnalyzer with a ticker symbol and date range"""
        self.original_ticker = ticker_symbol
        self.ticker_symbol = self.format_ticker(ticker_symbol)
        self.start_date = start_date
        self.end_date = end_date
        self.news_data = []
        self.company_info = {}
        self.stock_data = None
        self.output_dir = os.path.join('output', self.ticker_symbol)

        logging.info(f"Initializing StockAnalyzer for ticker: {ticker_symbol}")
        logging.info(f"Date range: {start_date} to {end_date}")
        logging.info(f"Formatted ticker symbol: {self.ticker_symbol}")

        # Ensure directories exist
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'forecast_plots'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'technical_analysis'), exist_ok=True)
        logging.info(f"Created output directories in {self.output_dir}")

    @classmethod
    def format_ticker(cls, ticker):
        """Format ticker symbols for different exchanges"""
        ticker = ticker.strip()

        # Handle cases where ticker might contain multiple colons
        if ':' in ticker:
            parts = ticker.split(':')
            exchange = parts[0] + ':'
            symbol = ':'.join(parts[1:])
        else:
            return ticker  # Return as-is if no exchange prefix

        # Clean the symbol part
        symbol = symbol.strip()

        # Look up the suffix for the exchange
        suffix = cls.EXCHANGE_MAPPINGS.get(exchange, '')

        # Special handling for exchanges that need symbol modifications
        if exchange in ['SHA:', 'SHE:', 'TPE:', 'BOM:', 'NSE:']:
            # Ensure number-only tickers are padded to proper length
            if symbol.isdigit():
                if exchange in ['SHA:', 'SHE:']:
                    symbol = symbol.zfill(6)
                elif exchange == 'TPE:':
                    symbol = symbol.zfill(4)
                elif exchange in ['BOM:', 'NSE:']:
                    symbol = symbol.zfill(6)

        return f"{symbol}{suffix}"

    def fetch_stock_data(self):
        """Fetch historical stock data with improved error handling"""
        logging.info(f"Starting stock data fetch for {self.ticker_symbol}")
        start_time = datetime.now()

        try:
            # Use provided dates or default to last 2 years
            if self.end_date is None:
                self.end_date = datetime.now()
            if self.start_date is None:
                self.start_date = self.end_date - timedelta(days=730)

            logging.info(f"Fetching data from {self.start_date} to {self.end_date}")

            ticker = yf.Ticker(self.ticker_symbol)
            logging.debug(f"Created yfinance Ticker object for {self.ticker_symbol}")

            info = ticker.info
            if not info:
                logging.error(f"No information found for ticker {self.ticker_symbol}")
                raise ValueError(f"No information found for ticker {self.ticker_symbol}")

            logging.info(f"Successfully validated ticker {self.ticker_symbol}")

            self.stock_data = ticker.history(
                start=self.start_date,
                end=self.end_date,
                interval="1d"
            )
            fetch_time = (datetime.now() - start_time).total_seconds()
            logging.info(f"Data fetch completed in {fetch_time:.2f} seconds")

            if not self.stock_data.empty:
                self.stock_data.index = self.stock_data.index.tz_localize(None)
                output_path = os.path.join(self.output_dir, f'{self.ticker_symbol}_stock_data.csv')
                self.stock_data.to_csv(output_path)
                logging.info(f"Saved stock data to {output_path}")
                logging.info(f"Retrieved {len(self.stock_data)} data points")

                # Log basic company info
                logging.info(f"Company Name: {info.get('longName', 'N/A')}")
                logging.info(f"Exchange: {info.get('exchange', 'N/A')}")
                logging.info(f"Currency: {info.get('currency', 'N/A')}")
            else:
                logging.warning(f"No data available for {self.ticker_symbol}")

        except ValueError as ve:
            logging.error(f"Validation error for {self.ticker_symbol}: {str(ve)}")
            self.stock_data = None
        except requests.exceptions.RequestException as re:
            logging.error(f"Network error while fetching {self.ticker_symbol}: {str(re)}")
            self.stock_data = None
        except Exception as e:
            logging.error(f"Unexpected error fetching data for {self.ticker_symbol}: {str(e)}", exc_info=True)
            self.stock_data = None

    def fetch_financial_news(self):
        """Fetch financial news from various sources"""
        logging.info(f"Starting financial news fetch for {self.ticker_symbol}")
        start_time = datetime.now()

        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }

            search_ticker = self.ticker_symbol.split('.')[0]
            logging.info(f"Using search ticker: {search_ticker}")

            all_articles = []
            for source_name, source_config in NEWS_SOURCES.items():
                logging.info(f"Fetching news from {source_name}")
                url = source_config['url'].format(search_ticker, search_ticker)
                logging.debug(f"Requesting URL: {url}")

                try:
                    response = requests.get(url, headers=headers, timeout=10)
                    soup = BeautifulSoup(response.text, 'html.parser')

                    for selector in source_config['selectors']:
                        articles = soup.select(selector)
                        logging.debug(f"Found {len(articles)} articles from {source_name}")
                        all_articles.extend(articles)

                except Exception as e:
                    logging.error(f"Error fetching from {source_name}: {str(e)}")
                    continue

            processed_articles = 0
            for article in all_articles[:10]:
                title = article.text.strip()
                if title and len(title) > 5:
                    sentiment = TextBlob(title).sentiment.polarity
                    self.news_data.append({
                        'title': title,
                        'sentiment': round(sentiment, 3),
                        'date': datetime.now().strftime('%Y-%m-%d'),
                        'source': source_name,
                        'url': article.get('href')
                    })
                    processed_articles += 1

            logging.info(f"Processed {processed_articles} articles")

            if self.news_data:
                news_df = pd.DataFrame(self.news_data).drop_duplicates(subset=['title'])
                output_path = os.path.join(self.output_dir, f'{self.ticker_symbol}_news.csv')
                news_df.to_csv(output_path, index=False)
                logging.info(f"Saved news data to {output_path}")

            fetch_time = (datetime.now() - start_time).total_seconds()
            logging.info(f"News fetch completed in {fetch_time:.2f} seconds")

        except Exception as e:
            logging.error(f"Error in news fetch for {self.ticker_symbol}: {str(e)}", exc_info=True)

    def fetch_company_info(self):
        """Fetch comprehensive company information"""
        logging.info(f"Starting company information fetch for {self.ticker_symbol}")
        start_time = datetime.now()

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

        clean_ticker = get_clean_ticker(self.ticker_symbol)
        company_data = {
            'ticker': self.ticker_symbol,
            'original_ticker': self.original_ticker,
            'sources': {}
        }

        try:
            logging.info("Fetching data from Yahoo Finance API")
            yf_ticker = yf.Ticker(self.ticker_symbol)
            info = yf_ticker.info or {}

            company_data['yahoo_api'] = {
                'name': info.get('longName', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'sector': info.get('sector', 'N/A'),
                'country': info.get('country', 'N/A'),
                'employees': info.get('fullTimeEmployees', 'N/A'),
                'website': info.get('website', 'N/A'),
                'market_cap': info.get('marketCap', 'N/A'),
                'currency': info.get('currency', 'N/A')
            }

            logging.info("Successfully retrieved Yahoo Finance data")

        except Exception as e:
            logging.error(f"Error fetching Yahoo Finance API data: {str(e)}", exc_info=True)
            company_data['yahoo_api'] = {}

        self.company_info = company_data
        self._save_company_info()

        fetch_time = (datetime.now() - start_time).total_seconds()
        logging.info(f"Company info fetch completed in {fetch_time:.2f} seconds")

    def _save_company_info(self):
        """Save company information to a JSON file."""
        try:
            output_file = os.path.join(self.output_dir, f"{self.ticker_symbol}_company_info.json")
            with open(output_file, 'w') as json_file:
                json.dump(self.company_info, json_file, indent=4)
            print(f"Company information saved to {output_file}")
        except Exception as e:
            print(f"Error saving company information for {self.ticker_symbol}: {str(e)}")

    def prepare_data_for_prophet(self):
        """Prepare data for Prophet model"""
        if self.stock_data is None or self.stock_data.empty:
            print(f"No valid stock data for {self.ticker_symbol}, cannot prepare data for Prophet")
            return None

        df = self.stock_data.reset_index()
        prophet_df = pd.DataFrame()
        prophet_df['ds'] = df['Date']

        try:
            prophet_df['y'] = df['Close'].values.flatten()
        except ValueError as e:
            print(f"Error preparing data for {self.ticker_symbol}: {str(e)}")
            return None

        return prophet_df.dropna()

    def perform_technical_analysis(self):
        """Perform technical analysis on the stock data"""
        try:
            if self.stock_data is None or self.stock_data.empty:
                logging.warning(f"No data available for technical analysis of {self.ticker_symbol}")
                return None, None

            logging.info(f"Performing technical analysis for {self.ticker_symbol}...")

            tech_analysis_dir = os.path.join(self.output_dir, 'technical_analysis')
            results = pd.DataFrame(index=self.stock_data.index)

            # Calculate technical indicators using modern pandas methods
            close_series = pd.Series(self.stock_data['Close'].values, index=self.stock_data.index)

            # RSI
            rsi_indicator = ta.momentum.RSIIndicator(close_series)
            results.loc[:, 'RSI'] = rsi_indicator.rsi()
            results.loc[:, 'RSI_High'] = 70
            results.loc[:, 'RSI_Low'] = 20

            # MACD with custom parameters
            macd_indicator = ta.trend.MACD(
                close_series,
                window_slow=26,  # Long EMA
                window_fast=12,  # Short EMA
                window_sign=9  # Signal line EMA
            )
            results.loc[:, 'MACD'] = macd_indicator.macd()
            results.loc[:, 'MACD_Signal'] = macd_indicator.macd_signal()
            results.loc[:, 'MACD_Histogram'] = macd_indicator.macd_diff()

            # Bollinger Bands
            bollinger = ta.volatility.BollingerBands(close_series)
            results.loc[:, 'BB_High'] = bollinger.bollinger_hband()
            results.loc[:, 'BB_Low'] = bollinger.bollinger_lband()
            results.loc[:, 'BB_Mid'] = bollinger.bollinger_mavg()

            # Moving Averages
            results.loc[:, 'EMA_20'] = ta.trend.EMAIndicator(close_series, window=20).ema_indicator()
            results.loc[:, 'SMA_20'] = ta.trend.SMAIndicator(close_series, window=20).sma_indicator()
            results.loc[:, 'SMA_50'] = ta.trend.SMAIndicator(close_series, window=50).sma_indicator()

            # Parabolic SAR
            psar = ta.trend.PSARIndicator(
                self.stock_data['High'],
                self.stock_data['Low'],
                self.stock_data['Close']
            )
            results.loc[:, 'PSAR'] = psar.psar()

            # On-Balance Volume
            obv = ta.volume.OnBalanceVolumeIndicator(
                self.stock_data['Close'],
                self.stock_data['Volume']
            )
            results.loc[:, 'OBV'] = obv.on_balance_volume()

            # Add close price
            results.loc[:, 'Close'] = self.stock_data['Close']

            # Save results with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(tech_analysis_dir, f'technical_analysis_{timestamp}.csv')
            results.to_csv(output_file)

            # Generate summary statistics
            summary = pd.DataFrame({
                'Current_Value': results.iloc[-1],
                'Mean': results.mean(),
                'Std': results.std(),
                'Min': results.min(),
                'Max': results.max()
            })

            summary_file = os.path.join(tech_analysis_dir, f'summary_{timestamp}.csv')
            summary.to_csv(summary_file)

            # Create visualization with modern plotting
            self._create_technical_analysis_plot(results, timestamp, tech_analysis_dir)

            logging.info(f"Technical analysis completed for {self.ticker_symbol}")
            logging.info(f"Results saved to: {output_file}")
            logging.info(f"Summary saved to: {summary_file}")

            return results, summary

        except Exception as e:
            logging.error(f"Error performing technical analysis for {self.ticker_symbol}: {str(e)}", exc_info=True)
            return None, None

    def analyze_stock(self):
        """Main analysis pipeline"""
        try:
            print(f"\nStarting analysis for {self.ticker_symbol}...")

            # Data Collection
            self.fetch_company_info()
            self.fetch_financial_news()
            self.fetch_stock_data()

            # Perform technical analysis
            if self.stock_data is not None and not self.stock_data.empty:
                results, summary = self.perform_technical_analysis()
                logging.info(f"Technical analysis summary for {self.ticker_symbol}: {summary}")

                self.generate_buy_sell_signals(results)



        except Exception as e:
            logging.error(f"Error analyzing stock {self.ticker_symbol}: {str(e)}")

    def generate_buy_sell_signals(self, results):
        """Generate buy/sell signals based on technical indicators."""
        try:
            # Example: Using RSI and Moving Averages for signals
            buy_signals = []
            sell_signals = []

            for i in range(1, len(results)):
                # Buy signal: RSI < 30 (oversold) and price crosses above SMA
                if results['RSI'].iloc[i] < 30 and results['Close'].iloc[i] > results['SMA_20'].iloc[i]:
                    buy_signals.append(results.index[i])
                    logging.info(f"Buy signal generated for {self.ticker_symbol} on {results.index[i]}")

                # Sell signal: RSI > 70 (overbought) and price crosses below SMA
                elif results['RSI'].iloc[i] > 70 and results['Close'].iloc[i] < results['SMA_20'].iloc[i]:
                    sell_signals.append(results.index[i])
                    logging.info(f"Sell signal generated for {self.ticker_symbol} on {results.index[i]}")

            # Save signals to a file
            signals_df = pd.DataFrame({
                'Buy Signals': pd.Series(buy_signals),
                'Sell Signals': pd.Series(sell_signals)
            })
            signals_file_path = os.path.join(self.output_dir, f'{self.ticker_symbol}_signals.csv')
            signals_df.to_csv(signals_file_path, index=False)
            logging.info(f"Buy/Sell signals saved to {signals_file_path}")

        except Exception as e:
            logging.error(f"Error generating buy/sell signals for {self.ticker_symbol}: {str(e)}")

    def _create_technical_analysis_plot(self, results, timestamp, tech_analysis_dir):
        """Create technical analysis visualization with candlesticks and MACD histogram"""
        try:
            plt.style.use('seaborn-v0_8-darkgrid')

            # Create figure with subplots
            fig = plt.figure(figsize=(15, 12))
            gs = fig.add_gridspec(3, 1, height_ratios=[2, 1, 1])

            # Main price chart with candlesticks
            ax1 = fig.add_subplot(gs[0])

            # Plot candlesticks
            width = np.timedelta64(12, 'h')  # Adjust candlestick width
            up = self.stock_data[self.stock_data.Close >= self.stock_data.Open]
            down = self.stock_data[self.stock_data.Close < self.stock_data.Open]

            # Up candlesticks
            ax1.bar(up.index, up.Close - up.Open, width, bottom=up.Open, color='g', alpha=0.6)
            ax1.bar(up.index, up.High - up.Close, width / 5, bottom=up.Close, color='g', alpha=0.6)
            ax1.bar(up.index, up.Low - up.Open, width / 5, bottom=up.Open, color='g', alpha=0.6)

            # Down candlesticks
            ax1.bar(down.index, down.Close - down.Open, width, bottom=down.Open, color='r', alpha=0.6)
            ax1.bar(down.index, down.High - down.Open, width / 5, bottom=down.Open, color='r', alpha=0.6)
            ax1.bar(down.index, down.Low - down.Close, width / 5, bottom=down.Close, color='r', alpha=0.6)

            # Add Bollinger Bands
            if all(col in results.columns for col in ['BB_High', 'BB_Low', 'BB_Mid']):
                ax1.plot(results.index, results['BB_High'], 'b--', alpha=0.5, label='BB Upper')
                ax1.plot(results.index, results['BB_Mid'], 'b-', alpha=0.5, label='BB Middle')
                ax1.plot(results.index, results['BB_Low'], 'b--', alpha=0.5, label='BB Lower')

            # Add Moving Averages
            if 'EMA_20' in results.columns:
                ax1.plot(results.index, results['EMA_20'], 'orange', label='EMA 20', alpha=0.7)
            if 'SMA_50' in results.columns:
                ax1.plot(results.index, results['SMA_50'], 'purple', label='SMA 50', alpha=0.7)

            ax1.set_title(f'{self.ticker_symbol} Technical Analysis')
            ax1.set_ylabel('Price')
            ax1.grid(True, alpha=0.3)
            ax1.legend(loc='upper left')

            # RSI subplot
            ax2 = fig.add_subplot(gs[1], sharex=ax1)
            if 'RSI' in results.columns:
                ax2.plot(results.index, results['RSI'], 'purple', label='RSI')
                ax2.axhline(y=70, color='r', linestyle='--', alpha=0.5)
                ax2.axhline(y=30, color='g', linestyle='--', alpha=0.5)
                ax2.set_ylabel('RSI')
                ax2.grid(True, alpha=0.3)
                ax2.legend(loc='upper left')

            # MACD subplot with histogram
            ax3 = fig.add_subplot(gs[2], sharex=ax1)
            if all(col in results.columns for col in ['MACD', 'MACD_Signal', 'MACD_Histogram']):
                # Plot MACD Histogram
                colors = np.where(results['MACD_Histogram'] >= 0, 'g', 'r')
                ax3.bar(results.index, results['MACD_Histogram'], color=colors, alpha=0.5, label='MACD Histogram')

                # Plot MACD and Signal lines
                ax3.plot(results.index, results['MACD'], 'b-', label='MACD')
                ax3.plot(results.index, results['MACD_Signal'], 'orange', label='Signal')
                ax3.set_ylabel('MACD')
                ax3.grid(True, alpha=0.3)
                ax3.legend(loc='upper left')

            # Format x-axis
            plt.xticks(rotation=45)
            plt.tight_layout()

            # Save the plot
            plot_file = os.path.join(tech_analysis_dir, f'technical_analysis_plot_{timestamp}.png')
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()

            logging.info(f"Technical analysis plot saved to: {plot_file}")

        except Exception as e:
            logging.error(f"Error creating technical analysis plot: {str(e)}", exc_info=True)

    def perform_backtest(self):
        """Perform backtesting on the stock data using multiple strategies."""
        if self.stock_data is None or self.stock_data.empty:
            logging.warning(f"No data available for backtesting of {self.ticker_symbol}")
            return None

        logging.info(f"Starting backtest for {self.ticker_symbol}")

        try:
            # Create strategies directory
            strategies_dir = os.path.join(self.output_dir, 'strategies')
            os.makedirs(strategies_dir, exist_ok=True)

            # Test both strategies
            strategies = {
                'SMA_Cross': SmaCrossWithRisk,
                'Bollinger_RSI': BollingerBandStrategy
            }

            results = {}

            for strategy_name, strategy_class in strategies.items():
                # Run basic backtest
                bt = Backtest(self.stock_data, strategy_class, cash=10000, commission=.002)
                stats = bt.run()

                # Convert stats to dictionary
                strategy_results = {
                    'Return [%]': float(stats['Return [%]']),
                    'Sharpe Ratio': float(stats['Sharpe Ratio']),
                    'Max. Drawdown [%]': float(stats['Max. Drawdown [%]']),
                    'Win Rate [%]': float(stats['Win Rate [%]']),
                    'Profit Factor': float(stats.get('Profit Factor', 0)),
                    '# Trades': int(stats['# Trades'])
                }

                results[strategy_name] = strategy_results

                # Save individual strategy results
                strategy_file = os.path.join(strategies_dir, f'{strategy_name}_results.json')
                with open(strategy_file, 'w') as f:
                    json.dump(strategy_results, f, indent=4)

                # Create and save strategy performance plot
                self._create_strategy_plot(bt, stats, strategies_dir, strategy_name)

            # Optimize Bollinger Band strategy
            bb_params = {
                'bb_window': range(10, 31, 5),
                'bb_std': [1.5, 2.0, 2.5],
                'rsi_window': range(10, 21, 5),
                'rsi_upper': range(65, 81, 5),
                'rsi_lower': range(20, 36, 5)
            }

            best_params, optimization_results = optimize_strategy(
                self,
                BollingerBandStrategy,
                bb_params
            )

            if best_params and optimization_results is not None:
                # Save optimization results
                opt_results_path = os.path.join(strategies_dir, 'optimization_results.csv')
                optimization_results.to_csv(opt_results_path)

                # Create and test optimized strategy
                OptimizedStrategy = type('OptimizedStrategy', (BollingerBandStrategy,), best_params)
                bt = Backtest(self.stock_data, OptimizedStrategy, cash=10000, commission=.002)
                stats = bt.run()

                # Save optimized strategy results
                optimized_results = {
                    'parameters': best_params,
                    'performance': {
                        'Return [%]': float(stats['Return [%]']),
                        'Sharpe Ratio': float(stats['Sharpe Ratio']),
                        'Max. Drawdown [%]': float(stats['Max. Drawdown [%]']),
                        'Win Rate [%]': float(stats['Win Rate [%]']),
                        'Profit Factor': float(stats.get('Profit Factor', 0)),
                        '# Trades': int(stats['# Trades'])
                    }
                }

                opt_stats_path = os.path.join(strategies_dir, 'optimized_strategy_results.json')
                with open(opt_stats_path, 'w') as f:
                    json.dump(optimized_results, f, indent=4)

                # Create and save optimized strategy plot
                self._create_strategy_plot(bt, stats, strategies_dir, 'Optimized_Strategy')

            # Save comparison of all strategies
            comparison_file = os.path.join(strategies_dir, 'strategy_comparison.json')
            with open(comparison_file, 'w') as f:
                json.dump(results, f, indent=4)

            return results

        except Exception as e:
            logging.error(f"Error in backtesting: {str(e)}", exc_info=True)
            return None

    def _create_strategy_plot(self, bt, stats, strategies_dir, strategy_name):
        """Create and save strategy performance plot"""
        try:
            plt.style.use('seaborn-v0_8-darkgrid')
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), height_ratios=[2, 1])

            # Plot equity curve
            equity_curve = pd.Series(stats['_equity_curve']['Equity'])
            equity_curve.plot(ax=ax1, label='Strategy Equity', color='blue', linewidth=2)

            # Plot buy/sell signals
            trades = stats['_trades']
            for _, trade in trades.iterrows():
                if trade['Size'] > 0:  # Buy signal
                    ax1.plot(trade['EntryTime'], trade['EntryPrice'], '^',
                             color='green', markersize=10, alpha=0.7)
                else:  # Sell signal
                    ax1.plot(trade['EntryTime'], trade['EntryPrice'], 'v',
                             color='red', markersize=10, alpha=0.7)

            ax1.set_title(f'{self.ticker_symbol} - {strategy_name} Performance')
            ax1.set_xlabel('')
            ax1.set_ylabel('Portfolio Value ($)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Plot returns
            returns = equity_curve.pct_change()
            returns.plot(ax=ax2, label='Returns', color='gray', alpha=0.7)
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Returns (%)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()

            # Save plot
            plot_file = os.path.join(strategies_dir, f'{strategy_name}_performance.png')
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()

        except Exception as e:
            logging.error(f"Error creating strategy plot: {str(e)}", exc_info=True)

    def _create_backtest_plot(self, bt, stats, backtest_dir):
        """Create custom backtest visualization"""
        try:
            plt.style.use('seaborn-v0_8-darkgrid')  # Use the updated style

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), height_ratios=[2, 1], facecolor='#f0f0f0')

            for ax in [ax1, ax2]:
                ax.set_facecolor('white')
                ax.grid(True, linestyle='--', alpha=0.7)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)

            equity_curve = pd.Series(stats['_equity_curve']['Equity'])
            equity_curve.plot(ax=ax1, label='Strategy Equity', color='#2E86C1', linewidth=2)

            trades = stats['_trades']
            for _, trade in trades.iterrows():
                if trade['Size'] > 0:  # Buy signal
                    ax1.plot(trade['EntryTime'], trade['EntryPrice'], '^', markersize=10, color='g', alpha=0.7)
                else:  # Sell signal
                    ax1.plot(trade['EntryTime'], trade['EntryPrice'], 'v', markersize=10, color='r', alpha=0.7)

            sma1 = self.stock_data['Close'].rolling(window=10).mean()
            sma2 = self.stock_data['Close'].rolling(window=20).mean()
            sma1.plot(ax=ax1, label='SMA(10)', color='#8E44AD', alpha=0.7, linewidth=1)
            sma2.plot(ax=ax1, label='SMA(20)', color='#F39C12', alpha=0.7, linewidth=1)

            ax1.set_title(f'{self.ticker_symbol} Backtest Results', pad=20, fontsize=14, fontweight='bold')
            ax1.set_xlabel('')
            ax1.set_ylabel('Price/Equity', fontsize=12)
            ax1.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='none')

            returns = equity_curve.pct_change()
            returns.plot(ax=ax2, label='Returns', color='#34495E', alpha=0.6)

            ax2.set_xlabel('Date', fontsize=12)
            ax2.set_ylabel('Returns', fontsize=12)
            ax2.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='none')

            plt.tight_layout()

            plot_file = os.path.join(backtest_dir, f'{self.ticker_symbol}_backtest_plot.png')
            plt.savefig(plot_file, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()

            stats_summary = {
                'Total Return': f"{stats['Return [%]']:.2f}%",
                'Sharpe Ratio': f"{stats['Sharpe Ratio']:.2f}",
                'Max Drawdown': f"{stats['Max. Drawdown [%]']:.2f}%",
                'Win Rate': f"{stats['Win Rate [%]']:.2f}%",
                'Profit Factor': f"{stats.get('Profit Factor', 0):.2f}",
                'Total Trades': stats['# Trades'],
                'Average Trade': f"{stats.get('Avg. Trade [%]', 0):.2f}%",
                'Max Trade': f"{stats.get('Best Trade [%]', 0):.2f}%",
                'Min Trade': f"{stats.get('Worst Trade [%]', 0):.2f}%",
                'Exposure Time': f"{stats.get('Exposure Time [%]', 0):.2f}%"
            }

            stats_file = os.path.join(backtest_dir, f'{self.ticker_symbol}_backtest_summary.json')
            with open(stats_file, 'w') as f:
                json.dump(stats_summary, f, indent=4)

            logging.info(f"Backtest plot saved to: {plot_file}")
            logging.info(f"Backtest summary saved to: {stats_file}")

        except Exception as e:
            logging.error(f"Error creating backtest plot: {str(e)}", exc_info=True)

    def analyze_stock(self):
        """Main analysis pipeline"""
        try:
            print(f"\nStarting analysis for {self.ticker_symbol}...")

            # Data Collection
            self.fetch_company_info()
            self.fetch_financial_news()
            self.fetch_stock_data()

            # Fetch additional data
            economic_data = self.fetch_economic_indicators()
            options_data = self.fetch_options_data()

            # Analysis
            if self.stock_data is not None and not self.stock_data.empty:
                # Existing analyses
                results, summary = self.perform_technical_analysis()
                lstm_predictions, lstm_metrics = self.perform_lstm_analysis()
                forecast = self.forecast_prices()
                backtest_results = self.perform_backtest()

                # New market condition analysis
                market_analysis = self.analyze_market_conditions(economic_data, options_data)

                if forecast is not None:
                    self.print_forecast_summary(forecast)

                # Add optimization and new strategy testing here
                # Define parameter ranges for optimization
                bb_params = {
                    'bb_window': range(10, 31, 5),  # 10, 15, 20, 25, 30
                    'bb_std': [1.5, 2.0, 2.5, 3.0],
                    'rsi_window': range(10, 21, 5),  # 10, 15, 20
                    'rsi_upper': range(65, 81, 5),  # 65, 70, 75, 80
                    'rsi_lower': range(20, 36, 5)  # 20, 25, 30, 35
                }

                # Create strategies directory
                strategies_dir = os.path.join(self.output_dir, 'strategies')
                os.makedirs(strategies_dir, exist_ok=True)

                # Optimize and test Bollinger Band strategy
                logging.info("Starting strategy optimization...")
                try:
                    best_params, optimization_results = optimize_strategy(
                        self,
                        BollingerBandStrategy,
                        bb_params
                    )

                    if best_params and optimization_results is not None:
                        # Save optimization results
                        opt_results_path = os.path.join(strategies_dir,
                                                        f'{self.ticker_symbol}_optimization_results.csv')
                        optimization_results.to_csv(opt_results_path)
                        logging.info(f"Saved optimization results to {opt_results_path}")

                        # Create and test optimized strategy
                        OptimizedStrategy = type('OptimizedStrategy',
                                                 (BollingerBandStrategy,),
                                                 best_params)

                        bt = Backtest(self.stock_data, OptimizedStrategy,
                                      cash=10000, commission=.002)
                        stats = bt.run()

                        # Save optimized strategy results
                        opt_stats_path = os.path.join(strategies_dir,
                                                      f'{self.ticker_symbol}_optimized_strategy_stats.json')
                        with open(opt_stats_path, 'w') as f:
                            json.dump(stats._asdict(), f, indent=4)

                        # Create and save strategy performance plot
                        self._create_backtest_plot(bt, stats, strategies_dir)
                        logging.info("Completed strategy optimization and testing")

                except Exception as e:
                    logging.error(f"Error in strategy optimization: {str(e)}")

                # Perform original backtesting strategy
                self.perform_backtest()

                # Print market analysis summary if available
                if market_analysis:
                    self._print_market_analysis(market_analysis)

            else:
                print(f"Cannot proceed with analysis for {self.ticker_symbol} due to missing stock data.")

        except Exception as e:
            logging.error(f"Error analyzing stock {self.ticker_symbol}: {str(e)}")

    def _print_market_analysis(self, analysis):
        """Print a summary of the market condition analysis"""
        print("\nMarket Analysis Summary:")

        if 'economic' in analysis:
            print("\nEconomic Indicators:")
            print(f"GDP Trend: {analysis['economic']['gdp_trend']:.2%}")
            print(f"Unemployment Rate: {analysis['economic']['unemployment_rate']:.1f}%")
            print(f"Inflation Rate: {analysis['economic']['inflation_rate']:.1f}%")
            print(f"Interest Rate: {analysis['economic']['interest_rate']:.2f}%")
            print(f"Yield Curve Spread: {analysis['economic']['yield_curve']:.2f}")
            print(f"VIX Index: {analysis['economic']['market_volatility']:.2f}")

        if 'options' in analysis:
            print("\nOptions Market Indicators:")
            print(f"Put-Call Ratio: {analysis['options']['put_call_ratio']:.2f}")
            print(f"Average Implied Volatility: {analysis['options']['avg_implied_volatility']:.1f}%")
            print(f"Total Open Interest: {analysis['options']['total_open_interest']:,}")
            print("\nMost Active Strike Prices:")
            for strike, volume in analysis['options']['most_active_strikes'].items():
                print(f"${strike}: {volume:,} contracts")

    def _print_company_summary(self):
        """Print a summary of the fetched company information."""
        try:
            yahoo_info = self.company_info.get('yahoo_api', {})
            print("\nCompany Summary:")
            print(f"Name: {yahoo_info.get('name', 'N/A')}")
            print(f"Industry: {yahoo_info.get('industry', 'N/A')}")
            print(f"Sector: {yahoo_info.get('sector', 'N/A')}")
            print(f"Country: {yahoo_info.get('country', 'N/A')}")
            print(f"Employees: {yahoo_info.get('employees', 'N/A')}")
            print(f"Website: {yahoo_info.get('website', 'N/A')}")
            print(f"Market Cap: {yahoo_info.get('market_cap', 'N/A')}")
            print(f"Currency: {yahoo_info.get('currency', 'N/A')}")
        except Exception as e:
            print(f"Error printing company summary for {self.ticker_symbol}: {str(e)}")

    def forecast_prices(self):
        """Forecast future stock prices using Prophet."""
        try:
            prophet_df = self.prepare_data_for_prophet()
            if prophet_df is None or prophet_df.empty:
                logging.warning(f"No data available for forecasting for {self.ticker_symbol}")
                return None

            # Initialize and fit the Prophet model
            model = Prophet(
                daily_seasonality=True,
                yearly_seasonality=True,
                weekly_seasonality=True,
                changepoint_prior_scale=0.05,
                seasonality_prior_scale=10.0
            )

            # Convert datetime to numpy array explicitly before fitting
            prophet_df['ds'] = np.array(prophet_df['ds'].dt.to_pydatetime())
            model.fit(prophet_df)

            # Create future dataframe
            future = model.make_future_dataframe(periods=180)
            future['ds'] = np.array(future['ds'].dt.to_pydatetime())

            # Make forecast
            forecast = model.predict(future)

            # Convert forecast dates to numpy array
            forecast['ds'] = np.array(pd.to_datetime(forecast['ds']).dt.to_pydatetime())

            # Save forecast results
            self._save_forecast_results(forecast, model)

            return forecast

        except Exception as e:
            logging.error(f"Error forecasting prices for {self.ticker_symbol}: {str(e)}", exc_info=True)
            return None

    def _save_forecast_results(self, forecast, model):
        """Save forecast results and create visualizations"""
        try:
            # Create forecast directory if it doesn't exist
            forecast_dir = os.path.join(self.output_dir, 'forecast_plots')
            os.makedirs(forecast_dir, exist_ok=True)

            # Save forecast data
            forecast_file = os.path.join(forecast_dir, f'{self.ticker_symbol}_forecast.csv')
            forecast.to_csv(forecast_file, index=False)

            # Create figure for the forecast plot
            plt.figure(figsize=(12, 8))

            # Plot forecast
            forecast_dates = np.array(forecast['ds'])
            plt.plot(forecast_dates, forecast['yhat'], 'b-', label='Forecast')
            plt.fill_between(forecast_dates,
                             forecast['yhat_lower'],
                             forecast['yhat_upper'],
                             color='b',
                             alpha=0.2,
                             label='Confidence Interval')

            # Add actual prices to the plot
            if self.stock_data is not None:
                actual_dates = np.array(pd.to_datetime(self.stock_data.index).to_pydatetime())
                plt.plot(actual_dates, self.stock_data['Close'],
                         'k.', alpha=0.5, label='Actual Prices')

            plt.title(f'{self.ticker_symbol} Price Forecast')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.legend()

            # Save the forecast plot
            plot_file = os.path.join(forecast_dir, f'{self.ticker_symbol}_forecast_plot.png')
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()

            # Create and save components plot
            self._create_components_plot(forecast, model, forecast_dir)

            logging.info(f"Forecast saved to: {forecast_file}")
            logging.info(f"Forecast plots saved to: {plot_file}")

        except Exception as e:
            logging.error(f"Error saving forecast results: {str(e)}", exc_info=True)

    def _create_components_plot(self, forecast, model, forecast_dir):
        """Create and save the components plot"""
        try:
            # Create figure for components
            fig = plt.figure(figsize=(12, 10))

            # Plot trend
            plt.subplot(311)
            plt.plot(np.array(forecast['ds']), forecast['trend'])
            plt.title('Trend')

            # Plot yearly seasonality if it exists
            if 'yearly' in forecast.columns:
                plt.subplot(312)
                yearly_points = forecast['yearly'].groupby(forecast['ds'].dt.dayofyear).mean()
                plt.plot(yearly_points.index, yearly_points.values)
                plt.title('Yearly Seasonality')

            # Plot weekly seasonality if it exists
            if 'weekly' in forecast.columns:
                plt.subplot(313)
                weekly_points = forecast['weekly'].groupby(forecast['ds'].dt.dayofweek).mean()
                plt.plot(weekly_points.index, weekly_points.values)
                plt.title('Weekly Seasonality')

            plt.tight_layout()

            # Save components plot
            components_file = os.path.join(forecast_dir, f'{self.ticker_symbol}_forecast_components.png')
            plt.savefig(components_file, dpi=300, bbox_inches='tight')
            plt.close()

            logging.info(f"Components plot saved to: {components_file}")

        except Exception as e:
            logging.error(f"Error creating components plot: {str(e)}", exc_info=True)

    def prepare_data_for_prophet(self):
        """Prepare data for Prophet model"""
        if self.stock_data is None or self.stock_data.empty:
            logging.warning(f"No valid stock data for {self.ticker_symbol}")
            return None

        try:
            df = self.stock_data.reset_index()
            prophet_df = pd.DataFrame()

            # Convert dates to numpy array explicitly
            prophet_df['ds'] = pd.to_datetime(df['Date'])
            prophet_df['y'] = df['Close'].values

            return prophet_df.dropna()

        except Exception as e:
            logging.error(f"Error preparing data for Prophet: {str(e)}", exc_info=True)
            return None

    def _save_forecast_results(self, forecast, model):
        """Save forecast results and create visualizations"""
        try:
            # Create forecast directory if it doesn't exist
            forecast_dir = os.path.join(self.output_dir, 'forecast_plots')
            os.makedirs(forecast_dir, exist_ok=True)

            # Save forecast data
            forecast_file = os.path.join(forecast_dir, f'{self.ticker_symbol}_forecast.csv')
            forecast.to_csv(forecast_file, index=False)

            # Create and save plots
            fig = model.plot(forecast, uncertainty=True)
            plt.title(f'{self.ticker_symbol} Price Forecast')

            # Convert datetime values explicitly
            forecast['ds'] = pd.to_datetime(forecast['ds']).dt.to_pydatetime()

            # Add actual prices to the plot
            if self.stock_data is not None:
                actual_dates = pd.to_datetime(self.stock_data.index).to_pydatetime()
                plt.plot(actual_dates, self.stock_data['Close'],
                         'k.', alpha=0.5, label='Actual Prices')

            plt.legend()

            # Save the plot
            plot_file = os.path.join(forecast_dir, f'{self.ticker_symbol}_forecast_plot.png')
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()

            # Create components plot
            fig = model.plot_components(forecast)
            components_file = os.path.join(forecast_dir, f'{self.ticker_symbol}_forecast_components.png')
            plt.savefig(components_file, dpi=300, bbox_inches='tight')
            plt.close()

            logging.info(f"Forecast saved to: {forecast_file}")
            logging.info(f"Forecast plots saved to: {plot_file} and {components_file}")

        except Exception as e:
            logging.error(f"Error saving forecast results: {str(e)}", exc_info=True)

    def print_forecast_summary(self, forecast):
        """Print a summary of the forecast results."""
        try:
            # Extract key statistics from the forecast
            last_date = forecast['ds'].iloc[-1]
            last_price = forecast['yhat'].iloc[-1]
            price_upper = forecast['yhat_upper'].iloc[-1]
            price_lower = forecast['yhat_lower'].iloc[-1]

            print("\nForecast Summary:")
            print(f"Forecast End Date: {last_date}")
            print(f"Predicted Price: {last_price:.2f}")
            print(f"Upper Confidence Interval: {price_upper:.2f}")
            print(f"Lower Confidence Interval: {price_lower:.2f}")

        except Exception as e:
            print(f"Error printing forecast summary for {self.ticker_symbol}: {str(e)}")

    def _validate_data(self, data):
        """Validate input data"""
        if data is None or len(data) == 0:
            raise ValueError("Empty or invalid data provided")

        if isinstance(data, pd.DataFrame):
            if data.isnull().any().any():
                logging.warning("Data contains null values. Cleaning...")
                data = data.dropna()

        return data

    def _handle_outliers(self, data, columns, n_std=3):
        """Handle outliers using z-score method"""
        for column in columns:
            if column in data.columns:
                mean = data[column].mean()
                std = data[column].std()
                data[column] = data[column].clip(mean - n_std * std, mean + n_std * std)
        return data

    def perform_lstm_analysis(self):
        """Perform LSTM-based time series analysis"""
        try:
            if self.stock_data is None or self.stock_data.empty:
                logging.warning(f"No data available for LSTM analysis of {self.ticker_symbol}")
                return None

            logging.info(f"Performing LSTM analysis for {self.ticker_symbol}...")

            # Prepare data
            data = self.stock_data['Close'].values

            # Initialize and train LSTM model
            lstm_predictor = LSTMPredictor(sequence_length=60)
            history, (X_test, y_test) = lstm_predictor.train(data)

            # Make predictions
            predictions = lstm_predictor.predict(data)

            # Calculate metrics
            mse = mean_squared_error(y_test, lstm_predictor.scaler.transform(predictions[-len(y_test):]))
            mae = mean_absolute_error(y_test, lstm_predictor.scaler.transform(predictions[-len(y_test):]))

            # Create visualization
            plt.figure(figsize=(15, 8))
            plt.plot(self.stock_data.index[-len(predictions):], data[-len(predictions):],
                     label='Actual Price', alpha=0.6)
            plt.plot(self.stock_data.index[-len(predictions):], predictions,
                     label='LSTM Prediction', alpha=0.6)
            plt.title(f'{self.ticker_symbol} LSTM Price Prediction')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.legend()

            # Save results
            lstm_dir = os.path.join(self.output_dir, 'lstm_analysis')
            os.makedirs(lstm_dir, exist_ok=True)

            # Save plot
            plot_file = os.path.join(lstm_dir, f'{self.ticker_symbol}_lstm_prediction.png')
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()

            # Save metrics
            metrics = {
                'MSE': float(mse),
                'MAE': float(mae),
                'RMSE': float(np.sqrt(mse))
            }

            metrics_file = os.path.join(lstm_dir, f'{self.ticker_symbol}_lstm_metrics.json')
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=4)

            logging.info(f"LSTM analysis completed for {self.ticker_symbol}")
            logging.info(f"Results saved to: {lstm_dir}")

            return predictions, metrics

        except Exception as e:
            logging.error(f"Error performing LSTM analysis for {self.ticker_symbol}: {str(e)}", exc_info=True)
            return None

    def fetch_economic_indicators(self):
        """Fetch relevant economic indicators from alternative sources"""
        try:
            # Use Yahoo Finance to get major economic indicators via ETFs/indices
            indicators = {
                'SPY': '^GSPC',  # S&P 500
                'VIX': '^VIX',  # Volatility Index
                'TNX': '^TNX',  # 10-Year Treasury Yield
                'DXY': 'DX-Y.NYB',  # US Dollar Index
                'GLD': 'GC=F',  # Gold Futures
                'CL': 'CL=F',  # Crude Oil Futures
            }

            economic_data = pd.DataFrame()

            for indicator_name, symbol in indicators.items():
                try:
                    data = yf.download(symbol,
                                       start=self.start_date,
                                       end=self.end_date,
                                       progress=False)['Close']
                    economic_data[indicator_name] = data
                except Exception as e:
                    logging.warning(f"Could not fetch {indicator_name}: {str(e)}")

            # Add derived indicators
            if not economic_data.empty:
                # Calculate daily returns
                economic_data['SPY_Returns'] = economic_data['SPY'].pct_change()

                # Calculate rolling volatility
                economic_data['Market_Volatility'] = economic_data['SPY_Returns'].rolling(window=20).std() * np.sqrt(
                    252)

                # Calculate yield curve spread (if available)
                if 'TNX' in economic_data.columns:
                    try:
                        tyx_data = yf.download('^TYX', start=self.start_date, end=self.end_date, progress=False)[
                            'Close']
                        # Reindex both series to match dates
                        common_dates = economic_data.index.intersection(tyx_data.index)
                        economic_data['TYX'] = tyx_data[common_dates]
                        economic_data['Yield_Spread'] = economic_data['TYX'] - economic_data['TNX']
                    except Exception as e:
                        logging.warning(f"Could not calculate yield spread: {str(e)}")

                # Save economic indicators
                output_path = os.path.join(self.output_dir,
                                           f'{self.ticker_symbol}_economic_indicators.csv')
                economic_data.to_csv(output_path)
                logging.info(f"Saved economic indicators to {output_path}")

            return economic_data

        except Exception as e:
            logging.error(f"Error fetching economic indicators: {str(e)}")
            return None

    def analyze_market_conditions(self, economic_data, options_data):
        """Analyze market conditions using economic indicators and options data"""
        try:
            analysis = {}

            # Helper function to convert numpy types to native Python types
            def convert_to_native(obj):
                if isinstance(obj, (np.int64, np.int32)):
                    return int(obj)
                elif isinstance(obj, (np.float64, np.float32)):
                    return float(obj)
                elif isinstance(obj, np.bool_):
                    return bool(obj)
                elif isinstance(obj, pd.Timestamp):
                    return obj.strftime('%Y-%m-%d')
                return obj

            # Analyze economic indicators
            if economic_data is not None and not economic_data.empty:
                latest_data = economic_data.iloc[-1]
                month_ago = economic_data.iloc[-22] if len(economic_data) >= 22 else economic_data.iloc[0]

                analysis['economic'] = {
                    'market_trend': {
                        'current_level': convert_to_native(latest_data['SPY']),
                        'monthly_change': convert_to_native(((latest_data['SPY'] / month_ago['SPY']) - 1) * 100),
                        'volatility': convert_to_native(latest_data['Market_Volatility'])
                    },
                    'vix_level': convert_to_native(latest_data['VIX']),
                    'treasury_yield': convert_to_native(latest_data['TNX']),
                    'dollar_strength': {
                        'current_level': convert_to_native(latest_data['DXY']),
                        'monthly_change': convert_to_native(((latest_data['DXY'] / month_ago['DXY']) - 1) * 100)
                    },
                    'gold_price': convert_to_native(latest_data['GLD']),
                    'oil_price': convert_to_native(latest_data['CL'])
                }

                if 'Yield_Spread' in economic_data.columns:
                    analysis['economic']['yield_spread'] = convert_to_native(latest_data['Yield_Spread'])

            # Analyze options data
            if options_data is not None and not options_data.empty:
                try:
                    # Calculate put-call ratio
                    put_volume = int(options_data[options_data['type'] == 'put']['volume'].sum())
                    call_volume = int(options_data[options_data['type'] == 'call']['volume'].sum())
                    put_call_ratio = float(put_volume / call_volume) if call_volume > 0 else None

                    # Calculate average implied volatility
                    avg_iv = float(options_data['impliedVolatility'].mean())

                    # Find most active strikes
                    most_active = options_data.groupby('strike').agg({
                        'volume': 'sum',
                        'openInterest': 'sum'
                    }).sort_values('volume', ascending=False).head(5)

                    # Convert most_active to a dictionary with native Python types
                    most_active_dict = {}
                    for strike, row in most_active.iterrows():
                        most_active_dict[str(convert_to_native(strike))] = {
                            'volume': convert_to_native(row['volume']),
                            'openInterest': convert_to_native(row['openInterest'])
                        }

                    analysis['options'] = {
                        'put_call_ratio': put_call_ratio,
                        'avg_implied_volatility': convert_to_native(avg_iv * 100),
                        'total_open_interest': convert_to_native(options_data['openInterest'].sum()),
                        'most_active_strikes': most_active_dict,
                        'volume_distribution': {
                            'calls': convert_to_native(call_volume),
                            'puts': convert_to_native(put_volume)
                        }
                    }
                except Exception as e:
                    logging.warning(f"Error processing options data: {str(e)}")
                    analysis['options'] = None

            # Add market regime analysis
            if economic_data is not None and not economic_data.empty:
                vix_level = convert_to_native(latest_data['VIX'])
                market_volatility = convert_to_native(latest_data['Market_Volatility'])

                # Determine market regime
                if vix_level <= 15:
                    regime = "Low Volatility"
                elif vix_level <= 25:
                    regime = "Normal Volatility"
                elif vix_level <= 35:
                    regime = "High Volatility"
                else:
                    regime = "Extreme Volatility"

                analysis['market_regime'] = {
                    'current_regime': regime,
                    'vix_percentile': convert_to_native(economic_data['VIX'].rank(pct=True).iloc[-1] * 100),
                    'volatility_percentile': convert_to_native(
                        economic_data['Market_Volatility'].rank(pct=True).iloc[-1] * 100)
                }

            # Save analysis results
            output_path = os.path.join(self.output_dir,
                                       f'{self.ticker_symbol}_market_analysis.json')
            with open(output_path, 'w') as f:
                json.dump(analysis, f, indent=4)

            return analysis

        except Exception as e:
            logging.error(f"Error analyzing market conditions: {str(e)}")
            return None

    def _print_market_analysis(self, analysis):
        """Print a summary of the market condition analysis"""
        try:
            print("\nMarket Analysis Summary:")

            if 'economic' in analysis:
                print("\nEconomic Indicators:")
                print(f"S&P 500 Monthly Change: {analysis['economic']['market_trend']['monthly_change']:.1f}%")
                print(f"Market Volatility: {analysis['economic']['market_trend']['volatility']:.1f}%")
                print(f"VIX Index: {analysis['economic']['vix_level']:.2f}")
                print(f"10-Year Treasury Yield: {analysis['economic']['treasury_yield']:.2f}%")
                print(f"US Dollar Index Change: {analysis['economic']['dollar_strength']['monthly_change']:.1f}%")
                print(f"Gold Price: ${analysis['economic']['gold_price']:.2f}")
                print(f"Oil Price: ${analysis['economic']['oil_price']:.2f}")

                if 'yield_spread' in analysis['economic']:
                    print(f"Yield Curve Spread: {analysis['economic']['yield_spread']:.2f}%")

            if 'market_regime' in analysis:
                print(f"\nMarket Regime: {analysis['market_regime']['current_regime']}")
                print(f"VIX Percentile: {analysis['market_regime']['vix_percentile']:.1f}th")
                print(f"Volatility Percentile: {analysis['market_regime']['volatility_percentile']:.1f}th")

            if 'options' in analysis and analysis['options'] is not None:
                print("\nOptions Market Indicators:")
                print(f"Put-Call Ratio: {analysis['options']['put_call_ratio']:.2f}")
                print(f"Average Implied Volatility: {analysis['options']['avg_implied_volatility']:.1f}%")
                print(f"Total Open Interest: {analysis['options']['total_open_interest']:,}")
                print("\nVolume Distribution:")
                print(f"Calls: {analysis['options']['volume_distribution']['calls']:,}")
                print(f"Puts: {analysis['options']['volume_distribution']['puts']:,}")
                print("\nMost Active Strike Prices:")
                for strike, data in analysis['options']['most_active_strikes'].items():
                    print(f"${strike}: Volume {data['volume']:,}, OI {data['openInterest']:,}")

        except Exception as e:
            logging.error(f"Error printing market analysis: {str(e)}")

    def fetch_economic_indicators(self):
        """Fetch relevant economic indicators from alternative sources"""
        try:
            # Use Yahoo Finance to get major economic indicators via ETFs/indices
            indicators = {
                'SPY': '^GSPC',  # S&P 500
                'VIX': '^VIX',  # Volatility Index
                'TNX': '^TNX',  # 10-Year Treasury Yield
                'DXY': 'DX-Y.NYB',  # US Dollar Index
                'GLD': 'GC=F',  # Gold Futures
                'CL': 'CL=F',  # Crude Oil Futures
            }

            economic_data = pd.DataFrame()

            # Fetch data for each indicator
            for indicator_name, symbol in indicators.items():
                try:
                    data = yf.download(symbol,
                                       start=self.start_date,
                                       end=self.end_date,
                                       progress=False)['Close']
                    economic_data[indicator_name] = data
                except Exception as e:
                    logging.warning(f"Could not fetch {indicator_name}: {str(e)}")

            # Add derived indicators
            if not economic_data.empty:
                # Calculate daily returns
                economic_data['SPY_Returns'] = economic_data['SPY'].pct_change()

                # Calculate rolling volatility
                economic_data['Market_Volatility'] = economic_data['SPY_Returns'].rolling(window=20).std() * np.sqrt(
                    252)

                # Calculate yield curve spread (if available)
                if 'TNX' in economic_data.columns:
                    try:
                        # Fetch 30-year Treasury yield
                        tyx_data = yf.download('^TYX',
                                               start=self.start_date,
                                               end=self.end_date,
                                               progress=False)['Close']

                        # Ensure indexes match
                        economic_data = economic_data.loc[~economic_data.index.duplicated(keep='first')]
                        tyx_data = tyx_data.loc[~tyx_data.index.duplicated(keep='first')]

                        # Add TYX to economic data
                        economic_data['TYX'] = tyx_data

                        # Calculate spread
                        economic_data['Yield_Spread'] = economic_data['TYX'] - economic_data['TNX']

                        logging.info("Successfully calculated yield spread")
                    except Exception as e:
                        logging.warning(f"Could not calculate yield spread: {str(e)}")
                        economic_data['Yield_Spread'] = np.nan

                # Remove any remaining NaN values
                economic_data = economic_data.fillna(method='ffill').fillna(method='bfill')

                # Save economic indicators
                output_path = os.path.join(self.output_dir,
                                           f'{self.ticker_symbol}_economic_indicators.csv')
                economic_data.to_csv(output_path)
                logging.info(f"Saved economic indicators to {output_path}")

                # Log some basic statistics for verification
                logging.info(f"Economic data shape: {economic_data.shape}")
                logging.info(f"Columns: {economic_data.columns.tolist()}")

            return economic_data

        except Exception as e:
            logging.error(f"Error fetching economic indicators: {str(e)}")
            return None

    def fetch_options_data(self):
        """Fetch options data for the stock"""
        try:
            # Get options data using yfinance
            ticker = yf.Ticker(self.ticker_symbol)

            # Get all available expiration dates
            expirations = ticker.options

            if not expirations:
                logging.warning(f"No options data available for {self.ticker_symbol}")
                return None

            options_data = []

            # Get options for the next 3 expiration dates
            for expiration in expirations[:3]:
                try:
                    # Fetch calls and puts
                    opt_chain = ticker.option_chain(expiration)
                    calls = opt_chain.calls
                    puts = opt_chain.puts

                    # Process calls
                    calls['type'] = 'call'
                    calls = calls[['strike', 'lastPrice', 'volume', 'openInterest', 'impliedVolatility', 'type']]

                    # Process puts
                    puts['type'] = 'put'
                    puts = puts[['strike', 'lastPrice', 'volume', 'openInterest', 'impliedVolatility', 'type']]

                    # Combine calls and puts
                    chain_data = pd.concat([calls, puts])
                    chain_data['expiration'] = expiration

                    # Append to options data list
                    options_data.append(chain_data)

                except Exception as e:
                    logging.warning(f"Error fetching options for expiration {expiration}: {str(e)}")
                    continue

            if options_data:
                # Combine all options data
                options_df = pd.concat(options_data, ignore_index=True)

                # Fill NaN values
                options_df['volume'] = options_df['volume'].fillna(0).astype(int)
                options_df['openInterest'] = options_df['openInterest'].fillna(0).astype(int)

                # Save options data
                output_path = os.path.join(self.output_dir,
                                           f'{self.ticker_symbol}_options_data.csv')
                options_df.to_csv(output_path, index=False)
                logging.info(f"Saved options data to {output_path}")

                return options_df

            return None

        except Exception as e:
            logging.error(f"Error fetching options data: {str(e)}")
            return None



def main():
  # Set up logging
  setup_logging()
  logging.info("Starting stock analysis script")

  try:
      # Get date range from user
      start_date = get_date_input("Enter start date (YYYY-MM-DD) or press Enter for default: ")
      end_date = get_date_input("Enter end date (YYYY-MM-DD) or press Enter for default: ")

      # Set default dates if None
      if start_date is None:
          start_date = datetime.now() - timedelta(days=730)  # Default to 2 years ago
      if end_date is None:
          end_date = datetime.now()  # Default to today

      logging.info(f"Using date range: {start_date} to {end_date}")

      # Read tickers from file
      logging.info("Reading tickers from tickers.txt")
      with open('tickers.txt', 'r') as file:
          tickers = [line.strip() for line in file if line.strip()]
      logging.info(f"Found {len(tickers)} tickers to process")

      # Process each ticker
      for ticker in tickers:
          logging.info(f"\n{'=' * 50}")
          logging.info(f"Processing ticker: {ticker}")

          try:
              analyzer = StockAnalyzer(ticker, start_date, end_date)
              analyzer.analyze_stock()

              # Launch the real-time plot for the ticker
              os.system(f"bokeh serve --show real_time_plot.py --args {ticker}")

          except Exception as e:
              logging.error(f"Error processing ticker {ticker}: {str(e)}", exc_info=True)
              continue

      logging.info("Stock analysis script completed")

  except Exception as e:
      logging.error(f"Fatal error in main execution: {str(e)}", exc_info=True)

if __name__ == "__main__":
  main()

