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


class StockAnalyzer:
    def __init__(self, ticker_symbol):
        self.ticker_symbol = ticker_symbol
        self.news_data = []
        self.stock_data = None
        self.output_dir = os.path.join(ticker_symbol)  # Set directory for each ticker

        # Ensure the directory exists for each ticker
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'forecast_plots'), exist_ok=True)

    def fetch_financial_news(self):
        """Fetch financial news from various sources"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Connection': 'keep-alive',
            }
            yahoo_url = f"https://finance.yahoo.com/quote/{self.ticker_symbol}/news"
            response = requests.get(yahoo_url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            articles = []
            selectors = [
                'h3.Mb-5px',
                'h3 a',
                'h3',
                '.js-stream-content',
                '.caas-title'
            ]
            for selector in selectors:
                try:
                    found_articles = soup.select(selector)
                    if found_articles:
                        articles.extend(found_articles)
                except Exception:
                    continue

            for article in articles[:10]:
                title = article.text.strip() if article.find('a') is None else article.find('a').text.strip()
                if title and len(title) > 5:
                    sentiment = TextBlob(title).sentiment.polarity
                    self.news_data.append({
                        'title': title,
                        'sentiment': round(sentiment, 3),
                        'date': datetime.now().strftime('%Y-%m-%d'),
                        'source': 'Yahoo Finance'
                    })

            if self.news_data:
                news_df = pd.DataFrame(self.news_data).drop_duplicates(subset=['title']).sort_values('sentiment')
                news_df.to_csv(os.path.join(self.output_dir, f'{self.ticker_symbol}_news.csv'), index=False)
                print(f"News data saved for {self.ticker_symbol}")

        except Exception as e:
            print(f"Error fetching news for {self.ticker_symbol}: {str(e)}")

    def fetch_stock_data(self):
        """Fetch historical stock data"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=730)
            self.stock_data = yf.download(
                self.ticker_symbol,
                start=start_date,
                end=end_date,
                progress=False
            )
            if not self.stock_data.empty:
                self.stock_data.index = self.stock_data.index.tz_localize(None)
                self.stock_data.to_csv(os.path.join(self.output_dir, f'{self.ticker_symbol}_stock_data.csv'))
                print(f"Stock data saved for {self.ticker_symbol}")
            else:
                print(f"No stock data found for {self.ticker_symbol}")

        except Exception as e:
            print(f"Error fetching stock data for {self.ticker_symbol}: {str(e)}")
            self.stock_data = None

    def prepare_data_for_prophet(self):
        """Prepare data for Prophet model"""
        if self.stock_data is None or self.stock_data.empty:
            return None
        df = self.stock_data.reset_index()
        prophet_df = pd.DataFrame()
        prophet_df['ds'] = df['Date']
        prophet_df['y'] = df['Close'].values.flatten()
        return prophet_df.dropna()

    def forecast_prices(self, days_to_forecast=30):
        """Forecast future prices using Prophet"""
        df = self.prepare_data_for_prophet()
        if df is None:
            return None
        try:
            model = Prophet(
                daily_seasonality=True,
                weekly_seasonality=True,
                yearly_seasonality=True,
                changepoint_prior_scale=0.05
            )
            model.fit(df)
            future_dates = model.make_future_dataframe(periods=days_to_forecast)
            forecast = model.predict(future_dates)
            forecast.to_csv(os.path.join(self.output_dir, f'{self.ticker_symbol}_forecast.csv'))
            fig = model.plot(forecast)
            plt.title(f'{self.ticker_symbol} Stock Price Forecast')
            plt.savefig(os.path.join(self.output_dir, 'forecast_plots', f'{self.ticker_symbol}_forecast_plot.png'))
            plt.close(fig)
            fig2 = model.plot_components(forecast)
            plt.savefig(os.path.join(self.output_dir, 'forecast_plots', f'{self.ticker_symbol}_forecast_components.png'))
            plt.close(fig2)
            return forecast

        except Exception as e:
            print(f"Error forecasting prices for {self.ticker_symbol}: {str(e)}")
            return None

    def print_forecast_summary(self, forecast):
        """Print a detailed summary of the forecast results"""
        if forecast is not None:
            print("\n" + "=" * 50)
            print(f"FORECAST SUMMARY FOR {self.ticker_symbol}")
            print("=" * 50)
            print(f"\nProjected Growth: {((forecast['yhat'].iloc[-1] - forecast['yhat'].iloc[0]) / forecast['yhat'].iloc[0]) * 100:.1f}%")
            print(f"Forecast saved in folder: {self.output_dir}")


def main():
    # Read tickers from a text file
    with open('tickers.txt', 'r') as file:
        tickers = [line.strip() for line in file.readlines() if line.strip()]

    for ticker in tickers:
        print(f"\nProcessing {ticker}...")
        analyzer = StockAnalyzer(ticker)
        analyzer.fetch_financial_news()
        analyzer.fetch_stock_data()

        if analyzer.stock_data is not None and not analyzer.stock_data.empty:
            forecast = analyzer.forecast_prices()
            if forecast is not None:
                analyzer.print_forecast_summary(forecast)
            else:
                print(f"Forecast not generated for {ticker} due to data issues.")
        else:
            print(f"Cannot proceed with {ticker} due to missing stock data.")

if __name__ == "__main__":
    main()
