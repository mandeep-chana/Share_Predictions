import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from prophet import Prophet
from bs4 import BeautifulSoup
import requests
import matplotlib.pyplot as plt
from textblob import TextBlob


class StockAnalyzer:
    def __init__(self, ticker_symbol):
        self.ticker_symbol = ticker_symbol
        self.news_data = []
        self.stock_data = None

    def fetch_financial_news(self):
        """Fetch financial news from various sources"""
        try:
            # Use a more complete headers configuration
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Connection': 'keep-alive',
            }

            # Try multiple news sources
            # 1. Yahoo Finance
            yahoo_url = f"https://finance.yahoo.com/quote/{self.ticker_symbol}/news"
            response = requests.get(yahoo_url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')

            # Find news articles using multiple possible selectors
            articles = []

            # Try different HTML selectors
            selectors = [
                'h3.Mb-5px',  # Corrected Yahoo Finance class
                'h3 a',  # Generic h3 with anchor
                'h3',  # Generic h3
                '.js-stream-content',  # News stream content
                '.caas-title'  # Another possible class
            ]

            for selector in selectors:
                try:
                    found_articles = soup.select(selector)
                    if found_articles:
                        articles.extend(found_articles)
                except Exception as e:
                    continue  # Skip failed selectors silently

            # Process found articles
            for article in articles[:10]:
                title = ''
                if article.name == 'a':
                    title = article.text.strip()
                elif article.find('a'):
                    title = article.find('a').text.strip()
                else:
                    title = article.text.strip()

                if title and len(title) > 5:  # Basic validation
                    sentiment = TextBlob(title).sentiment.polarity
                    self.news_data.append({
                        'title': title,
                        'sentiment': round(sentiment, 3),
                        'date': datetime.now().strftime('%Y-%m-%d'),
                        'source': 'Yahoo Finance'
                    })

            # Alternative source: Finviz
            try:
                finviz_url = f"https://finviz.com/quote.ashx?t={self.ticker_symbol}"
                response = requests.get(finviz_url, headers=headers, timeout=10)
                soup = BeautifulSoup(response.text, 'html.parser')

                news_table = soup.find('table', {'class': 'news-table'})
                if news_table:
                    for row in news_table.find_all('tr')[:10]:
                        cells = row.find_all('td')
                        if len(cells) == 2:
                            title = cells[1].text.strip()
                            if title and len(title) > 5:
                                sentiment = TextBlob(title).sentiment.polarity
                                self.news_data.append({
                                    'title': title,
                                    'sentiment': round(sentiment, 3),
                                    'date': datetime.now().strftime('%Y-%m-%d'),
                                    'source': 'Finviz'
                                })
            except Exception as e:
                print(f"Note: Finviz news fetch failed: {str(e)}")

            # Save to CSV if we have any news
            if self.news_data:
                news_df = pd.DataFrame(self.news_data)
                news_df = news_df.drop_duplicates(subset=['title'])  # Remove duplicates

                # Sort by absolute sentiment value to show most significant news first
                news_df['abs_sentiment'] = news_df['sentiment'].abs()
                news_df = news_df.sort_values('abs_sentiment', ascending=False)
                news_df = news_df.drop('abs_sentiment', axis=1)

                # Save to CSV
                news_df.to_csv(f'{self.ticker_symbol}_news.csv', index=False)
                print(f"\nNews data saved to {self.ticker_symbol}_news.csv")
                print(f"Found {len(news_df)} unique news articles")

                # Print sample of news with sentiment analysis
                print("\nTop news articles by sentiment impact:")
                pd.set_option('display.max_colwidth', 100)
                print(news_df[['title', 'sentiment', 'source']].head().to_string())

                # Print sentiment summary
                print("\nSentiment Summary:")
                print(f"Average sentiment: {news_df['sentiment'].mean():.3f}")
                print(f"Most positive: {news_df['sentiment'].max():.3f}")
                print(f"Most negative: {news_df['sentiment'].min():.3f}")

            else:
                print("No news articles found")

        except Exception as e:
            print(f"Error fetching news: {str(e)}")

    def fetch_stock_data(self):
        """Fetch historical stock data"""
        try:
            # Get stock data for the last 2 years
            end_date = datetime.now()
            start_date = end_date - timedelta(days=730)  # 2 years

            # Fetch data directly as pandas DataFrame
            self.stock_data = yf.download(
                self.ticker_symbol,
                start=start_date,
                end=end_date,
                progress=False
            )

            if self.stock_data.empty:
                print(f"No data found for ticker {self.ticker_symbol}")
                return

            # Remove timezone information
            self.stock_data.index = self.stock_data.index.tz_localize(None)

            # Save stock data to CSV
            self.stock_data.to_csv(f'{self.ticker_symbol}_stock_data.csv')
            print(f"Stock data saved to {self.ticker_symbol}_stock_data.csv")
            print(f"Downloaded {len(self.stock_data)} rows of stock data")

        except Exception as e:
            print(f"Error fetching stock data: {str(e)}")
            self.stock_data = None

    def prepare_data_for_prophet(self):
        """Prepare data for Prophet model"""
        if self.stock_data is None or self.stock_data.empty:
            print("No stock data available for preparation")
            return None

        try:
            # Create a copy of the data and reset index
            df = self.stock_data.reset_index()

            # Remove timezone information from dates
            df['Date'] = df['Date'].dt.tz_localize(None)

            # Create Prophet dataframe with required columns
            prophet_df = pd.DataFrame()
            prophet_df['ds'] = df['Date']  # Date column without timezone
            prophet_df['y'] = df['Close'].values.flatten()  # Close price column

            # Ensure data is sorted by date
            prophet_df = prophet_df.sort_values('ds')

            # Remove any missing values
            prophet_df = prophet_df.dropna()

            print(f"Prepared {len(prophet_df)} rows of data for forecasting")
            print("Sample of prepared data:")
            print(prophet_df.head())

            # Verify no timezone information
            print("\nVerifying datetime format:")
            print(f"Timezone information present: {prophet_df['ds'].dt.tz is not None}")

            return prophet_df

        except Exception as e:
            print(f"Error preparing data: {str(e)}")
            print("DataFrame shape:", df.shape if 'df' in locals() else "Not created")
            print("Close column shape:", df['Close'].shape if 'df' in locals() else "Not created")
            return None

    def forecast_prices(self, days_to_forecast=30):
        """Forecast future prices using Prophet"""
        df = self.prepare_data_for_prophet()
        if df is None:
            print("Cannot perform forecast without prepared data")
            return None

        try:
            # Create and fit Prophet model
            print("Creating Prophet model...")
            model = Prophet(
                daily_seasonality=True,
                weekly_seasonality=True,
                yearly_seasonality=True,
                changepoint_prior_scale=0.05
            )

            print("Fitting Prophet model...")
            model.fit(df)

            # Create future dates for forecasting
            print("Creating future dates dataframe...")
            future_dates = model.make_future_dataframe(periods=days_to_forecast)

            print("Generating forecast...")
            # Make predictions
            forecast = model.predict(future_dates)

            # Save forecast to CSV
            forecast_file = f'{self.ticker_symbol}_forecast.csv'
            forecast.to_csv(forecast_file)
            print(f"Forecast data saved to {forecast_file}")

            # Create directory for plots if it doesn't exist
            import os
            plots_dir = 'forecast_plots'
            os.makedirs(plots_dir, exist_ok=True)

            # Plot the forecast
            print("Creating forecast plot...")
            fig = model.plot(forecast)
            plt.title(f'{self.ticker_symbol} Stock Price Forecast')
            forecast_plot = os.path.join(plots_dir, f'{self.ticker_symbol}_forecast_plot.png')
            plt.savefig(forecast_plot)
            plt.close(fig)  # Close the figure to free memory
            print(f"Forecast plot saved to {forecast_plot}")

            # Plot the components
            print("Creating components plot...")
            fig2 = model.plot_components(forecast)
            components_plot = os.path.join(plots_dir, f'{self.ticker_symbol}_forecast_components.png')
            plt.savefig(components_plot)
            plt.close(fig2)  # Close the figure to free memory
            print(f"Forecast components plot saved to {components_plot}")

            # Print some forecast statistics
            print("\nForecast Statistics:")
            print(f"Number of forecasted days: {days_to_forecast}")
            print(f"Forecast date range: {forecast['ds'].min()} to {forecast['ds'].max()}")
            print(f"Average forecasted price: ${forecast['yhat'].mean():.2f}")
            print(f"Price range: ${forecast['yhat'].min():.2f} to ${forecast['yhat'].max():.2f}")

            return forecast

        except Exception as e:
            print(f"Error in forecasting: {str(e)}")
            import traceback
            print("Detailed error:")
            print(traceback.format_exc())
            return None


    def print_forecast_summary(self, forecast):
        """Print a detailed summary of the forecast results"""
        print("\n" + "=" * 50)
        print(f"FORECAST SUMMARY FOR {self.ticker_symbol}")
        print("=" * 50)

        # Date range
        print("\nForecast Period:")
        print(f"Start Date: {forecast['ds'].min().strftime('%Y-%m-%d')}")
        print(f"End Date: {forecast['ds'].max().strftime('%Y-%m-%d')}")

        # Price Statistics
        print("\nPrice Predictions:")
        print(f"Current Price: ${forecast['yhat'].iloc[-1]:.2f}")
        print(f"30-Day Average: ${forecast['yhat'].tail(30).mean():.2f}")
        print(f"30-Day Range: ${forecast['yhat'].tail(30).min():.2f} - ${forecast['yhat'].tail(30).max():.2f}")

        # Confidence Intervals
        print("\nConfidence Intervals (Last Day):")
        last_day = forecast.iloc[-1]
        print(f"Lower Bound: ${last_day['yhat_lower']:.2f}")
        print(f"Upper Bound: ${last_day['yhat_upper']:.2f}")
        print(f"Range: ${(last_day['yhat_upper'] - last_day['yhat_lower']):.2f}")

        # Growth Analysis
        first_price = forecast['yhat'].iloc[0]
        last_price = forecast['yhat'].iloc[-1]
        growth = ((last_price - first_price) / first_price) * 100
        print(f"\nProjected Growth: {growth:.1f}%")

        # Volatility Measure
        volatility = forecast['yhat'].std()
        print(f"Forecast Volatility: ${volatility:.2f}")

        print("\nGenerated Files:")
        print(f"- Detailed forecast data: {self.ticker_symbol}_forecast.csv")
        print(f"- Forecast visualization: forecast_plots/{self.ticker_symbol}_forecast_plot.png")
        print(f"- Components analysis: forecast_plots/{self.ticker_symbol}_forecast_components.png")


def main():
  # Example usage
  ticker = "AAPL"  # Change this to any stock symbol you want to analyze
  analyzer = StockAnalyzer(ticker)

  # Fetch news and stock data
  print(f"Fetching news for {ticker}...")
  analyzer.fetch_financial_news()

  print(f"Fetching stock data for {ticker}...")
  analyzer.fetch_stock_data()

  if analyzer.stock_data is not None and not analyzer.stock_data.empty:
      # Verify the data
      print("\nStock data verification:")
      print(f"Number of records: {len(analyzer.stock_data)}")
      print("First few rows of stock data:")
      print(analyzer.stock_data.head())

      # Perform forecasting
      print("\nGenerating forecast...")
      forecast = analyzer.forecast_prices()

      if forecast is not None:
          # Print detailed forecast summary
          analyzer.print_forecast_summary(forecast)

          # Verify forecast output
          print("\nForecast verification:")
          print("Forecast files created:")
          import os
          expected_files = [
              f'{ticker}_forecast.csv',
              os.path.join('forecast_plots', f'{ticker}_forecast_plot.png'),
              os.path.join('forecast_plots', f'{ticker}_forecast_components.png')
          ]
          for file in expected_files:
              if os.path.exists(file):
                  print(f"✓ {file} (Size: {os.path.getsize(file)} bytes)")
              else:
                  print(f"✗ {file} (Not created)")
  else:
      print("Unable to proceed with forecast due to missing stock data")

if __name__ == "__main__":
    main()
