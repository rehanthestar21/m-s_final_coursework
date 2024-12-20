import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from polygon import RESTClient
import yfinance as yf
from datetime import datetime
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


class PolygonDataFetcher:
    def __init__(self, api_key):
        self.api_key = api_key
        self.client = RESTClient(api_key=self.api_key)

    def fetch_option_data(self, options_ticker, start_date, end_date):
        """
        Fetch daily aggregate bars for the specified options contract and date range from Polygon.io.
        """
        aggs = []
        for agg in self.client.list_aggs(
            ticker=options_ticker,
            multiplier=1,
            timespan="day",
            from_=start_date,
            to=end_date,
            limit=5000,
        ):
            aggs.append(agg)

        # Extract close prices and timestamps (converted to Python datetime)
        last_price = [x.close for x in aggs]
        time = [datetime.fromtimestamp(x.timestamp / 1000) for x in aggs]

        return time, last_price


class YahooDataFetcher:
    def __init__(self, ticker, start_date, end_date):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date

    def fetch_stock_data(self):
        """
        Fetch historical stock price data from Yahoo Finance and save it to a CSV file.
        """
        data = yf.download(self.ticker, start=self.start_date, end=self.end_date)
        output_file = f"{self.ticker}_stock_prices.csv"
        data.to_csv(output_file)
        return output_file


class OptionCalculator:
    @staticmethod
    def black_scholes_call(S, K, T, r, sigma):
        """
        Black-Scholes call option price formula.
        """
        # Avoid log of non-positive numbers or zero division
        if S <= 0 or K <= 0 or T <= 0:
            return 0.0
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        return call_price

    def calculate_option_premiums(
        self, 
        csv_file, 
        strike_price, 
        maturity_date, 
        interest_rate_changes, 
        volatility, 
        noise_level
    ):
        """
        Calculate call option premiums from a given CSV file of stock prices.
        Add randomized noise if needed.
        Interest rates are determined by the date from the given CSV of interest rate changes.
        """

        # Convert interest rate changes DataFrame to a list of tuples: (effective_date, rate)
        interest_rate_list = []
        for idx, row in interest_rate_changes.iterrows():
            rate_date = pd.to_datetime(row['Date'])
            rate = float(row['Federal Funds Rate (%)']) / 100.0
            interest_rate_list.append((rate_date, rate))

        # Sort by date
        interest_rate_list.sort(key=lambda x: x[0])

        # Default interest rate before the first date in interest_rate_list
        default_interest_rate = 0.05

        def get_interest_rate_for_date(d):
            # Given a date d, return the most recent interest rate applicable
            applicable_rate = default_interest_rate
            for (rate_date, rate_val) in interest_rate_list:
                if d >= rate_date:
                    applicable_rate = rate_val
                else:
                    break
            return applicable_rate

        uploaded_data = pd.read_csv(csv_file)
        cleaned_data = uploaded_data.iloc[2:]
        cleaned_data.columns = ['Date', 'Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']
        cleaned_data['Date'] = pd.to_datetime(cleaned_data['Date'], errors='coerce')
        cleaned_data['Close'] = pd.to_numeric(cleaned_data['Close'], errors='coerce')
        cleaned_data = cleaned_data.dropna()
        cleaned_data = cleaned_data.sort_values('Date')

        option_prices = []
        option_prices_random = []
        dates = []

        for _, row in cleaned_data.iterrows():
            current_date = row['Date']
            price = row['Close']
            time_to_maturity = (maturity_date - current_date).days / 365.0

            current_interest_rate = get_interest_rate_for_date(current_date)

            if time_to_maturity > 0:
                option_price = self.black_scholes_call(price, strike_price, time_to_maturity, current_interest_rate, volatility)
                random_factor = np.random.normal(1, noise_level)  # Random noise multiplier
                option_price_random = option_price * random_factor
            else:
                option_price = 0  # Option expires worthless after maturity
                option_price_random = 0

            option_prices.append(option_price)
            option_prices_random.append(option_price_random)
            dates.append(current_date)

        results = pd.DataFrame({
            'Date': dates,
            'SPY Price': cleaned_data['Close'].values,
            'Call Option Premium': option_prices,
            'Randomized Call Option Premium': option_prices_random
        })

        results_file = "eth_spy_call_option_prices_with_noise.csv"
        results.to_csv(results_file, index=False)
        return results, results_file


class Plotter:
    @staticmethod
    def plot_fetched_option_data(time_converted, last_price):
        """
        Plot the actual option last prices over time.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(time_converted, last_price, label='Price Over Time')
        plt.title('Last Price vs. Time')
        plt.xlabel('Time')
        plt.ylabel('Last Price')
        plt.grid(True)
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_option_premiums(results):
        """
        Plot the original and randomized option premiums over time.
        """
        plt.figure(figsize=(12, 6))
        plt.plot(results['Date'], results['Call Option Premium'], label='Original Premium')
        plt.plot(results['Date'], results['Randomized Call Option Premium'], label='Randomized Premium', linestyle='--')
        plt.title('Call Option Premium: Original vs. Randomized')
        plt.xlabel('Date')
        plt.ylabel('Option Premium')
        plt.grid()
        plt.legend()
        plt.show()

    @staticmethod
    def plot_comparison(time_converted, last_price, results, random=True):
        """
        Plot comparison of real premium price and Simulated Black-Scholes premium on the same axis.
        """
        plt.figure(figsize=(12, 6))
        
        # Plot real premium price
        plt.plot(time_converted, last_price, label='Real Premium Price', color='blue')
        
        # Plot Simulated premium price
        if not random:
            plt.plot(results['Date'], results['Call Option Premium'], 
                     label='Simulated Premium (Black-Scholes)', color='orange')
        else:
            plt.plot(results['Date'], results['Randomized Call Option Premium'],
                     label='Simulated Premium (Black-Scholes w/ Noise)', color='orange', linestyle='--')
        
        plt.title('Comparison: Last Price vs. Call Option Premium (Same Axis)')
        plt.xlabel('Date')
        plt.ylabel('Premium Price')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # Parameters
    polygon_api_key = "MgiVqc5YtMoq2XMXevk8Ss46koUFb2g9"
    options_ticker = "O:SPY250117C00600000"
    polygon_start_date = "2023-01-01"
    polygon_end_date = "2023-12-31"

    ticker = "SPY"
    yahoo_start_date = "2023-01-01"
    yahoo_end_date = "2023-12-31"

    strike_price = 600
    maturity_date = pd.Timestamp("2025-01-17")
    volatility = 0.12
    noise_level = 0.1

    # Load the interest rate changes from CSV
    # CSV Format:
    # Date,Federal Funds Rate (%)
    # 2023-02-01,4.5
    # 2023-03-22,4.75
    # 2023-05-03,5.0
    # 2023-07-26,5.25
    interest_rate_changes_file = "US_Federal_Funds_Rate_2023.csv"  # Make sure this is correct
    interest_rate_changes = pd.read_csv(interest_rate_changes_file)

    # Step 1: Fetch Option Data from Polygon
    polygon_fetcher = PolygonDataFetcher(api_key=polygon_api_key)
    time_converted, last_price = polygon_fetcher.fetch_option_data(
        options_ticker,
        start_date=polygon_start_date,
        end_date=polygon_end_date
    )

    # Step 2: Fetch Stock Data from Yahoo
    yahoo_fetcher = YahooDataFetcher(ticker=ticker, start_date=yahoo_start_date, end_date=yahoo_end_date)
    csv_file = yahoo_fetcher.fetch_stock_data()

    # Step 3: Calculate Option Premiums using Black-Scholes with date-based interest rates
    calculator = OptionCalculator()
    results, results_file = calculator.calculate_option_premiums(
        csv_file=csv_file,
        strike_price=strike_price,
        maturity_date=maturity_date,
        interest_rate_changes=interest_rate_changes,
        volatility=volatility,
        noise_level=noise_level
    )

    # Step 4: Plotting on the same axis
    plotter = Plotter()
    plotter.plot_comparison(time_converted, last_price, results)
    plotter.plot_comparison(time_converted, last_price, results, random=False)

    # Print a sample of the results
    print(results.head())





import numpy as np
import pandas as pd
from scipy import stats
from datetime import datetime

# Assume you have the following from your previous steps:
# time_converted: list of Python datetime objects
# last_price: list of corresponding last prices of the real option
# results: DataFrame with columns ['Date', 'SPY Price', 'Call Option Premium', 'Randomized Call Option Premium']

# Convert the list of datetimes and prices into a DataFrame
real_data_df = pd.DataFrame({
    'Date': [d.date() for d in time_converted],  # Convert datetime to just date if needed
    'Real Premium': last_price
})

# Ensure 'Date' column in results is just a date
results['Date'] = results['Date'].dt.date

# Merge the two DataFrames on 'Date'
merged_df = pd.merge(real_data_df, results, on='Date', how='inner')

# Now merged_df contains:
# ['Date', 'Real Premium', 'SPY Price', 'Call Option Premium', 'Randomized Call Option Premium']

actual = merged_df['Real Premium'].values
predicted = merged_df['Call Option Premium'].values
predicted_random = merged_df['Randomized Call Option Premium'].values

def compute_stats(actual, predicted):
    # Filter out any NaNs and zeros in 'actual' to avoid division by zero for MPE
    mask = (~np.isnan(actual) & ~np.isnan(predicted) & (actual != 0))
    actual_clean = actual[mask]
    predicted_clean = predicted[mask]

    errors = actual_clean - predicted_clean

    # Mean Error
    mean_error = np.mean(errors)

    # Mean Squared Error (MSE)
    mse = np.mean(errors ** 2)

    # Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mse)

    # Mean Absolute Error (MAE)
    mae = np.mean(np.abs(errors))

    # Mean Percentage Error (MPE)
    mpe = np.mean((errors / actual_clean)) * 100  # in percentage terms

    # Correlation
    if len(actual_clean) > 1:
        corr = np.corrcoef(actual_clean, predicted_clean)[0, 1]
    else:
        corr = np.nan

    # 95% Confidence Interval for the mean error (using t-distribution)
    n = len(errors)
    if n > 1:
        sem = stats.sem(errors)
        ci = stats.t.interval(0.95, df=n-1, loc=mean_error, scale=sem)
    else:
        ci = (np.nan, np.nan)

    return {
        'Mean Error': mean_error,
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MPE (%)': mpe,
        'Correlation': corr,
        '95% CI Mean Error': ci
    }

# Compute stats for the original Black-Scholes predicted premium
original_stats = compute_stats(actual, predicted)

# Compute stats for the randomized Black-Scholes predicted premium
random_stats = compute_stats(actual, predicted_random)

# Print the stats
print("=== Statistics for Original Black-Scholes Predictions ===")
for k, v in original_stats.items():
    if isinstance(v, tuple):
        print(f"{k}: {v[0]:.4f} to {v[1]:.4f}")
    else:
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

print("\n=== Statistics for Randomized Black-Scholes Predictions ===")
for k, v in random_stats.items():
    if isinstance(v, tuple):
        print(f"{k}: {v[0]:.4f} to {v[1]:.4f}")
    else:
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")