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
        data = yf.download(self.ticker, start=self.start_date, end=self.end_date)
        output_file = f"{self.ticker}_stock_prices.csv"
        data.to_csv(output_file)
        return output_file


class PDEBlackScholesSolver:
    def __init__(self, S_max, K, T, r, sigma, M=100, N=1000):
        """
        PDE solver for Black-Scholes using a finite difference approach.
        S_max : Maximum underlying asset price considered in grid
        K : Strike price
        T : Time to maturity (in years)
        r : Risk-free interest rate
        sigma : Volatility
        M : Number of spatial steps
        N : Number of time steps
        """
        self.S_max = S_max
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.M = M    # number of spatial steps
        self.N = N    # number of time steps

    def solve(self):
        # Grid setup
        S = np.linspace(0, self.S_max, self.M+1)
        dt = self.T / self.N
        dS = self.S_max / self.M
        
        # Terminal condition at t = T
        V = np.maximum(S - self.K, 0.0)

        # Boundary conditions:
        # As S=0, V=0 for call
        # As S->S_max large, V ~ S - K*exp(-r*(T-t)), but we can approximate at S_max
        # We'll handle the boundary inside the time-stepping loop.

        # Set up coefficients for the implicit scheme
        # Using a fully implicit scheme:
        # a_i, b_i, c_i correspond to coefficients of V_{i-1}, V_i, V_{i+1}
        # from the discretization of the PDE.
        
        # Coefficients for the tridiagonal system:
        i = np.arange(1, self.M)  # interior points
        alpha = 0.5 * self.sigma**2 * i**2
        beta = 0.5 * self.r * i

        # Coefficients for the implicit scheme
        # V_i^{n} = a_i * V_{i-1}^{n+1} + b_i * V_{i}^{n+1} + c_i * V_{i+1}^{n+1}
        # after rearranging the PDE terms.
        a = -dt * (alpha - beta)
        b = 1 + dt * (self.r + 2*alpha)
        c = -dt * (alpha + beta)

        # We will solve from n = N-1 down to n = 0
        # Create matrix A and vector B for the implicit step
        # A is (M-1)x(M-1) tridiagonal
        A = np.zeros((self.M-1, self.M-1))
        # Fill the diagonals of A
        np.fill_diagonal(A, b)
        np.fill_diagonal(A[1:], a[1:])
        np.fill_diagonal(A[:,1:], c[:-1])

        # Backward in time
        for n in range(self.N-1, -1, -1):
            # At each step, we know V at time t_{n+1}, we want to find V at t_n.
            # Right-hand side
            B = V[1:self.M].copy()

            # Apply boundary conditions
            # At S=0, V=0 => no contribution needed since V[0]=0
            # At S_max, V(S_max,t) ≈ S_max - K*exp(-r*(tau)) with tau=(T-t)
            tau = n * dt
            B[-1] -= c[-1]*(self.S_max - self.K*np.exp(-self.r*(self.T - tau)))

            # Solve the linear system A * V_internal = B
            V_internal = np.linalg.solve(A, B)

            # Update the solution
            V[1:self.M] = V_internal
            V[0] = 0.0
            V[self.M] = self.S_max - self.K*np.exp(-self.r*(self.T - n*dt))

        self.S = S
        self.V = V
        return S, V

    def interpolate(self, S_values):
        # Given a list of stock prices S_values, interpolate to get the option prices
        return np.interp(S_values, self.S, self.V)


class OptionCalculator:
    def __init__(self):
        pass

    def calculate_option_premiums(self, csv_file, strike_price, maturity_date, interest_rate, volatility, noise_level):
        uploaded_data = pd.read_csv(csv_file)

        # Clean the uploaded data
        cleaned_data = uploaded_data.iloc[2:]
        cleaned_data.columns = ['Date', 'Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']
        cleaned_data['Date'] = pd.to_datetime(cleaned_data['Date'], errors='coerce')
        cleaned_data['Close'] = pd.to_numeric(cleaned_data['Close'], errors='coerce')
        cleaned_data = cleaned_data.dropna()
        cleaned_data = cleaned_data.sort_values('Date')

        # Time to maturity from the earliest date
        earliest_date = cleaned_data['Date'].min()
        T = (maturity_date - earliest_date).days / 365.0
        
        # Set up a PDE solver
        S_max = max(cleaned_data['Close'].max()*3, strike_price*3)  # Some large max S
        pde_solver = PDEBlackScholesSolver(S_max=S_max,
                                           K=strike_price,
                                           T=T,
                                           r=interest_rate,
                                           sigma=volatility,
                                           M=365,   # spatial steps
                                           N=5000)  # time steps

        S_grid, V_grid = pde_solver.solve()

        option_prices = []
        option_prices_random = []
        dates = []

        for _, row in cleaned_data.iterrows():
            current_date = row['Date']
            price = row['Close']
            time_to_maturity = (maturity_date - current_date).days / 365.0

            if time_to_maturity > 0:
                # If needed, re-solve PDE for each date or (for efficiency) 
                # assume a static interest rate and maturity, and just interpolate.
                # Strictly, we solved from earliest date. For a different t, we should re-solve or store intermediate steps.
                # For simplicity, let's just use the final PDE solution (at earliest_date), 
                # assuming no significant drift in intermediate steps. Ideally, you'd re-run with a different T.
                
                # As a simplification, we'll use the PDE solution we have and just interpolate at the given price.
                # If dates differ significantly from earliest_date, we'd need a more advanced scheme.
                
                option_price = np.interp(price, S_grid, V_grid)
                random_factor = np.random.normal(1, noise_level)  # Random noise multiplier
                option_price_random = option_price * random_factor
            else:
                option_price = 0
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
        plt.figure(figsize=(12, 6))
        plt.plot(results['Date'], results['Call Option Premium'], label='Original Premium')
        plt.plot(results['Date'], results['Randomized Call Option Premium'], label='Randomized Premium', linestyle='--')
        plt.title('Call Option Premium: Original vs. Randomized (PDE)')
        plt.xlabel('Date')
        plt.ylabel('Option Premium')
        plt.grid()
        plt.legend()
        plt.show()

    @staticmethod
    def plot_comparison(time_converted, last_price, results, random=True):
        fig, ax1 = plt.subplots(figsize=(12, 6))
        ax1.plot(time_converted, last_price, label='Real Premium Price', color='blue')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Last Price', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')

        ax2 = ax1.twinx()
        if not random:
            ax2.plot(results['Date'], results['Call Option Premium'], label='Predicted Premium (PDE)', color='orange')
        else:
            ax2.plot(results['Date'], results['Randomized Call Option Premium'], label='Predicted Premium (PDE)', color='orange')
        ax2.set_ylabel('Call Option Premium', color='orange')
        ax2.tick_params(axis='y', labelcolor='orange')

        plt.title('Comparison: Real Premium vs PDE-derived Option Premium')
        plt.grid(True)

        fig.tight_layout()
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')
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
    interest_rate = 0.05
    volatility = 0.12
    noise_level = 0.1

    polygon_fetcher = PolygonDataFetcher(api_key=polygon_api_key)
    time_converted, last_price = polygon_fetcher.fetch_option_data(options_ticker,
                                                                   start_date=polygon_start_date,
                                                                   end_date=polygon_end_date)

    yahoo_fetcher = YahooDataFetcher(ticker=ticker, start_date=yahoo_start_date, end_date=yahoo_end_date)
    csv_file = yahoo_fetcher.fetch_stock_data()

    calculator = OptionCalculator()
    results, results_file = calculator.calculate_option_premiums(csv_file=csv_file,
                                                                 strike_price=strike_price,
                                                                 maturity_date=maturity_date,
                                                                 interest_rate=interest_rate,
                                                                 volatility=volatility,
                                                                 noise_level=noise_level)

    plotter = Plotter()
    plotter.plot_comparison(time_converted, last_price, results)
    plotter.plot_comparison(time_converted, last_price, results, random=False)

    print(results.head())
