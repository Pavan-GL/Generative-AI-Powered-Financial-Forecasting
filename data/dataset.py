import numpy as np
import pandas as pd

def generate_synthetic_stock_data(start_price, mu, sigma, days):
    """Generate synthetic stock price data using geometric Brownian motion."""
    prices = [start_price]
    for _ in range(days):
        # Calculate the daily price change
        daily_return = np.random.normal(mu, sigma)
        new_price = prices[-1] * np.exp(daily_return)
        prices.append(new_price)
    return prices

# Parameters
start_price = 100  # Starting stock price
mu = 0.0005        # Expected daily return
sigma = 0.01       # Daily volatility
days = 365         # Number of days to simulate

# Generate synthetic data
synthetic_prices = generate_synthetic_stock_data(start_price, mu, sigma, days)

# Create a DataFrame
synthetic_data = pd.DataFrame({
    'Date': pd.date_range(start='2024-01-01', periods=days+1),
    'Synthetic Price': synthetic_prices
})

# Save to CSV
synthetic_data.to_csv('synthetic_stock_data.csv', index=False)
