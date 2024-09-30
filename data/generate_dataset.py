import pandas as pd
import numpy as np
import random

# Set random seed for reproducibility
np.random.seed(42)

# Parameters for the synthetic data
start_date = '1995-01-01'
end_date = '2024-09-30'
num_rows = pd.date_range(start=start_date, end=end_date, freq='B').size  # Count business days

initial_price = 100.0
daily_volatility = 0.02  # Daily volatility for price changes

# Generate date range for business days
dates = pd.date_range(start=start_date, end=end_date, freq='B')

# Generate synthetic stock prices using a random walk with volatility
prices = [initial_price]  # Starting price
for _ in range(1, num_rows):
    # Daily return based on a normal distribution
    daily_return = np.random.normal(0, daily_volatility)
    new_price = max(prices[-1] * (1 + daily_return), 0)  # Ensure price doesn't go negative
    prices.append(new_price)

# Generate random trading volume
volume = np.random.randint(100000, 2000000, size=num_rows)

# Prepare Open, Close, High, Low prices
open_prices = prices[:-1]  # All except the last price (49,999)
close_prices = prices[1:]   # All except the first price (49,999)
high_prices = []
low_prices = []

# Calculate high and low prices
for i in range(num_rows - 1):
    high_prices.append(max(open_prices[i], close_prices[i]))
    low_prices.append(min(open_prices[i], close_prices[i]))

# Insert the first price as the first high and low
high_prices.insert(0, prices[0])
low_prices.insert(0, prices[0])

# Now all lists should have 50,000 entries
data = {
    'Date': dates,
    'Open': open_prices + [prices[-1]],  # Include last price for Open
    'High': high_prices,                  # Keep High at 50,000
    'Low': low_prices,                    # Keep Low at 50,000
    'Close': close_prices + [prices[-1]],  # Include last price for Close
    'Volume': volume
}

# Check lengths of all arrays before creating DataFrame
for key, value in data.items():
    print(f"{key}: {len(value)}")

# Create DataFrame
df = pd.DataFrame(data)

# Optionally, you can add more features here (e.g., moving averages)
df['MA_10'] = df['Close'].rolling(window=10).mean()  # 10-day moving average
df['MA_50'] = df['Close'].rolling(window=50).mean()  # 50-day moving average

# Save to CSV
df.to_csv('D:/Generative AI-Powered Financial Forecasting/data/synthetic_financial_data_until_2024_09_30.csv', index=False)
print("Dataset created with rows until 2024-09-30 and saved to 'synthetic_financial_data_until_2024_09_30.csv'.")
