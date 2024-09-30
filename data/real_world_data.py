import pandas_datareader.data as web
import datetime

# Define the date range
start = datetime.datetime(1950, 1, 1)
end = datetime.datetime(2024, 9, 30)

# Fetch inflation data (CPI) from FRED
cpi_data = web.DataReader('CPIAUCNS', 'fred', start, end)

# Display the first few rows
print(cpi_data.head())

# Save to CSV
cpi_data.to_csv('D:/Generative AI-Powered Financial Forecasting/data/real_world_inflation_data.csv')
