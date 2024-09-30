import matplotlib.pyplot as plt
import pandas as pd
import datetime

def plot_forecast(dates, predictions):
    """Visualize the forecasted data."""
    # Convert string dates to datetime objects
    dates = pd.to_datetime(dates)

    plt.figure(figsize=(10, 5))
    plt.plot(dates, predictions, marker='o', color='blue', linestyle='-', linewidth=2, markersize=5)
    plt.title('Financial Forecast', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Predicted Value', fontsize=14)
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()  # Adjust layout to fit labels
    plt.show()

# Example usage
if __name__ == "__main__":
    # Sample data
    dates = ['2024-01-01', '2024-01-02', '2024-01-03']
    predictions = [100, 105, 110]
    plot_forecast(dates, predictions)
