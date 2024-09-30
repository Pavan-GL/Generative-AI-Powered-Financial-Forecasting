import pandas as pd
import logging

class FinancialDataProcessor:
    def __init__(self, input_file, output_file):
        self.input_file = input_file
        self.output_file = output_file
        self.df = None

        # Configure logging
        logging.basicConfig(
            filename='financial_data_processor.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def load_data(self):
        """Load financial data from a CSV file."""
        try:
            self.df = pd.read_csv(self.input_file)
            logging.info(f"Data loaded successfully from {self.input_file}.")
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            raise

    def preprocess_data(self):
        """Preprocess the data (handle missing values, normalization, etc.)."""
        if self.df is not None:
            self.df.fillna(method='ffill', inplace=True)
            logging.info("Data preprocessing completed successfully.")
        else:
            logging.error("No data to preprocess.")
            raise ValueError("Data not loaded.")

    def save_data(self):
        """Save the processed data to a CSV file."""
        if self.df is not None:
            self.df.to_csv(self.output_file, index=False)
            logging.info(f"Processed data saved to {self.output_file}.")
        else:
            logging.error("No data to save.")
            raise ValueError("Data not loaded.")

# Example usage
if __name__ == "__main__":
    input_path = 'D:/Generative AI-Powered Financial Forecasting/data/real_world_inflation_data.csv'
    output_path = 'D:/Generative AI-Powered Financial Forecasting/data/processed_real_world_inflation_data.csv'
    
    processor = FinancialDataProcessor(input_path, output_path)
    processor.load_data()
    processor.preprocess_data()
    processor.save_data()
