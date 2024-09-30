import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import logging
import pickle
import pandas as pd
import os

class FinancialForecastModel:
    def __init__(self, log_file='model.log'):
        # Set up logging
        logging.basicConfig(filename=log_file, level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')
        logging.info("Initializing FinancialForecastModel")

        # Load tokenizer and model
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.model = GPT2LMHeadModel.from_pretrained("gpt2")

        # Set the padding token
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.save_pretrained('D:/Generative AI-Powered Financial Forecasting/data/financial_forecast_model')
        self.tokenizer.save_pretrained('D:/Generative AI-Powered Financial Forecasting/data/financial_forecast_model')
        logging.info("Model and tokenizer loaded successfully with padding token set")

    def fine_tune(self, csv_file):
        try:
            logging.info(f"Loading training data from {csv_file}")
            # Load the CSV file
            data = pd.read_csv(csv_file)

            # Print the columns for debugging
            logging.info(f"Columns in the CSV: {data.columns.tolist()}")

            # Check if required columns exist
           # required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            required_columns = ['DATE', 'CPIAUCNS']
            for col in required_columns:
                if col not in data.columns:
                    raise ValueError(f"CSV must contain a '{col}' column")

            # Create training texts from the CSV data
            training_texts = []
            # for _, row in data.iterrows():
            #     training_text = (f"On {row['Date']}, the stock opened at {row['Open']}, "
            #                     f"reached a high of {row['High']}, "
            #                     f"hit a low of {row['Low']}, "
            #                     f"and closed at {row['Close']}.")
            #     training_texts.append(training_text)

            for _,row in data.iterrows():
                training_text = (f"On {row['DATE']}, the stock consumer price index {row['CPIAUCNS']}")
                training_texts.append(training_text)


            logging.info("Training data created successfully")

            # Tokenize the training texts
            inputs = self.tokenizer(training_texts, return_tensors='pt', padding=True, truncation=True)
            # Implement model training here (pseudo-code)
            # self.model.train()
            # ...
            logging.info("Fine-tuning completed")
        except Exception as e:
            logging.error(f"Error during fine-tuning: {e}")

    def predict(self, input_text, temperature=0.9, top_k=50, top_p=0.95):
        try:
            # Load the fine-tuned model and tokenizer
            model = GPT2LMHeadModel.from_pretrained('D:/Generative AI-Powered Financial Forecasting/data/financial_forecast_model')
            tokenizer = GPT2Tokenizer.from_pretrained('D:/Generative AI-Powered Financial Forecasting/data/financial_forecast_model')

            logging.info(f"Making prediction for input: {input_text}")
            inputs = self.tokenizer.encode(input_text, return_tensors='pt')
            attention_mask = torch.ones(inputs.shape, dtype=torch.long)
            attention_mask = (inputs != self.tokenizer.pad_token_id).long() 
            outputs = self.model.generate(inputs, max_length=1024,temperature=temperature, top_p=top_p, top_k=top_k,num_return_sequences=1, do_sample=True,attention_mask=attention_mask,eos_token_id=tokenizer.eos_token_id,)
            prediction = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            logging.info(f"Prediction successful: {prediction}")
            return prediction
        except Exception as e:
            logging.error(f"Error during prediction: {e}")
            return None

    def save_model(self, file_path='D:/Generative AI-Powered Financial Forecasting/data/financial_forecast_model.pt'):
        try:
            torch.save(self.model.state_dict(), file_path)
            logging.info(f"Model saved successfully to {file_path}")
        except Exception as e:
            logging.error(f"Error saving model: {e}")

    def load_model(self, file_path='D:/Generative AI-Powered Financial Forecasting/data/financial_forecast_model.pt'):
        try:
            self.model.load_state_dict(torch.load(file_path))
            self.model.eval()  # Set the model to evaluation mode
            logging.info(f"Model loaded successfully from {file_path}")
        except Exception as e:
            logging.error(f"Error loading model: {e}")

# Example usage
if __name__ == "__main__":
    model = FinancialForecastModel()
    
    # Fine-tune the model using a CSV file
    model.fine_tune('D:/Generative AI-Powered Financial Forecasting/data/processed_real_world_inflation_data.csv') # Replace with your actual CSV file path
    input_text = "Considering the recent trends in inflation and interest rates in 2024, alongside rising consumer prices, "
    input_text += "the stock market will likely face increased volatility and potential downgrades due to economic uncertainty."
    
    prediction = model.predict(input_text=input_text)
    print(prediction)

    # Save the model after training (when fine-tuned)
    model.save_model('D:/Generative AI-Powered Financial Forecasting/data/financial_forecast_model.pt')

   

# After training your model
    
    
    print("loading done")
