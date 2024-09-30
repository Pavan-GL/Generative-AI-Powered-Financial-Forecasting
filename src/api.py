from flask import Flask, request, jsonify
from .model import FinancialForecastModel

app = Flask(__name__)
model = FinancialForecastModel()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_text = request.json.get('input', '')
        if not input_text:
            raise ValueError("Input text is required")
        
        prediction = model.predict(input_text)
        return jsonify({'prediction': prediction})
    except Exception as e:
        logging.error(f"Error in prediction endpoint: {e}")
        return jsonify({'error': str(e)}), 400

if __name__ == "__main__":
    # Load the model if it exists
    model.load_model('D:/Generative AI-Powered Financial Forecasting/data/financial_forecast_model.pt')
    app.run(debug=True)