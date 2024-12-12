from flask import Flask, render_template, request
import joblib
import pandas as pd
import os

# Initialize Flask application
app = Flask(__name__)

# Define the model path
MODEL_PATH = 'financial_fraud_model.joblib'

# Load the trained model and feature names
model = None
expected_feature_names = None
model_error = None

def load_model():
    """Load the trained model and its expected feature names."""
    global model, expected_feature_names, model_error
    try:
        if os.path.exists(MODEL_PATH):
            model_data = joblib.load(MODEL_PATH)
            model = model_data.get('model')
            expected_feature_names = model_data.get('feature_names')
            if not model or not expected_feature_names:
                raise ValueError("Invalid model structure in joblib file.")
        else:
            raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")
    except Exception as e:
        model_error = f"Error loading model: {e}"

# Load the model at startup
load_model()

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle predictions based on user input."""
    if model_error:
        return render_template('index.html', prediction_text=model_error)

    try:
        # Extract input features from the form and ensure they are in the correct order
        features = [
            float(request.form.get(feature, 0)) for feature in expected_feature_names
        ]

        # Convert input features into a DataFrame with correct feature names
        input_data = pd.DataFrame([features], columns=expected_feature_names)

        # Predict using the loaded model
        prediction = model.predict(input_data)
        result = 'Fraudulent' if prediction[0] == 1 else 'Not Fraudulent'

        return render_template('index.html', prediction_text=f'This transaction is {result}')
    except Exception as e:
        # Display error message during prediction
        return render_template('index.html', prediction_text=f"Error during prediction: {str(e)}")

if __name__ == '__main__':
    # Run the application
    app.run(debug=True, host='0.0.0.0', port=8080)
