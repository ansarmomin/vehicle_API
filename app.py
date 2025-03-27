from flask import Flask, request, jsonify
import joblib
import pandas as pd
import logging

# Initialize Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load model
try:
    model = joblib.load("vehicle_price_model.pkl")
    logging.info("Model loaded successfully")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    model = None

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Vehicle Price Prediction API is running!"}), 200

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "Empty request body"}), 400

        if not isinstance(data, dict):
            return jsonify({"error": "Invalid input format, expected JSON object"}), 400

        df = pd.DataFrame([data])

        # Ensure input has same features as the trained model
        if hasattr(model, "feature_names_in_"):  # Check if attribute exists
            expected_features = model.feature_names_in_
            missing_features = [f for f in expected_features if f not in df.columns]

            if missing_features:
                return jsonify({"error": f"Missing required features: {missing_features}"}), 400

        prediction = model.predict(df)[0]

        return jsonify({"predicted_price": float(prediction)}), 200

    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":

    app.run(host="0.0.0.0", port=5000, debug=True)
