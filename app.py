

from flask import Flask, request, jsonify
import joblib
import numpy as np

# Initialize Flask app
app = Flask(__name__)


model = joblib.load("VotingClassifier.pkl")

@app.route("/")
def home():
    return "Fraud Detection API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        features = np.array([list(data.values())], dtype=np.float64).reshape(1, -1)
        prediction = int(model.predict(features)[0])
        probability = float(model.predict_proba(features)[0][1])

        return jsonify({
            "Cadidate Status": "Will Subscribe" if prediction == 1 else "Will not subscribe",
            "Prediction_probability": round(probability, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002)
