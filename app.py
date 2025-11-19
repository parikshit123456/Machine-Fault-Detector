from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np
import shap
from datetime import datetime
import csv
import os

CSV_FILE = "prediction_logs.csv"

# Store last 20 logs
prediction_logs = []
MAX_LOGS = 20

# Load models
binary_model = joblib.load("binary_fault_model.joblib")
multi_model = joblib.load("multiclass_error_model.joblib")
encoder = joblib.load("errorcode_label_encoder.joblib")

# Feature order
features = [
    "Temperature(°C)", "Pressure(bar)", "FlowRate(L/min)",
    "Vibration(mm/s)", "FillHeight(mm)", "PowerConsumption(kW)",
    "CO2_Level(ppm)", "Humidity(%)"
]

# Error descriptions
error_descriptions = {
    "E000": "No fault detected — all parameters are within normal range.",
    "E001": "Over Temperature — system temperature has exceeded the safe limit.",
    "E002": "High Pressure — pressure is above the safe operating range.",
    "E003": "Flow Rate Out of Range — flow is too low or too high.",
    "E004": "High Vibration — abnormal vibration detected, possible mechanical issue.",
    "E005": "Fill Height Abnormal — container level is outside the expected range.",
    "E006": "Power Consumption Abnormal — energy usage is not within normal limits.",
    "E007": "CO2 Out of Range — CO2 concentration is above or below normal.",
    "E008": "Humidity Out of Range — humidity level is outside the safe range."
}

app = Flask(__name__)

# Load SHAP explainer
try:
    shap_explainer = shap.TreeExplainer(binary_model)
except:
    shap_explainer = shap.Explainer(binary_model)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json or {}

        # Safe float conversion
        def safe_float(val):
            if isinstance(val, list) and len(val) > 0:
                val = val[0]
            return float(val)

        # Input order expected by model
        input_keys = ["temp", "pressure", "flow", "vibration",
                      "fillheight", "power", "co2", "humidity"]

        # Extract sensor inputs
        input_values = []
        for key in input_keys:
            try:
                value = safe_float(data.get(key, 0))
            except:
                value = 0
            input_values.append(value)

        # Create dataframe
        df = pd.DataFrame([input_values], columns=features)

        # === BINARY PREDICTION ===
        binary_prob = float(binary_model.predict_proba(df)[0][1])
        binary_pred = int(binary_model.predict(df)[0])

        # === MULTICLASS PREDICTION ===
        multi_raw = int(multi_model.predict(df)[0])
        error_code = encoder.inverse_transform([multi_raw])[0]

        # Force to normal if probability very low
        if binary_prob < 0.05:
            error_code = "E000"

        # === SHAP CONTRIBUTIONS ===
        try:
            shap_values = shap_explainer.shap_values(df)
            shap_values = np.array(shap_values)

            # If SHAP returns (1, n_features, 2) take class 1
            if shap_values.ndim == 3 and shap_values.shape[2] == 2:
                shap_values = shap_values[:, :, 1]

            shap_flat = shap_values.ravel()

            contributions = {
                features[i]: float(shap_flat[i]) for i in range(len(features))
            }

        except Exception as e:
            print("SHAP error:", e)
            contributions = {f: 0 for f in features}

        # === SAVE TO IN-MEMORY LOGS ===
        log_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "probability": round(binary_prob, 4),
            "error_code": error_code
        }

        prediction_logs.append(log_entry)
        if len(prediction_logs) > MAX_LOGS:
            prediction_logs.pop(0)

        # === SAVE TO CSV FILE ===
        csv_headers = [
            "timestamp", "probability", "error_code",
            "Temperature", "Pressure", "FlowRate",
            "Vibration", "FillHeight", "Power",
            "CO2", "Humidity"
        ]

        file_exists = os.path.isfile(CSV_FILE)

        with open(CSV_FILE, "a", newline="") as f:
            writer = csv.writer(f)

            if not file_exists:
                writer.writerow(csv_headers)

            writer.writerow([
                log_entry["timestamp"],
                log_entry["probability"],
                error_code,
                input_values[0], input_values[1], input_values[2],
                input_values[3], input_values[4], input_values[5],
                input_values[6], input_values[7]
            ])

        # === RESPONSE ===
        return jsonify({
            "binary": binary_pred,
            "probability": round(binary_prob, 4),
            "error_code": error_code,
            "error_description": error_descriptions.get(error_code, "No fault detected."),
            "contributions": contributions
        })

    except Exception as e:
        return jsonify({
            "binary": 0,
            "probability": 0,
            "error_code": "E000",
            "error_description": "No fault detected - normal.",
            "contributions": {},
            "error": str(e)
        }), 500


@app.route("/logs")
def logs():
    return jsonify(prediction_logs)


if __name__ == "__main__":
    app.run(debug=True)