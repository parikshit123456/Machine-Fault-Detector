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
    "E000": "Normal",
    "E001": "Over Temperature",
    "E002": "High Pressure",
    "E003": "Flow Rate Out of Range",
    "E004": "High Vibration",
    "E005": "Fill Height Abnormal",
    "E006": "Power Consumption Abnormal",
    "E007": "CO2 Out of Range",
    "E008": "Humidity Out of Range"
}

app = Flask(__name__)

# Load SHAP Explainer
try:
    shap_explainer = shap.TreeExplainer(binary_model)
except:
    shap_explainer = shap.Explainer(binary_model)


# ----------------------------------------------------------
# ✅ LOAD LAST 20 LOGS FROM CSV ON STARTUP
# ----------------------------------------------------------
def load_csv_logs():
    if not os.path.exists(CSV_FILE):
        return

    df = pd.read_csv(CSV_FILE)

    # Ensure compatible columns
    required = ["timestamp", "probability", "error_code", "error_description"]
    for r in required:
        if r not in df.columns:
            df[r] = ""

    last_logs = df.tail(MAX_LOGS)

    for _, row in last_logs.iterrows():
        prediction_logs.append({
            "timestamp": row["timestamp"],
            "probability": float(row["probability"]),
            "error_code": row["error_code"],
            "error_description": row["error_description"]
        })


load_csv_logs()


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json or {}

        def safe_float(val):
            if isinstance(val, list) and len(val) > 0:
                val = val[0]
            return float(val)

        input_keys = ["temp", "pressure", "flow", "vibration",
                      "fillheight", "power", "co2", "humidity"]

        input_values = []
        for key in input_keys:
            try:
                value = safe_float(data.get(key, 0))
            except:
                value = 0
            input_values.append(value)

        df = pd.DataFrame([input_values], columns=features)

        # BINARY
        binary_prob = float(binary_model.predict_proba(df)[0][1])
        binary_pred = int(binary_model.predict(df)[0])

        # MULTICLASS
        multi_raw = int(multi_model.predict(df)[0])
        error_code = encoder.inverse_transform([multi_raw])[0]

        if binary_prob < 0.05:
            error_code = "E000"

        # SHAP
        try:
            shap_values = shap_explainer.shap_values(df)
            shap_values = np.array(shap_values)
            if shap_values.ndim == 3 and shap_values.shape[2] == 2:
                shap_values = shap_values[:, :, 1]

            contributions = {features[i]: float(shap_values.ravel()[i]) for i in range(len(features))}
        except:
            contributions = {f: 0 for f in features}

        short_desc = error_descriptions.get(error_code, "Normal")

        # ----------- SAVE IN MEMORY ----------
        log_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "probability": round(binary_prob, 4),
            "error_code": error_code,
            "error_description": short_desc
        }

        prediction_logs.append(log_entry)
        if len(prediction_logs) > MAX_LOGS:
            prediction_logs.pop(0)

        # ----------- SAVE TO CSV -------------
        file_exists = os.path.isfile(CSV_FILE)

        with open(CSV_FILE, "a", newline="") as f:
            writer = csv.writer(f)

            if not file_exists:
                writer.writerow([
                    "timestamp", "probability", "error_code", "error_description",
                    "Temperature", "Pressure", "FlowRate",
                    "Vibration", "FillHeight", "Power", "CO2", "Humidity"
                ])

            writer.writerow([
                log_entry["timestamp"],
                log_entry["probability"],
                error_code,
                short_desc,
                *input_values
            ])

        return jsonify({
            "binary": binary_pred,
            "probability": round(binary_prob, 4),
            "error_code": error_code,
            "error_description": short_desc,
            "contributions": contributions
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/logs")
def logs():
    return jsonify(prediction_logs)


if __name__ == "__main__":
    app.run(debug=True)
