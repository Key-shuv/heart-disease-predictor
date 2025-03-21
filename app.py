from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)  # Create a Flask web application

# Load trained model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Collect input values in the correct order
        age = float(request.form["age"])
        sex = float(request.form["sex"])
        cp = float(request.form["cp"])
        trestbps = float(request.form["trestbps"])
        chol = float(request.form["chol"])
        fbs = float(request.form["fbs"])
        restecg = float(request.form["restecg"])
        thalach = float(request.form["thalach"])
        exang = float(request.form["exang"])
        oldpeak = float(request.form["oldpeak"])
        slope = float(request.form["slope"])
        ca = float(request.form["ca"])
        thal = float(request.form["thal"])

        # Convert to numpy array
        user_input = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])

        # Scale the input (only if scaler was used during training)
        user_input_scaled = scaler.transform(user_input)

        # Make a prediction
        prediction = model.predict(user_input_scaled)

        # Convert result to readable format
        result = "has heart disease" if prediction[0] == 1 else "does not have heart disease"

        return render_template("result.html", result=result)

    except Exception as e:
        return f"Error: {e}"

import os

if __name__ == "__main__":  # Ensures Flask only runs when executing the script directly
    port = int(os.environ.get("PORT", 10000))  # Get Render's assigned port
    app.run(host="0.0.0.0", port=port, debug=True)