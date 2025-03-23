import csv
from datetime import datetime
from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load trained model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Collect input values
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

        # Prepare input
        user_input = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                                thalach, exang, oldpeak, slope, ca, thal]])
        user_input_scaled = scaler.transform(user_input)
        prediction = model.predict(user_input_scaled)
        result = "has heart disease" if prediction[0] == 1 else "does not have heart disease"

        # Log to CSV
        with open("logs.csv", mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                age, sex, cp, trestbps, chol, fbs, restecg,
                thalach, exang, oldpeak, slope, ca, thal,
                result
            ])

        return render_template("result.html", result=result)

    except Exception as e:
        return f"Error: {e}"

@app.route("/summary")
def summary():
    has_disease = 0
    no_disease = 0

    try:
        with open("logs.csv", "r") as file:
            reader = csv.reader(file)
            for row in reader:
                if len(row) > 0 and row[-1] == "has heart disease":
                    has_disease += 1
                elif len(row) > 0 and row[-1] == "does not have heart disease":
                    no_disease += 1
    except FileNotFoundError:
        pass

    return render_template("summary.html", has_disease=has_disease, no_disease=no_disease)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)