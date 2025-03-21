# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Load the dataset
df = pd.read_csv("heart.csv")

# Ensure all data is numeric
df = df.apply(pd.to_numeric, errors='coerce')

# Drop rows with missing values (if any)
df = df.dropna()

# Prepare features and target variable
X = df.drop('target', axis=1)  # Features (now includes all 13 columns)
y = df['target']  # Target variable (0 = no disease, 1 = disease)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 🚀 **Now, let's predict for a new patient!**
try:
    print("\nEnter new patient data:")
    age = int(input("Age: ").strip())
    sex = int(input("Sex (1 = Male, 0 = Female): ").strip())
    cp = int(input("Chest Pain Type (0-3): ").strip())
    trestbps = int(input("Resting Blood Pressure: ").strip())
    chol = int(input("Cholesterol Level: ").strip())
    fbs = int(input("Fasting Blood Sugar (1 = >120 mg/dL, 0 = <=120 mg/dL): ").strip())  
    restecg = int(input("Resting ECG (0-2): ").strip())
    thalach = int(input("Max Heart Rate Achieved: ").strip())
    exang = int(input("Exercise-Induced Angina (1 = Yes, 0 = No): ").strip())
    oldpeak = float(input("Oldpeak (ST depression): ").strip())
    slope = int(input("Slope of Peak Exercise (0-2): ").strip())
    ca = int(input("Number of Major Vessels (0-4): ").strip())
    thal = int(input("Thalassemia (1-3): ").strip())

    # Convert input into a numpy array (Now 13 elements)
    new_patient = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])

    # Scale the input
    new_patient_scaled = scaler.transform(new_patient)

    # Make the prediction
    prediction = model.predict(new_patient_scaled)

    # Display result
    if prediction[0] == 1:
        print("\n🔴 The model predicts: The patient **has** heart disease.")
    else:
        print("\n🟢 The model predicts: The patient **does not** have heart disease.")

except Exception as e:
    print("\n❌ Error:", e)
    print("Please enter valid numerical values.")

import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Load the dataset
df = pd.read_csv("heart.csv")

# Prepare features and target variable
X = df.drop('target', axis=1)  # Features
y = df['target']  # Target variable

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Save the trained model
with open("model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

# Save the scaler
with open("scaler.pkl", "wb") as scaler_file:
    pickle.dump(scaler, scaler_file)

print("✅ Model and scaler saved successfully!")