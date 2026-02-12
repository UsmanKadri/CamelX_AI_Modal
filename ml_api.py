from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

model = joblib.load("Camel_Risk_LightGBM.pkl")

@app.get("/")
def home():
    return {"status": "CamelX AI Running"}

@app.post("/predict")
def predict(data: dict):

    features = np.array([[
        data["heart_rate"],
        data["spo2"],
        data["temperature"],
        data["bp_sys"],
        data["bp_dia"],        
    ]])

    result = model.predict(features)

    score = float(result[0])

    if score < 0.3:
        category = "Low Risk"
    elif score < 0.6:
        category = "Medium Risk"
    else:
        category = "High Risk"

    return {
        "risk_score": score,
        "risk_category": category
    }
