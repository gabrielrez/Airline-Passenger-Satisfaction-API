from fastapi import FastAPI
import pandas as pd
import joblib
import numpy as np

app = FastAPI(
    title="Airline Passenger Satisfaction API",
    description="API para prever satisfação de passageiros aéreos",
    version="1.0.0"
)

model = joblib.load("random_forest_model.pkl")
model_columns = joblib.load("model_columns.pkl")

@app.get("/")
def home():
    return {"message": "Hello World, FastAPI"}

@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])

    categorical_features = ['Gender', 'Customer Type', 'Type of Travel', 'Class']
    df = pd.get_dummies(df, columns=categorical_features, drop_first=True)

    df = df.reindex(columns=model_columns, fill_value=0)

    pred = model.predict(df)[0]
    proba = model.predict_proba(df)[0][1]

    return {
        "prediction": int(pred),
        "satisfaction": "satisfied" if pred == 1 else "neutral or dissatisfied",
        "probability": float(proba)
    }
