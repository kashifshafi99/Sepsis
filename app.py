from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import pickle
import pandas as pd
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
scaler = None
imputer = None
features = []

@app.on_event("startup")
def load_model():
    global model, scaler, imputer, features
    try:
        with open("model.pkl", "rb") as f:
            artifacts = pickle.load(f)
        model = artifacts['model']
        scaler = artifacts['scaler']
        imputer = artifacts['imputer']
        features = artifacts['features']
        print("Model loaded successfully.")
    except Exception as e:
        print("Warning: model.pkl not found or failed to load. Ensure train_model.py has run.", e)

class PatientData(BaseModel):
    data: Dict[str, float]

@app.post("/predict")
def predict(patient: PatientData):
    if not model:
        raise HTTPException(status_code=500, detail="Model is not loaded.")
    
    input_data = patient.data
    row = {}
    for f in features:
        row[f] = input_data.get(f, np.nan)
    
    df = pd.DataFrame([row])
    
    df_imp = imputer.transform(df)
    df_scaled = scaler.transform(df_imp)
    
    prob = float(model.predict_proba(df_scaled)[0][1])
    prediction = int(model.predict(df_scaled)[0])
    
    return {"prediction": prediction, "probability": prob}

from fastapi.staticfiles import StaticFiles

# Serve the static files from the 'frontend' directory
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
