from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import pickle
import pandas as pd
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model at global scope so it caches across serverless invocations
model = None
scaler = None
imputer = None
features = []

# Vercel sets CWD differently, use absolute path relative to this file
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "model.pkl")

try:
    with open(model_path, "rb") as f:
        artifacts = pickle.load(f)
    model = artifacts['model']
    scaler = artifacts['scaler']
    imputer = artifacts['imputer']
    features = artifacts['features']
    print("Model loaded successfully.")
except Exception as e:
    print(f"Warning: model.pkl not found at {model_path}.", e)

class PatientData(BaseModel):
    data: Dict[str, float]

@app.post("/api/predict")
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

@app.get("/api/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}
