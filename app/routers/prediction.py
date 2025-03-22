from fastapi import APIRouter
from pydantic import BaseModel, Field
import numpy as np
import joblib
import logging


router = APIRouter()

class PredictionRequest(BaseModel):
    greScore: int = Field(..., ge=0, le=340)
    toeflScore: int = Field(..., ge=92, le=120)
    universityRating: int = Field(..., ge=1, le=5)
    sop: float = Field(..., ge=0.0, le=5.0)
    lor: float = Field(..., ge=0.0, le=5.0)
    cgpa: float = Field(..., ge=0.0, le=10.0)
    research: bool

with open("app/model.pkl", "rb") as f:
    model = joblib.load(f)


@router.post("/predict")
async def predict (request: PredictionRequest):

    X_input = [request.greScore, request.toeflScore, request.universityRating, request.sop, request.lor, request.cgpa, request.research]

    X_input = np.array(X_input).reshape(1, -1)
    logging.info(f"X_input: {X_input}")
    prediction = model.predict(X_input)

    prediction = round(prediction[0]*100, 2)

    return {"prediction": prediction}