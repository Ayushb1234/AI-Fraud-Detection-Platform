from fastapi import FastAPI
from backend.api.schemas import Transaction
from backend.api.predictor import predict_transaction

app = FastAPI(title="AI Fraud Detection API")

@app.get("/")
def home():
    return {"message": "Fraud Detection API running"}


@app.post("/predict")
def predict(transaction: Transaction):

    result = predict_transaction(transaction)

    return result