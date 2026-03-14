from pydantic import BaseModel

class Transaction(BaseModel):

    amount: float
    transaction_velocity: int
    merchant_risk: float
    device_score: float