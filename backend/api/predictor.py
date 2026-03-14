import joblib
import numpy as np

MODEL_PATH = "models/fraud_model.pkl"

model = joblib.load(MODEL_PATH)


def build_features(transaction):

    features = [
        transaction.amount,
        transaction.transaction_velocity,
        transaction.merchant_risk,
        transaction.device_score
    ]

    # pad to 34 features
    features = features + [0]*(34-len(features))

    return np.array(features).reshape(1, -1)


def predict_transaction(transaction):

    features = build_features(transaction)

    prob = model.predict_proba(features)[0][1]

    decision = "BLOCK" if prob > 0.8 else "ALLOW"

    return {
        "fraud_probability": float(prob),
        "decision": decision
    }