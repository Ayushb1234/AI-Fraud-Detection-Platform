import streamlit as st
import pandas as pd
import random
import time
import requests

API_URL = "https://ai-fraud-detection-platform.onrender.com/predict"


st.set_page_config(page_title="Fraud Monitoring Dashboard", layout="wide")

st.title("💳 Real-Time Fraud Detection Dashboard")

placeholder = st.empty()

data = []

while True:

    # Generate simulated transaction
    transaction = {
        "amount": random.uniform(10, 2000),
        "transaction_velocity": random.randint(1, 10),
        "merchant_risk": random.uniform(0, 1),
        "device_score": random.uniform(0, 1)
    }

    try:
        # Send to API
        response = requests.post(API_URL, json=transaction)

        result = response.json()

        fraud_probability = result["fraud_probability"]
        decision = result["decision"]

    except:
        fraud_probability = 0
        decision = "API ERROR"

    transaction["fraud_probability"] = fraud_probability
    transaction["decision"] = decision

    data.append(transaction)

    df = pd.DataFrame(data)

    with placeholder.container():

        col1, col2, col3 = st.columns(3)

        col1.metric("Total Transactions", len(df))
        col2.metric("Frauds Detected", len(df[df["decision"] == "BLOCK"]))

        fraud_rate = (len(df[df["decision"] == "BLOCK"]) / len(df)) * 100

        col3.metric("Fraud Rate", f"{fraud_rate:.2f}%")

        st.line_chart(df["fraud_probability"])

        st.dataframe(df.tail(10))

    time.sleep(2)
