import streamlit as st
import pandas as pd
import random
import time

API_URL = "https://ai-fraud-detection-platform.onrender.com/predict"

st.set_page_config(page_title="Fraud Monitoring Dashboard", layout="wide")

st.title("💳 Real-Time Fraud Detection Dashboard")

placeholder = st.empty()

data = []

while True:

    transaction = {
        "amount": random.uniform(10,2000),
        "fraud_probability": random.uniform(0,1)
    }

    transaction["decision"] = "BLOCK" if transaction["fraud_probability"] > 0.8 else "ALLOW"

    data.append(transaction)

    df = pd.DataFrame(data)

    with placeholder.container():

        col1, col2, col3 = st.columns(3)

        col1.metric("Total Transactions", len(df))
        col2.metric("Frauds Detected", len(df[df["decision"]=="BLOCK"]))
        col3.metric("Fraud Rate", f"{(len(df[df['decision']=='BLOCK'])/len(df))*100:.2f}%")

        st.line_chart(df["fraud_probability"])

        st.dataframe(df.tail(10))

    time.sleep(2)