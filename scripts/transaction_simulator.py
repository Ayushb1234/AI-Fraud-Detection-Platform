import requests
import random
import time

API_URL = "http://127.0.0.1:8000/predict"


def generate_transaction():

    transaction = {
        "amount": random.uniform(10, 2000),
        "transaction_velocity": random.randint(1, 10),
        "merchant_risk": random.uniform(0, 1),
        "device_score": random.uniform(0, 1)
    }

    return transaction


def send_transaction():

    transaction = generate_transaction()

    response = requests.post(API_URL, json=transaction)

    result = response.json()

    print("Transaction:", transaction)
    print("Prediction:", result)
    print("-" * 50)


def run_simulation():

    print("Starting fraud simulation...\n")

    while True:

        send_transaction()

        time.sleep(2)


if __name__ == "__main__":

    run_simulation()