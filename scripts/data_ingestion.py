import pandas as pd
import os

RAW_DATA_PATH = "data/raw/creditcard.csv"
PROCESSED_PATH = "data/processed/transactions.csv"


def load_data():

    print("Loading raw dataset...")

    df = pd.read_csv(RAW_DATA_PATH)

    print(f"Dataset shape: {df.shape}")

    return df


def clean_data(df):

    print("Cleaning dataset...")

    df = df.drop_duplicates()

    df = df.dropna()

    return df


def save_data(df):

    os.makedirs("data/processed", exist_ok=True)

    df.to_csv(PROCESSED_PATH, index=False)

    print(f"Processed dataset saved at {PROCESSED_PATH}")


def run_pipeline():

    df = load_data()

    df = clean_data(df)

    save_data(df)


if __name__ == "__main__":

    run_pipeline()