import pandas as pd
import os

INPUT_PATH = "data/processed/transactions.csv"
OUTPUT_PATH = "data/features/transactions_features.csv"


def load_data():
    print("Loading processed data...")
    df = pd.read_csv(INPUT_PATH)
    return df


def create_time_features(df):

    print("Creating time features...")

    # convert seconds to hour
    df["transaction_hour"] = (df["Time"] // 3600) % 24

    return df


def create_amount_features(df):

    print("Creating amount features...")

    df["high_value_transaction"] = (df["Amount"] > 200).astype(int)

    df["amount_log"] = df["Amount"].apply(lambda x: 0 if x == 0 else x)

    return df


def create_velocity_features(df):

    print("Creating velocity features...")

    df["transaction_velocity"] = df["Amount"].rolling(5).mean()

    df["transaction_velocity"].fillna(0, inplace=True)

    return df


def save_features(df):

    os.makedirs("data/features", exist_ok=True)

    df.to_csv(OUTPUT_PATH, index=False)

    print("Feature dataset saved")


def run_feature_pipeline():

    df = load_data()

    df = create_time_features(df)

    df = create_amount_features(df)

    df = create_velocity_features(df)

    save_features(df)


if __name__ == "__main__":
    run_feature_pipeline()