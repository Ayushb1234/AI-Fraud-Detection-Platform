import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier
import joblib
import mlflow
import mlflow.xgboost

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("fraud_detection")

DATA_PATH = "data/features/transactions_features.csv"
MODEL_PATH = "models/fraud_model.pkl"


def load_data():
    print("Loading feature dataset...")
    return pd.read_csv(DATA_PATH)


def split_data(df):
    print("Splitting dataset...")

    X = df.drop("Class", axis=1)
    y = df["Class"]

    return train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )


def run_training():

    df = load_data()

    X_train, X_test, y_train, y_test = split_data(df)

    params = {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.1
    }

    with mlflow.start_run():

        print("Training XGBoost model...")

        model = XGBClassifier(
            **params,
            scale_pos_weight=50,
            n_jobs=-1
        )

        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, probs)

        print(classification_report(y_test, preds))
        print("ROC-AUC:", auc)

        mlflow.log_params(params)
        mlflow.log_metric("roc_auc", auc)

        mlflow.xgboost.log_model(model, name="fraud_model")

        os.makedirs("models", exist_ok=True)
        joblib.dump(model, MODEL_PATH)

        print("Model saved:", MODEL_PATH)


if __name__ == "__main__":
    run_training()