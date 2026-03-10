import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier
import joblib

DATA_PATH = "data/features/transactions_features.csv"
MODEL_PATH = "models/fraud_model.pkl"


def load_data():

    print("Loading feature dataset...")

    df = pd.read_csv(DATA_PATH)

    return df


def split_data(df):

    print("Splitting dataset...")

    X = df.drop("Class", axis=1)
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):

    print("Training XGBoost model...")

    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=50,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    return model


def evaluate_model(model, X_test, y_test):

    print("Evaluating model...")

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:,1]

    print(classification_report(y_test, preds))

    auc = roc_auc_score(y_test, probs)

    print("ROC-AUC:", auc)


def save_model(model):

    os.makedirs("models", exist_ok=True)

    joblib.dump(model, MODEL_PATH)

    print("Model saved:", MODEL_PATH)


def run_training():

    df = load_data()

    X_train, X_test, y_train, y_test = split_data(df)

    model = train_model(X_train, y_train)

    evaluate_model(model, X_test, y_test)

    save_model(model)


if __name__ == "__main__":

    run_training()