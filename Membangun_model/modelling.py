import pandas as pd
import numpy as np
import mlflow
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report


DATA_DIR = "diabetes_preprocessing"
MLFLOW_TRACKING_URI = "file:./mlruns"


def load_data(data_dir: str = DATA_DIR):
    train_df = pd.read_csv(f"{data_dir}/train.csv")
    test_df = pd.read_csv(f"{data_dir}/test.csv")

    X_train = train_df.drop("Outcome", axis=1)
    y_train = train_df["Outcome"]
    X_test = test_df.drop("Outcome", axis=1)
    y_test = test_df["Outcome"]

    print(f"[INFO] Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, y_train, X_test, y_test


def train_model(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)
    print("[INFO] Model training completed")
    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
    }
    print("\n[INFO] Evaluation Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Non-Diabetic', 'Diabetic'])}")
    return metrics, y_pred


def main():
    os.makedirs("model", exist_ok=True)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("Diabetes_Classification_Basic")

    X_train, y_train, X_test, y_test = load_data()

    with mlflow.start_run() as run:
        mlflow.sklearn.autolog()

        model = train_model(X_train, y_train)
        metrics, y_pred = evaluate_model(model, X_test, y_test)

        joblib.dump(model, "model/model.pkl")
        print(f"\n[INFO] Model saved to model/model.pkl")
        print(f"[INFO] MLflow Run ID: {run.info.run_id}")
        print(f"[INFO] Track at: {MLFLOW_TRACKING_URI}")


if __name__ == "__main__":
    main()
