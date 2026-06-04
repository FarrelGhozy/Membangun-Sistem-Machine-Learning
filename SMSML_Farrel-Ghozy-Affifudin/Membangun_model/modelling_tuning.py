import pandas as pd
import numpy as np
import mlflow
import joblib
import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
    roc_curve
)
import argparse


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


def create_artifacts(model, X_test, y_test, y_pred, output_dir: str = "artifacts"):
    os.makedirs(output_dir, exist_ok=True)

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Non-Diabetic', 'Diabetic'],
                yticklabels=['Non-Diabetic', 'Diabetic'])
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    cm_path = f"{output_dir}/confusion_matrix.png"
    plt.savefig(cm_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"[ARTIFACT] Saved: {cm_path}")

    feature_names = X_test.columns.tolist()
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(feature_names)))
    plt.barh(range(len(feature_names)), importances[indices], color=colors)
    plt.yticks(range(len(feature_names)), [feature_names[i] for i in indices])
    plt.xlabel('Feature Importance', fontsize=12)
    plt.title('Feature Importance (Random Forest)', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    fi_path = f"{output_dir}/feature_importance.png"
    plt.savefig(fi_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"[ARTIFACT] Saved: {fi_path}")

    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = roc_auc_score(y_test, y_prob)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='#2ecc71', lw=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    roc_path = f"{output_dir}/roc_curve.png"
    plt.savefig(roc_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"[ARTIFACT] Saved: {roc_path}")

    # Distribution of predictions
    plt.figure(figsize=(8, 5))
    pred_series = pd.Series(y_pred).map({0: 'Non-Diabetic', 1: 'Diabetic'})
    pred_counts = pred_series.value_counts()
    plt.bar(pred_counts.index, pred_counts.values, color=['#2ecc71', '#e74c3c'], edgecolor='black')
    plt.title('Prediction Distribution', fontsize=14, fontweight='bold')
    plt.ylabel('Count')
    for i, v in enumerate(pred_counts.values):
        plt.text(i, v + 1, str(v), ha='center', fontweight='bold')
    plt.tight_layout()
    pred_path = f"{output_dir}/prediction_distribution.png"
    plt.savefig(pred_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"[ARTIFACT] Saved: {pred_path}")

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1_score": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc),
    }
    metrics_path = f"{output_dir}/metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"[ARTIFACT] Saved: {metrics_path}")

    return metrics, cm_path, fi_path, roc_path, pred_path


def train_with_tuning(X_train, y_train, n_iter: int = 20):
    param_dist = {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [5, 10, 15, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None],
    }

    base_model = RandomForestClassifier(random_state=42, n_jobs=-1)
    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=5,
        scoring='f1',
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    random_search.fit(X_train, y_train)

    print(f"\n[INFO] Best Parameters: {random_search.best_params_}")
    print(f"[INFO] Best CV Score (F1): {random_search.best_score_:.4f}")

    return random_search.best_estimator_, random_search.best_params_, random_search.best_score_


def main(use_dagshub: bool = False):
    if use_dagshub:
        import dagshub
        dagshub.init(repo_owner='FarrelGhozy', repo_name='Membangun-Sistem-Machine-Learning', mlflow=True)
        print("[INFO] Using DagsHub remote tracking")
    else:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        print(f"[INFO] Using local tracking: {MLFLOW_TRACKING_URI}")

    os.makedirs("model", exist_ok=True)
    os.makedirs("artifacts", exist_ok=True)

    X_train, y_train, X_test, y_test = load_data()

    mlflow.set_experiment("Diabetes_Classification_Tuning")

    with mlflow.start_run() as run:
        print(f"[INFO] MLflow Run ID: {run.info.run_id}")

        model, best_params, best_cv_score = train_with_tuning(X_train, y_train)

        y_pred = model.predict(X_test)

        metrics, cm_path, fi_path, roc_path, pred_path = create_artifacts(
            model, X_test, y_test, y_pred, output_dir="artifacts"
        )

        mlflow.log_params(best_params)
        mlflow.log_metrics(metrics)
        mlflow.log_metric("best_cv_f1", best_cv_score)

        mlflow.log_artifact(cm_path, artifact_path="plots")
        mlflow.log_artifact(fi_path, artifact_path="plots")
        mlflow.log_artifact(roc_path, artifact_path="plots")
        mlflow.log_artifact(pred_path, artifact_path="plots")
        mlflow.log_artifact("artifacts/metrics.json", artifact_path="metrics")

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name="Diabetes_RandomForest_Tuned"
        )

        joblib.dump(model, "model/model_tuned.pkl")
        print(f"\n[INFO] Model saved to model/model_tuned.pkl")

        print(f"\n[INFO] Final Metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

        print(f"[INFO] MLflow Run ID: {run.info.run_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dagshub', action='store_true', help='Use DagsHub remote tracking')
    args = parser.parse_args()
    main(use_dagshub=args.dagshub)
