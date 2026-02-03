"""
Training script for the Customer Churn Prediction model.

This module handles the end-to-end training pipeline, including:
1. Loading processed feature data.
2. Splitting data into training and testing sets.
3. Building a Scikit-Learn Pipeline (Imputation + Scaling + Logistic Regression).
4. Logging parameters, metrics, and the trained model artifact to MLflow.

Example:
    Run this script directly to train the baseline model:

    $ python src/models/train_model.py
"""

import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature


# Always log to the project's local mlruns folder
mlflow.set_tracking_uri(f"file:///{Path('mlruns').resolve().as_posix()}")
mlflow.set_experiment("RetailGenius-Churn")

DATA_PATH = Path("data/processed/features.csv")


def main():
    """
    Main training routine.

    Steps:
        1. Loads features from ``data/processed/features.csv``.
        2. Splits data into train/test sets (80/20 split).
        3. Trains a LogisticRegression pipeline with Median Imputation.
        4. Logs accuracy, F1-score, and ROC-AUC to MLflow.
        5. Saves the model artifact to MLflow for later serving.

    Raises:
        ValueError: If the 'Churn' target column is missing from the input data.
    """
    df = pd.read_csv(DATA_PATH)

    target_col = "Churn"
    if target_col not in df.columns:
        raise ValueError("Target column 'Churn' not found in dataset")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Pipeline: impute missing values then train
    pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=False)),
        ("model", LogisticRegression(max_iter=1000))
    ])

    with mlflow.start_run(run_name="logreg_baseline"):
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)

        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("imputer_strategy", "median")
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", auc)

        # log the whole pipeline (imputer + model)
        signature = infer_signature(X_train, pipeline.predict_proba(X_train))
        mlflow.sklearn.log_model(pipeline, "model",
                                 signature=signature,
                                 input_example=X_train.iloc[:5])

        print(f"Accuracy: {acc:.3f}")
        print(f"F1 Score: {f1:.3f}")
        print(f"AUC: {auc:.3f}")


if __name__ == "__main__":
    main()
