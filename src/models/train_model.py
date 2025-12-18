import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

import mlflow
import mlflow.sklearn

DATA_PATH = Path("data/processed/features.csv")

def main():
    df = pd.read_csv(DATA_PATH)

    target_col = "Churn"
    if target_col not in df.columns:
        raise ValueError("Target column 'churn' not found in dataset")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Pipeline: impute missing values then train
    pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("model", LogisticRegression(max_iter=1000))
    ])

    with mlflow.start_run():
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
        mlflow.sklearn.log_model(pipeline, "model")

        print(f"Accuracy: {acc:.3f}")
        print(f"F1 Score: {f1:.3f}")
        print(f"AUC: {auc:.3f}")

if __name__ == "__main__":
    main()
 