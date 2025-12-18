import argparse
from pathlib import Path
import pandas as pd
import mlflow


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-uri", type=str, required=True,
                   help="MLflow model URI, e.g. runs:/<run_id>/model")
    p.add_argument("--input", type=str, required=True,
                   help="Path to input CSV (features, no target column)")
    p.add_argument("--output",
                   type=str,
                   default="data/processed/predictions.csv",
                   help="Path to output CSV")
    return p.parse_args()


def main():
    args = parse_args()

    X = pd.read_csv(args.input)

    # If churn label is accidentally present, drop it
    if "churn" in X.columns:
        X = X.drop(columns=["churn"])

    model = mlflow.pyfunc.load_model(args.model_uri)

    # For classifiers logged with sklearn, predict() returns class label
    y_pred = model.predict(X)

    out = X.copy()
    out["prediction"] = y_pred

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)

    print(f"Saved predictions to: {args.output}")


''' python src/models/predict_model.py \
    --model-uri runs:/1e2c6d7fbeab41f4a9787d45d1991601/model \
    --input data/processed/features.csv \
    --output data/processed/predictions.csv'''
if __name__ == "__main__":
    main()
