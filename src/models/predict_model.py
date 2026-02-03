"""
Batch Inference Script for RetailGenius.

This script loads a trained model from MLflow and generates predictions 
for a given input dataset. It is designed to be run from the command line.

Example:
    .. code-block:: bash

        python src/models/predict_model.py \\
            --model-uri runs:/<run_id>/model \\
            --input data/processed/features.csv \\
            --output data/processed/predictions.csv
"""
import argparse
from pathlib import Path
import pandas as pd
import mlflow


def parse_args():
    """
    Parses command-line arguments for model inference.

    Returns:
        argparse.Namespace: The parsed arguments containing:
            - model_uri (str): The MLflow URI of the model.
            - input (str): Path to input CSV.
            - output (str): Path to save output predictions.
    """
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
    """
    Main execution entry point.

    1. Parses arguments.
    2. Loads the input data CSV.
    3. Loads the specified model from MLflow.
    4. Generates predictions (classes).
    5. Saves the results to a new CSV file.
    """
    args = parse_args()

    X = pd.read_csv(args.input)

    # If churn label is accidentally present, drop it
    # if "churn" in X.columns:
    #     X = X.drop(columns=["churn"])
    
    for col in ["Churn", "churn"]:
        if col in X.columns:
            X = X.drop(columns=[col])

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
