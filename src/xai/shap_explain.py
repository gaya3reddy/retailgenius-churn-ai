"""
SHAP Explanation Script for RetailGenius.

This module generates comprehensive Explainable AI (XAI) reports using SHAP (SHapley Additive exPlanations).
It produces four key visualizations to help stakeholders understand model behavior:
1. **Summary Bar Plot:** Global feature importance ranking.
2. **Beeswarm Plot:** Detailed view of feature impact and directionality (e.g., "Does high tenure increase or decrease churn?").
3. **Waterfall Plot:** Local explanation for a single customer (e.g., "Why did Customer #5 churn?").
4. **Dependence Plot:** Analysis of the interaction between the most important feature and the target.

Example:
    Run this script from the command line after training a model:

    .. code-block:: bash

        python src/xai/shap_explain.py \\
            --model-uri "models:/RetailGenius_Churn_Model/1" \\
            --data "data/processed/features.csv" \\
            --sample-size 500
"""


import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import mlflow
import shap
import matplotlib.pyplot as plt


def main():
    """
    Main XAI routine.

    Steps:
        1. Loads the feature dataset and drops the target column if present.
        2. Samples the data (default 500 rows) to ensure reasonable computation time.
        3. Loads the full pipeline (Preprocessing + XGBoost) from MLflow.
        4. Splits the pipeline to isolate the XGBoost model and the preprocessor.
        5. Computes SHAP values using the TreeExplainer.
        6. Generates and saves 4 types of plots to the reports directory.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-uri", required=True,
                        help="MLflow model URI, e.g. runs:/<run_id>/model")
    parser.add_argument("--data", default="data/processed/features.csv")
    parser.add_argument("--outdir", default="reports/xai_outputs")
    parser.add_argument("--sample-size", type=int, default=500)
    parser.add_argument("--row-index", type=int, default=0)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load data
    df = pd.read_csv(args.data)
    if "Churn" in df.columns:
        df = df.drop(columns=["Churn"])

    # Sample for speed
    df_sample = df.sample(
        min(len(df), args.sample_size), random_state=42
    )

    # Load pipeline from MLflow
    pipeline = mlflow.sklearn.load_model(args.model_uri)

    # Split pipeline
    preprocessor = pipeline[:-1]
    model = pipeline[-1]

    X_sample = preprocessor.transform(df_sample)

    feature_names = df.columns.tolist()

    # SHAP TreeExplainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # 1️⃣ Global importance (bar)
    shap.summary_plot(
        shap_values,
        X_sample,
        feature_names=feature_names,
        plot_type="bar",
        show=False
    )
    plt.tight_layout()
    plt.savefig(outdir / "shap_summary_bar.png", dpi=200)
    plt.close()

    # 2️⃣ Beeswarm
    shap.summary_plot(
        shap_values,
        X_sample,
        feature_names=feature_names,
        show=False
    )
    plt.tight_layout()
    plt.savefig(outdir / "shap_beeswarm.png", dpi=200)
    plt.close()

    # 3️⃣ Local explanation (waterfall)
    idx = min(args.row_index, X_sample.shape[0] - 1)
    explanation = shap.Explanation(
        values=shap_values[idx],
        base_values=explainer.expected_value,
        data=X_sample[idx],
        feature_names=feature_names
    )

    shap.plots.waterfall(explanation, show=False)
    plt.tight_layout()
    plt.savefig(outdir / f"shap_waterfall_row_{idx}.png", dpi=200)
    plt.close()

    # 4️⃣ Dependence plot (top feature)
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    top_feature = int(np.argmax(mean_abs))

    shap.dependence_plot(
        top_feature,
        shap_values,
        X_sample,
        feature_names=feature_names,
        show=False
    )
    plt.tight_layout()
    plt.savefig(outdir / "shap_dependence_top_feature.png", dpi=200)
    plt.close()

    print(f"SHAP plots saved to {outdir.resolve()}")


if __name__ == "__main__":
    main()
