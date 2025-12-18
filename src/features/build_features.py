"""
Feature Engineering Script for RetailGenius.

This module is responsible for transforming the cleaned data into machine-learning ready features.
It performs operations such as:
1. Binning continuous variables (e.g., tenure).
2. One-hot encoding categorical variables.
3. Saving the final feature set for model training.

Example:
    Run this script from the project root:

    $ python src/features/build_features.py
"""
import pandas as pd
from pathlib import Path

INPUT_PATH = Path("data/processed/processed_churn.csv")
OUTPUT_PATH = Path("data/processed/features.csv")


def main():
    """
    Main feature engineering pipeline.

    Steps:
        1. Loads the cleaned dataset from ``data/processed/processed_churn.csv``.
        2. Creates a new feature ``tenure_group`` by binning the ``tenure`` column
           into cohorts (0-1y, 1-2y, 2-4y, 4y+).
        3. Applies One-Hot Encoding (pd.get_dummies) to all categorical variables,
           dropping the first category to avoid multicollinearity.
        4. Saves the resulting dataframe to ``data/processed/features.csv``.

    Returns:
        None: The output is saved to disk as a CSV file.
    """
    df = pd.read_csv(INPUT_PATH)

    # Example feature engineering
    if 'tenure' in df.columns:
        df['tenure_group'] = pd.cut(
            df['tenure'],
            bins=[0, 12, 24, 48, 100],
            labels=['0-1y', '1-2y', '2-4y', '4y+']
        )

    # Convert categorical features
    df = pd.get_dummies(df, drop_first=True)

    df.to_csv(OUTPUT_PATH, index=False)
    print("Feature dataset saved.")


if __name__ == "__main__":
    main()
