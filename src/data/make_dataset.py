"""
Data Ingestion Script for RetailGenius.

This module is the starting point of the pipeline. It handles:
1. Reading the raw Excel dump provided by the business.
2. Performing initial cleaning (deduplication).
3. Saving the immutable raw data into a processed CSV format for downstream use.

Example:
    Run this script to refresh the processed dataset:

    $ python src/data/make_dataset.py
"""
import pandas as pd
from pathlib import Path

RAW_DATA_PATH = Path("src/data/raw/E_Commerce_Dataset.xlsx")
PROCESSED_DATA_PATH = Path("data/processed/processed_churn.csv")


def main():
    """
    Main data ingestion routine.

    Steps:
        1. Reads the 'E Comm' sheet from the raw Excel file.
        2. Drops duplicate rows to ensure data integrity.
        3. Creates the ``data/processed/`` directory if it doesn't exist.
        4. Saves the clean dataframe as a CSV file.

    Returns:
        None: The output is saved to disk at ``data/processed/processed_churn.csv``.
    """
    df = pd.read_excel(RAW_DATA_PATH, sheet_name="E Comm")
    # Basic cleaning
    df = df.drop_duplicates()
    # Save processed data
    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DATA_PATH, index=False)

    print("Processed dataset saved.")


if __name__ == "__main__":
    main()
