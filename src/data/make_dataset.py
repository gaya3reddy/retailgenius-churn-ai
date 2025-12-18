import pandas as pd
from pathlib import Path

RAW_DATA_PATH = Path("src/data/raw/E_Commerce_Dataset.xlsx")
PROCESSED_DATA_PATH = Path("data/processed/processed_churn.csv")


def main():
    df = pd.read_excel(RAW_DATA_PATH, sheet_name="E Comm")
    # Basic cleaning
    df = df.drop_duplicates()
    # Save processed data
    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DATA_PATH, index=False)

    print("Processed dataset saved.")


if __name__ == "__main__":
    main()
