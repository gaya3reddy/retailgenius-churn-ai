import pandas as pd
from pathlib import Path

INPUT_PATH = Path("data/processed/processed_churn.csv")
OUTPUT_PATH = Path("data/processed/features.csv")


def main():
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
