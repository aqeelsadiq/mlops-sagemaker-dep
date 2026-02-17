import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data", type=str, required=True)
    parser.add_argument("--train-output", type=str, required=True)
    parser.add_argument("--validation-output", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()

    df = pd.read_csv(args.input_data)

    if df.empty:
        raise ValueError("Input dataset is empty")

    train, validation = train_test_split(
        df,
        test_size=0.2,
        random_state=42
    )

    os.makedirs(args.train_output, exist_ok=True)
    os.makedirs(args.validation_output, exist_ok=True)

    train.to_csv(os.path.join(args.train_output, "train.csv"), index=False)
    validation.to_csv(os.path.join(args.validation_output, "validation.csv"), index=False)

    print("✅ Preprocessing completed")


if __name__ == "__main__":
    main()
