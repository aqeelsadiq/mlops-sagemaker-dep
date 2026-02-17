import argparse
import os
import json
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--validation-data", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()

    model = joblib.load(args.model_path)

    df = pd.read_csv(args.validation_data)

    y = df.iloc[:, -1]
    X = df.iloc[:, :-1]

    predictions = model.predict(X)
    accuracy = accuracy_score(y, predictions)

    os.makedirs(args.output_dir, exist_ok=True)

    metrics = {"accuracy": accuracy}

    with open(os.path.join(args.output_dir, "evaluation.json"), "w") as f:
        json.dump(metrics, f)

    print(f"✅ Evaluation accuracy: {accuracy}")


if __name__ == "__main__":
    main()
