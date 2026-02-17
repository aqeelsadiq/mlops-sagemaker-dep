import os
import json
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score


MODEL_PATH = "/opt/ml/processing/model/model.joblib"
TEST_PATH = "/opt/ml/processing/test/test.csv"
OUT_DIR = "/opt/ml/processing/evaluation"
OUT_FILE = os.path.join(OUT_DIR, "evaluation.json")


COMMON_LABEL_COLS = ["label", "Label", "churn", "Churn", "target", "Target", "y", "Y"]


def detect_label_col(df: pd.DataFrame) -> str:
    for c in COMMON_LABEL_COLS:
        if c in df.columns:
            return c
    return df.columns[-1]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(TEST_PATH)
    if df.empty:
        raise ValueError("Test dataset is empty.")

    label_col = detect_label_col(df)
    y = df[label_col]
    X = df.drop(columns=[label_col])

    preds = model.predict(X)
    acc = float(accuracy_score(y, preds))

    auc = None
    try:
        if y.nunique() == 2 and hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[:, 1]
            auc = float(roc_auc_score(y, proba))
    except Exception:
        auc = None

    payload = {
        "metrics": {
            "accuracy": acc,
        }
    }
    if auc is not None:
        payload["metrics"]["roc_auc"] = auc

    with open(OUT_FILE, "w") as f:
        json.dump(payload, f)

    print("✅ Evaluation written:", OUT_FILE)
    print(payload)


if __name__ == "__main__":
    main()
