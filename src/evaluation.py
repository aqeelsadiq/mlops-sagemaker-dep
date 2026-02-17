import os
import json
import tarfile
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

MODEL_TAR = "/opt/ml/processing/model/model.tar.gz"
MODEL_DIR = "/opt/ml/processing/model_extracted"
MODEL_FILE = os.path.join(MODEL_DIR, "model.joblib")

TEST_PATH = "/opt/ml/processing/test/test.csv"
OUT_DIR = "/opt/ml/processing/evaluation"
OUT_FILE = os.path.join(OUT_DIR, "evaluation.json")

COMMON_LABEL_COLS = ["label", "Label", "churn", "Churn", "target", "Target", "y", "Y"]


def detect_label_col(df: pd.DataFrame) -> str:
    for c in COMMON_LABEL_COLS:
        if c in df.columns:
            return c
    return df.columns[-1]


def extract_model():
    if not os.path.exists(MODEL_TAR):
        raise FileNotFoundError(
            f"Expected model artifact at {MODEL_TAR}. "
            "Training output is usually model.tar.gz"
        )

    os.makedirs(MODEL_DIR, exist_ok=True)
    with tarfile.open(MODEL_TAR, "r:gz") as tar:
        tar.extractall(MODEL_DIR)

    if not os.path.exists(MODEL_FILE):
        # helpful debug
        files = []
        for root, _, fnames in os.walk(MODEL_DIR):
            for f in fnames:
                files.append(os.path.join(root, f))
        raise FileNotFoundError(
            f"model.joblib not found after extracting. Looked for {MODEL_FILE}. "
            f"Extracted files: {files}"
        )


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    extract_model()
    model = joblib.load(MODEL_FILE)

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

    payload = {"metrics": {"accuracy": acc}}
    if auc is not None:
        payload["metrics"]["roc_auc"] = auc

    with open(OUT_FILE, "w") as f:
        json.dump(payload, f)

    print("✅ Evaluation written:", OUT_FILE)
    print(payload)


if __name__ == "__main__":
    main()
