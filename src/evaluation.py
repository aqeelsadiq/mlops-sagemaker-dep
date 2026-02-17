import os
import json
import tarfile
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score


MODEL_INPUT_DIR = "/opt/ml/processing/model"
EXTRACT_DIR = "/opt/ml/processing/model_extracted"

TEST_PATH = "/opt/ml/processing/test/test.csv"
OUT_DIR = "/opt/ml/processing/evaluation"
OUT_FILE = os.path.join(OUT_DIR, "evaluation.json")

COMMON_LABEL_COLS = ["label", "Label", "churn", "Churn", "target", "Target", "y", "Y"]


def ls_tree(root: str, max_files: int = 200):
    """Print directory tree for debugging."""
    print(f"\n📁 Listing: {root}")
    if not os.path.exists(root):
        print("  (does not exist)")
        return

    count = 0
    for dirpath, dirnames, filenames in os.walk(root):
        rel = os.path.relpath(dirpath, root)
        indent = "  " * (0 if rel == "." else rel.count(os.sep) + 1)
        print(f"{indent}{os.path.basename(dirpath)}/")
        for f in filenames:
            print(f"{indent}  - {f}")
            count += 1
            if count >= max_files:
                print(f"{indent}  ... (truncated)")
                return


def detect_label_col(df: pd.DataFrame) -> str:
    for c in COMMON_LABEL_COLS:
        if c in df.columns:
            return c
    return df.columns[-1]


def find_model_tar() -> str:
    """Find model tarball from the ProcessingInput mount."""
    if not os.path.exists(MODEL_INPUT_DIR):
        raise FileNotFoundError(f"{MODEL_INPUT_DIR} not found")

    # Prefer the standard name
    candidate = os.path.join(MODEL_INPUT_DIR, "model.tar.gz")
    if os.path.exists(candidate):
        return candidate

    # Otherwise pick any tar.gz file
    tars = [os.path.join(MODEL_INPUT_DIR, f) for f in os.listdir(MODEL_INPUT_DIR) if f.endswith(".tar.gz")]
    if not tars:
        raise FileNotFoundError(
            f"No .tar.gz model artifact found in {MODEL_INPUT_DIR}. "
            f"Contents: {os.listdir(MODEL_INPUT_DIR)}"
        )
    # pick first
    return tars[0]


def extract_tar(tar_path: str):
    os.makedirs(EXTRACT_DIR, exist_ok=True)
    print(f"\n📦 Extracting: {tar_path} -> {EXTRACT_DIR}")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(EXTRACT_DIR)


def find_model_file() -> str:
    """Find model.joblib anywhere under EXTRACT_DIR."""
    for dirpath, _, filenames in os.walk(EXTRACT_DIR):
        for f in filenames:
            if f == "model.joblib":
                return os.path.join(dirpath, f)
    raise FileNotFoundError("model.joblib not found after extracting model tarball")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("✅ Starting evaluation")
    print("MODEL_INPUT_DIR:", MODEL_INPUT_DIR)
    print("TEST_PATH:", TEST_PATH)

    # Debug: show what's mounted
    ls_tree(MODEL_INPUT_DIR)
    ls_tree(os.path.dirname(TEST_PATH))

    # Locate + extract model tar
    tar_path = find_model_tar()
    extract_tar(tar_path)

    # Debug: show extracted contents
    ls_tree(EXTRACT_DIR)

    # Load model
    model_path = find_model_file()
    print("\n✅ Loading model:", model_path)
    model = joblib.load(model_path)

    # Load test data
    df = pd.read_csv(TEST_PATH)
    if df.empty:
        raise ValueError("Test dataset is empty.")

    label_col = detect_label_col(df)
    print("✅ Detected label column:", label_col)

    y = df[label_col]
    X = df.drop(columns=[label_col])

    preds = model.predict(X)
    acc = float(accuracy_score(y, preds))

    auc = None
    try:
        if y.nunique() == 2 and hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[:, 1]
            auc = float(roc_auc_score(y, proba))
    except Exception as e:
        print("⚠️ ROC AUC skipped due to:", repr(e))
        auc = None

    payload = {"metrics": {"accuracy": acc}}
    if auc is not None:
        payload["metrics"]["roc_auc"] = auc

    with open(OUT_FILE, "w") as f:
        json.dump(payload, f)

    print("\n✅ Evaluation written:", OUT_FILE)
    print(payload)


if __name__ == "__main__":
    main()
