# import argparse
# import os
# from typing import Optional, List

# import joblib
# import pandas as pd

# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, roc_auc_score
# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.pipeline import Pipeline
# from sklearn.linear_model import LogisticRegression


# COMMON_LABEL_COLS = ["label", "Label", "churn", "Churn", "target", "Target", "y", "Y"]


# def parse_args():
#     p = argparse.ArgumentParser()
#     p.add_argument(
#         "--train",
#         type=str,
#         default="/opt/ml/input/data/training/customer_churn_processed.csv",
#         help="Path to training CSV inside the training container.",
#     )
#     p.add_argument(
#         "--label-col",
#         type=str,
#         default="",
#         help="Label column name. If empty, the script will auto-detect.",
#     )
#     p.add_argument("--model-dir", type=str, default="/opt/ml/model")
#     return p.parse_args()


# def detect_label_col(df: pd.DataFrame, preferred: str) -> str:
#     if preferred and preferred in df.columns:
#         return preferred

#     for c in COMMON_LABEL_COLS:
#         if c in df.columns:
#             return c

#     # fallback: last column
#     return df.columns[-1]


# def main():
#     args = parse_args()

#     df = pd.read_csv(args.train)
#     if df.empty:
#         raise ValueError(f"Training CSV is empty: {args.train}")

#     label_col = detect_label_col(df, args.label_col.strip())
#     if label_col not in df.columns:
#         raise ValueError(f"Label column '{label_col}' not found. Columns: {list(df.columns)}")

#     y = df[label_col]
#     X = df.drop(columns=[label_col])

#     # Identify categorical vs numeric
#     categorical_cols: List[str] = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
#     numeric_cols: List[str] = [c for c in X.columns if c not in categorical_cols]

#     preprocessor = ColumnTransformer(
#         transformers=[
#             ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
#             ("num", "passthrough", numeric_cols),
#         ],
#         remainder="drop",
#     )

#     clf = LogisticRegression(max_iter=2000)

#     model = Pipeline(
#         steps=[
#             ("preprocess", preprocessor),
#             ("model", clf),
#         ]
#     )

#     # split
#     X_train, X_val, y_train, y_val = train_test_split(
#         X,
#         y,
#         test_size=0.2,
#         random_state=42,
#         stratify=y if y.nunique() > 1 else None,
#     )

#     model.fit(X_train, y_train)
#     preds = model.predict(X_val)

#     acc = accuracy_score(y_val, preds)

#     # ROC AUC only if binary and we can get probabilities
#     auc: Optional[float] = None
#     try:
#         if y.nunique() == 2:
#             proba = model.predict_proba(X_val)[:, 1]
#             auc = roc_auc_score(y_val, proba)
#     except Exception:
#         auc = None

#     os.makedirs(args.model_dir, exist_ok=True)
#     joblib.dump(model, os.path.join(args.model_dir, "model.joblib"))

#     # metrics
#     with open(os.path.join(args.model_dir, "metrics.txt"), "w") as f:
#         f.write(f"label_col={label_col}\n")
#         f.write(f"accuracy={acc}\n")
#         if auc is not None:
#             f.write(f"roc_auc={auc}\n")

#     print(f"✅ Trained model saved to {args.model_dir}/model.joblib")
#     print(f"✅ label_col={label_col}")
#     print(f"✅ accuracy={acc}")
#     if auc is not None:
#         print(f"✅ roc_auc={auc}")


# if __name__ == "__main__":
#     main()











import argparse
import os
from typing import List, Optional

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score


COMMON_LABEL_COLS = ["label", "Label", "churn", "Churn", "target", "Target", "y", "Y"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--label-col", type=str, default="")
    # SageMaker training channel: /opt/ml/input/data/<channel_name>/
    p.add_argument("--train", type=str, default="/opt/ml/input/data/train/train.csv")
    p.add_argument("--model-dir", type=str, default="/opt/ml/model")
    return p.parse_args()


def detect_label_col(df: pd.DataFrame, preferred: str) -> str:
    if preferred and preferred in df.columns:
        return preferred
    for c in COMMON_LABEL_COLS:
        if c in df.columns:
            return c
    return df.columns[-1]


def main():
    args = parse_args()

    df = pd.read_csv(args.train)
    if df.empty:
        raise ValueError(f"Training data is empty: {args.train}")

    label_col = detect_label_col(df, args.label_col.strip())
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in dataset.")

    y = df[label_col]
    X = df.drop(columns=[label_col])

    cat_cols: List[str] = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    num_cols: List[str] = [c for c in X.columns if c not in cat_cols]

    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols),
        ]
    )

    clf = LogisticRegression(max_iter=3000)

    model = Pipeline(steps=[("preprocess", pre), ("model", clf)])

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if y.nunique() > 1 else None
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_val)
    acc = accuracy_score(y_val, preds)

    auc: Optional[float] = None
    try:
        if y.nunique() == 2:
            proba = model.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, proba)
    except Exception:
        auc = None

    os.makedirs(args.model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(args.model_dir, "model.joblib"))

    # Save training metrics too (optional)
    with open(os.path.join(args.model_dir, "train_metrics.txt"), "w") as f:
        f.write(f"label_col={label_col}\n")
        f.write(f"accuracy={acc}\n")
        if auc is not None:
            f.write(f"roc_auc={auc}\n")

    print("✅ Training complete")
    print("label_col:", label_col)
    print("accuracy:", acc)
    if auc is not None:
        print("roc_auc:", auc)


if __name__ == "__main__":
    main()
