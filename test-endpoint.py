# import json
# import pandas as pd
# import boto3

# REGION = "us-east-1"              # <-- change if needed
# ENDPOINT = "nasir-churn-endpoint" # <-- your endpoint name
# LABEL_COL = "Churn"

# df = pd.read_csv("data/customer_churn_processed.csv")

# # pick one row and drop label
# row = df.drop(columns=[LABEL_COL]).iloc[0].to_dict()

# payload = {"instances": [row]}

# runtime = boto3.client("sagemaker-runtime", region_name=REGION)

# resp = runtime.invoke_endpoint(
#     EndpointName=ENDPOINT,
#     ContentType="application/json",
#     Accept="application/json",
#     Body=json.dumps(payload).encode("utf-8"),
# )

# print(resp["Body"].read().decode("utf-8"))


import argparse
import json
import boto3

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--endpoint-name", required=True)
    p.add_argument("--region", default="us-east-1")
    args = p.parse_args()

    runtime = boto3.client("sagemaker-runtime", region_name=args.region)

    # ✅ Put one row here (features only, NO label column)
    payload = {
        "instances": [
            {
                "Age": 35,
                "Tenure": 12,
                "Usage Frequency": 20,
                "Support Calls": 1,
                "Payment Delay": 0,
                "Total Spend": 500.0,
                "Last Interaction": 3,
                "Gender_Male": 1,
                "Subscription Type_Premium": 1
            }
        ]
    }

    resp = runtime.invoke_endpoint(
        EndpointName=args.endpoint_name,
        ContentType="application/json",
        Accept="application/json",
        Body=json.dumps(payload),
    )

    out = resp["Body"].read().decode("utf-8")
    print("Raw response:", out)

if __name__ == "__main__":
    main()
