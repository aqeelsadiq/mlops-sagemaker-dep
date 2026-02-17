import json
import pandas as pd
import boto3

REGION = "us-east-1"              # <-- change if needed
ENDPOINT = "nasir-churn-endpoint" # <-- your endpoint name
LABEL_COL = "Churn"

df = pd.read_csv("data/customer_churn_processed.csv")

# pick one row and drop label
row = df.drop(columns=[LABEL_COL]).iloc[0].to_dict()

payload = {"instances": [row]}

runtime = boto3.client("sagemaker-runtime", region_name=REGION)

resp = runtime.invoke_endpoint(
    EndpointName=ENDPOINT,
    ContentType="application/json",
    Accept="application/json",
    Body=json.dumps(payload).encode("utf-8"),
)

print(resp["Body"].read().decode("utf-8"))


