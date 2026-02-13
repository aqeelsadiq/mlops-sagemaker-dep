import argparse
import boto3
import time


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--region", required=True)
    p.add_argument("--execution-role-arn", required=True)
    p.add_argument("--model-package-group-name", default="nasir-churn-model-group")
    p.add_argument("--endpoint-name", default="nasir-churn-endpoint")
    p.add_argument("--instance-type", default="ml.m5.large")
    return p.parse_args()


def latest_approved(sm, group_name):
    resp = sm.list_model_packages(
        ModelPackageGroupName=group_name,
        ModelApprovalStatus="Approved",
        SortBy="CreationTime",
        SortOrder="Descending",
        MaxResults=1,
    )
    items = resp.get("ModelPackageSummaryList", [])
    if not items:
        raise RuntimeError(f"No APPROVED model package found in: {group_name}")
    return items[0]["ModelPackageArn"]


def wait_endpoint(sm, endpoint_name):
    while True:
        d = sm.describe_endpoint(EndpointName=endpoint_name)
        s = d["EndpointStatus"]
        print(f"Endpoint status: {s}")
        if s in ("InService", "Failed"):
            return d
        time.sleep(30)


def main():
    args = parse_args()
    sm = boto3.client("sagemaker", region_name=args.region)

    pkg_arn = latest_approved(sm, args.model_package_group_name)
    print(f"✅ Deploying approved model package: {pkg_arn}")

    model_name = f"{args.endpoint_name}-model"
    cfg_name = f"{args.endpoint_name}-cfg"

    sm.create_model(
        ModelName=model_name,
        PrimaryContainer={"ModelPackageName": pkg_arn},
        ExecutionRoleArn=args.execution_role_arn,
    )

    sm.create_endpoint_config(
        EndpointConfigName=cfg_name,
        ProductionVariants=[
            {
                "VariantName": "AllTraffic",
                "ModelName": model_name,
                "InstanceType": args.instance_type,
                "InitialInstanceCount": 1,
            }
        ],
    )

    try:
        sm.describe_endpoint(EndpointName=args.endpoint_name)
        print("Endpoint exists → updating")
        sm.update_endpoint(EndpointName=args.endpoint_name, EndpointConfigName=cfg_name)
    except sm.exceptions.ClientError:
        print("Endpoint not found → creating")
        sm.create_endpoint(EndpointName=args.endpoint_name, EndpointConfigName=cfg_name)

    final = wait_endpoint(sm, args.endpoint_name)
    if final["EndpointStatus"] == "Failed":
        raise RuntimeError(f"Endpoint failed: {final}")
    print(f"✅ Endpoint InService: {args.endpoint_name}")


if __name__ == "__main__":
    main()
