import argparse
import boto3
import time


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--region", required=True)
    p.add_argument("--execution-role-arn", required=True)
    p.add_argument("--model-package-group-name", required=True)
    p.add_argument("--endpoint-name", required=True)
    p.add_argument("--instance-type", default="ml.m5.large")
    p.add_argument(
        "--allow-unapproved",
        action="store_true",
        help="If set, deploy latest model even if not Approved (NOT recommended for production).",
    )
    return p.parse_args()


def latest_by_status(sm, group_name, status):
    resp = sm.list_model_packages(
        ModelPackageGroupName=group_name,
        ModelApprovalStatus=status,
        SortBy="CreationTime",
        SortOrder="Descending",
        MaxResults=1,
    )
    items = resp.get("ModelPackageSummaryList", [])
    return items[0]["ModelPackageArn"] if items else None


def pick_model_package(sm, group_name, allow_unapproved):
    arn = latest_by_status(sm, group_name, "Approved")
    if arn:
        return arn, "Approved"

    if allow_unapproved:
        # Try pending manual approval
        arn = latest_by_status(sm, group_name, "PendingManualApproval")
        if arn:
            return arn, "PendingManualApproval"

    raise RuntimeError(
        f"No APPROVED model package found in: {group_name}. "
        f"{'--allow-unapproved was not set.' if not allow_unapproved else 'No PendingManualApproval found either.'}"
    )


def upsert_endpoint(sm, endpoint_name, model_name, config_name):
    try:
        sm.describe_endpoint(EndpointName=endpoint_name)
        sm.update_endpoint(EndpointName=endpoint_name, EndpointConfigName=config_name)
        return "updated"
    except sm.exceptions.ClientError:
        sm.create_endpoint(EndpointName=endpoint_name, EndpointConfigName=config_name)
        return "created"


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

    pkg_arn, status = pick_model_package(sm, args.model_package_group_name, args.allow_unapproved)
    print(f"✅ Deploying model package ({status}): {pkg_arn}")

    # Use unique names to avoid collisions across runs
    ts = str(int(time.time()))
    model_name = f"{args.endpoint_name}-model-{ts}"
    cfg_name = f"{args.endpoint_name}-cfg-{ts}"

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

    action = upsert_endpoint(sm, args.endpoint_name, model_name, cfg_name)
    print(f"✅ Endpoint {action}: {args.endpoint_name}")

    final = wait_endpoint(sm, args.endpoint_name)
    if final["EndpointStatus"] == "Failed":
        raise RuntimeError(f"Endpoint failed: {final}")

    print(f"✅ Endpoint InService: {args.endpoint_name}")


if __name__ == "__main__":
    main()

