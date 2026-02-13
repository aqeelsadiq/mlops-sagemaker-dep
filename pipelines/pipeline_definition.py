import argparse
import boto3
import sagemaker

from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.parameters import ParameterString
from sagemaker.workflow.steps import TrainingStep
from sagemaker.workflow.model_step import RegisterModel
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.inputs import TrainingInput


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--region", required=True)
    p.add_argument("--role-arn", required=True)
    p.add_argument("--pipeline-name", default="nasir-churn-pipeline")
    p.add_argument("--model-package-group-name", default="nasir-churn-model-group")
    p.add_argument("--default-train-data-s3-uri", required=True)
    p.add_argument("--label-col", default="")  # may be empty -> auto-detect in training.py
    return p.parse_args()


def main():
    args = parse_args()

    boto_sess = boto3.Session(region_name=args.region)
    sm_session = sagemaker.session.Session(boto_session=boto_sess)

    train_data_s3 = ParameterString(
        name="TrainDataS3Uri",
        default_value=args.default_train_data_s3_uri,
    )

    label_col = ParameterString(
        name="LabelCol",
        default_value=args.label_col,
    )

    estimator = SKLearn(
        entry_point="training.py",
        source_dir="src",
        role=args.role_arn,
        framework_version="1.2-1",
        instance_type="ml.m5.large",
        instance_count=1,
        sagemaker_session=sm_session,
        hyperparameters={
            "label-col": label_col,  # passed into script; can be empty
        },
    )

    train_step = TrainingStep(
        name="TrainChurnModel",
        estimator=estimator,
        inputs={
            "training": TrainingInput(
                s3_data=train_data_s3,
                content_type="text/csv",
            )
        },
    )

    register_step = RegisterModel(
        name="RegisterChurnModel",
        estimator=estimator,
        model_data=train_step.properties.ModelArtifacts.S3ModelArtifacts,
        model_package_group_name=args.model_package_group_name,
        approval_status="PendingManualApproval",
        content_types=["application/json", "text/csv"],
        response_types=["application/json"],
        inference_instances=["ml.m5.large", "ml.m5.xlarge"],
        transform_instances=["ml.m5.large"],
        entry_point="inference.py",
        source_dir="src",
    )

    pipeline = Pipeline(
        name=args.pipeline_name,
        parameters=[train_data_s3, label_col],
        steps=[train_step, register_step],
        sagemaker_session=sm_session,
    )

    pipeline.upsert(role_arn=args.role_arn)
    print(f"✅ Upserted pipeline: {args.pipeline_name}")
    print(f"✅ Model Package Group: {args.model_package_group_name}")
    print(f"✅ Default TrainDataS3Uri: {args.default_train_data_s3_uri}")


if __name__ == "__main__":
    main()
