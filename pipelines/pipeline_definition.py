import argparse
import boto3
import sagemaker

from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.parameters import ParameterString
from sagemaker.workflow.steps import TrainingStep
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.inputs import TrainingInput

from sagemaker.sklearn.model import SKLearnModel
from sagemaker.workflow.model_step import ModelStep


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--region", required=True)
    p.add_argument("--role-arn", required=True)
    p.add_argument("--pipeline-name", default="nasir-churn-pipeline")
    p.add_argument("--model-package-group-name", default="nasir-churn-model-group")
    p.add_argument("--default-train-data-s3-uri", required=True)
    p.add_argument("--label-col", default="")
    return p.parse_args()


def main():
    args = parse_args()

    boto_sess = boto3.Session(region_name=args.region)
    pipeline_session = PipelineSession(boto_session=boto_sess)

    # Pipeline parameters (overridable at execution time)
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
        sagemaker_session=pipeline_session,
        hyperparameters={
            "label-col": label_col,  # can be empty; training.py auto-detects
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

    # Create a Model object pointing to the training output artifacts
    model = SKLearnModel(
        model_data=train_step.properties.ModelArtifacts.S3ModelArtifacts,
        role=args.role_arn,
        entry_point="inference.py",
        source_dir="src",
        framework_version="1.2-1",
        sagemaker_session=pipeline_session,
    )

    # Register the model to Model Registry using a ModelStep
    register_args = model.register(
        content_types=["application/json", "text/csv"],
        response_types=["application/json"],
        inference_instances=["ml.m5.large", "ml.m5.xlarge"],
        transform_instances=["ml.m5.large"],
        model_package_group_name=args.model_package_group_name,
        approval_status="PendingManualApproval",
    )

    register_step = ModelStep(
        name="RegisterChurnModel",
        step_args=register_args,
    )

    pipeline = Pipeline(
        name=args.pipeline_name,
        parameters=[train_data_s3, label_col],
        steps=[train_step, register_step],
        sagemaker_session=pipeline_session,
    )

    pipeline.upsert(role_arn=args.role_arn)

    print(f"✅ Upserted pipeline: {args.pipeline_name}")
    print(f"✅ Model Package Group: {args.model_package_group_name}")
    print(f"✅ Default TrainDataS3Uri: {args.default_train_data_s3_uri}")


if __name__ == "__main__":
    main()
