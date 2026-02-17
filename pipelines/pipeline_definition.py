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
        instance_type="ml.t3.medium",
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
        inference_instances=["ml.t3.medium", "ml.t3.medium"],
        transform_instances=["ml.t3.medium"],
        model_package_group_name=args.model_package_group_name,
        approval_status="Approved",
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
# import argparse
# import boto3
# import sagemaker

# from sagemaker.workflow.pipeline import Pipeline
# from sagemaker.workflow.steps import ProcessingStep, TrainingStep
# from sagemaker.workflow.parameters import ParameterString
# from sagemaker.workflow.condition_step import ConditionStep
# from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
# from sagemaker.workflow.functions import JsonGet
# from sagemaker.workflow.pipeline_context import PipelineSession
# from sagemaker.workflow.properties import PropertyFile

# from sagemaker.sklearn.processing import SKLearnProcessor
# from sagemaker.sklearn.estimator import SKLearn
# from sagemaker.sklearn.model import SKLearnModel
# from sagemaker.workflow.model_step import ModelStep
# from sagemaker.inputs import TrainingInput


# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--region", required=True)
#     parser.add_argument("--role-arn", required=True)
#     parser.add_argument("--pipeline-name", default="churn-mlops-pipeline")
#     parser.add_argument("--model-package-group", default="churn-model-group")
#     parser.add_argument("--input-data-s3", required=True)
#     return parser.parse_args()


# def main():
#     args = parse_args()

#     boto_session = boto3.Session(region_name=args.region)
#     pipeline_session = PipelineSession(boto_session=boto_session)

#     input_data = ParameterString(
#         name="InputData",
#         default_value=args.input_data_s3
#     )

#     # ---------- PREPROCESSING ----------
#     processor = SKLearnProcessor(
#         framework_version="1.2-1",
#         role=args.role_arn,
#         instance_type="ml.m5.large",
#         instance_count=1,
#         sagemaker_session=pipeline_session
#     )

#     step_process = ProcessingStep(
#         name="PreprocessStep",
#         processor=processor,
#         inputs=[
#             sagemaker.processing.ProcessingInput(
#                 source=input_data,
#                 destination="/opt/ml/processing/input"
#             )
#         ],
#         outputs=[
#             sagemaker.processing.ProcessingOutput(
#                 output_name="train",
#                 source="/opt/ml/processing/train"
#             ),
#             sagemaker.processing.ProcessingOutput(
#                 output_name="validation",
#                 source="/opt/ml/processing/validation"
#             )
#         ],
#         code="src/preprocessing.py",
#         job_arguments=[
#             "--input-data", "/opt/ml/processing/input/customer_churn_processed.csv",
#             "--train-output", "/opt/ml/processing/train",
#             "--validation-output", "/opt/ml/processing/validation"
#         ]
#     )

#     # ---------- TRAINING ----------
#     estimator = SKLearn(
#         entry_point="training.py",
#         source_dir="src",
#         role=args.role_arn,
#         instance_type="ml.m5.large",
#         framework_version="1.2-1",
#         sagemaker_session=pipeline_session
#     )

#     step_train = TrainingStep(
#         name="TrainingStep",
#         estimator=estimator,
#         inputs={
#             "training": TrainingInput(
#                 s3_data=step_process.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
#                 content_type="text/csv"
#             )
#         }
#     )

#     # ---------- EVALUATION ----------
#     processor_eval = SKLearnProcessor(
#         framework_version="1.2-1",
#         role=args.role_arn,
#         instance_type="ml.m5.large",
#         instance_count=1,
#         sagemaker_session=pipeline_session
#     )

#     evaluation_report = PropertyFile(
#         name="EvaluationReport",
#         output_name="evaluation",
#         path="evaluation.json"
#     )

#     step_eval = ProcessingStep(
#         name="EvaluationStep",
#         processor=processor_eval,
#         inputs=[
#             sagemaker.processing.ProcessingInput(
#                 source=step_train.properties.ModelArtifacts.S3ModelArtifacts,
#                 destination="/opt/ml/processing/model"
#             ),
#             sagemaker.processing.ProcessingInput(
#                 source=step_process.properties.ProcessingOutputConfig.Outputs["validation"].S3Output.S3Uri,
#                 destination="/opt/ml/processing/validation"
#             )
#         ],
#         outputs=[
#             sagemaker.processing.ProcessingOutput(
#                 output_name="evaluation",
#                 source="/opt/ml/processing/output"
#             )
#         ],
#         code="src/evaluate.py",
#         property_files=[evaluation_report],
#         job_arguments=[
#             "--model-path", "/opt/ml/processing/model/model.joblib",
#             "--validation-data", "/opt/ml/processing/validation/validation.csv",
#             "--output-dir", "/opt/ml/processing/output"
#         ]
#     )

#     # ---------- MODEL REGISTRATION ----------
#     model = SKLearnModel(
#         model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
#         role=args.role_arn,
#         entry_point="inference.py",
#         source_dir="src",
#         framework_version="1.2-1",
#         sagemaker_session=pipeline_session
#     )

#     register_step = ModelStep(
#         name="RegisterModel",
#         step_args=model.register(
#             content_types=["application/json"],
#             response_types=["application/json"],
#             model_package_group_name=args.model_package_group,
#             approval_status="Approved"
#         )
#     )

#     # ---------- CONDITION ----------
#     cond = ConditionGreaterThanOrEqualTo(
#         left=JsonGet(
#             step_name="EvaluationStep",
#             property_file=evaluation_report,
#             json_path="accuracy"
#         ),
#         right=0.80
#     )

#     step_condition = ConditionStep(
#         name="AccuracyCondition",
#         conditions=[cond],
#         if_steps=[register_step],
#         else_steps=[]
#     )

#     pipeline = Pipeline(
#         name=args.pipeline_name,
#         parameters=[input_data],
#         steps=[step_process, step_train, step_eval, step_condition],
#         sagemaker_session=pipeline_session
#     )

#     pipeline.upsert(role_arn=args.role_arn)

#     print("✅ Pipeline created successfully")


# if __name__ == "__main__":
#     main()
