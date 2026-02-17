# import io
# import json
# import joblib
# import pandas as pd


# def model_fn(model_dir):
#     return joblib.load(f"{model_dir}/model.joblib")


# def input_fn(request_body, request_content_type):
#     if request_content_type == "application/json":
#         payload = json.loads(request_body)
#         if isinstance(payload, dict) and "instances" in payload:
#             data = payload["instances"]
#         else:
#             data = payload
#         return pd.DataFrame(data)

#     if request_content_type == "text/csv":
#         s = request_body.decode("utf-8") if isinstance(request_body, (bytes, bytearray)) else request_body
#         return pd.read_csv(io.StringIO(s))

#     raise ValueError(f"Unsupported content type: {request_content_type}")


# def predict_fn(input_data, model):
#     preds = model.predict(input_data)
#     return preds


# def output_fn(prediction, response_content_type):
#     if response_content_type == "application/json":
#         return json.dumps({"predictions": prediction.tolist()}), response_content_type
#     return "\n".join(map(str, prediction.tolist())), "text/plain"
import joblib
import pandas as pd
import json


def model_fn(model_dir):
    return joblib.load(f"{model_dir}/model.joblib")


def input_fn(request_body, content_type):
    if content_type == "application/json":
        data = json.loads(request_body)
        return pd.DataFrame(data)
    raise ValueError("Unsupported content type")


def predict_fn(input_data, model):
    return model.predict(input_data)


def output_fn(prediction, accept):
    return json.dumps({"predictions": prediction.tolist()})
