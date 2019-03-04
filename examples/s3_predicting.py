import os
from core.configures import PREDICTING_CONFIG
from core.utils.predicting_utils import predict_main


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICE"] = "0"
    predict_main(PREDICTING_CONFIG)