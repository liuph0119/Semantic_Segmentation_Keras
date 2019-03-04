import os
from core.utils.training_utils import parse_training_args, training_main


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    trainargs = parse_training_args()
    training_main(trainargs)