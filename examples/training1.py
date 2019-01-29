import os
import sys
sys.path.append(".")
from core.utils.training_utils import TrainingArgs, parse_params, training_main


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    parser = parse_params()
    args = parser.parse_args()
    trainargs = TrainingArgs(args.configure_file, args.model, args.dataset)

    training_main(trainargs)
