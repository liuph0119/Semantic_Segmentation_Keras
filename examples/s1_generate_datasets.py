import os
import json
from core.utils.data_utils.generate_dataset_utils import generate_dataset_main

if __name__ == "__main__":
    configure_file = "E:/SemanticSegmentation_Keras/configures/generate_dataset_configures.json"
    dataset_name = "voc"

    with open(configure_file, "r") as f:
        args = json.load(f)[dataset_name]

    generate_dataset_main(dataset_name, args)