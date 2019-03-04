import os
from keras.models import load_model
from .net_utils import BilinearUpSampling

def load_custom_model(model_path):
    """ load custom model, with custom objects like 'BilinearUpSampling', etc.

    :param model_path: the model path, string.

    :return: a Keras model instance.
    """
    if os.path.exists(model_path):
        model = load_model(model_path, custom_objects={
                                                        "BilinearUpSampling": BilinearUpSampling
                                                       }
                           )
        return model
    else:
        raise FileNotFoundError("[model_utils.py/load_custom_model] path: {} does not exist!".format(model_path))