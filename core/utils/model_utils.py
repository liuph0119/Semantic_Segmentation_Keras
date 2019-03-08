import os
from keras.models import load_model, Model
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation
from keras.regularizers import l2

from ..configures import *
from .net_utils import BilinearUpSampling


def load_custom_model(model_path):
    """ load custom model, with custom objects like 'BilinearUpSampling', etc.

    :param model_path: the model path, string.

    :return: a Keras model instance.
    """
    if os.path.exists(model_path):
        model = load_model(model_path,
                           custom_objects={
                                            "BilinearUpSampling": BilinearUpSampling
                                          }
                           )
        return model
    else:
        raise FileNotFoundError("[model_utils.py/load_custom_model] path: {} does not exist!".format(model_path))


def transfer_model_all(model1, model2_path):
    model2 = load_custom_model(model2_path)
    for layer in model2.layers:
        model1.set_weights(layer.get_weights())
    return model1


def transfer_model_custom_class(model, n_class=2):
    i = model.input
    o = model.get_layer(-3).output
    o = Conv2D(n_class, (1, 1), kernel_regularizer=l2(WEIGHT_DECAY), kernel_initializer=KERNEL_INITIALIZER)(o)
    o = Activation("softmax")(o)
    return Model(i, o)
