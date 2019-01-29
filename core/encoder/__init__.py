"""
    Script: __init__.py
    Author: Penghua Liu
    Date: 2019-01-13
    Email: liuphhhh@foxmail.com
    Functions: encoder builders given input_shape, encoder name and encoder weights,
    enables to access pool2, pool3,..., etc layers for hierarchical feature extraction.
    # TODO: the pool2, pool3, pool4, pool5 layer names of each resnet_v2 model need to be confirmed.
"""

from .resnet_v2 import resnet_v2_50, resnet_v2_101, resnet_v2_152, resnet_v2_200
from .resnet_v2_separable import resnet_v2_50_separable, resnet_v2_101_separable, resnet_v2_152_separable, resnet_v2_200_separable
from .xception import Xception_41
from .vgg import VGG_16, VGG_19
from keras.applications.resnet50 import ResNet50


# the alias mapping of the pool-like(not always be pooling layers, might caused by strides as well) layers
scope_table = {
    "resnet_v2_50": {"pool1": "conv1", "pool2": "activation_8", "pool3": "activation_20", "pool4": "activation_49"},
    "resnet_v2_101": {"pool1": "conv1", "pool2": "activation_8", "pool3": "activation_20", "pool4": "activation_100"},
    "resnet_v2_152": {"pool1": "conv1", "pool2": "activation_8", "pool3": "activation_32", "pool4": "activation_151"},
    "resnet_v2_200": {"pool1": "conv1", "pool2": "activation_8", "pool3": "activation_80", "pool4": "activation_199"},
    "resnet_v2_50_separable": {"pool1": "conv1", "pool2": "activation_8", "pool3": "activation_20", "pool4": "activation_49"},
    "resnet_v2_101_separable": {"pool1": "conv1", "pool2": "activation_8", "pool3": "activation_20", "pool4": "activation_100"},
    "resnet_v2_152_separable": {"pool1": "conv1", "pool2": "activation_8", "pool3": "activation_32", "pool4": "activation_151"},
    "resnet_v2_200_separable": {"pool1": "conv1", "pool2": "activation_8", "pool3": "activation_80", "pool4": "activation_199"},

    "xception_41": {"pool2": "add_1", "pool4": "exit_block2_sepconv3_bn"},

    "vgg_16":{"pool0": "activation_2", "pool1": "activation_4", "pool2": "activation_7", "pool3": "activation_10", "pool4": "activation_13"},
    "vgg_19":{"pool0": "activation_2", "pool1": "activation_4", "pool2": "activation_8", "pool3": "activation_12", "pool4": "activation_16"}
}


def build_encoder(input_shape=(256, 256, 3), encoder_name="resnet50",
                  encoder_weights="D:/keras_weights/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5"):
    """ the main api to build a encoder

    # Args:
        :param input_shape: input_shape, tuple of (height, width, channel), default (256, 256, 3)
        :param encoder_name: the encoder name, string, defaule "resnet50".
            supported encoder names: 'resnet50', 'resnet101', 'resnet152', 'resnet200',  'resnet50_v1', 'xception_41'
        :param encoder_weights: the pre-trained weights of the encoder

    # Returns:
        a Keras model instance
    """

    if encoder_name.lower() == "resnet_v1_50":
        encoder = ResNet50(input_shape=input_shape, weights=encoder_weights, include_top=False)
    elif encoder_name.lower() == "resnet_v2_50":
        encoder = resnet_v2_50(input_shape)
    elif encoder_name.lower() == "resnet_v2_101":
        encoder = resnet_v2_101(input_shape)
    elif encoder_name.lower() == "resnet_v2_152":
        encoder = resnet_v2_152(input_shape)
    elif encoder_name.lower() == "resnet_v2_200":
        encoder = resnet_v2_200(input_shape)

    elif encoder_name.lower()=="resnet_v2_50_separable":
        encoder = resnet_v2_50_separable(input_shape)
    elif encoder_name.lower()=="resnet_v2_101_separable":
        encoder = resnet_v2_101_separable(input_shape)
    elif encoder_name.lower()=="resnet_v2_152_separable":
        encoder = resnet_v2_152_separable(input_shape)
    elif encoder_name.lower()=="resnet_v2_200_separable":
        encoder = resnet_v2_200_separable(input_shape)

    elif encoder_name.lower() == "xception_41":
        encoder = Xception_41(input_shape)

    elif encoder_name.lower()=="vgg_16":
        encoder = VGG_16(input_shape)
    elif encoder_name.lower()=="vgg_19":
        encoder = VGG_19(input_shape)
    else:
        raise ValueError("the 'encoder_name'={} is not supported."
                         "\nSupported encoder names: 'resnet_v1_50', 'resnet_v2_50', 'resnet_v2_101', 'resnet_v2_152', 'resnet_v2_200', "
                         "'resnet_v2_50_separable', 'resnet_v2_101_separable', 'resnet_v2_152_separable', 'resnet_v2_200_separable', 'xception_41', "
                         "'vgg_16', 'vgg_19'")

    return encoder
