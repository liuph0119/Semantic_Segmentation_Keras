import os
from keras.applications.resnet50 import ResNet50

from .resnet_v2 import resnet_v2_50, resnet_v2_101, resnet_v2_152, resnet_v2_200
from .resnet_v2_separable import resnet_v2_50_separable, resnet_v2_101_separable, resnet_v2_152_separable, resnet_v2_200_separable
from .xceptions import xception_41
from .vggs import vgg_16, vgg_19


# the alias mapping of the pool-like(not always be pooling layers, might caused by strides as well) layers
scope_table = {
    "resnet_v1_50": {"pool1": "activation_1", "pool2": "activation_10", "pool3": "activation_22", "pool4": "activation_40", "pool5": "activation_49"},
    "resnet_v2_50": {"pool1": "conv1", "pool2": "activation_8", "pool3": "activation_20", "pool4": "activation_49"},
    "resnet_v2_101": {"pool1": "conv1", "pool2": "activation_8", "pool3": "activation_20", "pool4": "activation_100"},
    "resnet_v2_152": {"pool1": "conv1", "pool2": "activation_8", "pool3": "activation_32", "pool4": "activation_151"},
    "resnet_v2_200": {"pool1": "conv1", "pool2": "activation_8", "pool3": "activation_80", "pool4": "activation_199"},
    "resnet_v2_50_separable": {"pool1": "conv1", "pool2": "activation_8", "pool3": "activation_20", "pool4": "activation_49"},
    "resnet_v2_101_separable": {"pool1": "conv1", "pool2": "activation_8", "pool3": "activation_20", "pool4": "activation_100"},
    "resnet_v2_152_separable": {"pool1": "conv1", "pool2": "activation_8", "pool3": "activation_32", "pool4": "activation_151"},
    "resnet_v2_200_separable": {"pool1": "conv1", "pool2": "activation_8", "pool3": "activation_80", "pool4": "activation_199"},

    "xception_41": {"pool1": "entry_block2_sepconv3_act", "pool2": "entry_block3_sepconv3_act", "pool3": "entry_block4_sepconv3_act", "pool4": "exit_block2_sepconv3_bn"},

    "vgg_16":{"pool0": "activation_2", "pool1": "activation_4", "pool2": "activation_7", "pool3": "activation_10", "pool4": "activation_13", "pool5": "maxpooling_5"},
    "vgg_19":{"pool0": "activation_2", "pool1": "activation_4", "pool2": "activation_8", "pool3": "activation_12", "pool4": "activation_16", "pool5": "maxpooling_5"}
}


def build_encoder(input_shape,
                  encoder_name,
                  encoder_weights=None,
                  weight_decay=1e-4,
                  kernel_initializer="he_normal",
                  bn_epsilon=1e-3,
                  bn_momentum=0.99):
    """ the main api to build a encoder.
    :param input_shape: tuple, i.e., (height, width. channel).
    :param encoder_name: string, name of the encoder, refer to 'scope_table' above.
    :param encoder_weights: string, path of the weight, default None.
    :param weight_decay: float, default 1e-4.
    :param kernel_initializer: string, default "he_normal".
    :param bn_epsilon: float, default 1e-3.
    :param bn_momentum: float, default 0.99.

    :return: a Keras Model instance.
    """
    encoder_name = encoder_name.lower()

    if encoder_name == "resnet_v1_50":
        encoder = ResNet50(input_shape=input_shape, weights=encoder_weights, include_top=False)

    elif encoder_name=="resnet_v2_50":
        encoder = resnet_v2_50(input_shape, weight_decay=weight_decay, kernel_initializer=kernel_initializer, bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)
    elif encoder_name=="resnet_v2_101":
        encoder = resnet_v2_101(input_shape, weight_decay=weight_decay, kernel_initializer=kernel_initializer, bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)
    elif encoder_name=="resnet_v2_152":
        encoder = resnet_v2_152(input_shape, weight_decay=weight_decay, kernel_initializer=kernel_initializer, bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)
    elif encoder_name=="resnet_v2_200":
        encoder = resnet_v2_200(input_shape, weight_decay=weight_decay, kernel_initializer=kernel_initializer, bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)

    elif encoder_name=="resnet_v2_50_separable":
        encoder = resnet_v2_50_separable(input_shape, kernel_size=5, weight_decay=weight_decay, kernel_initializer=kernel_initializer, bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)
    elif encoder_name=="resnet_v2_101_separable":
        encoder = resnet_v2_101_separable(input_shape, kernel_size=5, weight_decay=weight_decay, kernel_initializer=kernel_initializer, bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)
    elif encoder_name=="resnet_v2_152_separable":
        encoder = resnet_v2_152_separable(input_shape, kernel_size=5, weight_decay=weight_decay, kernel_initializer=kernel_initializer, bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)
    elif encoder_name=="resnet_v2_200_separable":
        encoder = resnet_v2_200_separable(input_shape, kernel_size=5, weight_decay=weight_decay, kernel_initializer=kernel_initializer, bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)

    elif encoder_name == "xception_41":
        encoder = xception_41(input_shape, weight_decay=weight_decay, kernel_initializer=kernel_initializer, bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)

    elif encoder_name=="vgg_16":
        encoder = vgg_16(input_shape, weight_decay=weight_decay, kernel_initializer=kernel_initializer, bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)
    elif encoder_name=="vgg_19":
        encoder = vgg_19(input_shape, weight_decay=weight_decay, kernel_initializer=kernel_initializer, bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)
    else:
        raise ValueError("Invalid encoder name: {}."
                         "Supported encoder names: 'resnet_v1_50', 'resnet_v2_50', 'resnet_v2_101', 'resnet_v2_152', 'resnet_v2_200', "
                         "'resnet_v2_50_separable', 'resnet_v2_101_separable', 'resnet_v2_152_separable', 'resnet_v2_200_separable', "
                         "'xception_41', 'vgg_16', 'vgg_19'".format(encoder_name))

    if encoder_weights is not None and os.path.exists(encoder_weights):
        encoder.load_weights(encoder_weights)

    return encoder