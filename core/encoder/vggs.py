from keras.layers import Input
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras.models import Model

from ..utils.net_utils import conv_bn_act_block


def vgg_16(input_shape,
           weight_decay=1e-4,
           kernel_initializer="he_normal",
           bn_epsilon=1e-3,
           bn_momentum=0.99):
    """ build a vgg-16 encoder.
    :param input_shape: tuple, i.e., (height, width, channel).
    :param weight_decay: float, default 1e-4.
    :param kernel_initializer: string, default "he_normal".
    :param bn_epsilon: float, default 1e-3.
    :param bn_momentum: float, default 0.99.

    :return: a Keras Moodel instance.
    """
    input_x = Input(shape=input_shape)
    x = BatchNormalization(epsilon=bn_epsilon, momentum=bn_momentum)(input_x)

    x = conv_bn_act_block(x, 64, weight_decay=weight_decay, kernel_initializer=kernel_initializer,
                          bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)
    x = conv_bn_act_block(x, 64, weight_decay=weight_decay, kernel_initializer=kernel_initializer,
                          bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = conv_bn_act_block(x, 128, weight_decay=weight_decay, kernel_initializer=kernel_initializer,
                          bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)
    x = conv_bn_act_block(x, 128, weight_decay=weight_decay, kernel_initializer=kernel_initializer,
                          bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = conv_bn_act_block(x, 256, weight_decay=weight_decay, kernel_initializer=kernel_initializer,
                          bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)
    x = conv_bn_act_block(x, 256, weight_decay=weight_decay, kernel_initializer=kernel_initializer,
                          bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)
    x = conv_bn_act_block(x, 256, weight_decay=weight_decay, kernel_initializer=kernel_initializer,
                          bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = conv_bn_act_block(x, 512, weight_decay=weight_decay, kernel_initializer=kernel_initializer,
                          bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)
    x = conv_bn_act_block(x, 512, weight_decay=weight_decay, kernel_initializer=kernel_initializer,
                          bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)
    x = conv_bn_act_block(x, 512, weight_decay=weight_decay, kernel_initializer=kernel_initializer,
                          bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = conv_bn_act_block(x, 512, weight_decay=weight_decay, kernel_initializer=kernel_initializer,
                          bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)
    x = conv_bn_act_block(x, 512, weight_decay=weight_decay, kernel_initializer=kernel_initializer,
                          bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)
    x = conv_bn_act_block(x, 512, weight_decay=weight_decay, kernel_initializer=kernel_initializer,
                          bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    return Model(input_x, x)


def vgg_19(input_shape,
           weight_decay=1e-4,
           kernel_initializer="he_normal",
           bn_epsilon=1e-3,
           bn_momentum=0.99):
    """ build a vgg-16 encoder.
    :param input_shape: tuple, i.e., (height, width, channel).
    :param weight_decay: float, default 1e-4.
    :param kernel_initializer: string, default "he_normal".
    :param bn_epsilon: float, default 1e-3.
    :param bn_momentum: float, default 0.99.

    :return: a Keras Moodel instance.
    """
    input_x = Input(shape=input_shape)
    x = BatchNormalization(epsilon=bn_epsilon, momentum=bn_momentum)(input_x)

    x = conv_bn_act_block(x, 64, weight_decay=weight_decay, kernel_initializer=kernel_initializer,
                          bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)
    x = conv_bn_act_block(x, 64, weight_decay=weight_decay, kernel_initializer=kernel_initializer,
                          bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = conv_bn_act_block(x, 128, weight_decay=weight_decay, kernel_initializer=kernel_initializer,
                          bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)
    x = conv_bn_act_block(x, 128, weight_decay=weight_decay, kernel_initializer=kernel_initializer,
                          bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = conv_bn_act_block(x, 256, weight_decay=weight_decay, kernel_initializer=kernel_initializer,
                          bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)
    x = conv_bn_act_block(x, 256, weight_decay=weight_decay, kernel_initializer=kernel_initializer,
                          bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)
    x = conv_bn_act_block(x, 256, weight_decay=weight_decay, kernel_initializer=kernel_initializer,
                          bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = conv_bn_act_block(x, 512, weight_decay=weight_decay, kernel_initializer=kernel_initializer,
                          bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)
    x = conv_bn_act_block(x, 512, weight_decay=weight_decay, kernel_initializer=kernel_initializer,
                          bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)
    x = conv_bn_act_block(x, 512, weight_decay=weight_decay, kernel_initializer=kernel_initializer,
                          bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)
    x = conv_bn_act_block(x, 512, weight_decay=weight_decay, kernel_initializer=kernel_initializer,
                          bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = conv_bn_act_block(x, 512, weight_decay=weight_decay, kernel_initializer=kernel_initializer,
                          bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)
    x = conv_bn_act_block(x, 512, weight_decay=weight_decay, kernel_initializer=kernel_initializer,
                          bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)
    x = conv_bn_act_block(x, 512, weight_decay=weight_decay, kernel_initializer=kernel_initializer,
                          bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)
    x = conv_bn_act_block(x, 512, weight_decay=weight_decay, kernel_initializer=kernel_initializer,
                          bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    return Model(input_x, x)