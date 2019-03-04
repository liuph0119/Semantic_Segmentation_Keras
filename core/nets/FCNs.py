from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.core import Dropout, Activation
from keras.layers.merge import Add
from keras.regularizers import l2
from keras.models import Model

from ..encoder import scope_table, build_encoder


def FCN_8s(input_shape,
           n_class,
           encoder_name,
           encoder_weights=None,
           fc_num=4096,
           weight_decay=1e-4,
           kernel_initializer="he_normal",
           bn_epsilon=1e-3,
           bn_momentum=0.99,
           dropout=0.5
           ):
    """ implementation of FCN-8s for semantic segmentation.
        ref: Long J, Shelhamer E, Darrell T. Fully Convolutional Networks for Semantic Segmentation[J].
            arXiv preprint arXiv:1411.4038, 2014.
    :param input_shape: tuple, i.e., (height, width, channel).
    :param n_class: int, number of class, must >= 2.
    :param encoder_name: string, name of encoder.
    :param encoder_weights: string, path of weights, default None.
    :param fc_num: int, number of filters of fully convolutions, default 4096.
    :param weight_decay: float, default 1e-4.
    :param kernel_initializer: string, default "he_normal".
    :param bn_epsilon: float, default 1e-3.
    :param bn_momentum: float, default 0.99.
    :param dropout: float, default 0.5.

    :return: a Keras Model instance.
     """
    # adapted from https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/voc-fcn8s/net.py
    encoder = build_encoder(input_shape=input_shape, encoder_name=encoder_name, encoder_weights=encoder_weights,
                                weight_decay=weight_decay, kernel_initializer=kernel_initializer,
                                bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)
    p3 = encoder.get_layer(scope_table["pool3"]).output
    p4 = encoder.get_layer(scope_table["pool4"]).output
    p5 = encoder.get_layer(scope_table["pool5"]).output

    # # # 1. merge pool5 & pool4
    # upsample prediction from pool5
    x1 = Conv2D(fc_num, (7, 7), padding="same", activation="relu",
                kernel_regularizer=l2(weight_decay), kernel_initializer=kernel_initializer)(p5)
    x1 = Dropout(dropout)(x1)
    x1 = Conv2D(fc_num, (1, 1), padding="same", activation="relu",
                kernel_regularizer=l2(weight_decay), kernel_initializer=kernel_initializer)(x1)
    x1 = Dropout(dropout)(x1)
    x1 = Conv2D(n_class, (1, 1), padding="same", activation=None,
                kernel_regularizer=l2(weight_decay), kernel_initializer=kernel_initializer)(x1)
    x1 = Conv2DTranspose(n_class, (4,4), strides=(2,2), use_bias=False, activation=None,
                         kernel_regularizer=l2(weight_decay), kernel_initializer=kernel_initializer)(x1)
    # upsample prediction from pool4
    x2 = Conv2D(n_class, (1,1), padding="same", activation=None,
                kernel_regularizer=l2(weight_decay), kernel_initializer=kernel_initializer)(p4)
    x1 = Add()([x1, x2])

    # # # 2. merge pool4 & pool3
    x1 = Conv2DTranspose(n_class, (4,4), strides=(2,2), use_bias=False, Activation=None,
                         kernel_regularizer=l2(weight_decay), kernel_initializer=kernel_initializer)(x1)
    x2 = Conv2D(n_class, (1,1), padding="same", activation=None,
                kernel_regularizer=l2(weight_decay), kernel_initializer=kernel_initializer)(p3)
    x1 = Add()([x1, x2])

    # # # 3. upsample and predict
    x1 = Conv2DTranspose(n_class, (16, 16), strides=(8, 8), use_bias=False, activation=None,
                         kernel_regularizer=l2(weight_decay), kernel_initializer=kernel_initializer)(x1)
    output = Activation("softmax")(x1)

    fcn_8s_model = Model(encoder.input, output)
    return fcn_8s_model


def FCN_16s(input_shape,
            n_class,
            encoder_name,
            encoder_weights=None,
            fc_num=4096,
            weight_decay=1e-4,
            kernel_initializer="he_normal",
            bn_epsilon=1e-3,
            bn_momentum=0.99,
            dropout=0.5):
    """ implementation of FCN-8s for semantic segmentation.
        ref: Long J, Shelhamer E, Darrell T. Fully Convolutional Networks for Semantic Segmentation[J].
            arXiv preprint arXiv:1411.4038, 2014.
    :param input_shape: tuple, i.e., (height, width, channel).
    :param n_class: int, number of class, must >= 2.
    :param encoder_name: string, name of encoder.
    :param encoder_weights: string, path of weights, default None.
    :param fc_num: int, number of filters of fully convolutions, default 4096.
    :param weight_decay: float, default 1e-4.
    :param kernel_initializer: string, default "he_normal".
    :param bn_epsilon: float, default 1e-3.
    :param bn_momentum: float, default 0.99.
    :param dropout: float, default 0.5.

    :return: a Keras Model instance.
     """
    # adapted from https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/voc-fcn16s/net.py
    encoder = build_encoder(input_shape=input_shape, encoder_name=encoder_name, encoder_weights=encoder_weights,
                                weight_decay=weight_decay, kernel_initializer=kernel_initializer,
                                bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)
    p4 = encoder.get_layer(scope_table["pool4"]).output
    p5 = encoder.get_layer(scope_table["pool5"]).output

    # # # 1. merge pool5 & pool4
    # upsamples prediction from pool5
    x1 = Conv2D(fc_num, (7, 7), padding="same", activation="relu",
                kernel_regularizer=l2(weight_decay), kernel_initializer=kernel_initializer)(p5)
    x1 = Dropout(dropout)(x1)
    x1 = Conv2D(fc_num, (1, 1), padding="same", activation="relu",
                kernel_regularizer=l2(weight_decay), kernel_initializer=kernel_initializer)(x1)
    x1 = Dropout(dropout)(x1)
    x1 = Conv2D(n_class, (1, 1), padding="same", activation=None,
                kernel_regularizer=l2(weight_decay), kernel_initializer=kernel_initializer)(x1)
    x1 = Conv2DTranspose(n_class, (4,4), strides=(2,2), use_bias=False, activation=None,
                         kernel_regularizer=l2(weight_decay), kernel_initializer=kernel_initializer)(x1)
    # upsamples from pool4
    x2 = Conv2D(n_class, (1,1), padding="same", activation=None,
                kernel_regularizer=l2(weight_decay), kernel_initializer=kernel_initializer)(p4)
    x1 = Add()([x1, x2])

    # # # 2. upsample and predict
    x1 = Conv2DTranspose(n_class, (32, 32), strides=(16, 16), use_bias=False, activation=False,
                         kernel_regularizer=l2(weight_decay), kernel_initializer=kernel_initializer)(x1)
    output = Activation("softmax")(x1)

    fcn_16s_model = Model(encoder.input, output)
    return fcn_16s_model


def FCN_32s(input_shape,
            n_class,
            encoder_name,
            encoder_weights=None,
            fc_num=4096,
            weight_decay=1e-4,
            kernel_initializer="he_normal",
            bn_epsilon=1e-3,
            bn_momentum=0.99,
            dropout=0.5):
    """ implementation of FCN-8s for semantic segmentation.
        ref: Long J, Shelhamer E, Darrell T. Fully Convolutional Networks for Semantic Segmentation[J].
            arXiv preprint arXiv:1411.4038, 2014.
    :param input_shape: tuple, i.e., (height, width, channel).
    :param n_class: int, number of class, must >= 2.
    :param encoder_name: string, name of encoder.
    :param encoder_weights: string, path of weights, default None.
    :param fc_num: int, number of filters of fully convolutions, default 4096.
    :param weight_decay: float, default 1e-4.
    :param kernel_initializer: string, default "he_normal".
    :param bn_epsilon: float, default 1e-3.
    :param bn_momentum: float, default 0.99.
    :param dropout: float, default 0.5.

    :return: a Keras Model instance.
     """
    # adapted from https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/voc-fcn16s/net.py
    encoder = build_encoder(input_shape=input_shape, encoder_name=encoder_name, encoder_weights=encoder_weights,
                                weight_decay=weight_decay, kernel_initializer=kernel_initializer,
                                bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)
    p5 = encoder.get_layer(scope_table["pool5"]).output

    # # # 1. upsamples from pool5
    # upsamples prediction from pool5
    x1 = Conv2D(fc_num, (7, 7), padding="same", activation="relu", kernel_regularizer=l2(weight_decay), kernel_initializer=kernel_initializer)(p5)
    x1 = Dropout(dropout)(x1)
    x1 = Conv2D(fc_num, (1, 1), padding="same", activation="relu", kernel_regularizer=l2(weight_decay), kernel_initializer=kernel_initializer)(x1)
    x1 = Dropout(dropout)(x1)
    x1 = Conv2D(n_class, (1, 1), padding="same", activation=None, kernel_regularizer=l2(weight_decay), kernel_initializer=kernel_initializer)(x1)
    x1 = Conv2DTranspose(n_class, (64, 64), strides=(32,32), use_bias=False, activation=None, kernel_regularizer=l2(weight_decay), kernel_initializer=kernel_initializer)(x1)

    output = Activation("softmax")(x1)

    fcn_32s_model = Model(encoder.input, output)
    return fcn_32s_model
