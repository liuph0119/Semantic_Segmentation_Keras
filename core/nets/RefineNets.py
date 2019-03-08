from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import Add
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.regularizers import l2

from ..utils.net_utils import BilinearUpSampling, bn_act_convtranspose, bn_act_conv_block
from ..encoder import scope_table, build_encoder


def residual_conv_unit(inputs,
                       n_filters=256,
                       kernel_size=3,
                       weight_decay=1e-4,
                       kernel_initializer="he_normal",
                       bn_epsilon=1e-3,
                       bn_momentum=0.99):
    """ residual convolutional unit.
    :param inputs: 4-D tensor, shape of (batch_size, height, width, channel).
    :param n_filters: int, number of filters, default 256.
    :param kernel_size: int, default 3.
    :param weight_decay: float, default 1e-4.
    :param kernel_initializer: string, default "he_normal".
    :param bn_epsilon: float, default 1e-3.
    :param bn_momentum: float, default 0.99.

    :return: 4-D tensor, shape of (batch_size, height, width, channel).
    """
    x = Activation("relu")(inputs)
    x = Conv2D(n_filters, (kernel_size, kernel_size), padding="same", activation=None, use_bias=False,
               kernel_regularizer=l2(weight_decay), kernel_initializer=kernel_initializer)(x)
    x = BatchNormalization(epsilon=bn_epsilon, momentum=bn_momentum)(x)
    x = Activation("relu")(x)
    x = Conv2D(n_filters, (kernel_size, kernel_size), padding="same", activation=None, use_bias=False,
               kernel_regularizer=l2(weight_decay), kernel_initializer=kernel_initializer)(x)
    x = BatchNormalization(epsilon=bn_epsilon, momentum=bn_momentum)(x)
    x = Add()([x, inputs])

    return x


def chained_residual_pooling(inputs,
                             pool_size=(5, 5),
                             n_filters=256,
                             weight_decay=1e-4,
                             kernel_initializer="he_normal",
                             bn_epsilon=1e-3,
                             bn_momentum=0.99):
    """ chained residual pooling.
    :param inputs: 4-D tensor, shape of (batch_size, height, width, channel).
    :param pool_size: tuple, default (5, 5).
    :param n_filters: int, number of filters, default 256.
    :param weight_decay: float, default 1e-4.
    :param kernel_initializer: string, default "he_normal".
    :param bn_epsilon: float, default 1e-3.
    :param bn_momentum: float, default 0.99.

    :return: 4-D tensor, shape of (batch_size, height, width, channel).
    """
    x_relu = Activation("relu")(inputs)

    x = MaxPooling2D(pool_size=pool_size, strides=(1, 1), padding="same")(x_relu)
    x = Conv2D(n_filters, (3, 3), padding="same", activation=None, use_bias=False,
               kernel_regularizer=l2(weight_decay), kernel_initializer=kernel_initializer)(x)
    x = BatchNormalization(epsilon=bn_epsilon, momentum=bn_momentum)(x)
    x_sum1 = Add()([x_relu, x])

    x = MaxPooling2D(pool_size=pool_size, strides=(1, 1), padding="same")(x)
    x = Conv2D(n_filters, (3, 3), padding="same", activation=None, use_bias=False,
               kernel_regularizer=l2(weight_decay), kernel_initializer=kernel_initializer)(x)
    x_sum2 = Add()([x, x_sum1])

    return x_sum2


def multi_resolution_fusion(high_inputs=None,
                            low_inputs=None,
                            n_filters=256,
                            weight_decay=1e-4,
                            kernel_initializer="he_normal",
                            bn_epsilon=1e-3,
                            bn_momentum=0.99):
    """ fuse multi resolution features.
    :param high_inputs: 4-D tensor,  shape of (batch_size, height, width, channel),
        features with high spatial resolutions.
    :param low_inputs: 4-D tensor,  shape of (batch_size, height, width, channel),
        features with low spatial resolutions.
    :param n_filters: int, number of filters, default 256.
    :param weight_decay: float, default 1e-4.
    :param kernel_initializer: string, default "he_normal".
    :param bn_epsilon: float, default 1e-3.
    :param bn_momentum: float, default 0.99.

    :return: 4-D tensor, shape of (batch_size, height, width, channel).
    """
    if high_inputs is None:
        fuse = Conv2D(n_filters, (3, 3), padding="same", activation=None, use_bias=False,
                      kernel_regularizer=l2(weight_decay), kernel_initializer=kernel_initializer)(low_inputs)
        fuse = BatchNormalization(epsilon=bn_epsilon, momentum=bn_momentum)(fuse)
    else:
        conv_low = Conv2D(n_filters, (3, 3), padding="same", activation=None, use_bias=False,
                          kernel_regularizer=l2(weight_decay), kernel_initializer=kernel_initializer)(low_inputs)
        conv_low = BatchNormalization(epsilon=bn_epsilon, momentum=bn_momentum)(conv_low)
        conv_high = Conv2D(n_filters, (3, 3), padding="same", activation=None, use_bias=False,
                           kernel_regularizer=l2(weight_decay), kernel_initializer=kernel_initializer)(high_inputs)
        conv_high = BatchNormalization(epsilon=bn_epsilon, momentum=bn_momentum)(conv_high)
        conv_low = BilinearUpSampling(target_size=(int(conv_high.shape[1]), int(conv_high.shape[2])))(conv_low)
        fuse = Add()([conv_high, conv_low])

    return fuse


def refine_block(high_inputs=None,
                 low_inputs=None,
                 base_filters=256):
    """ a complete refine block.
    :param high_inputs: 4-D tensor,  shape of (batch_size, height, width, channel),
        features with high spatial resolutions.
    :param low_inputs: 4-D tensor,  shape of (batch_size, height, width, channel),
        features with low spatial resolutions.
    :param base_filters: int, initial number of filters, default 256.
    :return:
    """
    if low_inputs is None:  # Block 4
        # 2 RCUs
        rcu_new_low = residual_conv_unit(high_inputs, n_filters=base_filters * 2)
        rcu_new_low = residual_conv_unit(rcu_new_low, n_filters=base_filters * 2)

        # feature fusion
        fuse = multi_resolution_fusion(high_inputs=None, low_inputs=rcu_new_low, n_filters=base_filters * 2)
        fuse_pooling = chained_residual_pooling(fuse, n_filters=base_filters * 2)
        output = residual_conv_unit(fuse_pooling, n_filters=base_filters * 2)
        return output
    else:
        rcu_high = residual_conv_unit(high_inputs, n_filters=base_filters)
        rcu_high = residual_conv_unit(rcu_high, n_filters=base_filters)

        fuse = multi_resolution_fusion(rcu_high, low_inputs, n_filters=base_filters)
        fuse_pooling = chained_residual_pooling(fuse, n_filters=base_filters)
        output = residual_conv_unit(fuse_pooling, n_filters=base_filters)
        return output


def RefineNet(input_shape,
              n_class,
              encoder_name,
              encoder_weights=None,
              weight_decay=1e-4,
              kernel_initializer="he_normal",
              bn_epsilon=1e-3,
              bn_momentum=0.99,
              init_filters=256,
              upscaling_method="bilinear"):
    """ 4 cascaded RefineNet implementation using keras
        ref: Lin G, Milan A, Shen C, et al. RefineNet: Multi-Path Refinement Networks for High-Resolution
             Semantic Segmentation[J]. arXiv preprint arXiv:1611.06612, 2016.
    :param input_shape: tuple, i.e., (height, width, channel).
    :param n_class: int, number of class, must >= 2.
    :param encoder_name: string, name of encoder.
    :param encoder_weights: string, path of weights, default None.
    :param weight_decay: float, default 1e-4.
    :param kernel_initializer: string, default "he_normal".
    :param bn_epsilon: float, default 1e-3.
    :param bn_momentum: float, default 0.99.
    :param init_filters: int, number of filters when apply refining.
    :param upscaling_method: string, "bilinear" of "conv", default "bilinear".

    :return: a Keras Model instance.
    """
    encoder = build_encoder(input_shape, encoder_name, encoder_weights=encoder_weights,
                            weight_decay=weight_decay, kernel_initializer=kernel_initializer,
                            bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)

    # actually are pool5, pool4, pool3 and pool2
    high_1 = encoder.get_layer(scope_table[encoder_name]["pool4"]).output
    high_2 = encoder.get_layer(scope_table[encoder_name]["pool3"]).output
    high_3 = encoder.get_layer(scope_table[encoder_name]["pool2"]).output
    high_4 = encoder.get_layer(scope_table[encoder_name]["pool1"]).output

    high_1 = Conv2D(init_filters * 2, (1, 1), padding="same", activation="relu",
                    kernel_regularizer=l2(weight_decay), kernel_initializer=kernel_initializer)(high_1)
    high_2 = Conv2D(init_filters, (1, 1), padding="same", activation="relu",
                    kernel_regularizer=l2(weight_decay), kernel_initializer=kernel_initializer)(high_2)
    high_3 = Conv2D(init_filters, (1, 1), padding="same", activation="relu",
                    kernel_regularizer=l2(weight_decay), kernel_initializer=kernel_initializer)(high_3)
    high_4 = Conv2D(init_filters, (1, 1), padding="same", activation="relu",
                    kernel_regularizer=l2(weight_decay), kernel_initializer=kernel_initializer)(high_4)

    low_1 = refine_block(high_1, low_inputs=None, base_filters=init_filters)
    low_2 = refine_block(high_2, low_1, base_filters=init_filters)
    low_3 = refine_block(high_3, low_2, base_filters=init_filters)
    low_4 = refine_block(high_4, low_3, base_filters=init_filters)
    x = low_4

    x = residual_conv_unit(x, init_filters)
    x = residual_conv_unit(x, init_filters)

    if upscaling_method == "conv":
        x = bn_act_convtranspose(x, 128, kernel_size=[3, 3], scale=2,
                                 weight_decay=weight_decay, kernel_initializer=kernel_initializer,
                                 bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)
        x = bn_act_conv_block(x, 128, weight_decay=weight_decay, kernel_initializer=kernel_initializer,
                              bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)
    else:
        x = BilinearUpSampling(target_size=(input_shape[0], input_shape[1]))(x)

    output = Conv2D(n_class, (1, 1), activation=None,
                    kernel_regularizer=l2(weight_decay), kernel_initializer=kernel_initializer)(x)
    output = Activation("softmax")(output)

    return Model(encoder.input, output)
