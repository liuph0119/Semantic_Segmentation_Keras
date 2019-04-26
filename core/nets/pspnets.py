from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation
from keras.layers.merge import Concatenate
from keras.models import Model
from keras.regularizers import l2

from ..utils.net_utils import BilinearUpSampling, bn_act_convtranspose, bn_act_conv_block
from ..encoder import scope_table, build_encoder


def interp_block(inputs,
                 feature_map_shape,
                 level=1,
                 weight_decay=1e-4,
                 kernel_initializer="he_normal",
                 bn_epsilon=1e-3,
                 bn_momentum=0.99):
    """
    :param inputs: 4-D tensor, shape of (batch_size, height, width, channel).
    :param feature_map_shape: tuple, target shape of feature map.
    :param level: int, default 1.
    :param weight_decay: float, default 1e-4.
    :param kernel_initializer: string, default "he_normal".
    :param bn_epsilon: float, default 1e-3.
    :param bn_momentum: float, default 0.99.

    :return: 4-D tensor, shape of (batch_size, height, width, channel).
    """
    ksize = (int(round(float(feature_map_shape[0]) / float(level))),
             int(round(float(feature_map_shape[1]) / float(level))))
    stride_size = ksize

    x = MaxPooling2D(pool_size=ksize, strides=stride_size)(inputs)
    x = Conv2D(512, (1, 1), activation=None,
               kernel_regularizer=l2(weight_decay), kernel_initializer=kernel_initializer)(x)
    x = BatchNormalization(epsilon=bn_epsilon, momentum=bn_momentum)(x)
    x = Activation("relu")(x)
    x = BilinearUpSampling(target_size=feature_map_shape)(x)

    return x


def pyramid_scene_pooling(inputs,
                          feature_map_shape,
                          weight_decay=1e-4,
                          kernel_initializer="he_normal",
                          bn_epsilon=1e-3,
                          bn_momentum=0.99):
    """ PSP module.
    :param inputs: 4-D tensor, shape of (batch_size, height, width, channel).
    :param feature_map_shape: tuple, target shape of feature map.
    :param weight_decay: float, default 1e-4.
    :param kernel_initializer: string, default "he_normal".
    :param bn_epsilon: float, default 1e-3.
    :param bn_momentum: float, default 0.99.

    :return: 4-D tensor, shape of (batch_size, height, width, channel).
    """
    interp_block1 = interp_block(inputs, feature_map_shape, level=1,
                                 weight_decay=weight_decay, kernel_initializer=kernel_initializer,
                                 bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)
    interp_block2 = interp_block(inputs, feature_map_shape, level=2,
                                 weight_decay=weight_decay, kernel_initializer=kernel_initializer,
                                 bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)
    interp_block3 = interp_block(inputs, feature_map_shape, level=3,
                                 weight_decay=weight_decay, kernel_initializer=kernel_initializer,
                                 bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)
    interp_block6 = interp_block(inputs, feature_map_shape, level=6,
                                 weight_decay=weight_decay, kernel_initializer=kernel_initializer,
                                 bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)

    return Concatenate()([interp_block1, interp_block2, interp_block3, interp_block6])


def PSPNet(input_shape,
           n_class,
           encoder_name,
           encoder_weights=None,
           weight_decay=1e-4,
           kernel_initializer="he_normal",
           bn_epsilon=1e-3,
           bn_momentum=0.99,
           upscaling_method="bilinear"):
    """ implementation of PSPNet for semantic segmentation.
        ref: Zhao H, Shi J, Qi X, et al. Pyramid Scene Parsing Network[J]. arXiv preprint arXiv:1612.01105, 2016.
    :param input_shape: tuple, i.e., (height, width, channel).
    :param n_class: int, number of class, must >= 2.
    :param encoder_name: string, name of encoder.
    :param encoder_weights: string, path of weights, default None.
    :param weight_decay: float, default 1e-4.
    :param kernel_initializer: string, default "he_normal".
    :param bn_epsilon: float, default 1e-3.
    :param bn_momentum: float, default 0.99.
    :param upscaling_method: string, "bilinear" of "conv", default "bilinear".

    :return: a Keras Model instance.
    """
    encoder = build_encoder(input_shape, encoder_name, encoder_weights=encoder_weights,
                            weight_decay=weight_decay, kernel_initializer=kernel_initializer,
                            bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)

    features = encoder.get_layer(scope_table[encoder_name]["pool4"]).output
    feature_map_shape = (int(input_shape[0]/16), int(input_shape[1]/16))

    features = pyramid_scene_pooling(features, feature_map_shape,
                                     weight_decay=weight_decay, kernel_initializer=kernel_initializer,
                                     bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)
    features = Conv2D(512, (3, 3), padding="same", use_bias=False, activation=None,
                      kernel_regularizer=l2(weight_decay), kernel_initializer=kernel_initializer)(features)
    features = BatchNormalization(epsilon=bn_epsilon, momentum=bn_momentum)(features)
    features = Activation("relu")(features)

    # upsample
    if upscaling_method == "conv":
        features = bn_act_convtranspose(features, 512, (3, 3), 2,
                                        weight_decay=weight_decay, kernel_initializer=kernel_initializer,
                                        bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)
        features = bn_act_conv_block(features, 512, (3, 3),
                                     weight_decay=weight_decay, kernel_initializer=kernel_initializer,
                                     bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)
        features = bn_act_convtranspose(features, 256, (3, 3), 2,
                                        weight_decay=weight_decay, kernel_initializer=kernel_initializer,
                                        bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)
        features = bn_act_conv_block(features, 256, (3, 3),
                                     weight_decay=weight_decay, kernel_initializer=kernel_initializer,
                                     bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)
        features = bn_act_convtranspose(features, 128, (3, 3), 2,
                                        weight_decay=weight_decay, kernel_initializer=kernel_initializer,
                                        bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)
        features = bn_act_conv_block(features, 128, (3, 3),
                                     weight_decay=weight_decay, kernel_initializer=kernel_initializer,
                                     bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)
        features = bn_act_convtranspose(features, 64, (3, 3), 2,
                                        weight_decay=weight_decay, kernel_initializer=kernel_initializer,
                                        bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)
        features = bn_act_conv_block(features, 64, (3, 3),
                                     weight_decay=weight_decay, kernel_initializer=kernel_initializer,
                                     bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)
    else:
        features = BilinearUpSampling(target_size=(input_shape[0], input_shape[1]))(features)

    output = Conv2D(n_class, (1, 1), activation=None,
                    kernel_regularizer=l2(weight_decay), kernel_initializer=kernel_initializer)(features)
    output = Activation("softmax")(output)

    return Model(encoder.input, output)
