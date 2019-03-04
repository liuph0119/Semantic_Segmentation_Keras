from keras.layers.core import Activation, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.merge import Concatenate
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.models import Model

from ..encoder import build_encoder, scope_table
from ..utils.net_utils import BilinearUpSampling, atrous_spatial_pyramid_pooling, separable_conv_bn


def Deeplab_v3(input_shape,
               n_class,
               encoder_name,
               encoder_weights=None,
               weight_decay=1e-4,
               kernel_initializer="he_normal",
               bn_epsilon=1e-3,
               bn_momentum=0.99):
    """ implementation of Deeplab v3 for semantic segmentation.
        ref: Chen et al. Rethinking Atrous Convolution for Semantic Image Segmentation, 2017, arXiv:1706.05587.
    :param input_shape: tuple, i.e., (height, width, channel).
    :param n_class: int, number of class, must >= 2.
    :param encoder_name: string, name of encoder.
    :param encoder_weights: string, path of weights, default None.
    :param weight_decay: float, default 1e-4.
    :param kernel_initializer: string, default "he_normal".
    :param bn_epsilon: float, default 1e-3.
    :param bn_momentum: float, default 0.99.

    :return: a Keras Model instance.
    """
    encoder = build_encoder(input_shape, encoder_name, encoder_weights=encoder_weights,
                            weight_decay=weight_decay, kernel_initializer=kernel_initializer,
                            bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)

    net = encoder.get_layer(scope_table[encoder_name]["pool4"]).output
    net = atrous_spatial_pyramid_pooling(net, 256, rates=[6, 12, 18], imagelevel=True,
                                         weight_decay=weight_decay, kernel_initializer=kernel_initializer,
                                         bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)

    net = BilinearUpSampling(target_size=(input_shape[0], input_shape[1]))(net)
    x = Conv2D(n_class, (1, 1), activation=None, kernel_initializer=kernel_initializer,
               kernel_regularizer=l2(weight_decay))(net)
    output = Activation("softmax")(x)

    return Model(encoder.input, output)


def Deeplab_v3p(input_shape,
                n_class,
                encoder_name,
                encoder_weights=None,
                weight_decay=1e-4,
                kernel_initializer="he_normal",
                bn_epsilon=1e-3,
                bn_momentum=0.99):
    """ implementation of Deeplab v3+ for semantic segmentation.
        ref: Chen et al. Chen L C, Zhu Y, Papandreou G, et al. Encoder-Decoder with Atrous Separable
             Convolution for Semantic Image Segmentation[J]. arXiv preprint arXiv:1802.02611, 2018.,
             2018, arXiv:1802.02611.
    :param input_shape: tuple, i.e., (height, width, channel).
    :param n_class: int, number of class, must >= 2.
    :param encoder_name: string, name of encoder.
    :param encoder_weights: string, path of weights, default None.
    :param weight_decay: float, default 1e-4.
    :param kernel_initializer: string, default "he_normal".
    :param bn_epsilon: float, default 1e-3.
    :param bn_momentum: float, default 0.99.

    :return: a Keras Model instance.
    """
    encoder = build_encoder(input_shape, encoder_name, encoder_weights=encoder_weights,
                            weight_decay=weight_decay, kernel_initializer=kernel_initializer,
                            bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)
    net = encoder.get_layer(scope_table[encoder_name]["pool4"]).output
    net = atrous_spatial_pyramid_pooling(net, n_filters=256, rates=[6, 12, 18], imagelevel=True,
                                         weight_decay=weight_decay, kernel_initializer=kernel_initializer,
                                         bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)
    net = Conv2D(256, (1, 1), use_bias=False, activation=None, kernel_regularizer=l2(weight_decay),
                 kernel_initializer=kernel_initializer)(net)
    net = BatchNormalization(epsilon=bn_epsilon, momentum=bn_momentum)(net)
    net = Activation("relu")(net)
    net = Dropout(0.1)(net)
    decoder_features = BilinearUpSampling(target_size=(input_shape[0] // 4, input_shape[1] // 4))(net)

    encoder_features = encoder.get_layer(scope_table[encoder_name]["pool2"]).output
    encoder_features = Conv2D(48, (1, 1), use_bias=False, activation=None, kernel_regularizer=l2(weight_decay),
                              kernel_initializer=kernel_initializer)(encoder_features)
    encoder_features = BatchNormalization(epsilon=bn_epsilon, momentum=bn_momentum)(encoder_features)
    encoder_features = Activation("relu")(encoder_features)
    net = Concatenate()([encoder_features, decoder_features])

    net = separable_conv_bn(net, 256, 'decoder_conv1', depth_activation=True,
                            weight_decay=weight_decay, kernel_initializer=kernel_initializer,
                            bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)
    net = separable_conv_bn(net, 256, 'decoder_conv2', depth_activation=True,
                            weight_decay=weight_decay, kernel_initializer=kernel_initializer,
                            bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)
    net = Dropout(0.1)(net)

    net = BilinearUpSampling(target_size=(input_shape[0], input_shape[1]))(net)
    output = Conv2D(n_class, (1, 1), activation=None, kernel_regularizer=l2(weight_decay),
                    kernel_initializer=kernel_initializer)(net)
    output = Activation("softmax")(output)

    return Model(encoder.input, output)
