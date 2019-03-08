from keras.engine import Layer
from keras.layers.pooling import AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation
from keras.layers.convolutional import Conv2D, Conv2DTranspose, DepthwiseConv2D, ZeroPadding2D
from keras.layers.merge import Concatenate
from keras.regularizers import l2
import tensorflow as tf


class BilinearUpSampling(Layer):
    def __init__(self, target_size, **kwargs):
        super(BilinearUpSampling, self).__init__(**kwargs)
        self.target_size = target_size

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.target_size[0], self.target_size[1], input_shape[-1])

    def _resize_function(self, inputs):
        return tf.cast(tf.image.resize_bilinear(inputs, self.target_size, align_corners=True), dtype=tf.float32)

    def call(self, inputs):
        return self._resize_function(inputs=inputs)

    def get_config(self):
        config = {'target_size': self.target_size}
        base_config = super(BilinearUpSampling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def separable_conv_bn(x,
                      n_filters,
                      prefix,
                      stride=1,
                      kernel_size=3,
                      rate=1,
                      depth_activation=False,
                      weight_decay=1e-4,
                      kernel_initializer="he_normal",
                      bn_epsilon=1e-3,
                      bn_momentum=0.99):
    """ Separable convolution, with BN between depthwise and pointwise.
    :param x: 4-D tensor, shape of (batch_size, height, width, channel).
    :param n_filters: int, number of filters in pointwise convolution.
    :param prefix: string, prefix of name.
    :param stride: int, default 1.
    :param kernel_size: int, default 3.
    :param rate: int, default 1.
    :param depth_activation: bool, whether to add activation after BN, default False.
    :param weight_decay: float, default 1e-4.
    :param kernel_initializer: string, default "he_normal".
    :param bn_epsilon: float, default 1e-3.
    :param bn_momentum: float, default 0.99.

    :return: 4-D tensor, shape of (batch_size, height, width, channel)
    """

    if stride == 1:
        depth_padding = 'same'
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        x = ZeroPadding2D((pad_beg, pad_end))(x)
        depth_padding = 'valid'

    if not depth_activation:
        x = Activation('relu')(x)
    x = DepthwiseConv2D((kernel_size, kernel_size), strides=(stride, stride), dilation_rate=(rate, rate),
                        padding=depth_padding, use_bias=False, name=prefix + '_depthwise',
                        kernel_regularizer=l2(weight_decay), kernel_initializer=kernel_initializer)(x)
    x = BatchNormalization(name=prefix + '_depthwise_BN', epsilon=bn_epsilon, momentum=bn_momentum)(x)
    if depth_activation:
        x = Activation('relu')(x)
    x = Conv2D(n_filters, (1, 1), padding='same', use_bias=False, name=prefix + '_pointwise',
               kernel_regularizer=l2(weight_decay), kernel_initializer=kernel_initializer)(x)
    x = BatchNormalization(name=prefix + '_pointwise_BN', epsilon=bn_epsilon, momentum=bn_momentum)(x)
    if depth_activation:
        x = Activation('relu')(x)

    return x


def atrous_spatial_pyramid_pooling(inputs,
                                   n_filters=256,
                                   rates=[6, 12, 18],
                                   imagelevel=True,
                                   weight_decay=1e-4,
                                   kernel_initializer="he_normal",
                                   bn_epsilon=1e-3,
                                   bn_momentum=0.99):
    """ ASPP consists of one 1×1 convolution and three 3×3 convolutions with rates = (6, 12, 18)
    when output stride = 16 (all with 256 filters and batch normalization), and (b) the image-level features
    :param inputs: 4-D tensor, shape of (batch_size, height, width, channel).
    :param n_filters: int, number of filters, default 256.
    :param rates: list of dilation rates, default [6, 12, 18].
    :param imagelevel: bool, default True.
    :param weight_decay: float, default 1e-4.
    :param kernel_initializer: string, default "he_normal".
    :param bn_epsilon: float, default 1e-3.
    :param bn_momentum: float, default 0.99.

    :return: 4-D tensor, shape of (batch_size, height, width, channel).
    """
    branch_features = []
    if imagelevel:
        # image level features
        image_feature = AveragePooling2D(pool_size=(int(inputs.shape[1]), int(inputs.shape[2])))(inputs)
        image_feature = Conv2D(n_filters, (1, 1), use_bias=False, activation=None, kernel_regularizer=l2(weight_decay),
                               kernel_initializer=kernel_initializer)(image_feature)
        image_feature = BatchNormalization(epsilon=bn_epsilon, momentum=bn_momentum)(image_feature)
        image_feature = Activation("relu")(image_feature)
        image_feature = BilinearUpSampling(target_size=(int(inputs.shape[1]), int(inputs.shape[2])))(image_feature)
        branch_features.append(image_feature)

    # 1×1 conv
    atrous_pool_block_1 = Conv2D(n_filters, (1, 1), padding="same", use_bias=False, activation=None,
                                 kernel_regularizer=l2(weight_decay), kernel_initializer=kernel_initializer)(inputs)
    atrous_pool_block_1 = BatchNormalization(epsilon=bn_epsilon, momentum=bn_momentum)(atrous_pool_block_1)
    atrous_pool_block_1 = Activation("relu")(atrous_pool_block_1)
    branch_features.append(atrous_pool_block_1)

    for i, rate in enumerate(rates):
        atrous_pool_block_i = separable_conv_bn(inputs, 256, 'aspp'+str(i+1), rate=rate, depth_activation=True,
                                                weight_decay=weight_decay, kernel_initializer=kernel_initializer,
                                                bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)
        branch_features.append(atrous_pool_block_i)

    # concatenate multi-scale features
    aspp_features = Concatenate()(branch_features)
    return aspp_features


def conv_bn_act_block(inputs,
                      n_filters,
                      weight_decay=1e-4,
                      kernel_initializer="he_normal",
                      bn_epsilon=1e-3,
                      bn_momentum=0.99):
    """ Conv + BN + Act
    :param inputs: 4-D tensor, shape of (batch_size, height, width, channel).
    :param n_filters: int, number of convolution filters.
    :param weight_decay: float, default 1e-4.
    :param kernel_initializer: string, default "he_normal".
    :param bn_epsilon: float, default 1e-3.
    :param bn_momentum: float, default 0.99.

    :return: 4-D tensor, shape of (batch_size, height, width, channel).
    """
    x = inputs
    x = Conv2D(n_filters, (3, 3), strides=(1, 1), padding="same", use_bias=False, activation=None,
               kernel_initializer=kernel_initializer, kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization(epsilon=bn_epsilon, momentum=bn_momentum)(x)
    x = Activation("relu")(x)

    return x


def bn_act_conv_block(inputs,
                      n_filters,
                      kernel_size=3,
                      rate=1,
                      weight_decay=1e-4,
                      kernel_initializer="he_normal",
                      bn_epsilon=1e-3,
                      bn_momentum=0.99):
    """ BN + Act + Conv
    :param inputs: 4-D tensor, shape of (batch_size, height, width, channel).
    :param n_filters: int, number of convolution filters.
    :param kernel_size: int, default 3.
    :param rate: int, default 1.
    :param weight_decay: float, default 1e-4.
    :param kernel_initializer: string, default "he_normal".
    :param bn_epsilon: float, default 1e-3.
    :param bn_momentum: float, default 0.99.

    :return: 4-D tensor, shape of (batch_size, height, width, channel).
    """
    x = BatchNormalization(epsilon=bn_epsilon, momentum=bn_momentum)(inputs)
    x = Activation("relu")(x)
    x = Conv2D(n_filters, kernel_size, padding="same", activation=None, use_bias=False, dilation_rate=rate,
               kernel_regularizer=l2(weight_decay), kernel_initializer=kernel_initializer)(x)
    return x


def bn_act_convtranspose(inputs,
                         n_filters,
                         kernel_size=3,
                         scale=2,
                         weight_decay=1e-4,
                         kernel_initializer="he_normal",
                         bn_epsilon=1e-3,
                         bn_momentum=0.99):
    """ BN + Act + Transpose Conv
        :param inputs: 4-D tensor, shape of (batch_size, height, width, channel).
        :param n_filters: int, number of convolution filters.
        :param kernel_size: int, default 3.
        :param scale: int, default 2.
        :param weight_decay: float, default 1e-4.
        :param kernel_initializer: string, default "he_normal".
        :param bn_epsilon: float, default 1e-3.
        :param bn_momentum: float, default 0.99.

        :return: 4-D tensor, shape of (batch_size, height, width, channel).
        """
    x = BatchNormalization(epsilon=bn_epsilon, momentum=bn_momentum)(inputs)
    x = Activation("relu")(x)
    x = Conv2DTranspose(n_filters, kernel_size, padding="same", activation=None, use_bias=False,
                        strides=(scale, scale), kernel_regularizer=l2(weight_decay),
                        kernel_initializer=kernel_initializer)(x)
    return x
