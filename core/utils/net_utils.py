"""
    Script: net_utils.py
    Author: Penghua Liu
    Date: 2019-01-13
    Email: liuphhhh@foxmail.com
    Functions: some util functions for building FCNs, including:
        crop: crop tensor function for FCN_8s
        ResizeImageLayer: resize image size layer
        resize_to: resize image size to a given size
        resize_scale: resize image size according to a scale
        GlobalAveragePooling2D_keepdim: global average pooling that keeps dims, used in ASPP
        ResidualBlock: a common residual block, might be useless.
        ConvUpscaleBlock: Conv2DTranspose with a BN and a ReLU
        ConvBlock: Conv2D with a BN and a ReLU
        AtrousSpatialPyramidPooling: ASPP

"""

from keras.models import Model
from keras.layers import Activation, BatchNormalization, Add, Concatenate, Dropout
from keras.layers.convolutional import Conv2D, SeparableConv2D, Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Concatenate
from keras.layers import Cropping2D
from keras.regularizers import l2
from keras.engine import Layer
import tensorflow as tf
from keras.layers.pooling import _GlobalPooling2D, GlobalAveragePooling2D
from keras import layers
import keras.backend as K


def crop(o1, o2, i):
    """ crop the output
    :param o1: output 1
    :param o2: output 2
    :param i: input
    :return: the processed outputs
    """
    o_shape2 = Model(i, o2).output_shape
    outputHeight2 = o_shape2[1]
    outputWidth2 = o_shape2[2]

    o_shape1 = Model(i, o1).output_shape
    outputHeight1 = o_shape1[1]
    outputWidth1 = o_shape1[2]

    cx = abs(outputWidth1 - outputWidth2)
    cy = abs(outputHeight2 - outputHeight1)

    if outputWidth1 > outputWidth2:
        o1 = Cropping2D(cropping=((0, 0), (0, cx)))(o1)
    else:
        o2 = Cropping2D(cropping=((0, 0), (0, cx)))(o2)

    if outputHeight1 > outputHeight2:
        o1 = Cropping2D(cropping=((0, cy), (0, 0)))(o1)
    else:
        o2 = Cropping2D(cropping=((0, cy), (0, 0)))(o2)

    return o1, o2


class _ResizeImage(Layer):
    def __init__(self, target_size, resize_method,**kwargs):
        super(_ResizeImage, self).__init__(**kwargs)
        self.target_size = target_size
        self.resize_method = resize_method
        pass

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.target_size[0], self.target_size[1], input_shape[-1])

    def _resize_function(self, inputs, resize_method):
        raise NotImplementedError

    def call(self, inputs):
        return self._resize_function(inputs=inputs, resize_method=self.resize_method)

    def get_config(self):
        config = {'target_size': self.target_size, "resize_method": self.resize_method}
        base_config = super(_ResizeImage, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ResizeImageLayer(_ResizeImage):
    # @interfaces.legacy_resizeimage_support
    def __init__(self, target_size=(256, 256, 3), resize_method="linear", **kwargs):
        super(ResizeImageLayer, self).__init__(target_size=target_size, resize_method=resize_method, **kwargs)

    def _resize_function(self, inputs, resize_method):
        if resize_method=="linear":
            return tf.cast(tf.image.resize_bilinear(inputs, self.target_size, align_corners=True), dtype=tf.float32)
        elif resize_method=="cubic":
            return tf.cast(tf.image.resize_bicubic(inputs, self.target_size, align_corners=True), dtype=tf.float32)
        elif resize_method=="area":
            return tf.cast(tf.image.resize_area(inputs, self.target_size, align_corners=True), dtype=tf.float32)
        elif resize_method=="nearest":
            return tf.cast(tf.image.resize_nearest_neighbor(inputs, self.target_size, align_corners=True), dtype=tf.float32)
        else:
            raise KeyError("resize method is not valid! options:['linear', 'cubic', 'area', 'nearest']")


def resize_to(inputs, target_shape):
    return tf.image.resize_bilinear(inputs, size=[target_shape[0], target_shape[1]], align_corners=True)


def resize_scale(inputs, scale=2):
    return tf.image.resize_bilinear(inputs, size=[tf.shape(inputs)[1]*scale, tf.shape(inputs)[2]*scale], align_corners=True)


class GlobalAveragePooling2D_keepdim(_GlobalPooling2D):
    def call(self, inputs):
        if self.data_format == 'channels_last':
            return K.mean(inputs, axis=[1, 2], keepdims=True)
        else:
            return K.mean(inputs, axis=[2, 3], keepdims=True)



def ResidualBlock(inputs, filters_list, ksize, skip_connection_type="conv"):
    X = inputs

    # main stream of Conv->BN->ReLU
    for i in range(len(filters_list)-1):
        X = Conv2D(filters_list[i], kernel_size=ksize, strides=1, padding="same")(X)
        X = BatchNormalization()(X)
        X = Activation("relu")(X)

    # the last part has no activation
    X = Conv2D(filters_list[-1], kernel_size=ksize, strides=1, padding="same")(X)
    X = BatchNormalization()(X)

    if skip_connection_type=="conv":    # convolutional block
        short_cut = Conv2D(filters_list[-1], kernel_size=ksize, strides=1, padding="same")(inputs)
        short_cut = BatchNormalization()(short_cut)
        X = Add()([short_cut, X])
        return Activation("relu")(X)
    elif skip_connection_type=="sum":   # identity block
        short_cut = inputs
        X = Add()([short_cut, X])
        return Activation("relu")(X)
    else:                               # no residual connection
        return Activation("relu")(X)


def ConvUpscaleBlock(inputs, n_filters, kernel_size=(3, 3), scale=2):
    x = BatchNormalization()(inputs)
    x = Activation("relu")(x)
    x = Conv2DTranspose(n_filters, kernel_size, padding="same", activation=None,strides=(scale, scale))(x)
    return x


def ConvBlock(inputs, n_filters, kernel_size=(3, 3)):
    x = BatchNormalization()(inputs)
    x = Activation("relu")(x)
    x = Conv2D(n_filters, kernel_size, padding="same", activation=None)(x)
    return x


def AtrousSpatialPyramidPooling(inputs, n_filters=256, rates=[6, 12, 18], imagelevel=True):
    """ASPP consists of (a) one 1×1 convolution and three 3×3 convolutions with rates = (6, 12, 18) when output stride = 16
    (all with 256 filters and batch normalization), and (b) the image-level features as described in the paper
    :param inputs: 4-d tensor, with the shape of [batch_size, height, width, channel]
    :param filters: number of filters, int, default 256
    :param rates: list of dilation rates, default [6, 12, 18]
    :return: the ASPP features
    """
    branch_features = []
    if imagelevel:
        # image level features
        image_feature = GlobalAveragePooling2D_keepdim()(inputs)
        image_feature = Conv2D(n_filters, (1, 1), use_bias=False, activation=None)(image_feature)
        image_feature = Activation("relu")(image_feature)
        image_feature = ResizeImageLayer(target_size=(int(inputs.shape[1]), int(inputs.shape[2])))(image_feature)
        branch_features.append(image_feature)

    # 1×1 conv
    atrous_pool_block_1 = Conv2D(n_filters, (1, 1), padding="same", strides=1, use_bias=False, activation=None)(inputs)
    atrous_pool_block_1 = BatchNormalization()(atrous_pool_block_1)
    atrous_pool_block_1 = Activation("relu")(atrous_pool_block_1)
    branch_features.append(atrous_pool_block_1)
    # atrous_pool_block_1 = BatchNormalization()(atrous_pool_block_1)

    for rate in rates:
        atrous_pool_block_i = Conv2D(n_filters, (3, 3), dilation_rate=rate, padding="same", strides=1, use_bias=False, activation=None)(inputs)
        atrous_pool_block_i = BatchNormalization()(atrous_pool_block_i)
        atrous_pool_block_i = Activation("relu")(atrous_pool_block_i)
        branch_features.append(atrous_pool_block_i)

    # concatenate the multi-scale features and apply a 1×1 conv to reduce depth
    aspp_features = layers.concatenate(branch_features)

    return aspp_features
