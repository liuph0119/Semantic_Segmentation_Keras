from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation
from keras.layers.merge import Concatenate, Add
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.regularizers import l2

from ..encoder import build_encoder, scope_table
from ..utils.net_utils import BilinearUpSampling
from .unets import convolutional_residual_block


def spatial_residual_inception(inputs, base_filters=256):
    x_short = Conv2D(base_filters + 64, (1, 1), use_bias=False, activation=None)(inputs)
    x_short = Activation("relu")(x_short)

    x_conv1x1 = Conv2D(base_filters + 64, (1, 1), use_bias=False, activation=None)(x_short)

    x_conv3x3 = Conv2D(base_filters, (1, 1), use_bias=False, activation=None)(x_short)
    x_conv3x3 = Conv2D(base_filters + 32, (1, 3), padding="same", use_bias=False, activation=None)(x_conv3x3)
    x_conv3x3 = Conv2D(base_filters + 64, (3, 1), padding="same", use_bias=False, activation=None)(x_conv3x3)

    x_conv7x7 = Conv2D(base_filters, (1, 1), use_bias=False, activation=None)(x_short)
    x_conv7x7 = Conv2D(base_filters + 32, (1, 7), padding="same", use_bias=False, activation=None)(x_conv7x7)
    x_conv7x7 = Conv2D(base_filters + 64, (7, 1), padding="same", use_bias=False, activation=None)(x_conv7x7)

    x_conv = Concatenate()([x_conv1x1, x_conv3x3, x_conv7x7])
    x_conv = Conv2D(base_filters+64, (1, 1), use_bias=False, activation=None)(x_conv)

    x = Add()([x_short, x_conv])
    return Activation("relu")(x)


def sri_net(input_shape,
            n_class,
            encoder_name="resnet_v2_101",
            encoder_weights=None,
            weight_decay=1e-4,
            kernel_initializer="he_normal",
            bn_epsilon=1e-3,
            bn_momentum=0.99):
    """ spatial residual inception net.
    :param input_shape: tuple, i.e., (height, width, channel).
    :param n_class: int, number of classes, at least 2.
    :param encoder_name: string, default "resnet_v2_101".
    :param encoder_weights: string, path of weights.
    :param weight_decay: float, default 1e-4.
    :param kernel_initializer: string, default "he_normal".
    :param bn_epsilon: float, default 1e-3.
    :param bn_momentum: float, default 0.99.

    :return: a Keras Model instance.
    """
    encoder = build_encoder(input_shape=input_shape, encoder_name=encoder_name, encoder_weights=encoder_weights,
                            weight_decay=weight_decay, kernel_initializer=kernel_initializer,
                            bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)
    p2 = encoder.get_layer(scope_table[encoder_name]["pool2"]).output  # 64 channels
    p3 = encoder.get_layer(scope_table[encoder_name]["pool3"]).output  # 256 channels
    p4 = encoder.get_layer(scope_table[encoder_name]["pool4"]).output  # 512 channels

    # 32->64
    net = spatial_residual_inception_v2(p4, 192, weight_decay=weight_decay, kernel_initializer=kernel_initializer)
    net = BilinearUpSampling(target_size=(input_shape[0] // 8, input_shape[0] // 8))(net)

    # 64->128
    p3 = Conv2D(int(net.shape[-1]//4), (1, 1), use_bias=False, activation=None,
                kernel_regularizer=l2(weight_decay), kernel_initializer=kernel_initializer)(p3)
    p3 = BatchNormalization(epsilon=bn_epsilon, momentum=bn_momentum)(p3)
    p3 = Activation("relu")(p3)
    net = Concatenate()([net, p3])
    net = spatial_residual_inception_v2(net, 192, weight_decay=weight_decay, kernel_initializer=kernel_initializer)
    net = BilinearUpSampling(target_size=(input_shape[0] // 4, input_shape[0] // 4))(net)

    # 128->512
    p2 = Conv2D(int(net.shape[-1]//4), (1, 1), use_bias=False, activation=None,
                kernel_regularizer=l2(weight_decay), kernel_initializer=kernel_initializer)(p2)
    p2 = BatchNormalization(epsilon=bn_epsilon, momentum=bn_momentum)(p2)
    p2 = Activation("relu")(p2)
    net = Concatenate()([net, p2])
    net = spatial_residual_inception_v2(net, 192, weight_decay=weight_decay, kernel_initializer=kernel_initializer)
    net = BilinearUpSampling(target_size=(input_shape[0], input_shape[1]))(net)

    net = Conv2D(256, (3, 3), use_bias=False, activation=None, padding="same",
                 kernel_regularizer=l2(weight_decay), kernel_initializer=kernel_initializer)(net)
    net = BatchNormalization(epsilon=bn_epsilon, momentum=bn_momentum)(net)
    net = Activation("relu")(net)

    output = Conv2D(n_class, (1, 1), activation=None,
                    kernel_regularizer=l2(weight_decay), kernel_initializer=kernel_initializer)(net)
    output = Activation("softmax")(output)

    return Model(encoder.input, output)


def spatial_residual_inception_v2(inputs, base_filters=192, weight_decay=1e-4, kernel_initializer="he_normal"):
    x_short = Activation("relu")(inputs)
    x_short = Conv2D(base_filters + 64, (1, 1), use_bias=False, activation=None,
                     kernel_regularizer=l2(weight_decay), kernel_initializer=kernel_initializer)(x_short)

    # 1x1
    x_conv1x1 = Conv2D(base_filters + 64, (1, 1), use_bias=False, activation=None,
                       kernel_regularizer=l2(weight_decay), kernel_initializer=kernel_initializer)(x_short)

    x_conv3x3_1 = Conv2D(base_filters + 32, (1, 3), padding="same", use_bias=False, activation=None, dilation_rate=1,
                         kernel_regularizer=l2(weight_decay), kernel_initializer=kernel_initializer)(x_short)
    x_conv3x3_1 = Conv2D(base_filters + 64, (3, 1), padding="same", use_bias=False, activation=None, dilation_rate=1,
                         kernel_regularizer=l2(weight_decay), kernel_initializer=kernel_initializer)(x_conv3x3_1)

    x_conv3x3_5 = Conv2D(base_filters + 32, (1, 3), padding="same", use_bias=False, activation=None, dilation_rate=2,
                         kernel_regularizer=l2(weight_decay), kernel_initializer=kernel_initializer)(x_short)
    x_conv3x3_5 = Conv2D(base_filters + 64, (3, 1), padding="same", use_bias=False, activation=None, dilation_rate=2,
                         kernel_regularizer=l2(weight_decay), kernel_initializer=kernel_initializer)(x_conv3x3_5)

    x_conv5x5_1 = Conv2D(base_filters + 32, (1, 3), padding="same", use_bias=False, activation=None, dilation_rate=5,
                         kernel_regularizer=l2(weight_decay), kernel_initializer=kernel_initializer)(x_short)
    x_conv5x5_1 = Conv2D(base_filters + 64, (3, 1), padding="same", use_bias=False, activation=None, dilation_rate=5,
                         kernel_regularizer=l2(weight_decay), kernel_initializer=kernel_initializer)(x_conv5x5_1)

    x_conv5x5_5 = Conv2D(base_filters + 32, (1, 3), padding="same", use_bias=False, activation=None, dilation_rate=7,
                         kernel_regularizer=l2(weight_decay), kernel_initializer=kernel_initializer)(x_short)
    x_conv5x5_5 = Conv2D(base_filters + 64, (3, 1), padding="same", use_bias=False, activation=None, dilation_rate=7,
                         kernel_regularizer=l2(weight_decay), kernel_initializer=kernel_initializer)(x_conv5x5_5)

    x_conv = Concatenate()([x_conv1x1, x_conv3x3_1, x_conv3x3_5, x_conv5x5_1, x_conv5x5_5])

    x_short = Conv2D(base_filters + 64, (1, 1), use_bias=False, activation=None,
                     kernel_regularizer=l2(weight_decay), kernel_initializer=kernel_initializer)(x_short)
    x_conv = Conv2D(base_filters + 64, (1, 1), use_bias=False, activation=None,
                    kernel_regularizer=l2(weight_decay), kernel_initializer=kernel_initializer)(x_conv)

    x = Add()([x_short, x_conv])
    return Activation("relu")(x)


def sri_net_v2(input_shape,
               n_class,
               weight_decay=1e-4,
               kernel_initializer="he_normal",
               bn_epsilon=1e-3,
               bn_momentum=0.99,
               init_filters=64,
               dropout=0.5
               ):
    from keras.layers import Input, MaxPooling2D, SpatialDropout2D
    input_x = Input(shape=input_shape)
    x = BatchNormalization(epsilon=bn_epsilon, momentum=bn_momentum)(input_x)

    conv1 = convolutional_residual_block(x, init_filters*1, weight_decay=weight_decay,
                                         kernel_initializer=kernel_initializer, bn_epsilon=bn_epsilon,
                                         bn_momentum=bn_momentum)
    pool1 = MaxPooling2D((2, 2))(conv1)
    pool1 = SpatialDropout2D(dropout / 2)(pool1)

    conv2 = convolutional_residual_block(pool1, init_filters*2, weight_decay=weight_decay,
                                         kernel_initializer=kernel_initializer, bn_epsilon=bn_epsilon,
                                         bn_momentum=bn_momentum)
    pool2 = MaxPooling2D((2, 2))(conv2)
    pool2 = SpatialDropout2D(dropout)(pool2)

    conv3 = convolutional_residual_block(pool2, init_filters*4, weight_decay=weight_decay,
                                         kernel_initializer=kernel_initializer, bn_epsilon=bn_epsilon,
                                         bn_momentum=bn_momentum)
    pool3 = MaxPooling2D((2, 2))(conv3)
    pool3 = SpatialDropout2D(dropout)(pool3)

    conv4 = convolutional_residual_block(pool3, init_filters*8, weight_decay=weight_decay,
                                         kernel_initializer=kernel_initializer, bn_epsilon=bn_epsilon,
                                         bn_momentum=bn_momentum)
    pool4 = MaxPooling2D((2, 2))(conv4)
    pool4 = SpatialDropout2D(dropout)(pool4)

    conv5 = convolutional_residual_block(pool4, init_filters*16, weight_decay=weight_decay,
                                         kernel_initializer=kernel_initializer, bn_epsilon=bn_epsilon,
                                         bn_momentum=bn_momentum)
    conv5 = (conv5, 256, [2, 5, 7])

    dconv4 = BilinearUpSampling(target_size=(input_shape[0]//8, input_shape[1]//8))(conv5)
    dconv4 = Concatenate()([conv4, dconv4])
    dconv4 = SpatialDropout2D(dropout)(dconv4)
    dconv4 = convolutional_residual_block(dconv4, init_filters*8, weight_decay=weight_decay,
                                          kernel_initializer=kernel_initializer, bn_epsilon=bn_epsilon,
                                          bn_momentum=bn_momentum)

    dconv3 = BilinearUpSampling(target_size=(input_shape[0]//4, input_shape[1]//4))(dconv4)
    dconv3 = Concatenate()([conv3, dconv3])
    dconv3 = SpatialDropout2D(dropout)(dconv3)
    dconv3 = convolutional_residual_block(dconv3, init_filters*4, weight_decay=weight_decay,
                                          kernel_initializer=kernel_initializer, bn_epsilon=bn_epsilon,
                                          bn_momentum=bn_momentum)

    dconv2 = BilinearUpSampling(target_size=(input_shape[0]//2, input_shape[1]//2))(dconv3)
    dconv2 = Concatenate()([conv2, dconv2])
    dconv2 = SpatialDropout2D(dropout)(dconv2)
    dconv2 = convolutional_residual_block(dconv2, init_filters*2, weight_decay=weight_decay,
                                          kernel_initializer=kernel_initializer, bn_epsilon=bn_epsilon,
                                          bn_momentum=bn_momentum)

    dconv1 = BilinearUpSampling(target_size=(input_shape[0], input_shape[1]))(dconv2)
    dconv1 = Concatenate()([conv1, dconv1])
    dconv1 = SpatialDropout2D(dropout)(dconv1)
    dconv1 = convolutional_residual_block(dconv1, init_filters*1, weight_decay=weight_decay,
                                          kernel_initializer=kernel_initializer, bn_epsilon=bn_epsilon,
                                          bn_momentum=bn_momentum)

    output = Conv2D(n_class, (1, 1), activation=None,
                    kernel_initializer=kernel_initializer, kernel_regularizer=l2(weight_decay))(dconv1)
    output = Activation("softmax")(output)

    return Model(input_x, output)
