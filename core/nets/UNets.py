from keras.engine import Input
from keras.layers.convolutional import Conv2D, Conv2DTranspose, SeparableConv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Concatenate, Add
from keras.models import Model
from keras.regularizers import l2

from ..utils.net_utils import conv_bn_act_block, bn_act_convtranspose


def UNet(input_shape,
         n_class,
         weight_decay=1e-4,
         kernel_initializer="he_normal",
         bn_epsilon=1e-3,
         bn_momentum=0.99,
         init_filters=64,
         dropout=0.5):
    """ Implementation of U-Net for semantic segmentation.
        ref: Ronneberger O , Fischer P , Brox T . U-Net: Convolutional Networks for Biomedical Image Segmentation[J].
             arXiv preprint arXiv: 1505.04597, 2015.
    :param input_shape: tuple, i.e., (width, height, channel).
    :param n_class: int, number of classes, at least 2.
    :param weight_decay: float, default 1e-4.
    :param kernel_initializer: string, default "he_normal".
    :param bn_epsilon: float, default 1e-3.
    :param bn_momentum: float, default 0.99.
    :param init_filters: int, initial filters, default 64.
    :param dropout: float, default 0.5.

    :return: a Keras Model instance.
    """
    input_x = Input(shape=input_shape)
    x = BatchNormalization(epsilon=bn_epsilon, momentum=bn_momentum)(input_x)

    conv1 = Conv2D(init_filters * 1, (3, 3), activation='relu', padding='same',
                   kernel_regularizer=l2(weight_decay), kernel_initializer=kernel_initializer)(x)
    conv1 = Dropout(dropout)(conv1)
    conv1 = Conv2D(init_filters * 1, (3, 3), activation='relu', padding='same',
                   kernel_regularizer=l2(weight_decay), kernel_initializer=kernel_initializer)(conv1)
    pool1 = MaxPooling2D()(conv1)

    conv2 = Conv2D(init_filters * 2, (3, 3), activation='relu', padding='same',
                   kernel_regularizer=l2(weight_decay), kernel_initializer=kernel_initializer)(pool1)
    conv2 = Dropout(dropout)(conv2)
    conv2 = Conv2D(init_filters * 2, (3, 3), activation='relu', padding='same',
                   kernel_regularizer=l2(weight_decay), kernel_initializer=kernel_initializer)(conv2)
    pool2 = MaxPooling2D()(conv2)

    conv3 = Conv2D(init_filters * 4, (3, 3), activation='relu', padding='same',
                   kernel_regularizer=l2(weight_decay), kernel_initializer=kernel_initializer)(pool2)
    conv3 = Dropout(dropout)(conv3)
    conv3 = Conv2D(init_filters * 4, (3, 3), activation='relu', padding='same',
                   kernel_regularizer=l2(weight_decay), kernel_initializer=kernel_initializer)(conv3)
    pool3 = MaxPooling2D()(conv3)

    conv4 = Conv2D(init_filters * 8, (3, 3), activation='relu', padding='same',
                   kernel_regularizer=l2(weight_decay), kernel_initializer=kernel_initializer)(pool3)
    conv4 = Dropout(dropout)(conv4)
    conv4 = Conv2D(init_filters * 8, (3, 3), activation='relu', padding='same',
                   kernel_regularizer=l2(weight_decay), kernel_initializer=kernel_initializer)(conv4)
    pool4 = MaxPooling2D()(conv4)

    conv5 = Conv2D(init_filters * 16, (3, 3), activation='relu', padding='same',
                   kernel_regularizer=l2(weight_decay), kernel_initializer=kernel_initializer)(pool4)
    conv5 = Dropout(dropout)(conv5)
    conv5 = Conv2D(init_filters * 16, (3, 3), activation='relu', padding='same',
                   kernel_regularizer=l2(weight_decay), kernel_initializer=kernel_initializer)(conv5)

    up1 = Concatenate()([Conv2DTranspose(init_filters * 8, (3, 3), padding="same", strides=(2, 2),
                                         kernel_regularizer=l2(weight_decay),
                                         kernel_initializer=kernel_initializer)(conv5), conv4])
    conv6 = Conv2D(init_filters * 8, (3, 3), activation='relu', padding='same',
                   kernel_regularizer=l2(weight_decay), kernel_initializer=kernel_initializer)(up1)
    conv6 = Dropout(dropout)(conv6)
    conv6 = Conv2D(init_filters * 8, (3, 3), activation='relu', padding='same',
                   kernel_regularizer=l2(weight_decay), kernel_initializer=kernel_initializer)(conv6)

    up2 = Concatenate()([Conv2DTranspose(init_filters * 4, (3, 3), padding="same", strides=(2, 2),
                                         kernel_regularizer=l2(weight_decay),
                                         kernel_initializer=kernel_initializer)(conv6), conv3])
    conv7 = Conv2D(init_filters * 4, (3, 3), activation='relu', padding='same',
                   kernel_regularizer=l2(weight_decay), kernel_initializer=kernel_initializer)(up2)
    conv7 = Dropout(dropout)(conv7)
    conv7 = Conv2D(init_filters * 4, (3, 3), activation='relu', padding='same',
                   kernel_regularizer=l2(weight_decay), kernel_initializer=kernel_initializer)(conv7)

    up3 = Concatenate()([Conv2DTranspose(init_filters * 2, (3, 3), padding="same", strides=(2, 2),
                                         kernel_regularizer=l2(weight_decay),
                                         kernel_initializer=kernel_initializer)(conv7), conv2])
    conv8 = Conv2D(init_filters * 2, (3, 3), activation='relu', padding='same',
                   kernel_regularizer=l2(weight_decay), kernel_initializer=kernel_initializer)(up3)
    conv8 = Dropout(dropout)(conv8)
    conv8 = Conv2D(init_filters * 2, (3, 3), activation='relu', padding='same',
                   kernel_regularizer=l2(weight_decay), kernel_initializer=kernel_initializer)(conv8)

    up4 = Concatenate()([Conv2DTranspose(init_filters, (3, 3), padding="same", strides=(2, 2),
                                         kernel_regularizer=l2(weight_decay),
                                         kernel_initializer=kernel_initializer)(conv8), conv1])
    conv9 = Conv2D(init_filters, (3, 3), activation='relu', padding='same',
                   kernel_regularizer=l2(weight_decay), kernel_initializer=kernel_initializer)(up4)
    conv9 = Conv2D(init_filters, (3, 3), activation='relu', padding='same',
                   kernel_regularizer=l2(weight_decay), kernel_initializer=kernel_initializer)(conv9)

    output = Conv2D(n_class, (1, 1), activation=None,
                    kernel_regularizer=l2(weight_decay), kernel_initializer=kernel_initializer)(conv9)
    output = Activation("softmax")(output)

    return Model(input_x, output)


################################################ ResUNet ################
# def convolution_block(x, filters, size, strides=(1, 1), padding='same', activation=True):
#     x = Conv2D(filters, size, strides=strides, padding=padding)(x)
#     if activation == True:
#         x = BatchNormalization()(x)
#         x = Activation("relu")(x)
#     return x
#
#
# def residual_block(blockInput, num_filters=16, batch_activate=False):
#     x = BatchNormalization()(blockInput)
#     x = Activation("relu")(x)
#     x = convolution_block(x, num_filters, (3, 3))
#     x = convolution_block(x, num_filters, (3, 3), activation=False)
#     x = Add()([x, blockInput])
#     if batch_activate:
#         x = BatchNormalization()(x)
#         x = Activation("relu")(x)
#     return x


def convolutional_residual_block(inputs, n_filters, weight_decay=1e-4, kernel_initializer="he_normal", bn_epsilon=1e-3, bn_momentum=0.99):
    x = conv_bn_act_block(inputs, n_filters, weight_decay, kernel_initializer, bn_epsilon, bn_momentum)
    x = conv_bn_act_block(x, n_filters, weight_decay, kernel_initializer, bn_epsilon, bn_momentum)
    x = Conv2D(n_filters, kernel_size=(3, 3), padding="same", activation=None, use_bias=False,
               kernel_regularizer=l2(weight_decay), kernel_initializer=kernel_initializer)(x)
    x = Add()([inputs, x])
    _x = x
    x = conv_bn_act_block(_x, n_filters, weight_decay, kernel_initializer, bn_epsilon, bn_momentum)
    x = conv_bn_act_block(x, n_filters, weight_decay, kernel_initializer, bn_epsilon, bn_momentum)
    x = Conv2D(n_filters, kernel_size=(3, 3), padding="same", activation=None, use_bias=False,
               kernel_regularizer=l2(weight_decay), kernel_initializer=kernel_initializer)(x)
    x = Add()([_x, x])
    x = BatchNormalization(epsilon=bn_epsilon, momentum=bn_momentum)(x)
    x = Activation("relu")

    return x



def ResUNet(input_shape,
            n_class,
            weight_decay=1e-4,
            kernel_initializer="he_normal",
            bn_epsilon=1e-3,
            bn_momentum=0.99,
            init_filters=64,
            dropout=0.5):
    """ modification of U-Net.
        replace the Conv+BN+Act with Residual Convolutions.
    :param input_shape: tuple, i.e., (width, height, channel).
    :param n_class: int, number of classes, at least 2.
    :param weight_decay: float, default 1e-4.
    :param kernel_initializer: string, default "he_normal".
    :param bn_epsilon: float, default 1e-3.
    :param bn_momentum: float, default 0.99.
    :param init_filters: int, initial filters, default 64.
    :param dropout: float, default 0.5.

    :return: a Keras Model instance.
    """
    input_x = Input(shape=input_shape)
    x = BatchNormalization(epsilon=bn_epsilon, momentum=bn_momentum)(input_x)

    conv1 = convolutional_residual_block(x, init_filters*1, weight_decay,
                                         kernel_initializer, bn_epsilon, bn_momentum)
    pool1 = MaxPooling2D((2, 2))(conv1)
    pool1 = Dropout(dropout / 2)(pool1)

    conv2 = convolutional_residual_block(pool1, init_filters*2, weight_decay,
                                         kernel_initializer, bn_epsilon, bn_momentum)
    pool2 = MaxPooling2D((2, 2))(conv2)
    pool2 = Dropout(dropout)(pool2)

    conv3 = convolutional_residual_block(pool2, init_filters*4, weight_decay,
                                         kernel_initializer, bn_epsilon, bn_momentum)
    pool3 = MaxPooling2D((2, 2))(conv3)
    pool3 = Dropout(dropout)(pool3)

    conv4 = convolutional_residual_block(pool3, init_filters*8, weight_decay,
                                         kernel_initializer, bn_epsilon, bn_momentum)
    pool4 = MaxPooling2D((2, 2))(conv4)
    pool4 = Dropout(dropout)(pool4)

    convm = convolutional_residual_block(pool4, init_filters*16, weight_decay,
                                         kernel_initializer, bn_epsilon, bn_momentum)

    deconv4 = Conv2DTranspose(init_filters * 8, (3, 3), strides=(2, 2), padding="same",
                              kernel_regularizer=l2(weight_decay), kernel_initializer=kernel_initializer)(convm)
    uconv4 = Concatenate()([deconv4, conv4])
    uconv4 = Dropout(dropout)(uconv4)
    uconv4 = convolutional_residual_block(uconv4, init_filters*8, weight_decay,
                                          kernel_initializer, bn_epsilon, bn_momentum)

    deconv3 = Conv2DTranspose(init_filters * 4, (3, 3), strides=(2, 2), padding="same",
                              kernel_regularizer=l2(weight_decay), kernel_initializer=kernel_initializer)(uconv4)
    uconv3 = Concatenate()([deconv3, conv3])
    uconv3 = Dropout(dropout)(uconv3)
    uconv3 = convolutional_residual_block(uconv3, init_filters*4, weight_decay,
                                          kernel_initializer, bn_epsilon, bn_momentum)

    deconv2 = Conv2DTranspose(init_filters * 2, (3, 3), strides=(2, 2), padding="same",
                              kernel_regularizer=l2(weight_decay), kernel_initializer=kernel_initializer)(uconv3)
    uconv2 = Concatenate()([deconv2, conv2])
    uconv2 = Dropout(dropout)(uconv2)
    uconv2 = convolutional_residual_block(uconv2, init_filters*2, weight_decay,
                                          kernel_initializer, bn_epsilon, bn_momentum)


    deconv1 = Conv2DTranspose(init_filters * 1, (3, 3), strides=(2, 2), padding="same",
                              kernel_regularizer=l2(weight_decay), kernel_initializer=kernel_initializer)(uconv2)
    uconv1 = Concatenate()([deconv1, conv1])
    uconv1 = Dropout(dropout)(uconv1)
    uconv1 = convolutional_residual_block(uconv1, init_filters*1, weight_decay,
                                          kernel_initializer, bn_epsilon, bn_momentum)

    output = Conv2D(n_class, (1, 1), padding="same", activation=None,
                    kernel_regularizer=l2(weight_decay), kernel_initializer=kernel_initializer)(uconv1)
    output = Activation("softmax")(output)

    return Model(input_x, output)


# # # ===========================================================================================================
def DepthwiseSeparableConvBlock(inputs,
                                n_filters,
                                weight_decay=1e-4,
                                kernel_initializer="he_normal",
                                bn_epsilon=1e-3,
                                bn_momentum=0.99):
    """ Depthwise separable convolutional block
    :param inputs: 4-D tensor, shape of (batch_size, hwight, width, channel).
    :param n_filters: int, number of filters.
    :param weight_decay: float, default 1e-4.
    :param kernel_initializer: string, default "he_normal".
    :param bn_epsilon: float, default 1e-3.
    :param bn_momentum: float, default 0.99.

    :return: 4-D tensor, shape of (batch_size, height, width, channel).
    """
    x = SeparableConv2D(inputs, (3, 3), activation=None, padding="same", depth_multiplier=1,
                        kernel_regularizer=l2(weight_decay), kernel_initializer=kernel_initializer)(inputs)
    x = BatchNormalization(epsilon=bn_epsilon, momentum=bn_momentum)(x)
    x = Activation("relu")(x)
    x = Conv2D(n_filters, (1, 1), activation=None,
               kernel_regularizer=l2(weight_decay), kernel_initializer=kernel_initializer)(x)
    x = BatchNormalization(epsilon=bn_epsilon, momentum=bn_momentum)(x)
    x = Activation("relu")(x)
    return x


def MobileUNet(input_shape,
               n_class,
               weight_decay=1e-4,
               kernel_initializer="he_normal",
               bn_epsilon=1e-3,
               bn_momentum=0.99,
               preset_model="MobileUNet-Skip"):
    """
    :param input_shape: 3-D tuple, i.e., (height, width, channel).
    :param n_class: int, number of classes, at least 2.
    :param weight_decay: float, default 1e-4.
    :param kernel_initializer: string, default "he_normal".
    :param bn_epsilon: float, default 1e-3.
    :param bn_momentum: float, default 0.99.
    :param preset_model: string, "MobileUNet-Skip" or "MobileUNet".

    :return: a Keras Model instance.
    """
    if preset_model == "MobileUNet":
        has_skip = False
    elif preset_model == "MobileUNet-Skip":
        has_skip = True
    else:
        raise ValueError(
            "Unsupported MobileUNet model '%s'. This function only supports MobileUNet and MobileUNet-Skip" % (
                preset_model))

    input_x = Input(shape=input_shape)
    x = BatchNormalization(epsilon=bn_epsilon, momentum=bn_momentum)(input_x)

    x = conv_bn_act_block(x, 64, weight_decay=weight_decay,
                          kernel_initializer=kernel_initializer, bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)
    x = DepthwiseSeparableConvBlock(x, 64, weight_decay=weight_decay, kernel_initializer=kernel_initializer,
                                    bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)
    x = MaxPooling2D()(x)
    skip_1 = x

    x = DepthwiseSeparableConvBlock(x, 128, weight_decay=weight_decay, kernel_initializer=kernel_initializer,
                                    bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)
    x = DepthwiseSeparableConvBlock(x, 128, weight_decay=weight_decay, kernel_initializer=kernel_initializer,
                                    bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)
    x = MaxPooling2D()(x)
    skip_2 = x

    x = DepthwiseSeparableConvBlock(x, 256, weight_decay=weight_decay, kernel_initializer=kernel_initializer,
                                    bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)
    x = DepthwiseSeparableConvBlock(x, 256, weight_decay=weight_decay, kernel_initializer=kernel_initializer,
                                    bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)
    x = DepthwiseSeparableConvBlock(x, 256, weight_decay=weight_decay, kernel_initializer=kernel_initializer,
                                    bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)
    x = MaxPooling2D()(x)
    skip_3 = x

    x = DepthwiseSeparableConvBlock(x, 512, weight_decay=weight_decay, kernel_initializer=kernel_initializer,
                                    bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)
    x = DepthwiseSeparableConvBlock(x, 512, weight_decay=weight_decay, kernel_initializer=kernel_initializer,
                                    bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)
    x = DepthwiseSeparableConvBlock(x, 512, weight_decay=weight_decay, kernel_initializer=kernel_initializer,
                                    bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)
    x = MaxPooling2D()(x)
    skip_4 = x

    x = DepthwiseSeparableConvBlock(x, 512, weight_decay=weight_decay, kernel_initializer=kernel_initializer,
                                    bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)
    x = DepthwiseSeparableConvBlock(x, 512, weight_decay=weight_decay, kernel_initializer=kernel_initializer,
                                    bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)
    x = DepthwiseSeparableConvBlock(x, 512, weight_decay=weight_decay, kernel_initializer=kernel_initializer,
                                    bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)
    x = MaxPooling2D()(x)

    x = bn_act_convtranspose(x, 512, kernel_size=3, scale=2, weight_decay=weight_decay,
                             kernel_initializer=kernel_initializer, bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)
    x = DepthwiseSeparableConvBlock(x, 512, weight_decay=weight_decay, kernel_initializer=kernel_initializer,
                                    bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)
    x = DepthwiseSeparableConvBlock(x, 512, weight_decay=weight_decay, kernel_initializer=kernel_initializer,
                                    bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)
    x = DepthwiseSeparableConvBlock(x, 512, weight_decay=weight_decay, kernel_initializer=kernel_initializer,
                                    bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)
    if has_skip:
        x = Add()([x, skip_4])

    x = bn_act_convtranspose(x, 512, kernel_size=3, scale=2, weight_decay=weight_decay,
                             kernel_initializer=kernel_initializer, bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)
    x = DepthwiseSeparableConvBlock(x, 512, weight_decay=weight_decay, kernel_initializer=kernel_initializer,
                                    bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)
    x = DepthwiseSeparableConvBlock(x, 512, weight_decay=weight_decay, kernel_initializer=kernel_initializer,
                                    bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)
    x = DepthwiseSeparableConvBlock(x, 256, weight_decay=weight_decay, kernel_initializer=kernel_initializer,
                                    bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)
    if has_skip:
        x = Add()([x, skip_3])

    x = bn_act_convtranspose(x, 256, kernel_size=3, scale=2, weight_decay=weight_decay,
                             kernel_initializer=kernel_initializer, bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)
    x = DepthwiseSeparableConvBlock(x, 256, weight_decay=weight_decay, kernel_initializer=kernel_initializer,
                                    bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)
    x = DepthwiseSeparableConvBlock(x, 256, weight_decay=weight_decay, kernel_initializer=kernel_initializer,
                                    bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)
    x = DepthwiseSeparableConvBlock(x, 128, weight_decay=weight_decay, kernel_initializer=kernel_initializer,
                                    bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)
    if has_skip:
        x = Add()([x, skip_2])

    x = bn_act_convtranspose(x, 128, kernel_size=3, scale=2, weight_decay=weight_decay,
                             kernel_initializer=kernel_initializer, bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)
    x = DepthwiseSeparableConvBlock(x, 128, weight_decay=weight_decay, kernel_initializer=kernel_initializer,
                                    bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)
    x = DepthwiseSeparableConvBlock(x, 128, weight_decay=weight_decay, kernel_initializer=kernel_initializer,
                                    bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)
    x = DepthwiseSeparableConvBlock(x, 64, weight_decay=weight_decay, kernel_initializer=kernel_initializer,
                                    bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)
    if has_skip:
        x = Add()([x, skip_1])

    x = bn_act_convtranspose(x, 64, kernel_size=3, scale=2, weight_decay=weight_decay,
                             kernel_initializer=kernel_initializer, bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)
    x = DepthwiseSeparableConvBlock(x, 64, weight_decay=weight_decay, kernel_initializer=kernel_initializer,
                                    bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)
    x = DepthwiseSeparableConvBlock(x, 64, weight_decay=weight_decay, kernel_initializer=kernel_initializer,
                                    bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)

    x = Conv2D(n_class, (1, 1), activation=None,
               kernel_regularizer=l2(weight_decay), kernel_initializer=kernel_initializer)(x)
    output = Activation("softmax")(x)

    return Model(input_x, output)
