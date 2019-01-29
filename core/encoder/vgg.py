from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation, Dropout
from keras.engine import Input
from keras.models import Model


def ConvBlock(inputs, n_filters):
    x = inputs
    x = Conv2D(n_filters, (3, 3), strides=(1, 1), padding="same", use_bias=False, activation=None, kernel_initializer="he_uniform")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x


def VGG_16(input_shape=(256, 256, 3)):
    input_x = Input(shape=input_shape)
    x = input_x

    x = ConvBlock(x, 64)
    x = ConvBlock(x, 64)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = ConvBlock(x, 128)
    x = ConvBlock(x, 128)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = ConvBlock(x, 256)
    x = ConvBlock(x, 256)
    x = ConvBlock(x, 256)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = ConvBlock(x, 512)
    x = ConvBlock(x, 512)
    x = ConvBlock(x, 512)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = ConvBlock(x, 512)
    x = ConvBlock(x, 512)
    x = ConvBlock(x, 512)
    #x = MaxPooling2D(pool_size=(2, 2))(x)

    return Model(input_x, x)


def VGG_19(input_shape=(256, 256, 3)):
    input_x = Input(shape=input_shape)
    x = input_x

    x = ConvBlock(x, 64)
    x = ConvBlock(x, 64)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = ConvBlock(x, 128)
    x = ConvBlock(x, 128)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = ConvBlock(x, 256)
    x = ConvBlock(x, 256)
    x = ConvBlock(x, 256)
    x = ConvBlock(x, 256)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = ConvBlock(x, 512)
    x = ConvBlock(x, 512)
    x = ConvBlock(x, 512)
    x = ConvBlock(x, 512)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = ConvBlock(x, 512)
    x = ConvBlock(x, 512)
    x = ConvBlock(x, 512)
    x = ConvBlock(x, 512)
    #x = MaxPooling2D(pool_size=(2, 2))(x)

    return Model(input_x, x)

