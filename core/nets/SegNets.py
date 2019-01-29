from keras.layers.convolutional import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Input
from keras.layers.normalization import BatchNormalization
from keras.models import Model


def ConvBNPool(inputs, ConvBN_count=2, filters=64, pooling=True):
    x = inputs
    for i in range(ConvBN_count):
        x = Conv2D(filters, (3, 3), padding='same', activation='relu')(x)
        x = BatchNormalization()(x)

    if pooling:
        x = MaxPooling2D()(x)
    return x


def SegNet_Basic(input_shape=(256, 256, 3), n_class=1, init_filters=64):
    input_x = Input(shape=input_shape)

    # encoder
    x = ConvBNPool(input_x, 2, init_filters * 1)
    x = ConvBNPool(x, 2, init_filters * 2)
    x = ConvBNPool(x, 3, init_filters * 4)
    x = ConvBNPool(x, 3, init_filters * 8)
    x = ConvBNPool(x, 3, init_filters * 8)

    # upsampling
    x = UpSampling2D()(x)
    x = ConvBNPool(x, 3, init_filters * 8, pooling=False)
    x = UpSampling2D()(x)
    x = ConvBNPool(x, 3, init_filters * 8, pooling=False)
    x = UpSampling2D()(x)
    x = ConvBNPool(x, 3, init_filters * 4, pooling=False)
    x = UpSampling2D()(x)
    x = ConvBNPool(x, 2, init_filters * 2, pooling=False)
    x = UpSampling2D()(x)
    x = ConvBNPool(x, 2, init_filters * 1, pooling=False)

    output = Conv2D(n_class, (1, 1), strides=(1, 1), padding='same', activation="sigmoid")(x)

    segnet_model = Model(input_x, output)

    return segnet_model