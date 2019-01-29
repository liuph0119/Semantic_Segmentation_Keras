from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation
from keras.layers.merge import Concatenate
from keras.models import Model

from ..utils.net_utils import ResizeImageLayer, ConvBlock, ConvUpscaleBlock
from ..encoder import build_encoder, scope_table


def DilatedConvBlock(inputs, n_filters, rate=1, kernel_size=(3, 3)):
    x = Activation("relu")(inputs)
    x = BatchNormalization()(x)
    x = Conv2D(n_filters, kernel_size=kernel_size, padding="same", dilation_rate=rate, activation=None)(x)

    return x


def DenseASPP(input_shape=(256, 256, 3), n_class=1, encoder_name="resnet50"):
    encoder = build_encoder(input_shape, encoder_name=encoder_name)

    init_features = encoder.get_layer(scope_table[encoder_name]["pool3"]).output

    ### First block, rate = 3
    d_3_features = DilatedConvBlock(init_features, n_filters=256, kernel_size=(1, 1))
    d_3 = DilatedConvBlock(d_3_features, n_filters=64, rate=3, kernel_size=(3, 3))

    ### Second block, rate = 6
    d_4 = Concatenate()([init_features, d_3])
    d_4 = DilatedConvBlock(d_4, n_filters=256, kernel_size=(1, 1))
    d_4 = DilatedConvBlock(d_4, n_filters=64, rate=6, kernel_size=(3, 3))

    ### Third block, rate = 12
    d_5 = Concatenate()([init_features, d_3, d_4])
    d_5 = DilatedConvBlock(d_5, n_filters=256, kernel_size=(1, 1))
    d_5 = DilatedConvBlock(d_5, n_filters=64, rate=12, kernel_size=(3, 3))

    ### Fourth block, rate = 18
    d_6 = Concatenate()([init_features, d_3, d_4, d_5])
    d_6 = DilatedConvBlock(d_6, n_filters=256, kernel_size=(1, 1))
    d_6 = DilatedConvBlock(d_6, n_filters=64, rate=18, kernel_size=(3, 3))

    ### Fifth block, rate = 24
    d_7 = Concatenate()([init_features, d_3, d_4, d_5, d_6])
    d_7 = DilatedConvBlock(d_7, n_filters=256, kernel_size=(1, 1))
    d_7 = DilatedConvBlock(d_7, n_filters=64, rate=24, kernel_size=(3, 3))

    full_block = Concatenate()([init_features, d_3, d_4, d_5, d_6, d_7])

    output = Conv2D(n_class, (1, 1), activation=None)(full_block)
    output = ResizeImageLayer(target_size=(input_shape[0], input_shape[1]))(output)
    output = Activation("sigmoid")(output)

    return Model(encoder.input, output)