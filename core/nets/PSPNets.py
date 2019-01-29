from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation
from keras.layers.merge import Concatenate
from keras.applications.resnet50 import ResNet50
from keras.models import Model
import numpy as np
from ..utils.net_utils import ResizeImageLayer, ConvUpscaleBlock, ConvBlock
from ..encoder import scope_table, build_encoder


def interp_block(inputs, feature_map_shape, level=1):
    ksize = (int(np.round(float(feature_map_shape[0]) / float(level))), int(np.round(float(feature_map_shape[1]) / float(level))))
    stride_size = ksize

    x = MaxPooling2D(pool_size=ksize, strides=stride_size)(inputs)
    x = Conv2D(512, (1, 1), activation=None)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = ResizeImageLayer(target_size=feature_map_shape)(x)

    return x


def PyramidScenePoolingSPModule(inputs, feature_map_shape):
    interp_block1 = interp_block(inputs, feature_map_shape, level=1)
    interp_block2 = interp_block(inputs, feature_map_shape, level=2)
    interp_block3 = interp_block(inputs, feature_map_shape, level=3)
    interp_block6 = interp_block(inputs, feature_map_shape, level=6)

    return Concatenate()([interp_block1, interp_block2, interp_block3, interp_block6])


def PSPNet(input_shape=(256, 256, 3), n_class = 1, encoder_name="resnet50", upscaling_method="conv"):
    encoder = build_encoder(input_shape, encoder_name=encoder_name)
    input_x = encoder.input

    features = encoder.get_layer(scope_table[encoder_name]["pool3"]).output
    feature_map_shape = (int(input_shape[0]/8), int(input_shape[1]/8))

    features = PyramidScenePoolingSPModule(features, feature_map_shape)
    features = Conv2D(512, (3, 3), padding="same", activation=None)(features)
    features = BatchNormalization()(features)
    features = Activation("relu")(features)

    # upsample
    if upscaling_method=="conv":
        features = ConvUpscaleBlock(features, 256, (3, 3), 2)
        features = ConvBlock(features, 256, (3, 3))
        features = ConvUpscaleBlock(features, 128, (3, 3), 2)
        features = ConvBlock(features, 128, (3, 3))
        features = ConvUpscaleBlock(features, 64, (3, 3), 2)
        features = ConvBlock(features, 64, (3, 3))
    else:
        features = ResizeImageLayer(target_size=(input_shape[0], input_shape[1]))(features)

    output = Conv2D(n_class, (1, 1), activation=None)(features)
    output = Activation("sigmoid")(output)

    pspnet_model = Model(input_x, output)
    return pspnet_model
