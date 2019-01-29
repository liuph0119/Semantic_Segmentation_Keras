from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.core import Dropout, Activation
from keras.layers.merge import Add

from ..utils.net_utils import crop


def FCN_8s(input_shape=(256, 256, 3), n_class=1, vgg_weight=None):
    # adapted from https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/voc-fcn8s/net.py
    vgg_encoder = VGG16(include_top=False, weights=vgg_weight, input_shape=input_shape)
    p3 = vgg_encoder.get_layer("block3_pool").output
    p4 = vgg_encoder.get_layer("block4_pool").output
    p5 = vgg_encoder.get_layer("block5_pool").output

    # # # 1. merge pool5 & pool4
    # upsample prediction from pool5
    x1 = Conv2D(4096, (7, 7), padding="same", activation="relu")(p5)
    x1 = Dropout(0.5)(x1)
    x1 = Conv2D(4096, (1, 1), padding="same", activation="relu")(x1)
    x1 = Dropout(0.5)(x1)
    x1 = Conv2D(n_class, (1, 1), padding="same")(x1)
    x1 = Conv2DTranspose(n_class, (4,4), strides=(2,2), use_bias=False)(x1)
    # upsample prediction from pool4
    x2 = Conv2D(n_class, (1,1), padding="same")(p4)
    x1, x2 = crop(x1, x2, vgg_encoder.input)
    x1 = Add()([x1, x2])

    # # # 2. merge pool4 & pool3
    x1 = Conv2DTranspose(n_class, (4,4), strides=(2,2), use_bias=False)(x1)
    x2 = Conv2D(n_class, (1,1), padding="same")(p3)
    x1, x2 = crop(x1, x2, vgg_encoder.input)
    x1 = Add()([x1, x2])

    # # # 3. upsample and predict
    x1 = Conv2DTranspose(n_class, (16, 16), strides=(8, 8), use_bias=False)(x1)
    output = Activation("sigmoid")(x1)

    fcn_8s_model = Model(vgg_encoder.input, output)
    return fcn_8s_model


def FCN_16s(input_shape=(256, 256, 3), n_class=1, vgg_weight=None):
    # adapted from https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/voc-fcn8s/net.py
    vgg_encoder = VGG16(include_top=False, weights=vgg_weight, input_shape=input_shape)
    p4 = vgg_encoder.get_layer("block4_pool").output
    p5 = vgg_encoder.get_layer("block5_pool").output

    # # # 1. merge pool5 & pool4
    # upsamples prediction from pool5
    x1 = Conv2D(4096, (7, 7), padding="same", activation="relu")(p5)
    x1 = Dropout(0.5)(x1)
    x1 = Conv2D(4096, (1, 1), padding="same", activation="relu")(x1)
    x1 = Dropout(0.5)(x1)
    x1 = Conv2D(n_class, (1, 1), padding="same")(x1)
    x1 = Conv2DTranspose(n_class, (4,4), strides=(2,2), use_bias=False)(x1)
    # upsamples from pool4
    x2 = Conv2D(n_class, (1,1), padding="same")(p4)
    x1, x2 = crop(x1, x2, vgg_encoder.input)
    x1 = Add()([x1, x2])

    # # # 2. upsample and predict
    x1 = Conv2DTranspose(n_class, (32, 32), strides=(16, 16), use_bias=False)(x1)
    output = Activation("sigmoid")(x1)

    fcn_16s_model = Model(vgg_encoder.input, output)
    return fcn_16s_model



def FCN_32s(input_shape=(256, 256, 3), n_class=1, vgg_weight=None):
    # adapted from https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/voc-fcn8s/net.py
    vgg_encoder = VGG16(include_top=False, weights=vgg_weight, input_shape=input_shape)
    p5 = vgg_encoder.get_layer("block5_pool").output

    # # # 1. upsamples from pool5
    # upsamples prediction from pool5
    x1 = Conv2D(4096, (7, 7), padding="same", activation="relu")(p5)
    x1 = Dropout(0.5)(x1)
    x1 = Conv2D(4096, (1, 1), padding="same", activation="relu")(x1)
    x1 = Dropout(0.5)(x1)
    x1 = Conv2D(n_class, (1, 1), padding="same")(x1)
    x1 = Conv2DTranspose(n_class, (64, 64), strides=(32,32), use_bias=False)(x1)

    output = Activation("sigmoid")(x1)

    fcn_32s_model = Model(vgg_encoder.input, output)
    return fcn_32s_model

