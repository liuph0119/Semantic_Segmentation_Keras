"""
    Script: xception.py
    Author: Penghua Liu
    Date: 2019-01-28
    Email: liuphhhh@foxmail.com
    Functions: Implementation of xception models
    # TODO: the pool2, pool3, pool4, pool5 layer names of the models need to be confirmed.

"""

from keras.applications.xception import Xception
from keras.layers import Input
from keras.layers.convolutional import Conv2D, SeparableConv2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation
from keras.layers.merge import Add
from keras import layers
from keras.models import Model


def SeparableResidualBlock(inputs, n_filters_list=[256, 256, 256], block_id="entry_block2", skip_type="sum", stride=1, rate=1):
    x = Activation("relu", name=block_id+"sepconv1_act")(inputs)
    x = SeparableConv2D(n_filters_list[0], (3, 3), padding='same', use_bias=False, name=block_id+'_sepconv1', dilation_rate=rate)(x)
    x = BatchNormalization(name=block_id+'_sepconv1_bn')(x)

    x = Activation('relu', name=block_id+'_sepconv2_act')(x)
    x = SeparableConv2D(n_filters_list[1], (3, 3), padding='same', use_bias=False, name=block_id+'_sepconv2', dilation_rate=rate)(x)
    x = BatchNormalization(name=block_id+'_sepconv2_bn')(x)

    x = Activation("relu", name=block_id+"_sepconv3_act")(x)
    x = SeparableConv2D(n_filters_list[2], (3, 3), padding="same", use_bias=False, strides=stride, name=block_id+"_sepconv3", dilation_rate=rate)(x)
    x = BatchNormalization(name=block_id+"_sepconv3_bn")(x)

    if skip_type=="sum":
        x = Add()([inputs, x])
    elif skip_type=="conv":
        shortcut = Conv2D(n_filters_list[2], (1, 1), strides=stride, padding='same', use_bias=False)(inputs)
        shortcut = BatchNormalization()(shortcut)
        x = layers.add([shortcut, x])
    else:
        x = x

    return x




def Xception_41(input_shape=(256, 256, 3)):
    input_x = Input(shape=input_shape)

    x = Conv2D(32, (3, 3), strides=(2, 2), use_bias=False, name='entry_block1_conv1', padding="same")(input_x)
    x = BatchNormalization(name='entry_block1_conv1_bn')(x)
    x = Activation('relu', name='entry_block1_conv1_act')(x)
    x = Conv2D(64, (3, 3), use_bias=False, name='entry_block1_conv2', padding="same")(x)
    x = BatchNormalization(name='entry_block1_conv2_bn')(x)
    x = Activation('relu', name='entry_block1_conv2_act')(x)

    x = SeparableResidualBlock(x, [128, 128, 128], "entry_block2", skip_type="conv", stride=2, rate=1)
    x = SeparableResidualBlock(x, [256, 256, 256], "entry_block3", skip_type="conv", stride=2, rate=1)
    x = SeparableResidualBlock(x, [728, 728, 728], "entry_block4", skip_type="conv", stride=2, rate=1)

    for i in range(1, 17):
        x = SeparableResidualBlock(x, [728, 728, 728], "middle_block"+str(i), skip_type="sum", stride=1, rate=1)

    x = SeparableResidualBlock(x, [728, 1024, 1024], "exit_block1", skip_type="conv", stride=1, rate=1)
    x = SeparableConv2D(1536, (3, 3), strides=1, dilation_rate=2, use_bias=False, padding="same", activation="relu", name="exit_block2_sepconv1")(x)
    x = BatchNormalization(name="exit_block2_sepconv1_bn")(x)
    x = SeparableConv2D(1536, (3, 3), strides=1, dilation_rate=2, use_bias=False, padding="same", activation="relu", name="exit_block2_sepconv2")(x)
    x = BatchNormalization(name="exit_block2_sepconv2_bn")(x)
    x = SeparableConv2D(2048, (3, 3), strides=1, dilation_rate=2, use_bias=False, padding="same", activation="relu", name="exit_block2_sepconv3")(x)
    x = BatchNormalization(name="exit_block2_sepconv3_bn")(x)

    return Model(input_x, x)


