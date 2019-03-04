from keras.layers import Input
from keras.layers.core import Activation
from keras.layers.convolutional import Conv2D, SeparableConv2D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Add
from keras.models import Model
from keras.regularizers import l2


def separable_residual_block(inputs,
                           n_filters_list=[256, 256, 256],
                           block_id="entry_block2",
                           skip_type="sum",
                           stride=1,
                           rate=1,
                           weight_decay=1e-4,
                           kernel_initializer="he_normal",
                           bn_epsilon=1e-3,
                           bn_momentum=0.99):
    """ separable residual block
    :param inputs: 4-D tensor, shape of (batch_size, height, width, channel).
    :param n_filters_list: list of int, numbers of filters in the separable convolutions, default [256, 256, 256].
    :param block_id: string, default "entry_block2".
    :param skip_type: string, one of {"sum", "conv", "none"}, default "sum".
    :param stride: int, default 1.
    :param rate: int, default 1.
    :param weight_decay: float, default 1e-4.
    :param kernel_initializer: string, default "he_normal".
    :param bn_epsilon: float, default 1e-3.
    :param bn_momentum: float, default 0.99.

    :return: 4-D tensor, shape of (batch_size, height, width, channel).
    """
    x = Activation("relu", name=block_id+"sepconv1_act")(inputs)
    x = SeparableConv2D(n_filters_list[0], (3, 3), padding='same', use_bias=False,
                        name=block_id+'_sepconv1', dilation_rate=rate,
                        kernel_initializer=kernel_initializer, kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization(name=block_id+'_sepconv1_bn', epsilon=bn_epsilon, momentum=bn_momentum)(x)

    x = Activation('relu', name=block_id+'_sepconv2_act')(x)
    x = SeparableConv2D(n_filters_list[1], (3, 3), padding='same', use_bias=False,
                        name=block_id+'_sepconv2', dilation_rate=rate,
                        kernel_initializer=kernel_initializer, kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization(name=block_id+'_sepconv2_bn', epsilon=bn_epsilon, momentum=bn_momentum)(x)

    x = Activation("relu", name=block_id+"_sepconv3_act")(x)
    x = SeparableConv2D(n_filters_list[2], (3, 3), padding="same", use_bias=False,
                        strides=stride, name=block_id+"_sepconv3", dilation_rate=rate,
                        kernel_initializer=kernel_initializer, kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization(name=block_id+"_sepconv3_bn", epsilon=bn_epsilon, momentum=bn_momentum)(x)

    if skip_type=="sum":
        x = Add()([inputs, x])
    elif skip_type=="conv":
        shortcut = Conv2D(n_filters_list[2], (1, 1), strides=stride, padding='same', use_bias=False,
                          kernel_initializer=kernel_initializer, kernel_regularizer=l2(weight_decay))(inputs)
        shortcut = BatchNormalization(epsilon=bn_epsilon, momentum=bn_momentum)(shortcut)
        x = Add()([shortcut, x])
    else:
        x = x

    return x




def xception_41(input_shape,
                weight_decay=1e-4,
                kernel_initializer="he_normal",
                bn_epsilon=1e-3,
                bn_momentum=0.99):
    """
    :param input_shape: tuple, i.e., (height, width, channel).
    :param weight_decay: float, default 1e-4.
    :param kernel_initializer: string, default "he_normal".
    :param bn_epsilon: float, default 1e-3.
    :param bn_momentum: float, default 0.99.

    :return: a Keras Model instance.
    """
    input_x = Input(shape=input_shape)
    x = BatchNormalization(epsilon=bn_epsilon, momentum=bn_momentum)(input_x)

    x = Conv2D(32, (3, 3), strides=(2, 2), use_bias=False, name='entry_block1_conv1', padding="same",
               kernel_initializer=kernel_initializer, kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization(name='entry_block1_conv1_bn', epsilon=bn_epsilon, momentum=bn_momentum)(x)
    x = Activation('relu', name='entry_block1_conv1_act')(x)
    x = Conv2D(64, (3, 3), use_bias=False, name='entry_block1_conv2', padding="same",
               kernel_initializer=kernel_initializer, kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization(name='entry_block1_conv2_bn', epsilon=bn_epsilon, momentum=bn_momentum)(x)
    x = Activation('relu', name='entry_block1_conv2_act')(x)

    x = separable_residual_block(x, [128, 128, 128], "entry_block2", skip_type="conv", stride=2, rate=1,
                               weight_decay=weight_decay, kernel_initializer=kernel_initializer,
                               bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)
    x = separable_residual_block(x, [256, 256, 256], "entry_block3", skip_type="conv", stride=2, rate=1,
                               weight_decay=weight_decay, kernel_initializer=kernel_initializer,
                               bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)
    x = separable_residual_block(x, [728, 728, 728], "entry_block4", skip_type="conv", stride=2, rate=1,
                               weight_decay=weight_decay, kernel_initializer=kernel_initializer,
                               bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)

    for i in range(1, 17):
        x = separable_residual_block(x, [728, 728, 728], "middle_block"+str(i), skip_type="sum", stride=1, rate=1,
                                   weight_decay=weight_decay, kernel_initializer=kernel_initializer,
                                   bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)

    x = separable_residual_block(x, [728, 1024, 1024], "exit_block1", skip_type="conv", stride=1, rate=1,
                               weight_decay=weight_decay, kernel_initializer=kernel_initializer,
                               bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)
    x = SeparableConv2D(1536, (3, 3), strides=1, dilation_rate=2, use_bias=False, padding="same",
                        activation="relu", name="exit_block2_sepconv1",
                        kernel_regularizer=l2(weight_decay), kernel_initializer=kernel_initializer)(x)
    x = BatchNormalization(name="exit_block2_sepconv1_bn", epsilon=bn_epsilon, momentum=bn_momentum)(x)
    x = SeparableConv2D(1536, (3, 3), strides=1, dilation_rate=2, use_bias=False, padding="same",
                        activation="relu", name="exit_block2_sepconv2",
                        kernel_regularizer=l2(weight_decay), kernel_initializer=kernel_initializer)(x)
    x = BatchNormalization(name="exit_block2_sepconv2_bn", epsilon=bn_epsilon, momentum=bn_momentum)(x)
    x = SeparableConv2D(2048, (3, 3), strides=1, dilation_rate=2, use_bias=False, padding="same",
                        activation="relu", name="exit_block2_sepconv3",
                        kernel_regularizer=l2(weight_decay), kernel_initializer=kernel_initializer)(x)
    x = BatchNormalization(name="exit_block2_sepconv3_bn", epsilon=bn_epsilon, momentum=bn_momentum)(x)

    return Model(input_x, x)