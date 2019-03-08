from keras.layers import Input
from keras.layers.core import Activation
from keras.layers.convolutional import Conv2D, SeparableConv2D
from keras.layers.merge import Add
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras.models import Model
from keras.regularizers import l2


def residual_block(inputs,
                   base_depth,
                   depth,
                   kernel_size,
                   stride=1,
                   rate=1,
                   block_name="block1",
                   unit_name="unit1",
                   weight_decay=1e-4,
                   kernel_initializer="he_normal",
                   bn_epsilon=1e-3,
                   bn_momentum=0.99):
    """Implementation of a residual block, with 3 conv layers. Each convolutional layer is followed
        with a batch normalization layer and a relu layer.
    The corresponding kernel sizes are (1, kernel_size, 1),
        corresponding strides are (1->stride->1),
        corresponding filters are (base_depth, base_depth, depth).
    If the depth of the inputs is equal to the 'depth', this is a identity block, else a convolutional
        block.
    :param inputs: 4-D tensor, shape of (batch_size, height, width, channel).
    :param base_depth: int, base depth of the residual block.
    :param depth: int, output depth.
    :param kernel_size: int.
    :param stride: int, default 1.
    :param rate: int, dilation rate, default 1.
    :param block_name: string, name of the bottleneck block, default "block1".
    :param unit_name: string, name of the unit(residual block), default "unit1".
    :param weight_decay: float, default 1e-4.
    :param kernel_initializer: string, default "he_normal".
    :param bn_epsilon: float, default 1e-3.
    :param bn_momentum: float, default 0.99.

    :return: 4-D tensor, shape of (batch_size, height, width, channel).
    """
    depth_in = int(inputs.shape[-1])
    conv_name_base = block_name+"_"+unit_name+"_conv"
    bn_name_base = block_name+"_"+unit_name+"_bn"

    # pre-activation and batch normalization
    preact = BatchNormalization(name=bn_name_base+"0", epsilon=bn_epsilon, momentum=bn_momentum)(inputs)
    preact = Activation("relu")(preact)
    # determine convolutional or identity connection
    if depth_in == depth:
        x_shortcut = MaxPooling2D(pool_size=(1, 1), strides=stride)(inputs) if stride > 1 else inputs
    else:
        x_shortcut = Conv2D(depth, (1, 1), strides=(stride, stride), name=conv_name_base + "short",
                            use_bias=False, activation=None, kernel_initializer=kernel_initializer,
                            kernel_regularizer=l2(weight_decay))(preact)
        x_shortcut = BatchNormalization(name=bn_name_base + "short", epsilon=bn_epsilon,
                                        momentum=bn_momentum)(x_shortcut)

    x = SeparableConv2D(base_depth, (1, 1), strides=(1, 1), padding="same", name=conv_name_base + "2a",
                        use_bias=False, activation=None, kernel_initializer=kernel_initializer,
                        kernel_regularizer=l2(weight_decay))(preact)
    x = BatchNormalization(name=bn_name_base + "2a", epsilon=bn_epsilon, momentum=bn_momentum)(x)
    x = Activation("relu")(x)

    x = Conv2D(base_depth, (kernel_size, kernel_size), strides=(stride, stride), dilation_rate=rate,
               padding="same", name=conv_name_base + "2b", use_bias=False, activation=None,
               kernel_initializer=kernel_initializer, kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization(name=bn_name_base + "2b", epsilon=bn_epsilon, momentum=bn_momentum)(x)
    x = Activation("relu")(x)

    x = Conv2D(depth, (1, 1), strides=(1, 1), padding="same", name=conv_name_base + "2c", use_bias=False,
               activation=None, kernel_initializer=kernel_initializer, kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization(name=bn_name_base + "2c", epsilon=bn_epsilon, momentum=bn_momentum)(x)

    output = Add()([x_shortcut, x])
    return output


def residual_bottleneck(inputs,
                        params_list,
                        weight_decay=1e-4,
                        kernel_initializer="he_normal",
                        bn_epsilon=1e-3,
                        bn_momentum=0.99):
    """ Building a res-net bottleneck(or a stage) according to the parameters generated from function
        'bottleneck_param()'
    :param inputs: 4-D tensor, shape of (batch_size, height, width, depth).
    :param params_list: list, each element of the list is used to build a residual block.
    :param weight_decay: float, default 1e-4.
    :param kernel_initializer: string, default "he_normal".
    :param bn_epsilon: float, default 1e-3.
    :param bn_momentum: float, default 0.99.

    :return: 4-D tensor, shape of (batch_size, height, width, depth).
    """

    x = inputs
    for i, param in enumerate(params_list):
        x = residual_block(x, base_depth=param["base_depth"], depth=param["depth"],
                           kernel_size=param["kernel_size"], stride=param["stride"],
                           rate=param["rate"], unit_name="unit"+str(i+1),
                           block_name=param["block_name"], weight_decay=weight_decay,
                           kernel_initializer=kernel_initializer, bn_epsilon=bn_epsilon,
                           bn_momentum=bn_momentum)
    return x


def bottleneck_param(scope,
                     base_depth=64,
                     kernel_size=3,
                     num_units=3,
                     stride=2,
                     rate=1):
    """ Generate parameters for each stage in a res-net.
    :param scope: string, the name of the stage(bottleneck), e.g, "block1".
    :param base_depth: int, the base depth of each residual block, default 64.
    :param kernel_size: int, default 3.
    :param num_units: int, the total number of residual blocks in this stage, default 3.
    :param stride: int, the stride of the middle convolution layer in each residual block,
        default 2
    :param rate: int, default 1.

    :return: list of parameters, can be passed to the function 'ResidualBottleneck(...)'
    """
    return [{
        "base_depth": base_depth,
        "depth": base_depth*4,
        "kernel_size": kernel_size,
        "stride": 1,
        "rate": rate,
        "block_name": scope
    }]*(num_units-1) + [{
        "base_depth": base_depth,
        "depth": base_depth*4,
        "kernel_size": kernel_size,
        "stride": stride,
        "rate": rate,
        "block_name": scope
    }]


def resnet_v2_50_separable(input_shape,
                           kernel_size=3,
                           include_root=True,
                           weight_decay=1e-4,
                           kernel_initializer="he_normal",
                           bn_epsilon=1e-3,
                           bn_momentum=0.99):
    """ Build a Resnet_v2_50_separable encoder
    :param input_shape: tuple, i.e., (height, width, channel)
    :param kernel_size: int, default 3.
    :param include_root: bool, default True.
    :param weight_decay: float, default 1e-4.
    :param kernel_initializer: string, default "he_normal".
    :param bn_epsilon: float, default 1e-3.
    :param bn_momentum: float, default 0.99.

    :return: a Keras model instance.
    """
    input_x = Input(shape=input_shape)
    x = BatchNormalization(epsilon=bn_epsilon, momentum=bn_momentum)(input_x)

    if include_root:
        x = Conv2D(64, (7, 7), strides=(2, 2), padding="same", name="conv1", use_bias=False,
                   activation=None, kernel_initializer=kernel_initializer,
                   kernel_regularizer=l2(weight_decay))(x)
        x = MaxPooling2D((3, 3), (2, 2), padding="same")(x)

    x = residual_bottleneck(x, bottleneck_param(scope="block1", base_depth=64, kernel_size=kernel_size,
                                                num_units=3, stride=2, rate=1),
                            weight_decay=weight_decay, kernel_initializer=kernel_initializer,
                            bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)
    x = residual_bottleneck(x, bottleneck_param(scope="block2", base_depth=128, kernel_size=kernel_size,
                                                num_units=4, stride=2, rate=1),
                            weight_decay=weight_decay, kernel_initializer=kernel_initializer,
                            bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)
    x = residual_bottleneck(x, bottleneck_param(scope="block3", base_depth=256, kernel_size=kernel_size,
                                                num_units=6, stride=1, rate=2),
                            weight_decay=weight_decay, kernel_initializer=kernel_initializer,
                            bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)
    x = residual_bottleneck(x, bottleneck_param(scope="block4", base_depth=512, kernel_size=kernel_size,
                                                num_units=3, stride=1, rate=4),
                            weight_decay=weight_decay, kernel_initializer=kernel_initializer,
                            bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)

    x = BatchNormalization(epsilon=bn_epsilon, momentum=bn_momentum)(x)
    x = Activation("relu")(x)

    return Model(input_x, x)


def resnet_v2_101_separable(input_shape,
                            kernel_size=3,
                            include_root=True,
                            weight_decay=1e-4,
                            kernel_initializer="he_normal",
                            bn_epsilon=1e-3,
                            bn_momentum=0.99):
    """ Build a Resnet_v2_101_separable encoder
    :param input_shape: tuple, i.e., (height, width, channel)
    :param kernel_size: int, default 3.
    :param include_root: bool, default True.
    :param weight_decay: float, default 1e-4.
    :param kernel_initializer: string, default "he_normal".
    :param bn_epsilon: float, default 1e-3.
    :param bn_momentum: float, default 0.99.

    :return: a Keras model instance.
    """
    input_x = Input(shape=input_shape)
    x = BatchNormalization(epsilon=bn_epsilon, momentum=bn_momentum)(input_x)

    if include_root:
        x = Conv2D(64, (7, 7), strides=(2, 2), padding="same", name="conv1", use_bias=False,
                   activation=None, kernel_initializer=kernel_initializer,
                   kernel_regularizer=l2(weight_decay))(x)
        x = MaxPooling2D((3, 3), (2, 2), padding="same")(x)

    x = residual_bottleneck(x, bottleneck_param(scope="block1", base_depth=64, kernel_size=kernel_size,
                                                num_units=3, stride=2, rate=1),
                            weight_decay=weight_decay, kernel_initializer=kernel_initializer,
                            bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)
    x = residual_bottleneck(x, bottleneck_param(scope="block2", base_depth=128, kernel_size=kernel_size,
                                                num_units=4, stride=2, rate=1),
                            weight_decay=weight_decay, kernel_initializer=kernel_initializer,
                            bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)
    x = residual_bottleneck(x, bottleneck_param(scope="block3", base_depth=256, kernel_size=kernel_size,
                                                num_units=23, stride=1, rate=2),
                            weight_decay=weight_decay, kernel_initializer=kernel_initializer,
                            bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)
    x = residual_bottleneck(x, bottleneck_param(scope="block4", base_depth=512, kernel_size=kernel_size,
                                                num_units=3, stride=1, rate=4),
                            weight_decay=weight_decay, kernel_initializer=kernel_initializer,
                            bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)

    x = BatchNormalization(epsilon=bn_epsilon, momentum=bn_momentum)(x)
    x = Activation("relu")(x)

    return Model(input_x, x)


def resnet_v2_152_separable(input_shape,
                            kernel_size=3,
                            include_root=True,
                            weight_decay=1e-4,
                            kernel_initializer="he_normal",
                            bn_epsilon=1e-3,
                            bn_momentum=0.99):
    """ Build a Resnet_v2_152_separable encoder
    :param input_shape: tuple, i.e., (height, width, channel)
    :param kernel_size: int, default 3.
    :param include_root: bool, default True.
    :param weight_decay: float, default 1e-4.
    :param kernel_initializer: string, default "he_normal".
    :param bn_epsilon: float, default 1e-3.
    :param bn_momentum: float, default 0.99.

    :return: a Keras model instance.
    """
    input_x = Input(shape=input_shape)
    x = BatchNormalization(epsilon=bn_epsilon, momentum=bn_momentum)(input_x)

    if include_root:
        x = Conv2D(64, (7, 7), strides=(2, 2), padding="same", name="conv1", use_bias=False,
                   activation=None, kernel_initializer=kernel_initializer,
                   kernel_regularizer=l2(weight_decay))(x)
        x = MaxPooling2D((3, 3), (2, 2), padding="same")(x)

    x = residual_bottleneck(x, bottleneck_param(scope="block1", base_depth=64, kernel_size=kernel_size,
                                                num_units=3, stride=2, rate=1),
                            weight_decay=weight_decay, kernel_initializer=kernel_initializer,
                            bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)
    x = residual_bottleneck(x, bottleneck_param(scope="block2", base_depth=128, kernel_size=kernel_size,
                                                num_units=8, stride=2, rate=1),
                            weight_decay=weight_decay, kernel_initializer=kernel_initializer,
                            bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)
    x = residual_bottleneck(x, bottleneck_param(scope="block3", base_depth=256, kernel_size=kernel_size,
                                                num_units=36, stride=1, rate=2),
                            weight_decay=weight_decay, kernel_initializer=kernel_initializer,
                            bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)
    x = residual_bottleneck(x, bottleneck_param(scope="block4", base_depth=512, kernel_size=kernel_size,
                                                num_units=3, stride=1, rate=4),
                            weight_decay=weight_decay, kernel_initializer=kernel_initializer,
                            bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)

    x = BatchNormalization(epsilon=bn_epsilon, momentum=bn_momentum)(x)
    x = Activation("relu")(x)

    return Model(input_x, x)


def resnet_v2_200_separable(input_shape,
                            kernel_size=3,
                            include_root=True,
                            weight_decay=1e-4,
                            kernel_initializer="he_normal",
                            bn_epsilon=1e-3,
                            bn_momentum=0.99):
    """ Build a Resnet_v2_200_separable encoder
    :param input_shape: tuple, i.e., (height, width, channel)
    :param kernel_size: int, default 3.
    :param include_root: bool, default True.
    :param weight_decay: float, default 1e-4.
    :param kernel_initializer: string, default "he_normal".
    :param bn_epsilon: float, default 1e-3.
    :param bn_momentum: float, default 0.99.

    :return: a Keras model instance.
    """
    input_x = Input(shape=input_shape)
    x = BatchNormalization(epsilon=bn_epsilon, momentum=bn_momentum)(input_x)

    if include_root:
        x = Conv2D(64, (7, 7), strides=(2, 2), padding="same", name="conv1", use_bias=False,
                   activation=None, kernel_initializer=kernel_initializer,
                   kernel_regularizer=l2(weight_decay))(x)
        x = MaxPooling2D((3, 3), (2, 2), padding="same")(x)

    x = residual_bottleneck(x, bottleneck_param(scope="block1", base_depth=64, kernel_size=kernel_size,
                                                num_units=3, stride=2, rate=1),
                            weight_decay=weight_decay, kernel_initializer=kernel_initializer,
                            bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)
    x = residual_bottleneck(x, bottleneck_param(scope="block2", base_depth=128, kernel_size=kernel_size,
                                                num_units=24, stride=2, rate=1),
                            weight_decay=weight_decay, kernel_initializer=kernel_initializer,
                            bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)
    x = residual_bottleneck(x, bottleneck_param(scope="block3", base_depth=256, kernel_size=kernel_size,
                                                num_units=36, stride=1, rate=2),
                            weight_decay=weight_decay, kernel_initializer=kernel_initializer,
                            bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)
    x = residual_bottleneck(x, bottleneck_param(scope="block4", base_depth=512, kernel_size=kernel_size,
                                                num_units=3, stride=1, rate=4),
                            weight_decay=weight_decay, kernel_initializer=kernel_initializer,
                            bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)

    x = BatchNormalization(epsilon=bn_epsilon, momentum=bn_momentum)(x)
    x = Activation("relu")(x)

    return Model(input_x, x)
