"""
    Script: resnet_v2.py
    Author: Penghua Liu
    Date: 2019-01-13
    Email: liuphhhh@foxmail.com
    Functions: Implementation of components to building resnet_v2 models and the various resnet_v2 model builder
    # TODO: the pool2, pool3, pool4, pool5 layer names of each resnet_v2 model need to be confirmed.

"""
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation
from keras.layers.convolutional import Conv2D, SeparableConv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import Add
from keras.layers import Input
from keras.models import Model
from keras.regularizers import l2

BLOCK_IDS = "ABCDEFGHIJKLMNOPQRSTUVWEYZabcdefghijklmnopqrstuvwxyz"

def residual_block(inputs, base_depth, depth, kernel_size, stride=1, rate=1, block_name="block1", unit_name="unit1", weight_decay=4e-5):
    """ Implementation of a residual block, with 3 conv layers. Each convolutional layer is followed with
    a batch normalization layer and a relu layer.
    The corresponding kernel sizes are (1, kernel_size, 1),
        corresponding strides are (1->stride->1),
        corresponding filters are (base_depth, base_depth, depth)

    If the depth of the inputs is equal to the 'depth', this is a identity block, else a convolutional block.

    # Args:
        :param inputs: input tensor, 4-d tensor,[batch_size, height, width, depth]
        :param base_depth: the base depth of the residual block, int
        :param depth: the output depth, int
        :param kernel_size: kernel size, int
        :param stride: the stride, int, default 1
        :param rate: the dilation rate, int, default 1
        :param stage: the id of the stage, int, default 1
        :param block: the block id, string, default "a"

    # Returns:
        a 4-d tensor, with the shape of [batch_size, height, width, depth]
    """

    depth_in = int(inputs.shape[-1])
    conv_name_base = block_name+"_"+unit_name+"_conv"
    bn_name_base = block_name+"_"+unit_name+"_bn"

    # pre-activation and batch normalization
    preact = BatchNormalization(name=bn_name_base+"0")(inputs)
    preact = Activation("relu")(preact)
    # determine convolutional or identity connection
    if depth_in == depth:
        if stride > 1:
            x_shortcut = MaxPooling2D(pool_size=(1, 1), strides=stride)(inputs)
        else:
            x_shortcut = inputs
    else:
        x_shortcut = Conv2D(depth, (1, 1), strides=(stride, stride), name=conv_name_base + "short", activation=None, kernel_regularizer=l2(weight_decay))(
            preact)
        x_shortcut = BatchNormalization(name=bn_name_base + "short")(x_shortcut)


    x = Conv2D(base_depth, (1, 1), strides=(1, 1), padding="same", name=conv_name_base + "2a", kernel_regularizer=l2(weight_decay))(preact)
    x = BatchNormalization(name=bn_name_base + "2a")(x)
    x = Activation("relu")(x)

    x = SeparableConv2D(base_depth, (kernel_size, kernel_size), strides=(stride, stride), dilation_rate= rate, padding="same", name=conv_name_base + "2b", kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization(name=bn_name_base + "2b")(x)
    x = Activation("relu")(x)

    x = Conv2D(depth, (1, 1), strides=(1, 1), padding="same", name=conv_name_base + "2c", kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization(name=bn_name_base + "2c")(x)

    output = Add()([x_shortcut, x])
    return output


def ResidualBottleneck(inputs, params_list):
    """ Building a resnet bottleneck(or a stage) according to the parameters generated from function 'bottleneck_param()'

    # Args:
        :param inputs: input tensor, 4-d tensor,[batch_size, height, width, depth]
        :param params_list: a list of parameters, each element of the list is used to build a residual block

    # Returns:
        a 4-d tensor, [batch_size, height, width, depth]
    """

    x = inputs
    for i, param in enumerate(params_list):
        x = residual_block(x, base_depth=param["base_depth"], depth=param["depth"], kernel_size=param["kernel_size"],
                           stride=param["stride"], rate=param["rate"], unit_name="unit"+str(i+1), block_name=param["block_name"])
    return x


def bottleneck_param(scope, base_depth=64, kernel_size=3, num_units=3, stride=2, rate=1):
    """Generate parameters for each stage in a resnet

    # Args:
        :param scope: the name of the stage(bottleneck), e.g, "block1"
        :param base_depth: the base depth of each residual block, int, default 64
        :param num_units: the total number of residual blocks in this stage, int, default 3
        :param stride: the stride of the middle convolutional layer in each residual block, int, default 2

    # Returns:
        a list of parameters, can be passed to the function 'ResidualBottleneck(...)'
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


def resnet_v2_50_separable(input_shape, kernel_size=3, include_root=True):
    """ Build a Resnet 50 model
    # Args:
        :param input_shape: shape of the inputs, [height, width, channel]

    # Returns:
        a Keras model instance.
    """
    input_x = Input(shape=input_shape)
    x = input_x

    if include_root:
        x = Conv2D(64, (7, 7), strides=(2, 2), padding="same", activation=None, kernel_regularizer=l2(4e-5), name="conv1")(input_x)
        x = MaxPooling2D((3, 3), (2, 2), padding="same")(x)

    x = ResidualBottleneck(x, bottleneck_param(scope="block1", base_depth=64, kernel_size=kernel_size, num_units=3, stride=2, rate=1))
    x = ResidualBottleneck(x, bottleneck_param(scope="block2", base_depth=128, kernel_size=kernel_size, num_units=4, stride=2, rate=1))
    x = ResidualBottleneck(x, bottleneck_param(scope="block3", base_depth=256, kernel_size=kernel_size, num_units=6, stride=1, rate=2))
    x = ResidualBottleneck(x, bottleneck_param(scope="block4", base_depth=512, kernel_size=kernel_size, num_units=3, stride=1, rate=4))

    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    model = Model(input_x, x)
    return model


def resnet_v2_101_separable(input_shape, kernel_size=3, include_root=True):
    """ Build a Resnet 101 model
    # Args:
        :param input_shape: shape of the inputs, [height, width, channel]

    # Returns:
        a Keras model instance.
    """
    input_x = Input(shape=input_shape)
    x = input_x

    if include_root:
        x = Conv2D(64, (7, 7), strides=(2, 2), padding="same", activation=None, kernel_regularizer=l2(4e-5),
                   name="conv1")(input_x)
        x = MaxPooling2D((3, 3), (2, 2), padding="same")(x)

    x = ResidualBottleneck(x, bottleneck_param(scope="block1", base_depth=64, kernel_size=kernel_size, num_units=3, stride=2, rate=1))
    x = ResidualBottleneck(x, bottleneck_param(scope="block2", base_depth=128, kernel_size=kernel_size, num_units=4, stride=2, rate=1))
    x = ResidualBottleneck(x, bottleneck_param(scope="block3", base_depth=256, kernel_size=kernel_size, num_units=23, stride=1, rate=2))
    x = ResidualBottleneck(x, bottleneck_param(scope="block4", base_depth=512, kernel_size=kernel_size, num_units=3, stride=1, rate=4))

    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    model = Model(input_x, x)
    return model


def resnet_v2_152_separable(input_shape, kernel_size=3, include_root=True):
    """ Build a Resnet 152 model
    # Args:
        :param input_shape: shape of the inputs, [height, width, channel]

    # Returns:
        a Keras model instance.
    """
    input_x = Input(shape=input_shape)
    x = input_x

    if include_root:
        x = Conv2D(64, (7, 7), strides=(2, 2), padding="same", activation=None, kernel_regularizer=l2(4e-5),
                   name="conv1")(input_x)
        x = MaxPooling2D((3, 3), (2, 2), padding="same")(x)

    x = ResidualBottleneck(x, bottleneck_param(scope="block1", base_depth=64, kernel_size=kernel_size, num_units=3, stride=2, rate=1))
    x = ResidualBottleneck(x, bottleneck_param(scope="block2", base_depth=128, kernel_size=kernel_size, num_units=8, stride=2, rate=1))
    x = ResidualBottleneck(x, bottleneck_param(scope="block3", base_depth=256, kernel_size=kernel_size, num_units=36, stride=1, rate=2))
    x = ResidualBottleneck(x, bottleneck_param(scope="block4", base_depth=512, kernel_size=kernel_size, num_units=3, stride=1, rate=4))

    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    model = Model(input_x, x)
    return model


def resnet_v2_200_separable(input_shape, kernel_size=3, include_root=True):
    """ Build a Resnet 200 model
    # Args:
        :param input_shape: shape of the inputs, [height, width, channel]

    # Returns:
        a Keras model instance.
    """
    input_x = Input(shape=input_shape)
    x = input_x

    if include_root:
        x = Conv2D(64, (7, 7), strides=(2, 2), padding="same", activation=None, kernel_regularizer=l2(4e-5),
                   name="conv1")(input_x)
        x = MaxPooling2D((3, 3), (2, 2), padding="same")(x)

    x = ResidualBottleneck(x, bottleneck_param(scope="block1", base_depth=64, kernel_size=kernel_size, num_units=3, stride=2, rate=1))
    x = ResidualBottleneck(x, bottleneck_param(scope="block2", base_depth=128, kernel_size=kernel_size, num_units=24, stride=2, rate=1))
    x = ResidualBottleneck(x, bottleneck_param(scope="block3", base_depth=256, kernel_size=kernel_size, num_units=36, stride=1, rate=2))
    x = ResidualBottleneck(x, bottleneck_param(scope="block4", base_depth=512, kernel_size=kernel_size, num_units=3, stride=1, rate=4))

    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    model = Model(input_x, x)
    return model
