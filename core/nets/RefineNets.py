from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import Add
from keras.models import Model

from ..utils.net_utils import ResizeImageLayer, ConvBlock, ConvUpscaleBlock
from ..encoder import scope_table, build_encoder


def ResidualConvUnit(inputs, n_filters=256, kernel_size=3):
    x = Activation("relu")(inputs)
    x = Conv2D(n_filters, (kernel_size, kernel_size), padding="same", activation=None)(x)
    x = Activation("relu")(x)
    x = Conv2D(n_filters, (kernel_size, kernel_size), padding="same", activation=None)(x)
    x = Add()([x, inputs])

    return x


def ChainedResidualPooling(inputs, pool_size=(5,5), n_filters=256):
    X_relu = Activation("relu")(inputs)
    
    X = MaxPooling2D(pool_size=pool_size, strides=(1, 1), padding="same")(X_relu)
    X = Conv2D(n_filters, (3, 3), activation=None, padding="same")(X)
    X_sum1 = Add()([X_relu, X])

    X = MaxPooling2D(pool_size=pool_size, strides=(1, 1), padding="same")(X)
    X = Conv2D(n_filters, (3, 3), activation=None, padding="same")(X)
    X_sum2 = Add()([X, X_sum1])

    return X_sum2


def MultiResolutionFusion(high_inputs=None,low_inputs=None,n_filters=256):
    if low_inputs is None:
        fuse = Conv2D(n_filters, (3, 3), padding="same", activation=None)(high_inputs)
    else:
        conv_low = Conv2D(n_filters, (3, 3), padding="same", activation=None)(low_inputs)
        conv_high = Conv2D(n_filters, (3, 3), padding="same", activation=None)(high_inputs)
        
        conv_low = ResizeImageLayer(target_size=(int(conv_high.shape[1]), int(conv_high.shape[2])))(conv_low)
        fuse = Add()([conv_high, conv_low])
    
    return fuse

def RefineBlock(high_inputs=None, low_inputs=None, base_filters=256):
    if low_inputs is None:      # Block 4
        # 2 RCUs
        rcu_new_high = ResidualConvUnit(high_inputs, n_filters=base_filters*2)
        rcu_new_high = ResidualConvUnit(rcu_new_high, n_filters=base_filters*2)
        
        # feature fusion
        fuse = MultiResolutionFusion(high_inputs=rcu_new_high, low_inputs=None, n_filters=base_filters*2)
        fuse_pooling = ChainedResidualPooling(fuse, n_filters=base_filters*2)
        output = ResidualConvUnit(fuse_pooling, n_filters=base_filters*2)
        return output
    else:
        rcu_high = ResidualConvUnit(high_inputs, n_filters=base_filters)
        rcu_high = ResidualConvUnit(rcu_high, n_filters=base_filters)

        fuse = MultiResolutionFusion(rcu_high, low_inputs, n_filters=base_filters)
        fuse_pooling = ChainedResidualPooling(fuse, n_filters=base_filters)
        output = ResidualConvUnit(fuse_pooling, n_filters=base_filters)
        return output
    # sum_logits = []
    # for inputs in inputs_list:
    #     # 2 Residual Convolutional Units
    #     inputs = ResidualConvUnit(inputs, filters_list, ksize=(3, 3), skip_connection_type="sum")
    #     inputs = ResidualBlock(inputs, filters_list, ksize=(3, 3), skip_connection_type="sum")
    # 
    #     # Multi-resolution Fusion
    #     inputs = Conv2D(filters_list[-1], kernel_size=(3, 3), padding="same", strides=1)(inputs)
    #     inputs = ResizeImageLayer(target_size=target_size, resize_method="linear")(inputs)
    #     sum_logits.append(inputs)
    # 
    # if len(inputs_list)==1:
    #     sum_logits = sum_logits[0]
    # else:
    #     sum_logits = Add()(sum_logits)
    # 
    # # Chained Residual Pooling
    # outputs = ChainedResidualPooling(sum_logits, n_iters=2, pool_size=(5, 5))
    # # another Residual Convolutional Unit
    # for i in range(last_rcu_count):
    #     outputs = ResidualBlock(outputs, filters_list, ksize=(3,3), skip_connection_type="sum")
    # return outputs


def RefineNet(input_shape=(256, 256, 3), n_class=1, encoder_name="resnet50", encoder_weights=None, init_filters=256, upscaling_method="bilinear"):
    """ 4 cascaded RefineNet implementation using keras
    :param input_shape: tuple of integers, input_shape of the input tensor, like (image_height, image_width, image_channel)
    :param n_class: int, number of object classes, default 1
    :param res50_weights: string, the path of the weights of pre-trained resnet50 model in imagenet
    :param init_filters: int, number of initial filters, default 32
    :return: keras model instance
    """
    encoder = build_encoder(input_shape, encoder_name, encoder_weights=encoder_weights)

    high_1 = encoder.get_layer(scope_table[encoder_name]["pool5"]).output
    high_2 = encoder.get_layer(scope_table[encoder_name]["pool4"]).output
    high_3 = encoder.get_layer(scope_table[encoder_name]["pool3"]).output
    high_4 = encoder.get_layer(scope_table[encoder_name]["pool2"]).output

    high_1 = Conv2D(init_filters*2, (1, 1), padding="same", activation="relu")(high_1)
    high_2 = Conv2D(init_filters, (1, 1), padding="same", activation="relu")(high_2)
    high_3 = Conv2D(init_filters, (1, 1), padding="same", activation="relu")(high_3)
    high_4 = Conv2D(init_filters, (1, 1), padding="same", activation="relu")(high_4)

    low_1 = RefineBlock(high_1, low_inputs=None, base_filters=init_filters)
    low_2 = RefineBlock(high_2, low_1, base_filters=init_filters)
    low_3 = RefineBlock(high_3, low_2, base_filters=init_filters)
    low_4 = RefineBlock(high_4, low_3, base_filters=init_filters)
    x = low_4
    
    x = ResidualConvUnit(x, init_filters)
    x = ResidualConvUnit(x, init_filters)

    if upscaling_method=="conv":
        x = ConvUpscaleBlock(x, 128, kernel_size=[3, 3], scale=2)
        x = ConvBlock(x, 128)
        x = ConvUpscaleBlock(x, 64, kernel_size=[3, 3], scale=2)
        x = ConvBlock(x, 64)
    else:
        x = ResizeImageLayer(target_size=(input_shape[0], input_shape[1]))(x)
        
    output = Conv2D(n_class, (1,1), activation=None)(x)
    output = Activation("sigmoid")(output)

    return Model(encoder.input, output)