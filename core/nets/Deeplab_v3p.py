from keras.layers.convolutional import Conv2D
from keras.layers.merge import Concatenate
from keras.layers.core import Activation, Dropout
from keras.layers.normalization import BatchNormalization
from keras.models import Model, load_model
import os


from ..encoder import build_encoder, scope_table
from ..utils.net_utils import AtrousSpatialPyramidPooling, ResizeImageLayer, ConvUpscaleBlock, ConvBlock



def Deeplab_v3p(input_shape=(256, 256, 3), n_class=1, encoder_name="resnet50", encoder_weights="D:/keras_weights/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5"):
    """ Deeplab_v3+ implementation.

    # Args:
        :param input_shape: the shape of inputs. tuple of (height, width, channel)
        :param n_class: number of classes. int, default 1
        :param encoder_name: the encoder name, default "resnet50". More details can be found in '.encoder'
        :param encoder_weights: the pre-trained weights of the encoder model.

    # Returns:
        a Keras model instance.
    """
    if encoder_weights is not None and os.path.exists(encoder_weights):
        encoder = load_model(encoder_weights)
    else:
        encoder = build_encoder(input_shape, encoder_name=encoder_name, encoder_weights=encoder_weights)

    net = AtrousSpatialPyramidPooling(encoder.get_layer(scope_table[encoder_name]["pool4"]).output, n_filters=256,
                                      rates=[6, 12, 18], imagelevel=True)
    net = Conv2D(256, (1, 1), use_bias=False, activation=None)(net)
    net = BatchNormalization()(net)
    net = Activation("relu")(net)
    net = Dropout(0.5)(net)
    decoder_features = ResizeImageLayer(target_size=(input_shape[0] // 4, input_shape[1] // 4))(net)
    # encoder_features = ConvUpscaleBlock(encoder_features, 256, (3, 3), 2)
    # encoder_features = ConvBlock(encoder_features, 256, (3, 3))
    # encoder_features = ConvUpscaleBlock(encoder_features, 256, (3, 3), 2)
    # encoder_features = ConvBlock(encoder_features, 256, (3, 3))
    encoder_features = encoder.get_layer(scope_table[encoder_name]["pool2"]).output
    encoder_features = Conv2D(48, (1, 1), use_bias=False, activation=None)(encoder_features)
    encoder_features = BatchNormalization()(encoder_features)
    encoder_features = Activation("relu")(encoder_features)
    net = Concatenate()([encoder_features, decoder_features])
    
    net = Conv2D(256, (3, 3), use_bias=False, activation=None, padding="same")(net)
    net = BatchNormalization()(net)
    net = Activation("relu")(net)
    net = Conv2D(256, (3, 3), use_bias=False, activation=None, padding="same")(net)
    net = BatchNormalization()(net)
    net = Activation("relu")(net)
    net = Dropout(0.1)(net)

    net = ResizeImageLayer(target_size=(input_shape[0], input_shape[1]))(net)
    # features = ConvUpscaleBlock(features, 128, (3, 3), 2)
    # features = ConvBlock(features, 128, (3, 3))
    # features = ConvUpscaleBlock(features, 64, (3, 3), 2)
    # features = ConvBlock(features, 64, (3, 3))

    output = Conv2D(n_class, (1, 1), activation=None)(net)
    output = Activation("sigmoid")(output)

    return Model(encoder.input, output)
