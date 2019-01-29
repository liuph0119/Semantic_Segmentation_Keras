from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation
from keras.models import Model

from ..utils.net_utils import AtrousSpatialPyramidPooling, ResizeImageLayer
from ..encoder import build_encoder, scope_table

def Deeplab_v3(input_shape=(256, 256, 3), n_class=1, encoder_name="resnet50", encoder_weights="D:/keras_weights/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5"):
    encoder = build_encoder(input_shape, encoder_name, encoder_weights=encoder_weights)

    x = encoder.get_layer(scope_table[encoder_name]["pool4"]).output
    x = AtrousSpatialPyramidPooling(x, 256)

    x = ResizeImageLayer(target_size=(input_shape[0], input_shape[1]))(x)
    x = Conv2D(n_class, (1, 1), activation=None)(x)
    output = Activation("sigmoid")(x)

    return Model(encoder.input, output)