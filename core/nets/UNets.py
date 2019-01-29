from keras.layers.convolutional import Conv2D, Conv2DTranspose, SeparableConv2D
from keras.layers.core import Dropout, Activation
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate, Add
from keras.layers.normalization import BatchNormalization
from keras.layers import Input
from keras.models import Model

def UNet(input_shape=(256, 256, 3), n_class=1, init_filters=64, dropout=0.25):
    input_x = Input(shape=input_shape)

    conv1 = Conv2D(init_filters * 1, (3, 3), activation='relu', padding='same')(input_x)
    conv1 = Dropout(dropout)(conv1)
    conv1 = Conv2D(init_filters * 1, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D()(conv1)

    conv2 = Conv2D(init_filters * 2, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Dropout(dropout)(conv2)
    conv2 = Conv2D(init_filters * 2, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D()(conv2)

    conv3 = Conv2D(init_filters * 4, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Dropout(dropout)(conv3)
    conv3 = Conv2D(init_filters * 4, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D()(conv3)

    conv4 = Conv2D(init_filters * 8, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Dropout(dropout)(conv4)
    conv4 = Conv2D(init_filters * 8, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D()(conv4)

    conv5 = Conv2D(init_filters * 16, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Dropout(dropout)(conv5)
    conv5 = Conv2D(init_filters * 16, (3, 3), activation='relu', padding='same')(conv5)

    up1 = concatenate([Conv2DTranspose(init_filters * 8, (3, 3), padding="same", strides=(2, 2))(conv5), conv4])
    conv6 = Conv2D(init_filters * 8, (3, 3), activation='relu', padding='same')(up1)
    conv6 = Dropout(dropout)(conv6)
    conv6 = Conv2D(init_filters * 8, (3, 3), activation='relu', padding='same')(conv6)

    up2 = concatenate([Conv2DTranspose(init_filters * 4, (3, 3), padding="same", strides=(2, 2))(conv6), conv3])
    conv7 = Conv2D(init_filters * 4, (3, 3), activation='relu', padding='same')(up2)
    conv7 = Dropout(dropout)(conv7)
    conv7 = Conv2D(init_filters * 4, (3, 3), activation='relu', padding='same')(conv7)

    up3 = concatenate([Conv2DTranspose(init_filters * 2, (3, 3), padding="same", strides=(2, 2))(conv7), conv2])
    conv8 = Conv2D(init_filters * 2, (3, 3), activation='relu', padding='same')(up3)
    conv8 = Dropout(dropout)(conv8)
    conv8 = Conv2D(init_filters * 2, (3, 3), activation='relu', padding='same')(conv8)

    up4 = concatenate([Conv2DTranspose(init_filters, (3, 3), padding="same", strides=(2, 2))(conv8), conv1])
    conv9 = Conv2D(init_filters, (3, 3), activation='relu', padding='same')(up4)
    # conv9 = Dropout(dropout)(conv9)
    conv9 = Conv2D(init_filters, (3, 3), activation='relu', padding='same')(conv9)

    output = Conv2D(n_class, (1, 1), activation='relu', padding='same')(conv9)
    unet_model = Model(input_x, output)

    return unet_model



def convolution_block(x, filters, size, strides=(1,1), padding='same', activation=True):
    x = Conv2D(filters, size, strides=strides, padding=padding)(x)
    if activation == True:
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
    return x


def residual_block(blockInput, num_filters=16, batch_activate = False):
    x = BatchNormalization()(blockInput)
    x = Activation("relu")(x)
    x = convolution_block(x, num_filters, (3,3) )
    x = convolution_block(x, num_filters, (3,3), activation=False)
    x = Add()([x, blockInput])
    if batch_activate:
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
    return x


def ResUNet(input_shape=(256, 256, 3), n_class=1, init_filters=64, dropout=0.5):
    input_x = Input(shape=input_shape)

    conv1 = Conv2D(init_filters * 1, (3, 3), activation=None, padding="same")(input_x)
    conv1 = residual_block(conv1, init_filters * 1)
    conv1 = residual_block(conv1, init_filters * 1, True)
    pool1 = MaxPooling2D((2, 2))(conv1)
    pool1 = Dropout(dropout / 2)(pool1)

    conv2 = Conv2D(init_filters * 2, (3, 3), activation=None, padding="same")(pool1)
    conv2 = residual_block(conv2, init_filters * 2)
    conv2 = residual_block(conv2, init_filters * 2, True)
    pool2 = MaxPooling2D((2, 2))(conv2)
    pool2 = Dropout(dropout)(pool2)

    conv3 = Conv2D(init_filters * 4, (3, 3), activation=None, padding="same")(pool2)
    conv3 = residual_block(conv3, init_filters * 4)
    conv3 = residual_block(conv3, init_filters * 4, True)
    pool3 = MaxPooling2D((2, 2))(conv3)
    pool3 = Dropout(dropout)(pool3)

    conv4 = Conv2D(init_filters * 8, (3, 3), activation=None, padding="same")(pool3)
    conv4 = residual_block(conv4, init_filters * 8)
    conv4 = residual_block(conv4, init_filters * 8, True)
    pool4 = MaxPooling2D((2, 2))(conv4)
    pool4 = Dropout(dropout)(pool4)

    convm = Conv2D(init_filters * 16, (3, 3), activation=None, padding="same")(pool4)
    convm = residual_block(convm, init_filters * 16)
    convm = residual_block(convm, init_filters * 16, True)

    # 16 -> 32
    deconv4 = Conv2DTranspose(init_filters * 8, (3, 3), strides=(2, 2), padding="same")(convm)
    uconv4 = concatenate([deconv4, conv4])
    uconv4 = Dropout(dropout)(uconv4)

    uconv4 = Conv2D(init_filters * 8, (3, 3), activation=None, padding="same")(uconv4)
    uconv4 = residual_block(uconv4, init_filters * 8)
    uconv4 = residual_block(uconv4, init_filters * 8, True)

    # 32 -> 64
    # deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="valid")(uconv4)
    deconv3 = Conv2DTranspose(init_filters * 4, (3, 3), strides=(2, 2), padding="same")(uconv4)
    uconv3 = concatenate([deconv3, conv3])
    uconv3 = Dropout(dropout)(uconv3)

    uconv3 = Conv2D(init_filters * 4, (3, 3), activation=None, padding="same")(uconv3)
    uconv3 = residual_block(uconv3, init_filters * 4)
    uconv3 = residual_block(uconv3, init_filters * 4, True)

    # 64 -> 128
    deconv2 = Conv2DTranspose(init_filters * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
    uconv2 = concatenate([deconv2, conv2])

    uconv2 = Dropout(dropout)(uconv2)
    uconv2 = Conv2D(init_filters * 2, (3, 3), activation=None, padding="same")(uconv2)
    uconv2 = residual_block(uconv2, init_filters * 2)
    uconv2 = residual_block(uconv2, init_filters * 2, True)

    # 128 -> 256
    # deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="valid")(uconv2)
    deconv1 = Conv2DTranspose(init_filters * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
    uconv1 = concatenate([deconv1, conv1])

    uconv1 = Dropout(dropout)(uconv1)
    uconv1 = Conv2D(init_filters * 1, (3, 3), activation=None, padding="same")(uconv1)
    uconv1 = residual_block(uconv1, init_filters * 1)
    uconv1 = residual_block(uconv1, init_filters * 1, True)

    output = Conv2D(n_class, (1, 1), padding="same", activation="sigmoid")(uconv1)

    return Model(input_x, output)

# # # ===========================================================================================================

def ConvBlock(inputs,  n_filters, kernel_size=(3, 3)):
    x = Conv2D(n_filters, kernel_size, padding="same", activation=None)(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x


def DepthwiseSeparableConvBlock(inputs, n_filters):
    x = SeparableConv2D(inputs, (3, 3), activation=None, padding="same", depth_multiplier=1)(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(n_filters, (1, 1), activation=None)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x


def conv_transpose_block(inputs, n_filters):
    x = Conv2DTranspose(n_filters, (3, 3), padding="same", strides=(2, 2), activation=None)(inputs)
    x = BatchNormalization()(x)
    return x


def MobileUNet(input_shape=(256, 256, 3), n_class=1, preset_model="MobileUNet"):
    if preset_model == "MobileUNet":
        has_skip = False
    elif preset_model == "MobileUNet-Skip":
        has_skip = True
    else:
        raise ValueError(
            "Unsupported MobileUNet model '%s'. This function only supports MobileUNet and MobileUNet-Skip" % (
                preset_model))

    inputs = Input(shape=input_shape)
    x = ConvBlock(inputs, 64)
    x = DepthwiseSeparableConvBlock(x, 64)
    x = MaxPooling2D()(x)
    skip_1 = x

    x = DepthwiseSeparableConvBlock(x, 128)
    x = DepthwiseSeparableConvBlock(x, 128)
    x = MaxPooling2D()(x)
    skip_2 = x

    x = DepthwiseSeparableConvBlock(x, 256)
    x = DepthwiseSeparableConvBlock(x, 256)
    x = DepthwiseSeparableConvBlock(x, 256)
    x = MaxPooling2D()(x)
    skip_3 = x

    x = DepthwiseSeparableConvBlock(x, 512)
    x = DepthwiseSeparableConvBlock(x, 512)
    x = DepthwiseSeparableConvBlock(x, 512)
    x = MaxPooling2D()(x)
    skip_4 = x

    x = DepthwiseSeparableConvBlock(x, 512)
    x = DepthwiseSeparableConvBlock(x, 512)
    x = DepthwiseSeparableConvBlock(x, 512)
    x = MaxPooling2D()(x)

    x = conv_transpose_block(x, 512)
    x = DepthwiseSeparableConvBlock(x, 512)
    x = DepthwiseSeparableConvBlock(x, 512)
    x = DepthwiseSeparableConvBlock(x, 512)
    if has_skip:
        x = Add()([x, skip_4])

    x = conv_transpose_block(x, 512)
    x = DepthwiseSeparableConvBlock(x, 512)
    x = DepthwiseSeparableConvBlock(x, 512)
    x = DepthwiseSeparableConvBlock(x, 256)
    if has_skip:
        x = Add()([x, skip_3])

    x = conv_transpose_block(x, 256)
    x = DepthwiseSeparableConvBlock(x, 256)
    x = DepthwiseSeparableConvBlock(x, 256)
    x = DepthwiseSeparableConvBlock(x, 128)
    if has_skip:
        x = Add()([x, skip_2])

    x = conv_transpose_block(x, 128)
    x = DepthwiseSeparableConvBlock(x, 128)
    x = DepthwiseSeparableConvBlock(x, 128)
    x = DepthwiseSeparableConvBlock(x, 64)
    if has_skip:
        x = Add()([x, skip_1])

    x = conv_transpose_block(x, 64)
    x = DepthwiseSeparableConvBlock(x, 64)
    x = DepthwiseSeparableConvBlock(x, 64)
    
    x = Conv2D(n_class, (1, 1), activation=None)(x)
    output = Activation("sigmoid")(x)
    
    return Model(inputs, output)
