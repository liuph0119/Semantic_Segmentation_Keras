from .unets import UNet, ResUNet, MobileUNet
from .pspnets import PSPNet
from .deeplabs import Deeplab_v3, Deeplab_v3p
from .dense_aspp import DenseASPP
from .fcns import FCN_8s, FCN_16s, FCN_32s
from .refinenets import RefineNet
from .segnets import SegNet
from .srinets import sri_net


def SemanticSegmentationModel(model_name,
                              input_shape,
                              n_class,
                              encoder_name,
                              encoder_weights=None,
                              init_filters=64,
                              weight_decay=1e-4,
                              kernel_initializer="he_normal",
                              bn_epsilon=1e-3,
                              bn_momentum=0.99,
                              dropout=0.5,
                              upscaling_method="bilinear"):
    """ the main api of model builder.
    :param model_name: string, name of FCN model.
    :param input_shape: tuple, i.e., (height, width, channel).
    :param n_class: int, number of classes, at least 2.
    :param encoder_name: string, name of the encoder.
    :param encoder_weights: string, path of the encoder.
    :param init_filters: int, initial filters, only used for some of the models like U-Net, default 64.
    :param weight_decay: float, default 1e-4.
    :param kernel_initializer: string, default "he_normal".
    :param bn_epsilon: float, default 1e-3.
    :param bn_momentum: float, default 0.99.
    :param dropout: float, default 0.5.
    :param upscaling_method: string, method for up-sampling, one of ['bilinear', 'conv'], default "conv".

    :return: a Keras Model instance.
    """
    model_name = model_name.lower()
    if model_name == "unet":
        model = UNet(input_shape=input_shape, n_class=n_class,
                     weight_decay=weight_decay, kernel_initializer=kernel_initializer,
                       bn_momentum=bn_momentum, bn_epsilon=bn_epsilon,
                     init_filters=init_filters, dropout=dropout)
    elif model_name == "resunet":
        model = ResUNet(input_shape=input_shape, n_class=n_class,
                        weight_decay=weight_decay, kernel_initializer=kernel_initializer,
                       bn_momentum=bn_momentum, bn_epsilon=bn_epsilon,
                        init_filters=init_filters, dropout=dropout)
    elif model_name == "mobile_unet":
        model = MobileUNet(input_shape=input_shape, n_class=n_class,
                           weight_decay=weight_decay, kernel_initializer=kernel_initializer,
                       bn_momentum=bn_momentum, bn_epsilon=bn_epsilon, preset_model="MobileUNet-Skip")
    elif model_name == "pspnet":
        model = PSPNet(input_shape=input_shape, n_class=n_class,
                       encoder_name=encoder_name, encoder_weights=encoder_weights,
                       weight_decay=weight_decay, kernel_initializer=kernel_initializer,
                       bn_momentum=bn_momentum, bn_epsilon=bn_epsilon, upscaling_method=upscaling_method)
    elif model_name == "refinenet":
        model = RefineNet(input_shape=input_shape, n_class=n_class,
                          encoder_name=encoder_name, encoder_weights=encoder_weights,
                          weight_decay=weight_decay, kernel_initializer=kernel_initializer,
                          bn_momentum=bn_momentum, bn_epsilon=bn_epsilon,
                          init_filters=init_filters, upscaling_method=upscaling_method)
    elif model_name == "deeplab_v3":
        model = Deeplab_v3(input_shape=input_shape, n_class=n_class,
                           encoder_name=encoder_name, encoder_weights=encoder_weights,
                           weight_decay=weight_decay, kernel_initializer=kernel_initializer,
                           bn_momentum=bn_momentum, bn_epsilon=bn_epsilon)
    elif model_name == "deeplab_v3p":
        model = Deeplab_v3p(input_shape=input_shape, n_class=n_class,
                            encoder_name=encoder_name, encoder_weights=encoder_weights,
                            weight_decay=weight_decay, kernel_initializer=kernel_initializer,
                            bn_momentum=bn_momentum, bn_epsilon=bn_epsilon)
    elif model_name == "dense_aspp":
        model = DenseASPP(input_shape=input_shape, n_class=n_class,
                          encoder_name=encoder_name, encoder_weights=encoder_weights,
                          weight_decay=weight_decay, kernel_initializer=kernel_initializer,
                          bn_momentum=bn_momentum, bn_epsilon=bn_epsilon)
    elif model_name == "fcn_8s":
        model = FCN_8s(input_shape=input_shape, n_class=n_class,
                       encoder_name=encoder_name, encoder_weights=encoder_weights,
                       weight_decay=weight_decay, kernel_initializer=kernel_initializer,
                       bn_momentum=bn_momentum, bn_epsilon=bn_epsilon, dropout=dropout)
    elif model_name == "fcn_16s":
        model = FCN_16s(input_shape=input_shape, n_class=n_class,
                        encoder_name=encoder_name, encoder_weights=encoder_weights,
                        weight_decay=weight_decay, kernel_initializer=kernel_initializer,
                        bn_momentum=bn_momentum, bn_epsilon=bn_epsilon, dropout=dropout)
    elif model_name == "fcn_32s":
        model = FCN_32s(input_shape=input_shape, n_class=n_class,
                        encoder_name=encoder_name, encoder_weights=encoder_weights,
                        weight_decay=weight_decay, kernel_initializer=kernel_initializer,
                        bn_momentum=bn_momentum, bn_epsilon=bn_epsilon, dropout=dropout)
    elif model_name == "segnet":
        model = SegNet(input_shape=input_shape, n_class=n_class,
                       encoder_name=encoder_name, encoder_weights=encoder_weights,
                       weight_decay=weight_decay, kernel_initializer=kernel_initializer,
                       bn_momentum=bn_momentum, bn_epsilon=bn_epsilon)
    elif model_name == "srinet":
        model = sri_net(input_shape=input_shape, n_class=n_class,
                        encoder_name=encoder_name, encoder_weights=encoder_weights,
                        weight_decay=weight_decay, kernel_initializer=kernel_initializer,
                        bn_epsilon=bn_epsilon, bn_momentum=bn_momentum)
    else:
        raise ValueError("Invalid model_name: {}. Expected to be one of ['fcn_8s', 'fcn_16s', 'fcn_32s', 'segnet', "
                         "'unet', 'resunet', 'mobile_unet', 'pspnet', 'refinenet', 'deeplab_v3, 'deeplab_v3p', "
                         "'dense_aspp']!!!".format(model_name))

    return model
