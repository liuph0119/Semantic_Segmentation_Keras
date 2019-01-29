#from .CascadedNets import CascadedUNet_v1, CascadedUNet_v2, CascadedUNet_v3
#from .MyNet import MyNet, MyNet_v2, MyNet_v3, MyNet_v4, MyNet_v4_1
from .Deeplab_v3 import Deeplab_v3
from .Deeplab_v3p import Deeplab_v3p
from .UNets import UNet, ResUNet, MobileUNet
#from .ResUNet_v2 import ResUNet_v2
from .SegNets import SegNet_Basic
from .FCNs import FCN_32s, FCN_16s, FCN_8s
#
from .RefineNets import RefineNet
from .PSPNets import PSPNet
from .DenseASPP import DenseASPP



def SemanticSegmentationModel(encoder_name="resnet50", encoder_weights=None,
                              model_name="MyNet", input_shape=(256, 256, 3),
                              n_class=1, init_filters=16):
    # if model_name.lower()=="mynet":
    #     model = MyNet(input_shape=input_shape, n_class=n_class, init_filters=16)
    # elif model_name.lower()=="mynet_v2":
    #     model = MyNet_v2(input_shape=input_shape, n_class=n_class, encoder_name=encoder_name, encoder_weights=encoder_weights)
    # elif model_name.lower()=="mynet_v3":
    #     model = MyNet_v3(input_shape=input_shape, n_class=n_class, encoder_name=encoder_name, encoder_weights=encoder_weights)
    # elif model_name.lower()=="mynet_v4":
    #     model = MyNet_v4(input_shape=input_shape, n_class=n_class, encoder_name=encoder_name, encoder_weights=encoder_weights)
    # elif model_name.lower() == "mynet_v4_1":
    #     model = MyNet_v4_1(input_shape=input_shape, n_class=n_class, encoder_name=encoder_name, encoder_weights=encoder_weights)
    if model_name.lower()=="deeplab_v3p":
        model = Deeplab_v3p(input_shape=input_shape, n_class=n_class, encoder_name=encoder_name, encoder_weights=encoder_weights)
    elif model_name.lower()=="deeplab_v3":
        model = Deeplab_v3(input_shape=input_shape, n_class=n_class, encoder_name=encoder_name, encoder_weights=encoder_weights)

    elif model_name.lower()=="unet":
        model = UNet(input_shape=input_shape, n_class=n_class, init_filters=init_filters, dropout=0.5)
    elif model_name.lower()=="resunet":
        model = ResUNet(input_shape=input_shape, n_class=n_class, init_filters=init_filters)
    elif model_name.lower()=="mobileunet":
        model = MobileUNet(input_shape=input_shape, n_class=n_class, preset_model="MobileUNet")
    elif model_name.lower()=="mobileunet_skip":
        model = MobileUNet(input_shape=input_shape, n_class=n_class, preset_model="MobileUNet-Skip")

    elif model_name.lower()=="fcn_8s":
        model = FCN_8s(input_shape=input_shape, n_class=n_class, vgg_weight=encoder_weights)
    elif model_name.lower()=="fcn_16s":
        model = FCN_16s(input_shape=input_shape, n_class=n_class, vgg_weight=encoder_weights)
    elif model_name.lower()=="fcn_32s":
        model = FCN_32s(input_shape=input_shape, n_class=n_class, vgg_weight=encoder_weights)
    elif model_name.lower()=="refinenet":
        model = RefineNet(input_shape=input_shape, n_class=n_class, encoder_name=encoder_name, encoder_weights=encoder_weights,
                          init_filters=init_filters, upscaling_method="conv")
    elif model_name.lower()=="pspnet":
        model = PSPNet(input_shape=input_shape, n_class=n_class, encoder_name="resnet50", upscaling_method="conv")
    else:
        raise ValueError("Supported model names: deeplab_v3, deeplab_v3p, unet, resunet, mobileunet, mobileunet_skip, fcn_8s, fcn_16s, fcn_32s, refinenet, pspnet")
    return model
