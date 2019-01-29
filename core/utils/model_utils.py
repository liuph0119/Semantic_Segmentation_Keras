"""
    Script: model_utils.py
    Author: Penghua Liu
    Date: 2019-01-13
    Email: liuphhhh@foxmail.com
    Functions: util function to load custom model

"""

import os
from keras.models import load_model
from .loss_utils import positive_iou_metric, mIoU_metric, lovasz_loss
from .net_utils import ResizeImageLayer, GlobalAveragePooling2D_keepdim


def load_custom_model(model_path):
    """ load custom model, with custom objects like 'ResizeImageLayer', 'mIoU_metric', etc.

    # Args:
        :param model_path: the model path, string.

    # Returns:
        a Keras model instance.
    """
    if os.path.exists(model_path):
        model = load_model(model_path,
                           custom_objects={"ResizeImageLayer": ResizeImageLayer,
                                            "GlobalAveragePooling2D_keepdim": GlobalAveragePooling2D_keepdim,
                                            "positive_iou_metric": positive_iou_metric,
                                            "mIoU_metric": mIoU_metric,
                                            "my_iou_metric": positive_iou_metric,
                                            "lovasz_loss": lovasz_loss})
        return model
    else:
        raise FileNotFoundError("model path: {} does not exist!".format(model_path))
