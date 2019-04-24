import os
import datetime
import numpy as np
from tqdm import tqdm
from osgeo import gdal
import sys
sys.path.append('.')

from core.utils.data_utils.image_io_utils import load_image, get_image_info, save_to_image_gdal, save_to_image
from core.utils.vis_utils import plot_segmentation
from core.configures import COLOR_MAP, NAME_MAP, predicting_config, net_config
from core.nets import SemanticSegmentationModel


def predict_stride(model, image_path, patch_height=256, patch_width=256, stride=None, to_prob=False, plot=False,
                   geo=False, dataset_name="voc"):
    """ predict labels of a large image, usually for remote sensing HSR tiles
        or for images that size are not consistent with the required input size.
    :param model: Keras Model instance
    :param image_path: string, path of input image.
    :param patch_height: int, default 256.
    :param patch_width: int, default 256.
    :param stride: int, default None.
    :param to_prob: bool, whether to return probability.
    :param plot: bool, whether to plot.
    :param geo: bool, whether to load geo images.
    :param: bool, whether to plot.
    :param dataset_name: string.

    :return: probability or color of prediction.
    """
    # load the image
    image = load_image(image_path, is_gray=False, value_scale=1.0, use_gdal=geo)

    h, w, c = image.shape
    if stride is None:
        stride = w//4
    # padding with 0
    padding_h = int(np.ceil(h / stride) * stride)
    padding_w = int(np.ceil(w / stride) * stride)
    padding_img = np.zeros((padding_h, padding_w, c), dtype=np.float16)
    padding_img[:h, :w] = image

    colour_mapping = COLOR_MAP[dataset_name]
    n_class = len(colour_mapping)
    mask_probas = np.zeros((padding_h, padding_w, n_class), dtype=np.float16)
    mask_counts = np.zeros_like(mask_probas, dtype=np.uint8)
    for i in range(padding_h // stride):
        for j in range(padding_w // stride):
            _image = padding_img[i*stride:i*stride+patch_height, j*stride:j*stride+patch_width]
            _h, _w, _c = _image.shape
            if _h != patch_height or _w != patch_width:
                continue
            pred = model.predict(np.expand_dims(_image, axis=0), verbose=2)[0]

            mask_probas[i*stride:i*stride+patch_height, j*stride:j*stride+patch_width] += pred
            mask_counts[i*stride:i*stride+patch_height, j*stride:j*stride+patch_width] += 1

    mask_probas[:h, :w] = mask_probas[:h, :w] / mask_counts[:h, :w]
    if to_prob:
        return mask_probas[:h, :w]
    else:
        # save image labels
        label = np.argmax(mask_probas[:h, :w], axis=-1)
        if plot:
            plot_segmentation(image / 255.0, label, COLOR_MAP[dataset_name], NAME_MAP[dataset_name])
        return label


def predict_per_image(model, image_path, image_width=256, image_height=256, to_prob=False, plot=False,
                      dataset_name="voc"):
    """ predict per image, with no spatial reference
    """
    # load image
    image = load_image(image_path, is_gray=False, value_scale=1, target_size=(image_height, image_width),
                       use_gdal=False)
    # predict
    pred = model.predict(np.expand_dims(image, axis=0))[0]

    if to_prob:
        return pred
    else:
        # save to labels and render with colours
        label = np.argmax(pred, axis=-1)
        if plot:
            plot_segmentation(image/255.0, label, COLOR_MAP[dataset_name], NAME_MAP[dataset_name])
        return label


def save_prediction(arr, dst_path, to_prob=False, with_geo=False, spatial_reference=None):
    """ save prediction.
    :param arr: 2D/3D array.
    :param dst_path: string.
    :param to_prob: bool, whether to save probability. If True, a *.npy or a float-formatted tiff file will be created.
    :param with_geo: bool, whether to save spatial reference.
    :param spatial_reference: string, or dict. If the data type is string, this must be a path of a existing tiff file,
        else, must be a dict contains "geotransform" and "projection"
    :return:
    """
    if with_geo:
        if type(spatial_reference) is str and os.path.exists(spatial_reference):
            img_info = get_image_info(spatial_reference, get_geotransform=True, get_projection=True)
        elif type(spatial_reference) is dict \
                and "geotransform" in spatial_reference and "projection" in spatial_reference:
            img_info = spatial_reference
        else:
            raise ValueError("Invalid 'spatial_reference': None. Expected to be a dict or a existing tiff image path.")
        if to_prob:
            save_to_image_gdal(arr, dst_path, datatype=gdal.GDT_Float32,
                               geoTransform=img_info["geotransform"], proj=img_info["projection"])
        else:
            save_to_image_gdal(arr, dst_path, datatype=gdal.GDT_Byte,
                               geoTransform=img_info["geotransform"], proj=img_info["projection"])
    else:
        if to_prob:
            if ".npy" not in dst_path:
                dst_path = dst_path + ".npy"
            np.save(dst_path, arr)
        else:
            save_to_image(arr, dst_path)


def predict_main():
    model_path = predicting_config.model_path
    image_fnames = os.listdir(predicting_config.image_dir)

    # predicting and save to file
    model = SemanticSegmentationModel(model_name=predicting_config.model_name,
                                      input_shape=(predicting_config.image_height,
                                                   predicting_config.image_width,
                                                   predicting_config.image_channel),
                                      n_class=len(COLOR_MAP[predicting_config.dataset_name]),
                                      encoder_name=predicting_config.encoder_name,
                                      encoder_weights=None,
                                      init_filters=net_config.init_filters,
                                      dropout=net_config.dropout,
                                      weight_decay=net_config.weight_decay,
                                      kernel_initializer=net_config.kernel_initializer,
                                      bn_epsilon=net_config.bn_epsilon,
                                      bn_momentum=net_config.bn_momentum,
                                      upscaling_method=net_config.upsampling_method)
    print("%s: loading network: %s..." % (datetime.datetime.now().strftime("%y-%m-%d %H:%M:%S"), model_path))
    model.load_weights(model_path)

    if predicting_config.mode == "stride":
        for image_fname in tqdm(image_fnames):
            src_fname = os.path.join(predicting_config.image_dir, image_fname)
            dst_fname = os.path.join(predicting_config.preds_dir, image_fname)
            pred = predict_stride(model, src_fname, predicting_config.image_height, predicting_config.image_width,
                                  predicting_config.stride, predicting_config.to_prob, predicting_config.plot,
                                  predicting_config.geo, predicting_config.dataset_name)
            save_prediction(pred, dst_fname, predicting_config.to_prob, predicting_config.geo, src_fname)
    else:
        for image_fname in tqdm(image_fnames):
            src_fname = os.path.join(predicting_config.image_dir, image_fname)
            dst_fname = os.path.join(predicting_config.preds_dir, image_fname)
            pred = predict_per_image(model, src_fname, predicting_config.image_height, predicting_config.image_width,
                                     predicting_config.to_prob, predicting_config.plot, predicting_config.dataset_name)
            save_prediction(pred, dst_fname, predicting_config.to_prob, predicting_config.geo, src_fname)


if __name__ == "__main__":
    print(">>>> Predicting...")
    predict_main()
