import os
import datetime
import numpy as np
from tqdm import tqdm
from osgeo import gdal
from .data_utils.image_io_utils import load_image, get_image_info, save_to_image_gdal, save_to_image
from .data_utils.label_transform_utils import index_to_color
from .model_utils import load_custom_model
from .vis_utils import plot_segmentation
from core.configures import COLOR_MAP, NAME_MAP


def predict_per_image(model, image_fname, image_dir, pred_dir, image_width=256, image_height=256, to_prob=False,
                      colour_mapping=None, model_name="model", dataset_name="voc"):
    """ predict per image, with no spatial reference
    """
    image = load_image(os.path.join(image_dir, image_fname), is_gray=False, value_scale=255.0, target_size=(image_height, image_width), use_gdal=False)
    pred = model.predict(np.expand_dims(image, axis=0))[0]

    # if save the probability, use gdal to save to geo-tiff files
    if to_prob:
        save_to_image_gdal(pred, os.path.join(pred_dir, image_fname.split(".")[0] + "_" + model_name + ".tif"), datatype=gdal.GDT_Float32)
    else:
        # save to labels and render with colours
        label = np.argmax(pred, axis=-1)
        label_color = index_to_color(label, colour_mapping).astype(np.uint8)
        plot_segmentation(image, label, COLOR_MAP[dataset_name], NAME_MAP[dataset_name])
        save_to_image(label_color, os.path.join(pred_dir, image_fname.split(".")[0] + "_" + model_name + "." + image_fname.split(".")[1]))


def predict_stride(model, image_fname, image_dir, pred_dir, patch_height=256, patch_width=256, stride = 32,
                   to_prob=False, colour_mapping=None, geo=False, model_name="model"):
    """ predict labels of a large image, usually for remote sensing HSR tiles
    :param model: the FCN model instance
    :param src_path: file path of the source hsr image
    :param pred_path: file path to save
    :param img_size: input size
    :param stride: stride of scanning
    :return: None
    """
    # load the image
    image = load_image(os.path.join(image_dir, image_fname), is_gray=False, value_scale=255.0, use_gdal=geo)
    img_info = get_image_info(os.path.join(image_dir, image_fname), get_geotransform=True, get_projection=True)
    h, w, c = image.shape

    # padding with 0
    padding_h = int(np.ceil(h / stride) * stride)
    padding_w = int(np.ceil(w / stride) * stride)
    padding_img = np.zeros((padding_h, padding_w, c))
    padding_img[:h, :w] = image
    print('>> padding image size: ', padding_img.shape[0], padding_img.shape[1])

    n_class = len(colour_mapping)
    mask_probas = np.zeros((padding_h, padding_w, n_class), dtype=np.float)
    mask_counts = np.zeros_like(mask_probas, dtype=np.uint8)
    for i in tqdm(range(padding_h // stride)):
        for j in range(padding_w // stride):
            _image = padding_img[i*stride:i*stride+patch_height, j*stride:j*stride+patch_width]
            _h, _w, _c = _image.shape
            if _h!=patch_height or _w!=patch_width:
                continue
            pred = model.predict(np.expand_dims(_image, axis=0), verbose=2)[0]

            mask_probas[i*stride:i*stride+patch_height, j*stride:j*stride+patch_width] += pred
            mask_counts[i*stride:i*stride+patch_height, j*stride:j*stride+patch_width] += 1

    if to_prob:
        # use gdal to save the probability to geo-tiff images
        save_to_image_gdal(mask_probas[:h, :w] / mask_counts[:h, :w], os.path.join(pred_dir, image_fname.split(".")[0] + "_" + model_name + ".tif"),
                           datatype=gdal.GDT_Float32, geoTransform=img_info["geotransform"], proj=img_info["projection"])
    else:
        # save image labels
        pred_label = index_to_color(np.argmax(mask_probas[:h, :w], axis=-1), colour_mapping)
        # if are spatial images, use gdal to save the labels to geo-tiff images
        if geo:
            save_to_image_gdal(pred_label,  os.path.join(pred_dir, image_fname.split(".")[0] + "_" + model_name + ".tif"), datatype=gdal.GDT_Byte,
                        geoTransform=img_info["geotransform"], proj=img_info["projection"])
        # else, save to common images
        else:
            save_to_image(pred_label, os.path.join(pred_dir, image_fname.split(".")[0] + "_" + model_name + "." + image_fname.split(".")[1]))


def predict_main(args):
    model_name = args["model_name"]
    model_path = args["model_path"]
    image_fnames = os.listdir(args["image_dir"])

    # predicting and save to file
    print("%s: loading network: %s..." % (datetime.datetime.now().strftime("%y-%m-%d %H:%M:%S"), model_path))
    model = load_custom_model(model_path)
    if args["mode"]=="stride":
        for image_fname in image_fnames:
            print("%s: predicting image: %s" % (datetime.datetime.now().strftime("%y-%m-%d %H:%M:%S"), image_fname))
            predict_stride(model, image_fname, args["image_dir"], args["preds_dir"], args["image_height"],
                           args["image_width"], args["stride"], args["to_prob"], COLOR_MAP[args["data_name"]], args["geo"], model_name)
    else:
        for image_fname in tqdm(image_fnames):
            predict_per_image(model, image_fname, args["image_dir"], args["preds_dir"], args["image_height"],
                              args["image_width"], args["to_prob"], COLOR_MAP[args["data_name"]], model_name, args["data_name"])