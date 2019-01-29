"""
    Script: predicting_utils.py
    Author: Penghua Liu
    Date: 2019-01-14
    Email: liuphhhh@foxmail.com
    Functions: function for predicting a large hsr image

"""
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal

from .image_utils import load_image
from .loss_utils import compute_positive_iou
from .gdal_utils import getGeoInfomation, arr_to_tif
# from .crf_postprocess import do_crf_inference

def predict_buildingfootprint(model, src_path, pred_path, img_size = 256, stride = 32, softmax=1):
    """ predict building footprint of a large hsr image using a window scanning method
    :param model: the FCN model instance
    :param src_path: file path of the source hsr image
    :param pred_path: file path to save
    :param img_size: input size
    :param stride: stride of scanning
    :return: None
    """
    # load the image
    image = load_image(src_path, grayscale=False, scale=255.0)
    geoTransform, proj = getGeoInfomation(src_path)
    h, w, _ = image.shape

    # padding with 0
    padding_h = int(np.ceil(h / stride) * stride)
    padding_w = int(np.ceil(w / stride) * stride)
    padding_img = np.zeros((padding_h, padding_w, 3))
    padding_img[:h, :w, :] = image
    print('>> source image shape: ', padding_img.shape)

    mask_probas = np.zeros((padding_h, padding_w), dtype=np.float)
    mask_counts = np.zeros_like(mask_probas, dtype=np.uint8)
    for i in tqdm(range(padding_h // stride)):
        for j in range(padding_w // stride):
            image_patch = padding_img[i * stride:i * stride + img_size, j * stride:j * stride + img_size]
            ch, cw, _ = image_patch.shape
            if ch != img_size or cw != img_size:
                continue
            image_patch = np.expand_dims(image_patch, axis=0)
            pred = model.predict(image_patch, verbose=2)
            pred = pred.reshape((img_size, img_size))

            mask_probas[i * stride:i * stride + img_size, j * stride:j * stride + img_size] += pred
            mask_counts[i * stride:i * stride + img_size, j * stride:j * stride + img_size] += 1

    if softmax==1:
        # save the probability to tif, scaled to 0~255, unsigned char 8 bit
        arr_to_tif(np.array(mask_probas[:h, :w] / mask_counts[:h, :w]*255, dtype=np.uint8), pred_path, datatype=gdal.GDT_Byte,
               geoTransform=geoTransform, proj=proj)
    else:
        # save the segmented label to tif, 0 for negative class, 255 for positive class
        arr_to_tif(((mask_probas[:h, :w] / mask_counts[:h, :w])>0.5).astype(np.uint8)*255, pred_path, datatype=gdal.GDT_Byte,
               geoTransform=geoTransform, proj=proj)



def getOptimalIoUThreshold(img_true, img_pred, thresholds, plot=True):
    """ get the optimal threshold to get the highest iou
    :param img_true: ground truth, 0/1
    :param img_pred: prediction of probability, 0~1
    :param thresholds: list of thresholds, e.g., [0.1, 0.2, ..., 1.0]
    :return: the best iou and the best threshold
    """
    ious = [compute_positive_iou(img_true, img_pred > threshold) for threshold in thresholds]
    best_iou = np.argmax(ious)
    best_threshold = thresholds[np.argmax(ious)]
    print(
        "best iou is achieved when threshold is set to {:.6f}, the best iou is {:.6f}".format(best_threshold, best_iou))

    if plot:
        plt.plot(thresholds, ious)
        plt.plot(best_threshold, best_iou, "xr")
        plt.show()

    return (best_iou, best_threshold)
