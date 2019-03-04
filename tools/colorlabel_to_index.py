from core.utils.data_utils.image_io_utils import load_image, save_to_image
from core.utils.data_utils.label_transform_utils import color_to_index, index_to_color
import numpy as np
import json
import os
import matplotlib.pyplot as plt
from tqdm import tqdm


def convert_color_to_index(src_path, color_mapping, src_color_mode="rgb", dst_path=None, plot=False):
    """ convert a colorful label image to a gray (1-channel) image
        (positive index from 1~n, 0 represents background.
        If there is no background classes, there will still be 0 values)
    :param src_path: string
        source label image, rgb/gray color mode
    :param dst_path: string
        destination label image, gray color mode, index from 0 to n (n is the number of non-background classes)
    :param src_color_mode: string, "rgb" or "gray"
        color mode of the source label image
    :param color_mapping: list or array
        a list like [0, 255], [[1, 59, 3], [56, 0, 0]]
    :param plor: bool
        whether to plot comparison

    :return: None
    """
    if color_mapping is None:
        raise ValueError("Invalid color mapping: None. Expected not None!")
    if src_color_mode=="rgb":
        label_color = load_image(src_path, is_gray=False).astype(np.uint8)
    elif src_color_mode=="gray":
        label_color = load_image(src_path, is_gray=True).astype(np.uint8)
    else:
        raise ValueError("Invalid src_color_mode: {}. Expected 'rgb' or 'gray'!".format(src_color_mode))

    label_index = color_to_index(label_color, color_mapping, to_sparse=True)
    if np.max(label_index)>=len(color_mapping):
        raise ValueError("max value is large than: {}：{}".format(len(color_mapping)+1, np.max(label_index)))

    if dst_path:
        save_to_image(label_index, dst_path)

    if plot:
        plt.figure(figsize=(16, 8))

        plt.subplot(1, 2, 1)
        if label_color.shape[-1]==3:
            label_color = label_color/255.0
        else:
            label_color = label_color[:, :, 0]
        plt.imshow(label_color)
        plt.xticks([])
        plt.yticks([])
        plt.title("colorful label")

        plt.subplot(1, 2, 2)
        plt.imshow(label_index, vmin=0, vmax=len(color_mapping)-1)
        plt.xticks([])
        plt.yticks([])
        plt.title("label index")

        plt.show()


def convert_index_to_color(src_path, color_mapping, dst_path=None, plot=False):
    if color_mapping is None:
        raise ValueError("Invalid color mapping: None. Expected not None!")
    label_index = load_image(src_path, is_gray=True).astype(np.uint8)[:, :, 0]
    label_color = index_to_color(label_array=label_index, color_mapping=color_mapping).astype(np.uint8)

    if dst_path is not None:
        save_to_image(label_color, dst_path)
    if plot:
        plt.figure(figsize=(16, 8))

        plt.subplot(1, 2, 1)
        plt.imshow(label_index, vmin=0, vmax=len(color_mapping)-1)
        plt.xticks([])
        plt.yticks([])
        plt.title("label index")

        plt.subplot(1, 2, 2)
        if label_color.ndim == 3:
            label_color = label_color / 255
        plt.imshow(label_color)
        plt.xticks([])
        plt.yticks([])
        plt.title("colorful label")

        plt.show()


if __name__ == "__main__":
    with open("../configures/colour_mapping.json", "r") as f:
        color_mapping = json.load(f)
    color_mapping = np.asarray(color_mapping["voc"])    # color map
    src_color_mode = "rgb"                              # color mode of the source image, "rgb" or "gray
    src_dir = "F:/Data/VOC/SegmentationClass"           # directory of the source labels
    dst_dir = "F:/Data/VOC/SegmentationLabel"           # directory of the destination labels
    _temp_dir = "F:/Data/VOC/_temp/label_color"
    suffix = ".png"                                     # label image suffix
    label_fnames_path = "F:/Data/VOC/train.txt"           # file that contains the base names of the label images, each line is a label name

    # convert colorful label to label index
    with open(label_fnames_path, "r", encoding="utf-8") as f:
        for line in tqdm(f):
            convert_color_to_index(src_path=os.path.join(src_dir, line.strip()+suffix), color_mapping=color_mapping,
                                src_color_mode=src_color_mode, dst_path=os.path.join(dst_dir, line.strip()+suffix), plot=False)
            convert_index_to_color(src_path=os.path.join(dst_dir, line.strip()+suffix), color_mapping=color_mapping,
                                   dst_path=os.path.join(_temp_dir, line.strip()+suffix), plot=False)


    # convert gray label image
    convert_color_to_index(src_path="F:\Data\AerialImageDataset\\val\label\\austin7.tif",
                           color_mapping=np.asarray([0, 255]), src_color_mode="gray",
                           dst_path="F:/Data/VOC/_temp/austin7.png", plot=True)
    convert_index_to_color(src_path="F:/Data/VOC/_temp/austin7.png", color_mapping=np.asarray([0, 255]),
                           dst_path="F:/Data/VOC/_temp/austin7_color.png", plot=True)
