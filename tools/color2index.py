import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import gridspec

import sys
sys.path.append('.')

from core.configures import COLOR_MAP, NAME_MAP, color2index_config
from core.utils.vis_utils import plot_image_label
from core.utils.data_utils.image_io_utils import load_image, save_to_image
from core.utils.data_utils.label_transform_utils import color_to_index, index_to_color


def convert_color_to_index(src_path, color_mapping, src_color_mode='rgb', dst_path=None, plot=False, names=None):
    """ convert a colorful label image to a gray (1-channel) image
        (positive index from 1~n, 0 represents background.
        If there is no background classes, there will still be 0 values)
    :param src_path: string
        path of source label image, rgb/gray color mode
    :param dst_path: string
        path of destination label image, gray color mode, index from 0 to n (n is the number of non-background classes)
    :param src_color_mode: string, "rgb" or "gray", default "rgb"
        color mode of the source label image
    :param color_mapping: list or array, default None
        a list like [0, 255], [[1, 59, 3], [56, 0, 0]]
    :param plor: bool, default False
        whether to plot comparison
    :param names: list.

    :return: None
    """
    if color_mapping is None:
        raise ValueError('Invalid color mapping: None. Expected not None!')
    if src_color_mode=='rgb':
        label_color = load_image(src_path, is_gray=False).astype(np.uint8)
    elif src_color_mode=='gray':
        label_color = load_image(src_path, is_gray=True).astype(np.uint8)
    else:
        raise ValueError('Invalid src_color_mode: {}. Expected "rgb" or "gray"!'.format(src_color_mode))

    label_index = color_to_index(label_color, color_mapping, to_sparse=True)
    if np.max(label_index)>=len(color_mapping):
        raise ValueError('max value is large than: {}：{}'.format(len(color_mapping)+1, np.max(label_index)))

    if dst_path:
        save_to_image(label_index, dst_path)

    if plot:
        if names is None:
            names = ['class_{}'.format(i) for i in range(len(color_mapping))]
        if label_color.shape[-1]==1:
            label_color = label_color[:, :, 0]
        plot_image_label(label_color, label_index, 0, len(color_mapping)-1, names, overlay=False)


def convert_index_to_color(src_path, color_mapping, dst_path=None, plot=False, names=None):
    """ recover label index to colorful image
    :param src_path: string, path of source label image, gray colored.
    :param color_mapping: list. for example [0, 255], or [[52, 33, 24], [60, 95, 87]].
    :param dst_path: string, path of destination image, default None
    :param plot: bool, whether to plot.

    :return: None
    """
    if color_mapping is None:
        raise ValueError('Invalid color mapping: None. Expected not None!')
    label_index = load_image(src_path, is_gray=True).astype(np.uint8)[:, :, 0]
    label_color = index_to_color(label_array=label_index, color_mapping=color_mapping).astype(np.uint8)

    if dst_path is not None:
        save_to_image(label_color, dst_path)
    if plot:
        grid_spec = gridspec.GridSpec(1, 3, width_ratios=[6, 6, 1])

        plt.subplot(grid_spec[0])
        plt.imshow(label_index, vmin=0, vmax=len(color_mapping)-1)
        plt.axis('off')
        plt.title('label index')

        plt.subplot(grid_spec[1])
        if label_color.ndim == 3:
            label_color = label_color / 255
        plt.imshow(label_color)
        plt.axis('off')
        plt.title('colorful label')

        ax = plt.subplot(grid_spec[2])
        if names is None:
            names = ['class_{}'.format(i) for i in range(len(color_mapping))]
        FULL_LABEL_MAP = np.arange(len(names)).reshape(len(names), 1)
        FULL_COLOR_MAP = index_to_color(FULL_LABEL_MAP, color_mapping)
        unique_labels = np.unique(label_index)
        plt.imshow(
            FULL_COLOR_MAP[unique_labels].astype(np.uint8), interpolation='nearest')
        ax.yaxis.tick_right()
        plt.yticks(range(len(unique_labels)), np.asarray(names)[unique_labels])
        plt.xticks([], [])
        ax.tick_params(width=0.0)
        plt.grid('off')

        plt.show()


if __name__ == "__main__":
    color_mapping = COLOR_MAP[color2index_config.dataset_name]
    name_mapping = NAME_MAP[color2index_config.dataset_name]
    src_color_mode = color2index_config.color_mode
    src_dir = color2index_config.src_dir
    dst_dir = color2index_config.dst_dir

    fnames = os.listdir(src_dir)
    for fname in tqdm(fnames):
        if color2index_config.mode == 'color2index':
            convert_color_to_index(src_path=os.path.join(src_dir, fname.strip()), color_mapping=color_mapping,
                                   src_color_mode=src_color_mode, dst_path=os.path.join(dst_dir, fname.strip()),
                                   plot=color2index_config.show_comparison, names=name_mapping)
        elif color2index_config.mode == 'index2color':
            convert_index_to_color(src_path=os.path.join(src_dir, fname.strip()), color_mapping=color_mapping,
                                   dst_path=os.path.join(dst_dir, fname.strip()),
                                   plot=color2index_config.show_comparison, names=name_mapping)
