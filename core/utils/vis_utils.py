import numpy as np
from matplotlib import gridspec
import matplotlib.pyplot as plt

from .data_utils.label_transform_utils import index_to_color

plt.rcParams['font.sans-serif'] = ['Cambria Math']
FONT_SIZE = 12


def plot_image_label(rgb_img, label_img, vmin, vmax, names):
    """ plot a rgb image and a label image.
    :param rgb_img: 3-D array or a PIL instance
    :param label_img: 2-D array
    :param vmin: int, minimum value.
    :param vmax: int, maximum value.
    :param names: 1-D array.

    :return: None
    """
    grid_spec = gridspec.GridSpec(1, 3, width_ratios=[6, 6, 1])
    # plot rgb image
    plt.subplot(grid_spec[0])
    plt.imshow(rgb_img)
    plt.axis('off')
    plt.title('image', fontdict={"fontsize": 12})

    # plot label image
    plt.subplot(grid_spec[1])
    plt.imshow(rgb_img)
    plt.imshow(label_img, vmin=vmin, vmax=vmax, alpha=0.7)
    plt.axis('off')
    plt.title('label', fontdict={"fontsize": 12})

    unique_labels = np.unique(label_img)
    FULL_LABEL_MAP = np.arange(len(names)).reshape(len(names), 1)
    ax = plt.subplot(grid_spec[2])
    plt.imshow(
        FULL_LABEL_MAP[unique_labels].astype(np.uint8), interpolation='nearest', vmin=vmin, vmax=vmax)
    ax.yaxis.tick_right()
    plt.yticks(range(len(unique_labels)), np.array(names)[unique_labels])
    plt.xticks([], [])
    ax.tick_params(width=0.0)
    plt.grid('off')

    plt.show()


def plot_segmentation(image, seg_map, colors, names):
    """
    :param image: 3-D array, while depth==3. source image.
    :param seg_map: 2-D array, the sparse label index (start from 0 to n_class-1).
    :param colors: 2-D array, color map.
    :param names: 1-D array.

    :return: None
    """
    FULL_LABEL_MAP = np.arange(len(names)).reshape(len(names), 1)
    FULL_COLOR_MAP = index_to_color(FULL_LABEL_MAP, colors)
    plt.figure(figsize=(15, 5))
    grid_spec = gridspec.GridSpec(1, 4, width_ratios=[6, 6, 6, 1])

    plt.subplot(grid_spec[0])
    plt.imshow(image)
    plt.axis('off')
    plt.title('input image')

    plt.subplot(grid_spec[1])
    seg_image = index_to_color(seg_map, colors).astype(np.uint8)
    plt.imshow(seg_image)
    plt.axis('off')
    plt.title('segmentation map')

    plt.subplot(grid_spec[2])
    plt.imshow(image)
    plt.imshow(seg_image, alpha=0.7)
    plt.axis('off')
    plt.title('segmentation overlay')

    unique_labels = np.unique(seg_map)
    ax = plt.subplot(grid_spec[3])
    plt.imshow(
        FULL_COLOR_MAP[unique_labels].astype(np.uint8), interpolation='nearest')
    ax.yaxis.tick_right()
    plt.yticks(range(len(unique_labels)), np.asarray(names)[unique_labels])
    plt.xticks([], [])
    ax.tick_params(width=0.0)
    plt.grid('off')
    plt.show()