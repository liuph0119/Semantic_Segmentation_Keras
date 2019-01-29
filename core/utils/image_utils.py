"""
    Script: image_utils.py
    Author: Penghua Liu
    Date: 2019-01-13
    Email: liuphhhh@foxmail.com
    Functions: some util functions for loading and plotting images

"""

import cv2
import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img, img_to_array

def load_image(fname, grayscale=False, scale=1.0, target_size=None):
    """ trying load images using PIL and OpenCV

    # Args:
        :param fname: input image file name
        :param grayscale: True or False
        :param scale: the scale to divide the values
        :param target_size: resize size

    # Returns:
        3-dim array if grayscale=False, else 2-dim array
    """

    if grayscale:
        try:
            img = img_to_array(load_img(fname, color_mode="grayscale", target_size=target_size))[:, :, 0]
        except:
            img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
            if target_size is not None:
                img = cv2.resize(img, target_size)
    else:
        try:
            img = img_to_array(load_img(fname, target_size=target_size))
        except:
            img = cv2.cvtColor(cv2.imread(fname), cv2.COLOR_BGR2RGB)
            if target_size is not None:
                img = cv2.resize(img, target_size)

    return img / scale


def plot_rgb_img(img):
    """ plot a rgb image
    :param img: a 3-d array or a PIL instance
    :return: None
    """
    assert img.ndim==3 and (img.shape[-1]==3 or img.shape[-1]==4)
    plt.imshow(img)
    plt.show()


def plot_label_img(img, ax=None):
    """ plot a label image
    :param img: a 2-d array
    :param ax: the plot handler
    :return: None
    """
    assert img.ndim==2
    if ax is None:
        plt.imshow(img)
        plt.show()
    else:
        ax.imshow(img)


def plot_rgb_label(rgb_img, label_img, label_alpha=0.5, ax=None):
    """ plot a rgb image and a label image
    :param rgb_img: a 3-d array or a PIL instance
    :param label_img: a 2-d array
    :param label_alpha: the transparency of the label image
    :param ax: the plot handler
    :return: None
    """
    if ax is None:
        plt.imshow(rgb_img)
        plt.imshow(label_img, alpha=label_alpha, cmap="Reds")
        plt.show()
    else:
        ax.imshow(rgb_img)
        ax.imshow(label_img, alpha=label_alpha, cmap="Reds")