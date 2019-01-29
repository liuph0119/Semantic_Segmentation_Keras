"""
Implementation of Data Augmentation, including:
    (1) up-down flip, left-right flip   √
    (2) random crop                     √
    (3) color enhancement               √
    (4) shift                           √
    (5) scale                           √
    (6) noise                           √
    (7) reflection / rotation           √

    Script: data_augmentation.py
    Author: Penghua Liu
    Date: 2019-01-13
    Email: liuphhhh@foxmail.com
    Dependencies: PIL, Numpy

"""

from PIL import Image, ImageEnhance, ImageChops
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2


def DataAugmentationFlip(image, label, flip_mode="left_right"):
    """ Flip the src and label images
    :param image: Image instance
    :param label: Image instance
    :param flip_mode: "left_right" or "top_bottom"
    :return: the processed Image instances
    """
    mode = Image.FLIP_LEFT_RIGHT if flip_mode=="left_right" else Image.FLIP_TOP_BOTTOM
    image = image.transpose(mode)
    label = label.transpose(mode)
    return image, label


def DataAugmentationRandomCrop(image, label, crop_size=(128,128)):
    """ Random Crop the src and label images
    :param image: Image instance
    :param label: Image instance
    :param crop_size: tuple, (crop_width, crop_height)
    :return: the processed Image instances
    """
    assert image.size[0]>=crop_size[0] and image.size[1]>=crop_size[1]
    image_width = image.size[0]
    image_height = image.size[1]
    random_region = ( (image_width - crop_size[0]) >> 1, (image_height - crop_size[1]) >> 1,
                      (image_width + crop_size[0]) >> 1, (image_height + crop_size[1]) >> 1)
    return image.crop(random_region), label.crop(random_region)


def DataAugmentationColorEnhancement(image, label):
    """ Apply color enhancements on the src and label images, including saturation, brightness, contrast, sharpness
    :param image: Image instance
    :param label: Image instance
    :return: the processed images
    """
    random_factor = np.random.randint(7, 13) / 10.
    color_image = ImageEnhance.Color(image).enhance(random_factor)  # adjust saturation
    random_factor = np.random.randint(10, 14) / 10.
    brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor)  # adjust brightness
    random_factor = np.random.randint(10, 14) / 10.
    contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)  # adjust Contrast
    random_factor = np.random.randint(10, 14) / 10.
    sharpness_image = ImageEnhance.Sharpness(contrast_image).enhance(random_factor)   # adjust Sharpness
    return sharpness_image, label


def DataAugmentationShift(image, label, factor=0.1, image_background_value=(0,0,0), label_background_value=0):
    """ applying shifting on the x-axis and the y-axis
    :param image: Image instance
    :param label: Image instance
    :param factor: the maximum off-set that might be applied in the shifting
    :param image_background_value: e.g, (0,0,0)
    :param label_background_value: e.g, 0
    :return: the processed images
    """
    def offset(img, xoff, yoff, background_value):
        c = ImageChops.offset(img, xoff, yoff)
        c.paste(background_value, (0, 0, xoff, img.size[1]))
        c.paste(background_value, (0, 0, img.size[0], yoff))
        return c

    xoff = np.random.randint(-image.size[0]*factor, image.size[0]*factor)
    yoff = np.random.randint(-image.size[1]*factor, image.size[1]*factor)
    return offset(image, xoff, yoff, image_background_value), offset(label, xoff, yoff, label_background_value)


def DataAugmentationScale(image, label, scale_factor=1.1, image_background_value=(0,0,0), label_background_value=0):
    """ apply scaling on the src and label images
    :param image: Image instance
    :param label: Image instance
    :param scale_factor: the scale factor, e.g, 0.9, 1.1
    :param image_background_value: e.g, (0,0,0)
    :param label_background_value: e.g, 0
    :return: the processed images
    """
    if scale_factor >= 1:
        # the scaled image is larger than the original one
        # select subregion equal to the original size from the scaled image
        resized_size = (int(image.size[0] * scale_factor), int(image.size[1] * scale_factor))
        image_resize = image.resize(resized_size)
        label_resize = label.resize(resized_size)

        margin_width = (resized_size[0]-image.size[0]) // 2
        margin_height = (resized_size[1]-image.size[1]) // 2

        image_result = image_resize.transform(image.size, Image.EXTENT, (margin_width, margin_height, margin_width + image.size[0], margin_height + image.size[1]))
        label_result = label_resize.transform(image.size, Image.EXTENT, (margin_width, margin_height, margin_width + image.size[0], margin_height + image.size[1]))
        return image_result, label_result
    else:
        # the scaled image is smaller than the original one
        # mapping the center part of the empty image with the scaled image
        resized_size = (int(image.size[0] * scale_factor), int(image.size[1]*scale_factor))
        image_resize = image.resize(resized_size)
        label_resize = label.resize(resized_size)

        image_result = np.ones((image.size[1], image.size[0], len(image_background_value)), dtype=np.uint8)*image_background_value[0]
        label_result = np.ones((image.size[1], image.size[0]), dtype=np.uint8)*label_background_value
        margin_width = (image.size[0]-resized_size[0])//2
        margin_height = (image.size[1]-resized_size[1])//2
        image_result[margin_height:margin_height+resized_size[1], margin_width:margin_width+resized_size[0],:] = image_resize
        label_result[margin_height:margin_height+resized_size[1], margin_width:margin_width+resized_size[0]] = label_resize

        return Image.fromarray(image_result), Image.fromarray(label_result)


def DataAugmentationNoise(image, label, factor=10):
    """ add noise to the src image
    :param image: Image instance
    :param label: Image instance
    :param factor: the sigma of the guess noise added the src image
    :return: the processed images
    """
    dataarr = np.array(image)
    noise = np.random.randn(image.size[1], image.size[0], dataarr.shape[2])*factor
    result = Image.fromarray(np.array(dataarr + noise, dtype=np.uint8))
    return result, label


def DataAugmentationRotation(image, label, angle=5):
    """ rotate the src and label images
    :param image: Image instance
    :param label: Image instance
    :param angle: the rotation angle
    :return: the processed images
    """
    return image.rotate(angle), label.rotate(angle)


def DataAugmentationBrighten(image, label, brightness=0.05):
    factor = 1.0 + random.uniform(-brightness, brightness)
    table = np.array([((i / 255.0) * factor) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
    image = cv2.LUT(np.array(image), table)
    return Image.fromarray(image.astype(np.uint8)), label



def __test(da_func):
    src_img = Image.open("test_data/src.png")
    label_img = Image.open("test_data/label.png")
    src_img_da, label_img_da = da_func(src_img, label_img)

    plt.subplot(2, 2, 1)
    plt.imshow(src_img)
    plt.subplot(2, 2, 2)
    plt.imshow(label_img)
    plt.subplot(2, 2, 3)
    plt.imshow(src_img_da)
    plt.subplot(2, 2, 4)
    plt.imshow(label_img_da)
    plt.show()


DataAugmentationFlip = DataAugmentationFlip
DataAugmentationRandomCrop = DataAugmentationRandomCrop
DataAugmentationColorEnhancement = DataAugmentationColorEnhancement
DataAugmentationShift = DataAugmentationShift
DataAugmentationScale = DataAugmentationScale
DataAugmentationNoise = DataAugmentationNoise
DataAugmentationRotation = DataAugmentationRotation
test_dataaugmentation = __test
