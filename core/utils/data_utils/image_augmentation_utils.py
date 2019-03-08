"""
Implementation of image Augmentation, including:
    (1) up-down flip, left-right flip   √
    (2) random crop                     √
    (3) blur                            √
    (4) shift                           √
    (5) scale                           √
    (6) add noise                       √
    (7) reflection / rotation           √
    (8) blur
"""
from PIL import Image, ImageChops
import matplotlib.pyplot as plt
import numpy as np
import cv2
from keras.preprocessing.image import array_to_img

def flip_axis(x, axis):
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x


def image_flip(image, label, left_right_axis=True):
    """ Flip the src and label images
    :param image: numpy array
    :param label: numpy array
    :param left_right_axis: True / False
    :return: the processed numpy arrays
    """
    axis = 1 if left_right_axis==True else 0
    image = np.flip(image, axis=axis)
    label = np.flip(label, axis=axis)
    return image, label


def image_randomcrop(image, label, crop_height, crop_width):
    """ Random Crop the src and label images
    :param image: numpy array
    :param label: numpy array
    :param crop_height: target height
    :param crop_width: target width
    :return: the processed numpy arrays
    """
    assert image.shape[1]>=crop_width and image.shape[0]>=crop_height
    image_width = image.shape[1]
    image_height = image.shape[0]

    x = np.random.randint(0, image_width-crop_width+1)
    y = np.random.randint(0, image_height-crop_height+1)

    return image[y:y+crop_height, x:x+crop_width], \
           label[y:y+crop_height, x:x+crop_width]


def image_centercrop(image, label, crop_height, crop_width):
    centerh, centerw = image.shape[0] // 2, image.shape[1] // 2
    lh, lw = crop_height // 2, crop_width // 2
    rh, rw = crop_height - lh, crop_width - lw

    h_start, h_end = centerh - lh, centerh + rh
    w_start, w_end = centerw - lw, centerw + rw

    return image[h_start:h_end, w_start:w_end], \
           label[h_start:h_end, w_start:w_end]
# def DataAugmentationColorEnhancement(image, label):
#     """ Apply color enhancements on the src and label images, including saturation, brightness, contrast, sharpness
#     :param image: Image instance
#     :param label: Image instance
#     :return: the processed images
#     """
#     random_factor = np.random.randint(7, 13) / 10.
#     color_image = ImageEnhance.Color(image).enhance(random_factor)  # adjust saturation
#     random_factor = np.random.randint(10, 14) / 10.
#     brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor)  # adjust brightness
#     random_factor = np.random.randint(10, 14) / 10.
#     contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)  # adjust Contrast
#     random_factor = np.random.randint(10, 14) / 10.
#     sharpness_image = ImageEnhance.Sharpness(contrast_image).enhance(random_factor)   # adjust Sharpness
#     return sharpness_image, label


def image_shift(image, label, xoff, yoff, image_background_value=(0,0,0), label_background_value=0):
    """ applying shifting on the x-axis and the y-axis
    :param image: numpy array
    :param label: numpy array
    :param image_background_value: e.g, (0,0,0)
    :param label_background_value: e.g, 0
    :return: the processed numpy arrays
    """
    def offset(img, xoff, yoff, background_value):
        c = ImageChops.offset(img, xoff, yoff)
        c.paste(background_value, (0, 0, xoff, img.size[1]))
        c.paste(background_value, (0, 0, img.size[0], yoff))
        return np.array(c, dtype=np.uint8)

    image = array_to_img(image.astype(np.uint8), scale=False, dtype=np.uint8)
    label = array_to_img(label.astype(np.uint8), scale=False, dtype=np.uint8)

    return offset(image, xoff, yoff, image_background_value), offset(label, xoff, yoff, label_background_value)


def image_scale(image, label, zoom_x, zoom_y, image_background_value=(0,0,0), label_background_value=0):
    """ apply scaling on the src and label images
    :param image: numpy array
    :param label: numpy array
    :param scale_factor: the scale factor, e.g, 0.9, 1.1
    :param image_background_value: e.g, (0,0,0)
    :param label_background_value: e.g, 0
    :return: the processed arrays
    """
    image = array_to_img(image.astype(np.uint8), scale=False, dtype=np.uint8)
    label = array_to_img(label.astype(np.uint8), scale=False, dtype=np.uint8)
    assert zoom_x==zoom_y
    if zoom_x > 1:
        # the scaled image is larger than the original one
        # select subregion equal to the original size from the scaled image
        resized_size = (int(image.size[0] * zoom_x), int(image.size[1] * zoom_y))
        image_resize = image.resize(resized_size)
        label_resize = label.resize(resized_size)

        margin_width = (resized_size[0]-image.size[0]) // 2
        margin_height = (resized_size[1]-image.size[1]) // 2

        image_result = image_resize.transform(image.size, Image.EXTENT, (margin_width, margin_height, margin_width + image.size[0], margin_height + image.size[1]))
        label_result = label_resize.transform(image.size, Image.EXTENT, (margin_width, margin_height, margin_width + image.size[0], margin_height + image.size[1]))
        return np.array(image_result, dtype=np.uint8), np.array(label_result, dtype=np.uint8)
    else:
        # the scaled image is smaller than the original one
        # mapping the center part of the empty image with the scaled image
        resized_size = (int(image.size[0] * zoom_x), int(image.size[1]*zoom_y))
        image_resize = image.resize(resized_size)
        label_resize = label.resize(resized_size)

        image_result = np.ones((image.size[1], image.size[0], len(image_background_value)), dtype=np.uint8)*image_background_value[0]
        label_result = np.ones((image.size[1], image.size[0]), dtype=np.uint8)*label_background_value
        margin_width = (image.size[0]-resized_size[0])//2
        margin_height = (image.size[1]-resized_size[1])//2
        image_result[margin_height:margin_height+resized_size[1], margin_width:margin_width+resized_size[0],:] = image_resize
        label_result[margin_height:margin_height+resized_size[1], margin_width:margin_width+resized_size[0]] = label_resize

        return image_result, label_result


def image_addnoise(image, label, factor=10, min_value=0, max_value=255):
    """ add noise to the src image
    :param image: numpy array
    :param label: numpy array
    :param factor: the sigma of the guess noise added the src image
    :return: the processed images
    """
    noise = np.random.randn(image.shape[0], image.shape[1], image.shape[2])*factor
    image = np.clip(image+noise, min_value, max_value)
    return image, label


def image_rotate(image, label, angle=5):
    """ rotate the src and label images
    :param image: numpy array
    :param label: numpy array
    :param angle: the rotation angle
    :return: the processed numpy arrays
    """
    image = array_to_img(image.astype(np.uint8), scale=False, dtype=np.uint8)
    label = array_to_img(label.astype(np.uint8), scale=False, dtype=np.uint8)
    return np.array(image.rotate(angle), dtype=np.uint8), np.array(label.rotate(angle), dtype=np.uint8)


def image_blur(image, label, ksize=(3, 3)):
    image = cv2.blur(image, ksize)
    return image, label


def __test(src_img, label_img, da_func):
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


def my_apply_affine_transform(x, y, rotate_angle, x_shift, y_shift, zoom_x, zoom_y, cval=0, label_cval=0):
    """ affine transform including rotate, shift and scale. """
    if x.ndim==3:
        cval = (cval, cval, cval)
    if y.ndim == 2:
        y = np.expand_dims(y, -1)
    if rotate_angle!=0:
        x, y = image_rotate(x, y, rotate_angle)
    if y.ndim == 2:
        y = np.expand_dims(y, -1)
    if x_shift!=0 or y_shift!=0:
        x, y = image_shift(x, y, x_shift, y_shift, image_background_value=cval, label_background_value=label_cval)
    if y.ndim == 2:
        y = np.expand_dims(y, -1)
    if zoom_x!=1 or zoom_y!=1:
        x, y = image_scale(x, y, zoom_x, zoom_y, cval, label_cval)

    return x, y



