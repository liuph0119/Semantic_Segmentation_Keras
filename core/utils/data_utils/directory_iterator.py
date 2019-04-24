import os
import datetime
import numpy as np
from keras.utils import to_categorical
from keras.preprocessing.image import Iterator

from .image_io_utils import load_image, save_to_image
from .image_augmentation_utils import image_randomcrop, image_centercrop
from ..vis_utils import plot_image_label
from ...configures import NAME_MAP


class SegDirectoryIterator(Iterator):

    def __init__(self,
                 base_fnames,
                 data_generator,
                 image_dir,
                 image_suffix,
                 image_color_mode,
                 label_dir,
                 label_suffix,
                 n_class,
                 feed_onehot_label=True,
                 cval=255.,
                 label_cval=0,
                 crop_mode="random",
                 target_size=None,
                 batch_size=1,
                 shuffle=True,
                 seed=None,
                 debug=False,
                 dataset_name="voc"
                 ):
        """
        :param base_fnames: list, basic file names
        :param data_generator: ImageDataGenerator instance
        :param image_dir: string
        :param image_suffix: string, one of ["npy", "jpg", "jpeg", "png", "tif"]
        :param image_color_mode: string, one of ["gray", "rgb", "multi"]
        :param label_dir: string
        :param label_suffix: string, one of ["npy", "jpg", "jpeg", "png", "tif"]
        :param n_class: int, number of classes, including background
        :param feed_onehot_label: bool, whether to apply one-hot encoding to labels
        :param cval: float, filling value for image
        :param label_cval: float, filling value for label
        :param crop_mode:string, one of ["none", "resize", "random", "center"]
        :param target_size: tuple, (height, width)
        :param batch_size: int
        :param shuffle: bool
        :param seed: float
        :param save_to_dir: string
        :param save_image_path: string
        :param save_label_path: string
        """
        self.base_fnames = base_fnames
        self.nb_sample = len(base_fnames)
        self.seg_data_generator = data_generator
        self.image_dir = image_dir
        self.image_suffix = image_suffix
        if self.image_suffix not in [".npy", ".jpg", ".jpeg", ".png", ".tif"]:
            raise ValueError(
                "Invalid image suffix: {}. Expected '.npy', '.jpg', '.jpeg', '.png' or '.tif'.".format(self.image_suffix))
        self.image_color_mode = image_color_mode
        if self.image_color_mode not in ["gray", "rgb", "multi"]:
            raise ValueError(
                "Invalid image color mode: {}. Expected 'gray', 'rgb', 'multi'.".format(self.image_color_mode))
        self.label_dir = label_dir
        self.label_suffix = label_suffix
        if self.label_suffix not in [".npy", ".jpg", ".jpeg", ".png", ".tif"]:
            raise ValueError(
                "Invalid label suffix: {}. Expected '.npy', '.jpg', '.jpeg', '.png' or '.tif'.".format(self.label_suffix))
        self.n_class = n_class
        self.feed_onehot_label = feed_onehot_label
        self.cval = cval
        self.label_cval = label_cval

        self.crop_mode = crop_mode
        if self.crop_mode not in ["random", "center", "resize", "none"]:
            raise ValueError(
                "Invalid crop mode: {}. Expected 'random', 'center', 'resize', 'none'.".format(self.crop_mode))
        self.target_size = target_size
        self.batch_size = batch_size
        self.debug = debug
        self.shuffle = shuffle
        self.seed = seed
        self.dataset_name = dataset_name

        super(SegDirectoryIterator, self).__init__(
            self.nb_sample, batch_size, shuffle, seed)


    def _get_batches_of_transformed_samples(self, indices):
        """ get a batch of samples
        :param indices: list
            list of sample indices
        :return: a batch of samples, including images and labels
        """
        batch_image = []
        batch_label = []
        _image_is_gray = self.image_color_mode=="gray"
        for ind in indices:
            ### 1. load image and label
            if self.image_suffix == ".npy":
                # using Numpy to load *.npy files
                # NOTE: RESIZE, GRAY, VALUE_SCALE are not valid here!

                _image = np.load(os.path.join(self.image_dir, self.base_fnames[ind] + self.image_suffix))
                _label = np.load(os.path.join(self.label_dir, self.base_fnames[ind] + self.label_suffix))
            else:
                # if the crop_mode is 'resize', resize the input image to target size,
                # but will not apply random cropping or center cropping

                if self.crop_mode=="resize":
                    _target_size = (self.target_size[0], self.target_size[1])
                else:
                    # if the _target_size is None, the input image will maintain it's original size
                    _target_size = None

                if self.image_suffix == ".tif" and self.image_color_mode == "multi":
                    # using GDAL to load multi-spectral images
                    ### NOTE: RESIZE AND GRAY ARE NOT VALID here!!!

                    _target_size = None
                    _image = load_image(os.path.join(self.image_dir, self.base_fnames[ind] + self.image_suffix),
                                        value_scale=1, use_gdal=True)
                else:
                    # RGB/Gray using PIL
                    _image = load_image(os.path.join(self.image_dir, self.base_fnames[ind] + self.image_suffix),
                                        is_gray=_image_is_gray, value_scale=1, target_size=_target_size)

                _label = load_image(os.path.join(self.label_dir, self.base_fnames[ind] + self.label_suffix),
                                    is_gray=True, value_scale=1, target_size=_target_size)

            ### 2. do padding if applying cropping
            img_h, img_w, img_c = _image.shape
            if self.crop_mode in ["random", "center"]:
                pad_h = max(self.target_size[0] - img_h, 0)
                pad_w = max(self.target_size[1] - img_w, 0)
                _image = np.lib.pad(_image,
                                    ((pad_h//2, pad_h - pad_h//2), (pad_w//2, pad_w - pad_w//2), (0, 0)),
                                    'constant', constant_values=self.cval)  # change 255 to 0
                _label = np.lib.pad(_label, ((pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2), (0, 0)),
                                    'constant', constant_values=self.label_cval)

            ## 3. do cropping from the padded image/label
            if self.crop_mode == 'center':
                _image, _label = image_centercrop(_image, _label, self.target_size[0], self.target_size[1])
            elif self.crop_mode == 'random':
                _image, _label = image_randomcrop(_image, _label, self.target_size[0], self.target_size[1])

            ### 4. do data augmentation for rgb images
            if self.image_color_mode=="rgb":
                _image, _label = self.seg_data_generator.random_transform(_image,
                                                                          _label,
                                                                          cval=self.cval,
                                                                          label_cval=self.label_cval,
                                                                          seed=None)
            # we do not apply a normalization here since a BN is firstly adopted in the FCN.
            ### 5.1. clip the label values to a valid range
            _label = np.clip(_label, 0, self.n_class - 1).astype(np.uint8)

            ### 5.2. save the generated images to local dir
            if self.debug:
                #time_flag = "_{}".format(datetime.datetime.now().strftime("%y%m%d%H%M%S"))
                plot_image_label(_image/255, _label, vmin=0, vmax=self.n_class - 1, names=NAME_MAP[self.dataset_name])
                #save_to_image(_image, os.path.join(self.save_image_path, self.base_fnames[ind] + time_flag + self.image_suffix))
                #save_to_image(_label, os.path.join(self.save_label_path, self.base_fnames[ind] + time_flag + self.label_suffix))

            if self.feed_onehot_label:
                _label = to_categorical(_label, self.n_class, dtype="uint8")
                assert _label.shape==(self.target_size[0], self.target_size[1], self.n_class)
            batch_image.append(_image)
            batch_label.append(_label)

        batch_image = np.stack(batch_image, axis=0)
        batch_label = np.stack(batch_label, axis=0)

        return (batch_image, batch_label)


    def next(self):
        with self.lock:
            index_array = next(self.index_generator)
        return self._get_batches_of_transformed_samples(index_array)