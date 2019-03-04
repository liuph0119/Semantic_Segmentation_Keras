import warnings
from keras.preprocessing.image import apply_brightness_shift, apply_channel_shift

from .image_augmentation_utils import *
from .directory_iterator import SegDirectoryIterator


class ImageDataGenerator(object):
    def __init__(self,
                 featurewise_center=False,
                 samplewise_center=False,
                 featurewise_std_normalization=False,
                 samplewise_std_normalization=False,
                 rotation_range=0,
                 width_shift_range=0.,
                 height_shift_range=0.,
                 brightness_range=None,
                 shear_range=0.,
                 zoom_range=0.,
                 zoom_maintain_shape=True,
                 channel_shift_range=0.,
                 channelwise_center=False,
                 horizontal_flip=False,
                 vertical_flip=False,
                 # above are data augmentation methods
                 # below are image standardize methods
                 fill_mode='constant',
                 cval=255.,
                 label_cval=255,
                 rescale=None,
                 preprocessing_function=None,
                 dtype='float32',
                 ):
        self.featurewise_center = featurewise_center
        self.samplewise_center = samplewise_center
        self.featurewise_std_normalization = featurewise_std_normalization
        self.samplewise_std_normalization = samplewise_std_normalization
        self.rotation_range = rotation_range
        self.width_shift_range = width_shift_range
        self.height_shift_range = height_shift_range
        self.brightness_range = brightness_range
        self.shear_range = shear_range
        self.zoom_range = zoom_range
        self.channel_shift_range = channel_shift_range
        self.fill_mode = fill_mode
        self.cval = cval
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.rescale = rescale
        self.preprocessing_function = preprocessing_function
        self.dtype = dtype

        self.zoom_maintain_shape = zoom_maintain_shape
        self.channelwise_center = channelwise_center
        self.label_cval = label_cval

        self.channel_axis = 3
        self.row_axis = 1
        self.col_axis = 2

        self.mean = None
        self.std = None
        self.ch_mean = None
        self.principal_components = None

        if np.isscalar(zoom_range):
            self.zoom_range = [1 - zoom_range, 1 + zoom_range]
        elif len(zoom_range) == 2:
            self.zoom_range = [zoom_range[0], zoom_range[1]]
        else:
            raise ValueError('`zoom_range` should be a float or '
                             'a tuple or list of two floats. '
                             'Received: %s' % (zoom_range,))

        if featurewise_std_normalization:
            if not featurewise_center:
                self.featurewise_center = True
                warnings.warn('This ImageDataGenerator specifies '
                              '`featurewise_std_normalization`, '
                              'which overrides setting of '
                              '`featurewise_center`.')
        if samplewise_std_normalization:
            if not samplewise_center:
                self.samplewise_center = True
                warnings.warn('This ImageDataGenerator specifies '
                              '`samplewise_std_normalization`, '
                              'which overrides setting of '
                              '`samplewise_center`.')


    def flow_from_directory(self, base_fnames,
                 image_dir, image_suffix, image_color_mode,
                 label_dir, label_suffix,
                 n_class, feed_onehot_label, ignore_label=0,
                 crop_mode="random", target_size=None,
                 batch_size=1, shuffle=True, seed=None,
                 save_to_dir=False, save_image_path="", save_label_path="",
                 dataset_name="voc"):
        return SegDirectoryIterator(
            base_fnames,
            self,
            image_dir, image_suffix, image_color_mode,
            label_dir, label_suffix, n_class, feed_onehot_label,
            cval=self.cval, label_cval=self.label_cval, ignore_label=ignore_label,
            crop_mode=crop_mode, target_size=target_size,
            batch_size=batch_size, shuffle=shuffle, seed=seed,
            save_to_dir=save_to_dir, save_image_path=save_image_path, save_label_path=save_label_path,
            dataset_name = dataset_name)

    def standardize(self, x):
        """Applies the normalization configuration to a batch of inputs.

        # Arguments
            x: single of inputs to be normalized.

        # Returns
            The inputs, normalized.
        """
        if self.preprocessing_function:
            x = self.preprocessing_function(x)
        if self.rescale:
            x *= self.rescale
        img_channel_index = self.channel_axis - 1
        if self.samplewise_center:
            x -= np.mean(x, axis=img_channel_index, keepdims=True)
        if self.samplewise_std_normalization:
            x /= (np.std(x, axis=img_channel_index, keepdims=True) + 1e-6)

        if self.featurewise_center:
            if self.mean is not None:
                x -= self.mean
        if self.featurewise_std_normalization:
            if self.std is not None:
                x /= (self.std + 1e-6)
        if self.channelwise_center:
            if self.ch_mean is not None:
                x -= self.ch_mean
        return x

    def get_random_transform(self, img_shape, seed=None):
        """Generates random parameters for a transformation.

        # Arguments
            seed: Random seed.
            img_shape: Tuple of integers.
                Shape of the image that is transformed.

        # Returns
            A dictionary containing randomly chosen parameters describing the
            transformation.
        """
        img_row_axis = self.row_axis - 1
        img_col_axis = self.col_axis - 1

        if seed is not None:
            np.random.seed(seed)

        if self.rotation_range:
            theta = np.random.uniform(-self.rotation_range, self.rotation_range)
        else:
            theta = 0

        if self.height_shift_range:
            try:  # 1-D array-like or int
                tx = np.random.choice(self.height_shift_range)
                tx *= np.random.choice([-1, 1])
            except ValueError:  # floating point
                tx = np.random.uniform(-self.height_shift_range, self.height_shift_range)
            if np.max(self.height_shift_range) < 1:
                tx *= img_shape[img_row_axis]
        else:
            tx = 0

        if self.width_shift_range:
            try:  # 1-D array-like or int
                ty = np.random.choice(self.width_shift_range)
                ty *= np.random.choice([-1, 1])
            except ValueError:  # floating point
                ty = np.random.uniform(-self.width_shift_range,
                                       self.width_shift_range)
            if np.max(self.width_shift_range) < 1:
                ty *= img_shape[img_col_axis]
        else:
            ty = 0

        if self.shear_range:
            shear = np.random.uniform(-self.shear_range, self.shear_range)
        else:
            shear = 0

        if self.zoom_range[0] == 1 and self.zoom_range[1] == 1:
            zx, zy = 1, 1
        else:
            zx, zy = np.random.uniform(self.zoom_range[0], self.zoom_range[1], 2)
        if self.zoom_maintain_shape:
            zy = zx
        flip_horizontal = (np.random.random() < 0.5) * self.horizontal_flip
        flip_vertical = (np.random.random() < 0.5) * self.vertical_flip

        channel_shift_intensity = None
        if self.channel_shift_range != 0:
            channel_shift_intensity = np.random.uniform(-self.channel_shift_range, self.channel_shift_range)

        brightness = None
        if self.brightness_range is not None:
            if len(self.brightness_range) != 2:
                raise ValueError(
                    '`brightness_range should be tuple or list of two floats. '
                    'Received: %s' % (self.brightness_range,))
            brightness = np.random.uniform(self.brightness_range[0], self.brightness_range[1])

        transform_parameters = {'theta': theta,
                                'tx': tx,
                                'ty': ty,
                                'shear': shear,
                                'zx': zx,
                                'zy': zy,
                                'flip_horizontal': flip_horizontal,
                                'flip_vertical': flip_vertical,
                                'channel_shift_intensity': channel_shift_intensity,
                                'brightness': brightness}

        return transform_parameters

    def apply_transform(self, x, y, transform_parameters, fill_mode="constant", cval=255., label_cval=0.):
        """Applies a transformation to an image according to given parameters.

        # Arguments
            x: 3D tensor, single image.
            y: 2D tensor, single label
            transform_parameters: Dictionary with string - parameter pairs
                describing the transformation.
                Currently, the following parameters
                from the dictionary are used:
                - `'theta'`: Float. Rotation angle in degrees.
                - `'tx'`: Float. Shift in the x direction.
                - `'ty'`: Float. Shift in the y direction.
                - `'shear'`: Float. Shear angle in degrees.
                - `'zx'`: Float. Zoom in the x direction.
                - `'zy'`: Float. Zoom in the y direction.
                - `'flip_horizontal'`: Boolean. Horizontal flip.
                - `'flip_vertical'`: Boolean. Vertical flip.
                - `'channel_shift_intencity'`: Float. Channel shift intensity.
                - `'brightness'`: Float. Brightness shift intensity.

        # Returns
            A transformed version of the input (same shape).
        """
        # x is a single image, so it doesn't have image number at index 0
        img_row_axis = self.row_axis - 1
        img_col_axis = self.col_axis - 1
        img_channel_axis = self.channel_axis - 1

        # x = apply_affine_transform(x, transform_parameters.get('theta', 0),
        #                            transform_parameters.get('tx', 0),
        #                            transform_parameters.get('ty', 0),
        #                            transform_parameters.get('shear', 0),
        #                            transform_parameters.get('zx', 1),
        #                            transform_parameters.get('zy', 1),
        #                            row_axis=img_row_axis,
        #                            col_axis=img_col_axis,
        #                            channel_axis=img_channel_axis,
        #                            fill_mode=fill_mode,
        #                            cval=cval)
        # y = apply_affine_transform(y, transform_parameters.get('theta', 0),
        #                            transform_parameters.get('tx', 0),
        #                            transform_parameters.get('ty', 0),
        #                            transform_parameters.get('shear', 0),
        #                            transform_parameters.get('zx', 1),
        #                            transform_parameters.get('zy', 1),
        #                            row_axis=img_row_axis,
        #                            col_axis=img_col_axis,
        #                            channel_axis=img_channel_axis,
        #                            fill_mode="constant",
        #                            cval=label_cval)

        from .image_augmentation_utils import my_apply_affine_transform
        x, y = my_apply_affine_transform(x, y, transform_parameters.get("theta", 0),
                                         transform_parameters.get("tx", 0),
                                         transform_parameters.get("ty", 0),
                                         transform_parameters.get("zx", 1),
                                         transform_parameters.get("zy", 1),
                                         cval=cval, label_cval=label_cval)
        if y.ndim==3:
            y = y[:, :, 0]
        if transform_parameters.get('channel_shift_intensity') is not None:
            x = apply_channel_shift(x,
                                    transform_parameters['channel_shift_intensity'],
                                    img_channel_axis)

        if transform_parameters.get('flip_horizontal', False):
            x = flip_axis(x, img_col_axis)
            y = flip_axis(y, img_col_axis)

        if transform_parameters.get('flip_vertical', False):
            x = flip_axis(x, img_row_axis)
            y = flip_axis(y, img_row_axis)

        if transform_parameters.get('brightness') is not None:
            x = apply_brightness_shift(x, transform_parameters['brightness'])

        return x, y

    def random_transform(self, x, y, fill_mode="constant", cval=255., label_cval=0, seed=None):
        """Applies a random transformation to an image.

        # Arguments
            x: 3D tensor, single image.
            seed: Random seed.

        # Returns
            A randomly transformed version of the input (same shape).
        """
        img_row_index = self.row_axis - 1
        img_col_index = self.col_axis - 1

        assert x.shape[img_row_index] == y.shape[img_row_index] and x.shape[img_col_index] == y.shape[
            img_col_index], 'DATA ERROR: Different shape of data and label!\ndata shape: %s, label shape: %s' % (
            str(x.shape), str(y.shape))

        params = self.get_random_transform(x.shape, seed)

        x, y = self.apply_transform(x, y, params, fill_mode, cval, label_cval)

        return x, y

    def set_ch_mean(self, ch_mean):
        self.ch_mean = ch_mean

