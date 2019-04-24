from keras.preprocessing.image import apply_brightness_shift, apply_channel_shift

from .image_augmentation_utils import *
from .directory_iterator import SegDirectoryIterator


class ImageDataGenerator(object):
    def __init__(self,
                 rotation_range=0,
                 width_shift_range=0.,
                 height_shift_range=0.,
                 brightness_range=None,
                 zoom_range=0,
                 zoom_maintain_shape=True,
                 channel_shift_range=0.,
                 horizontal_flip=False,
                 vertical_flip=False
                 ):
        self.rotation_range = rotation_range
        self.width_shift_range = width_shift_range
        self.height_shift_range = height_shift_range
        self.brightness_range = brightness_range
        self.zoom_range = zoom_range
        self.zoom_maintain_shape = zoom_maintain_shape
        self.channel_shift_range = channel_shift_range
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip

        self.channel_axis = 2
        self.row_axis = 0
        self.col_axis = 1
        #self.ch_mean = None

        if np.isscalar(zoom_range):
            self.zoom_range = [1 - zoom_range, 1 + zoom_range]
        elif len(zoom_range) == 2:
            self.zoom_range = [zoom_range[0], zoom_range[1]]
        else:
            raise ValueError('Invalid zoom_range: {}. Expected to be a float or a tuple or list of two floats. '.format(zoom_range))


    def flow_from_directory(self,
                            base_fnames,
                            image_dir,
                            image_suffix,
                            image_color_mode,
                            label_dir,
                            label_suffix,
                            n_class,
                            feed_onehot_label,
                            cval,
                            label_cval,
                            crop_mode="random",
                            target_size=None,
                            batch_size=1,
                            shuffle=True,
                            seed=None,
                            debug=False,
                            dataset_name="voc"):
        return SegDirectoryIterator(base_fnames,
                                    self,
                                    image_dir,
                                    image_suffix,
                                    image_color_mode,
                                    label_dir,
                                    label_suffix,
                                    n_class,
                                    feed_onehot_label,
                                    cval=cval,
                                    label_cval=label_cval,
                                    crop_mode=crop_mode,
                                    target_size=target_size,
                                    batch_size=batch_size,
                                    shuffle=shuffle,
                                    seed=seed,
                                    debug=debug,
                                    dataset_name = dataset_name)


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
                tx *= img_shape[self.row_axis]
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
                ty *= img_shape[self.col_axis]
        else:
            ty = 0


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
                                'zx': zx,
                                'zy': zy,
                                'flip_horizontal': flip_horizontal,
                                'flip_vertical': flip_vertical,
                                'channel_shift_intensity': channel_shift_intensity,
                                'brightness': brightness}

        return transform_parameters


    def apply_transform(self, x, y, transform_parameters, cval=255., label_cval=0.):
        """Applies a transformation to an image according to given parameters.

        # Arguments
            x: 3D tensor, single image.
            y: 2D tensor, single label
            transform_parameters: Dictionary with string - parameter pairs
                describing the transformation.

        # Returns
            A transformed version of the input (same shape).
        """
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
                                    self.channel_axis)

        if transform_parameters.get('flip_horizontal', False):
            x = flip_axis(x, self.col_axis)
            y = flip_axis(y, self.col_axis)

        if transform_parameters.get('flip_vertical', False):
            x = flip_axis(x, self.row_axis)
            y = flip_axis(y, self.row_axis)

        if transform_parameters.get('brightness') is not None:
            x = apply_brightness_shift(x, transform_parameters['brightness'])

        return x, y


    def random_transform(self, x, y, cval=255., label_cval=0, seed=None):
        """Applies a random transformation to an image.

        # Arguments
            x: 3D tensor, single image.
            seed: Random seed.

        # Returns
            A randomly transformed version of the input (same shape).
        """
        assert x.shape[self.row_axis] == y.shape[self.row_axis] and x.shape[self.col_axis] == y.shape[
            self.col_axis], 'DATA ERROR: Different shape of data and label!\ndata shape: %s, label shape: %s' % (
            str(x.shape), str(y.shape))

        params = self.get_random_transform(x.shape, seed)
        x, y = self.apply_transform(x, y, params, cval, label_cval)

        return x, y


    # def set_ch_mean(self, ch_mean):
    #     self.ch_mean = ch_mean
