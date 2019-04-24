COLOR_MAP = {
  "inria": [0, 255],
  "whu": [0, 255],
  "sz": [0, 255],
  'usa': [0, 255],
  "voc": [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128], [128, 0, 128], [0, 128, 128],
          [128, 128, 128], [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128],
          [64, 128, 128], [192, 128, 128], [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0], [0, 64, 128]],

  "cityscapes": [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156], [190, 153, 153], [153, 153, 153],
                 [250, 170, 30], [220, 220, 0], [107, 142, 35], [152, 251, 152], [0, 130, 180], [220, 20, 60],
                 [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32]]
}

# # # Data Set Color Map and Name Configurations
NAME_MAP = {
    "inria": ["non", "building"],
    "whu": ["non", "building"],
    "sz": ["non", "uv"],
    'usa': ['non', 'building'],
    "voc": ["background", "airplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
            "table", "dog", "horse", "bike", "person", "plant", "sheep", "sofa", "train", "monitor"]
}

# tool configurations
class GenerateDataSetConfig(object):
    image_dir ='F:/Data/AerialImageDataset/train/image'
    label_dir = 'F:/Data/AerialImageDataset/train/label'
    image_suffix = '.tif'
    label_suffix = '.tif'
    dst_dir = 'F:/inria/train'
    method = 'stride'
    image_height = 256
    image_width = 256
    image_number_per_tile = 512
    stride = 256
    label_is_gray = True
    use_gdal = False


class Color2IndexConfig(object):
    """
    # Args:
        dataset_name: dict key of `COLOR_MAP` below
        color_mode: 'rgb' or 'gray'
        mode: 'color2index' or 'index2color'
    """
    dataset_name = 'inria'
    color_mode = 'gray'
    src_dir = 'F:/ChinaBuilding/BingAerialBuilding/USA/test/label'
    dst_dir = 'F:/ChinaBuilding/BingAerialBuilding/USA/test/label_index'
    mode = 'color2index'
    show_comparison = False


class NetConfig(object):
    weight_decay = 1e-4
    kernel_initializer = 'he_normal'
    bn_epsilon = 1e-3
    bn_momentum = 0.99
    upsampling_method = 'conv'
    init_filters = 64
    dropout = 0.5


class AugmentConfig(object):
    rotation_range = 0
    width_shift_range = 0
    height_shift_range = 0
    brightness_range = None
    zoom_range = 0
    zoom_maintain_shape = True
    channel_shift_range = 20
    horizontal_flip = True
    vertical_flip = False


class TrainingConfig(object):
    """
    # Args:
        crop_mode: one of ["random", "center", "resize", "none"]
        lr_mode: one of ['power_decay', 'exp_decay', 'progressive_drops', 'cosine_cycle', 'none']
        lr_cycle: number of epochs per cycle, only valid if 'lr_mode' is 'cosine_cycle'
        lr_power: lr power, only valid if 'lr_mode' is 'power_decay'
    """
    dataset_name = 'usa'
    model_name = 'deeplab_v3p'
    loss_name = 'binary_crossentropy'
    metric_name = 'acc'
    encoder_name = 'xception_41'
    encoder_weights = None
    old_model_version = 'deeplab_v3p_usa'
    new_model_version = 'deeplab_v3p_usa'
    workspace = 'F:\ChinaBuilding\BingAerialBuilding\\USA\\train_val'

    image_dir = 'F:\ChinaBuilding\BingAerialBuilding\\USA\\train_val\image'
    label_dir = 'F:\ChinaBuilding\BingAerialBuilding\\USA\\train_val\label_index'
    train_fnames_path = 'F:\ChinaBuilding\BingAerialBuilding\\USA\\train_val\\train.txt'
    val_fnames_path = 'F:\ChinaBuilding\BingAerialBuilding\\USA\\train_val\\val.txt'
    image_suffix = '.tif'
    label_suffix = '.tif'
    image_color_mode = 'rgb'
    cval = 0
    label_cval = 0

    feed_onehot_label = True
    crop_mode = 'resize'

    image_width = 256
    image_height = 256
    image_channel = 3
    n_class = 2

    base_lr = 1e-4
    min_lr = 1e-7
    lr_mode = 'power_decay'
    lr_cycle = 10
    lr_power = 0.9
    optimizer_name = 'adam'
    batch_size = 2
    epoch = 2
    steps_per_epoch = 0
    steps_per_epoch_val = 0
    verbose = 1
    early_stop_patience = 0
    debug = False
    model_summary = False


class PredictingConfig(object):
    """
    # Args:
        mode: ['stride', 'per_image']
    """
    model_name = 'deeplab_v3p'
    encoder_name = 'xception_41'
    model_path = 'F:\ChinaBuilding\BingAerialBuilding\\USA\\train_val\models\\deeplab_v3p_usa.h5'
    image_dir = 'F:\ChinaBuilding\BingAerialBuilding\\USA\\test\image'
    preds_dir = 'F:\ChinaBuilding\BingAerialBuilding\\USA\\test\preds'
    dataset_name = 'usa'
    image_height = 256
    image_width = 256
    image_channel = 3
    mode = 'stride'
    stride = 64
    to_prob = False
    geo = True
    plot = True


class EvaluatingConfig(object):
    """
    # Args:
        mode: one of ["global", "per_image"]
        ignore_0: only valid if "mode" is global
    """
    preds_dir = 'F:\ChinaBuilding\BingAerialBuilding\\USA\\test\preds'
    label_dir = 'F:\ChinaBuilding\BingAerialBuilding\\USA\\test\label_index'
    dataset_name = 'usa'
    mode = 'global'
    ignore_0 = False


color2index_config = Color2IndexConfig()
net_config = NetConfig()
augment_config = AugmentConfig()
generate_dadaset_config = GenerateDataSetConfig()
training_config = TrainingConfig()
predicting_config = PredictingConfig()
evaluating_config = EvaluatingConfig()
