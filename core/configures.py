# # # Default Network Configurations
WEIGHT_DECAY = 1e-4
KERNEL_INITIALIZER = "he_normal"
BN_EPSILON = 1e-3
BN_MOMENTUM = 0.99
UPSCALING_METHOD = "conv"

INIT_FILTERS = 64
DROPOUT = 0.4


# # # Data Set Color Map and Name Configurations
COLOR_MAP = {
  "inria": [0, 255],
  "whu": [0, 255],
  "voc": [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128], [128, 0, 128], [0, 128, 128],
          [128, 128, 128], [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128],
          [64, 128, 128], [192, 128, 128], [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0], [0, 64, 128]],

  "cityscapes": [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156], [190, 153, 153], [153, 153, 153],
                 [250, 170, 30], [220, 220, 0], [107, 142, 35], [152, 251, 152], [0, 130, 180], [220, 20, 60],
                 [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32]]
}


NAME_MAP = {
    "inria": ["non", "building"],
    "whu": ["non", "building"],
    "voc": ["back-\nground", "airplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
            "table", "dog", "horse", "bike", "person", "plant", "sheep", "sofa", "train", "monitor"]
}

# # # Augmentation configurations
AUGMENTATION_CONFIG = {
    "rotation_range": 0,
    "width_shift_range": 0,
    "height_shift_range": 0,
    "brightness_range": None,
    "zoom_range": 0,
    "zoom_maintain_shape": True,
    "channel_shift_range": 20,
    "horizontal_flip": True,
    "vertical_flip": False
}


# # # Training Configurations
TRAINING_CONFIG = {
    "dataset_name": "whu",
    "model_name": "deeplab_v3p",
    "old_model_version": "deeplab_v3p_whu",
    "new_model_version": "deeplab_v3p_whu",
    "workspace": "F:/whu",

    "image_dir": "F:/whu/train/image",
    "label_dir": "F:/whu/train/label",
    "train_fnames_path": "F:/whu/train.txt",
    "val_fnames_path": "F:/whu/val.txt",
    "save_to_dir": False,
    "save_image_path": "F:/whu/_temp/image",
    "save_label_path": "F:/whu/_temp/label",
    "image_suffix": ".tif",
    "label_suffix": ".tif",
    "image_color_mode": "rgb",
    "cval": 255,
    "label_cval": 0,
    "feed_onehot_label": True,
    "crop_mode": "resize",                  # one of ["random", "center", "resize", "none"]

    "image_width": 256,
    "image_height": 256,
    "image_channel": 3,
    "n_class": 2,

    "base_lr": 1e-3,
    "min_lr": 1e-7,
    "lr_mode": "power_decay",               # one of ["power_decay", "exp_decay", "progressive_drops",
                                            # "cosine_cycle", "none"]
    "lr_cycle": 10,                         # number of epochs per cycle, only valid if 'lr_mode' is 'cosine_cycle'
    "lr_power": 0.9,                        # lr power, only valid if 'lr_mode' is 'power_decay'
    "optimizer_name": "adam",

    "batch_size": 4,
    "epoch": 100,
    "steps_per_epoch": 0,
    "steps_per_epoch_val": 0,
    "verbose": 1,

    "loss": "categorical_crossentropy",
    "metric_name": "acc",
    "encoder_name": "xception_41",
    "callbacks": {
        "early_stop": {
            "patience": 100
        }
      }
}
# TRAINING_CONFIG = {
#     "dataset_name": "voc",
#     "model_name": "deeplab_v3p",
#     "old_model_version": "deeplab_v3p_voc",
#     "new_model_version": "deeplab_v3p_voc_temp",
#     "workspace": "F:/Data/VOC",
#
#     "image_dir": "F:/Data/VOC/VOC_aug/img",
#     "label_dir": "F:/Data/VOC/VOC_aug/label",
#     "train_fnames_path": "F:/Data/VOC/VOC_aug/train.txt",
#     "val_fnames_path": "F:/Data/VOC/VOC_aug/val.txt",
#     "save_to_dir": False,
#     "save_image_path": "F:/Data/VOC/_temp/image",
#     "save_label_path": "F:/Data/VOC/_temp/label",
#     "image_suffix": ".jpg",
#     "label_suffix": ".png",
#     "image_color_mode": "rgb",
#
# 	"cval": 255,
#     "label_cval": 0,
#     "feed_onehot_label": True,
#     "crop_mode": "random",              # one of ["random", "center", "resize", "none"]
#
#     "image_width": 320,
#     "image_height": 320,
#     "image_channel": 3,
#     "n_class": 21,
#
#     "base_lr": 1e-7,
#     "min_lr": 1e-8,
#     "lr_mode": "power_decay",             # one of ["power_decay", "exp_decay", "progressive_drops",
#                                           # "cosine_cycle", "none"]
#     "lr_cycle": 10,                       # number of epochs per cycle, only valid if 'lr_mode' is 'cosine_cycle'
#     "lr_power": 0.9,                      # lr power, only valid if 'lr_mode' is 'power_decay'
#     "optimizer_name": "adam",
#
#     "batch_size": 2,
#     "epoch": 50,
#     "steps_per_epoch": 0,
#     "steps_per_epoch_val": 0,
#     "verbose": 1,
#
#     "loss": "categorical_crossentropy",
#     "metric_name": "acc",
#     "encoder_name": "xception_41",
#     "callbacks": {
#         "early_stop": {
#             "patience": 100
#         }
#       }
# }


# # # Predicting Configurations
PREDICTING_CONFIG = {
    "model_name": "refinenet",
    "model_path": "F:/whu/models/refinenet_whu.h5",
    "image_dir": "F:/whu/test/image",
    "preds_dir": "F:/whu/test/prediction/refinenet",
    "label_dir": "F:/whu/test/label",
    "dataset_name": "whu",
    "image_height": 256,
    "image_width": 256,
    "mode": "per_image",
    "stride": 64,
    "to_prob": 0,
    "geo": 0,
    "plot": 0
}


# # # Evaluation Configurations
EVALUATING_CONFIG = {
    "preds_dir": "F:/whu/test/prediction/refinenet",
    "label_dir": "F:/whu/test/label_index",
    "dataset_name": "whu",
    "mode": "global",                        # one of ["global", "per_image"]
    "ignore_0": False                        # only valid if "mode" is global
}
