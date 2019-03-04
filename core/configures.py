# # # Default Network Configurations
KERNEL_INITIALIZER="he_normal"
BN_MOMENTUM = 0.99
BN_EPSILON = 1e-3
WEIGHT_DECAY = 1e-4
INIT_FILTERS = 64
DROPOUT = 0.4


# # # Data Set Color Map and Name Configurations
COLOR_MAP = {
  "inria": [0, 255],
  "usa": [0, 255],
  "whu": [0, 255],
  "ade20k":[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
            29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
            56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82,
            83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107,
            108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128,
            129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150],
  "voc": [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128], [128, 0, 128], [0, 128, 128],
          [128, 128, 128], [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128],
          [64, 128, 128], [192, 128, 128], [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0], [0, 64, 128]],

  "cityscapes": [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156], [190, 153, 153], [153, 153, 153],
        [250, 170, 30], [220, 220, 0], [107, 142, 35], [152, 251, 152], [0, 130, 180], [220, 20, 60], [255, 0, 0],
        [0, 0, 142], [0, 0, 70], [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32]]
}


NAME_MAP = {
    "voc": ["back-\nground", "airplane","bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "table",
            "dog", "horse", "bike", "person", "plant", "sheep", "sofa", "train", "monitor"]
}


# # # Training Configurations
TRAINING_CONFIG = {
    "dataset_name": "voc",
    "model_name": "deeplab_v3p",
    "old_model_version": "deeplab_v3p_voc",
    "new_model_version": "deeplab_v3p_voc",
    "workspace": "F:/Data/VOC",

    "image_dir": "F:/Data/VOC/VOC_aug/img",
    "label_dir": "F:/Data/VOC/VOC_aug/label",
    "train_fnames_path": "F:/Data/VOC/VOC_aug/train.txt",
    "val_fnames_path": "F:/Data/VOC/VOC_aug/val.txt",
    "save_to_dir": False,
    "save_image_path": "F:/Data/VOC/_temp/image",
    "save_label_path": "F:/Data/VOC/_temp/label",
    "image_suffix": ".jpg",
    "label_suffix": ".png",
    "image_color_mode": "rgb",

    "ignore_label": 0,
	"cval": 255,
    "label_cval": 0,
    "feed_onehot_label": True,
    "crop_mode": "random",              # one of
    "featurewise_scale": 1,

    "image_width": 320,
    "image_height": 320,
    "image_channel": 3,
    "n_class": 21,

    "base_lr": 1e-4,
    "min_lr": 1e-7,
    "lr_mode": "power_decay",             # one of ["power_decay", "exp_decay", "progressive_drops", "cosine_cycle", "none"]
    "lr_cycle": 10,                       # number of epochs per cycle, only valid if 'lr_mode' is 'cosine_cycle'
    "lr_power": 0.9,                      # lr power, only valid if 'lr_mode' is 'power_decay'
    "optimizer_name": "adam",

    "batch_size": 2,
    "epoch": 200,
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


# # # Predicting Configurations
PREDICTING_CONFIG = {
    "model_name": "deeplab_v3p",
    "model_path": "F:/Data/VOC/models/deeplab_v3p_voc_new.h5",
    "image_dir": "F:/Data/VOC/_temp/image",
    "preds_dir": "F:/Data/VOC/_temp/prediction",
    "label_dir": "F:/Data/VOC/_temp/label",
    "data_name": "voc",
    "image_height": 320,
    "image_width": 320,
    "stride": 80,
    "to_prob": 0,
    "geo": 0,
    "mode": "per_image",
    "plot": 1
}


# # # Evaluation Configurations
EVALUATING_CONFIG = {
    "preds_dir": "E:/SegData/inria/data_ori/my_test/austin_test/prediction/deeplab_v3p",
    "label_dir": "E:/SegData/inria/data_ori/my_test/austin_test/label",
    "data_name": "inria",
    "ignore_label": -1,
    "label_is_gray": 1,
    "plot": 0
}
