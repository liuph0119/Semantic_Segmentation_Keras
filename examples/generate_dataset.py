from core.utils.data_augmentation import *
from core.utils.image_utils import load_image
import datetime
import numpy as np
from PIL import Image
import os


def data_augment(src_img, label_img, lr_flip=True, tb_flip=True, color=False, shitf=False, scale_in = True, scale_out=False, add_noise=False, rotate=False):
    if lr_flip and np.random.random() < 0.25:
        src_img, label_img = DataAugmentationFlip(src_img, label_img, flip_mode="left_right")
    if tb_flip and np.random.random() < 0.25:
        src_img, label_img = DataAugmentationFlip(src_img, label_img, flip_mode="top_buttom")
    if color and np.random.random() < 0.25:
        src_img, label_img = DataAugmentationColorEnhancement(src_img, label_img)
    if shitf and np.random.random() < 0.25:
        src_img, label_img = DataAugmentationShift(src_img, label_img)
    if scale_in and np.random.random() < 0.25:
        src_img, label_img = DataAugmentationScale(src_img, label_img, scale_factor=1.2)
    if scale_out and np.random.random() < 0.25:
        src_img, label_img = DataAugmentationScale(src_img, label_img, scale_factor=0.8)
    if add_noise and np.random.random() < 0.25:
        src_img, label_img = DataAugmentationNoise(src_img, label_img, factor=10)
    if rotate and np.random.random() < 0.25:
        src_img, label_img = DataAugmentationRotation(src_img, label_img, angle=5)

    return src_img, label_img



def create_dataset(image_sets, image_num=50000, mode='original', src_dir="./data", dst_dir = "./training_samples/building", img_h=256, img_w=256):
    """
    create data sets for training
    :param image_num: total number of training samples
    :param mode: 'original' or 'augment'
    :return:
    """
    # create dirs to store sample images and gts
    if not os.path.exists("{}/image".format(dst_dir)):
        os.mkdir("{}/image".format(dst_dir))
    if not os.path.exists("{}/label".format(dst_dir)):
        os.mkdir("{}/label".format(dst_dir))

    # number of samples for each image
    image_each = int(np.ceil(image_num // len(image_sets)))
    g_count = 0
    for i in range(len(image_sets)):
        src_img = load_image("{}/image/{}".format(src_dir, image_sets[i]), grayscale=False)
        label_img = load_image("{}/label/{}".format(src_dir, image_sets[i]), grayscale=True)

        X_height,X_width,_ = src_img.shape
        print("%s: sampling from %s..."%(datetime.datetime.now().strftime("%y-%m-%d %H:%M:%S"), image_sets[i]))

        l_count=0
        while l_count < image_each:
            #  randomly select a x and y for the upper left cornrr
            random_width = np.random.randint(0, X_width - img_w - 1)
            random_height = np.random.randint(0, X_height - img_h - 1)
            src_roi = Image.fromarray(src_img[random_height: random_height + img_h, random_width: random_width + img_w,:])
            label_roi = Image.fromarray(label_img[random_height: random_height + img_h, random_width: random_width + img_w])

            if np.mean(label_roi) <= 0:
                continue

            # data augmentation
            if mode == 'augment':
                src_roi,label_roi = data_augment(src_roi, label_roi)

            # save sample images
            src_roi.save('{}/image/{}.png'.format(dst_dir, g_count))
            label_roi.save('{}/label/{}.png'.format(dst_dir, g_count))

            l_count += 1
            g_count += 1


def create_dataset_scan(image_sets, mode='augment', src_dir="./data", dst_dir = "./training_samples/building", img_h=256, img_w=256, stride=256):
    """ create training samples by scanning from the top-left to the bottom-right with a certain stride to avoid overlap
    :param image_sets: source images
    :param mode: "augment" or "original"
    :param src_dir: source path
    :param dst_dir: destination path
    :param img_h: image height
    :param img_w: image width
    :param stride: stride
    :return: None
    """
    # create dirs to store sample images and gts
    if not os.path.exists("{}/image".format(dst_dir)):
        os.mkdir("{}/image".format(dst_dir))
    if not os.path.exists("{}/label".format(dst_dir)):
        os.mkdir("{}/label".format(dst_dir))


    for i in range(len(image_sets)):
        flag = image_sets[i].split(".")[0]
        src_img = load_image("{}/image/{}".format(src_dir, image_sets[i]), grayscale=False)
        label_img = load_image("{}/label/{}".format(src_dir, image_sets[i]), grayscale=True)

        X_height, X_width, _ = src_img.shape
        print("%s: sampling from %s..." % (datetime.datetime.now().strftime("%y-%m-%d %H:%M:%S"), image_sets[i]))

        l_count = 1
        for _row in range(0, X_height, stride):
            for _col in range(0, X_width, stride):
                src_roi = Image.fromarray(src_img[_row:_row+img_h, _col: _col+img_w].astype(np.uint8))
                label_roi = Image.fromarray(label_img[_row:_row+img_h, _col:_col+img_w].astype(np.uint8))
                if src_roi.size != (img_w, img_h):
                    continue

                # if np.mean(label_roi) <= 0:
                #     continue
                # data augmentation
                if mode == 'augment':
                    src_roi, label_roi = data_augment(src_roi, label_roi)

                if src_roi.size != (img_w, img_h) or label_roi.size!=(img_w, img_h):
                    continue
                # save sample images
                try:
                    src_roi.save('{}/image/{}_{}.png'.format(dst_dir, flag, l_count))
                    label_roi.save('{}/label/{}_{}.png'.format(dst_dir, flag, l_count))
                    l_count += 1
                except:
                    pass


if __name__=='__main__':
    # image_sets = os.listdir("F:/Data/AerialImageDataset/val/label")
    # create_dataset_scan(image_sets, mode="augment",
    #               src_dir="F:/Data/AerialImageDataset/val",
    #               dst_dir="F:/AerialBuildingFootprints/data/training_samples/validation",
    #               img_h=256, img_w=256, stride=256)
    #
    # image_sets = os.listdir("F:/Data/AerialImageDataset/train/label")
    # create_dataset_scan(image_sets, mode="augment",
    #               src_dir="F:/Data/AerialImageDataset/train",
    #               dst_dir="F:/AerialBuildingFootprints/data/training_samples/training",
    #               img_h=256, img_w=256, stride=256)

    image_sets = [fn for fn in os.listdir("F:/ChinaBuilding/samples/US_samples/image")]

    train_image_sets = [fn for fn in image_sets if "1" in fn or "2" in fn or "3" in fn]
    create_dataset_scan(train_image_sets, mode="augment",
                  src_dir="F:/ChinaBuilding/samples/US_samples",
                  dst_dir="F:/ChinaBuilding/training/training_samples/training",
                  img_h=256, img_w=256)

    val_image_sets = [fn for fn in image_sets if "4" in fn]
    create_dataset_scan(val_image_sets, mode="augment",
                  src_dir="F:/ChinaBuilding/samples/US_samples",
                  dst_dir="F:/ChinaBuilding/training/training_samples/validation",
                  img_h=256, img_w=256)

    # test_image_sets = [fn for fn in image_sets if "5" in fn]
