"""
remove the CityScape images and gts to the corresponding directories, and resize the image size to 512×1024
"""
import os
import fnmatch
from PIL import Image
from tqdm import tqdm

def find_all_files(path, suffix="*.jpg"):
    dirs = [path]
    files = []
    while True:
        _dir = dirs.pop(0)
        subdirs = [os.path.join(_dir, subdir) for subdir in os.listdir(_dir)]
        for subdir in subdirs:
            if os.path.isfile(subdir):
                files.append(subdir)
            else:
                dirs.append(subdir)
        if len(dirs)==0:
            break
    return fnmatch.filter(files, suffix)


def resize_and_copy_image(original_fnames, dst_dir, target_size=(1024, 512)):
    for fname in tqdm(original_fnames):
        img = Image.open(fname)
        img = img.resize(target_size, Image.ANTIALIAS)
        img.save(os.path.join(dst_dir, os.path.basename(fname)))
        img.close()

def resize_and_copy_label(original_fnames, dst_dir, target_size=(1024, 512)):
    for fname in tqdm(original_fnames):
        if "LabelIds" in fname:
            img = Image.open(fname)
            img = img.resize(target_size, Image.ANTIALIAS)
            img.save(os.path.join(dst_dir, os.path.basename(fname)))
            img.close()


if __name__=="__main__":
    # remove training images
    # root_dir = "E:/Data/CityScape/train/image"
    # dst_dir = root_dir
    # image_fnames = find_all_files(root_dir, "*.png")
    # resize_and_copy(image_fnames, dst_dir, (1024, 512))
    mode = "prepare_train_label"

    if mode=="prepare_train_image":
        root_dir = "E:/Data/CityScape/train/image"
        dst_dir = root_dir
        image_fnames = find_all_files(root_dir, "*.png")
    elif mode=="prepare_train_label":
        root_dir = "E:/Data/CityScape/train/label"
        dst_dir = root_dir
        image_fnames = find_all_files(root_dir, "*.png")
        image_fnames = [fn for fn in image_fnames if "labelIds" in fn]
    elif mode=="prepare_val_image":
        root_dir = "E:/Data/CityScape/val/image"
        dst_dir = root_dir
        image_fnames = find_all_files(root_dir, "*.png")
    else:
        root_dir = "E:/Data/CityScape/val/label"
        dst_dir = root_dir
        image_fnames = find_all_files(root_dir, "*.png")
        image_fnames = [fn for fn in image_fnames if "labelIds" in fn]

    resize_and_copy_image(image_fnames, dst_dir, (1024, 512))