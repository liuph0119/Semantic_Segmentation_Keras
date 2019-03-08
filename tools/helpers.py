import os
import fnmatch
from tqdm import tqdm
import numpy as np
from scipy import io as scio
from PIL import Image

from core.utils.data_utils.image_io_utils import save_to_image


def convert_vocmat_to_png(mat_dir, png_dir):
    """ convert *.mat to gray-colored *.png, might be useful for voc2012_aug dataset"""
    for fn in tqdm(os.listdir(mat_dir)):
        mat = scio.loadmat(os.path.join(mat_dir, fn))
        mat = np.asarray(mat["GTcls"][0][0][1])
        save_to_image(mat, os.path.join(png_dir, fn.split(".")[0] + ".png"))



def find_all_files(path, suffix="*.jpg"):
    """ find all the absolute path of files """
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



def resize_and_copy_image(fname, dst_dir, target_size=(1024, 512)):
    """ resize image and copy to another path, might useful for cityscape dataset"""
    img = Image.open(fname)
    img = img.resize(target_size, Image.ANTIALIAS)
    img.save(os.path.join(dst_dir, os.path.basename(fname)))
    img.close()

