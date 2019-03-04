from scipy import io as scio
import numpy as np
import os
from tqdm import tqdm
from core.utils.data_utils.image_io_utils import save_to_image


mat_dir = "F:/Data/VOC/VOC_aug/cls"
png_dir = "F:/Data/VOC/VOC_aug/label"
for fn in tqdm(os.listdir(mat_dir)):
    mat = scio.loadmat(os.path.join(mat_dir, fn))
    mat = np.asarray(mat["GTcls"][0][0][1])
    save_to_image(mat, os.path.join(png_dir, fn.split(".")[0]+".png"))