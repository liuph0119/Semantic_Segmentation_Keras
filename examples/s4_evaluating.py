import os
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from core.utils.data_utils.image_io_utils import load_image
from core.utils.data_utils.label_transform_utils import labelarray_to_onehot, onehot_to_labelarray
from core.utils.vis_utils import plot_label_img
from core.utils.metric_utils import evaluate_segmentation

def evaluating_main(args):
    with open(args["colour_mapping_path"], "r") as f:
        colour_mapping = json.load(f)[args["data_name"]]
    preds_fnames = os.listdir(args["preds_dir"])
    label_fnames = os.listdir(args["label_dir"])

    mIous, accs, precisions, recalls, f1s, ious_per_class = [], [], [], [], [], []
    for preds_fname, label_fname in tqdm(zip(preds_fnames, label_fnames)):
        label = load_image(os.path.join(args["label_dir"], label_fname), is_gray=args["label_is_gray"])
        label = onehot_to_labelarray(labelarray_to_onehot(label, colour_mapping))

        preds = load_image(os.path.join(args["preds_dir"], preds_fname), is_gray=args["label_is_gray"])
        preds = onehot_to_labelarray(labelarray_to_onehot(preds, colour_mapping))

        if args["plot"]:
            fig, (ax1, ax2) = plt.subplots(1, 2)
            plot_label_img(label, ax=ax1)
            plot_label_img(preds, ax=ax2)
            plt.show()

        acc, pre, rec, f1, miou, iou_per_class = evaluate_segmentation(label, preds, total_class=2)
        print("metrics of {}: acc={:.6f}  f1={:.6f}, mIoU={:.6f}, IoU_Per_Class={}".format(preds_fname, acc, f1, miou, ";".join(iou_per_class.astype(np.str))))
        mIous.append(miou)
        accs.append(acc)
        f1s.append(f1)
        precisions.append(pre)
        recalls.append(rec)
        ious_per_class.append(iou_per_class)

    print("avg_accuracy={}".format(np.mean(accs)))
    print("avg_f1={}".format(np.mean(f1s)))
    print("avg_mIoU={}".format(np.mean(mIous)))
    print("avg_IoU_per_class={}".format(np.mean(ious_per_class, axis=0)))
    print("avg_IoU_per_class={}".format(np.mean(ious_per_class)))


if __name__ == "__main__":
    configure_file = "E:/SemanticSegmentation_Keras/configures/evaluating_configures.json"
    task_id = "task_1"
    with open(configure_file, "r") as f:
        args = json.load(f)[task_id]

    evaluating_main(args)