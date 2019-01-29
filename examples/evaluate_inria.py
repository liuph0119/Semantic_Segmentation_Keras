import os
import json
import numpy as np
from core.utils.image_utils import load_image
from core.utils.metrics_utils import evaluate_segmentation

model_name = "deeplab_v3p"
metrics_save_path = "D:/results/inria/metrics/metrics_deeplab_v3p_inria.json"
root_dir = "E:/SegData/inria/data_ori/my_test"
sub_dirs = ["austin_test", "chicago_test", "kitsap_test", "tyrol-w_test", "vienna_test"]

metrics = dict()
for sub_dir in sub_dirs:
    _dir = os.path.join(root_dir, sub_dir)

    metrics[sub_dir] = []
    image_names = os.listdir(_dir + "/image")
    for image_name in image_names:
        image_path = _dir + "/image/" + image_name
        label_path = _dir + "/label/" + image_name
        pred_path = _dir + "/prediction/" + image_name.replace(".tif", "_{}.tif".format(model_name))

        label = load_image(label_path, scale=255, grayscale=True)
        pred = load_image(pred_path, scale=255, grayscale=True)

        acc, pre, rec, f1, miou, iou1 = evaluate_segmentation(label, pred)
        print("accuracy={}\tprecision={}\trecall={}\tf1={}\tmIoU={}\tIoU1={}".format(acc, pre, rec, f1, miou,
                                                                                              iou1))
        metrics[sub_dir].append([acc, pre, rec, f1, miou, iou1])
    metrics[sub_dir] = np.mean(np.array(metrics[sub_dir]), axis=0).tolist()
    print(metrics[sub_dir])

print(metrics)
with open(metrics_save_path, "w") as f:
    json.dump(metrics, f)
