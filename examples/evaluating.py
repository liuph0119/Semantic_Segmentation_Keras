import os
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import sys
sys.path.append('.')

from core.configures import NAME_MAP, evaluating_config
from core.utils.data_utils.image_io_utils import load_image
from core.utils.metric_utils import compute_global_metrics, compute_metrics_per_image


def evaluating_main():
    preds_fnames = os.listdir(evaluating_config.preds_dir)
    label_fnames = os.listdir(evaluating_config.label_dir)
    n_class = len(NAME_MAP[evaluating_config.dataset_name])

    if evaluating_config.mode == "global":
        mat = np.zeros((n_class, n_class))

        for preds_fname, label_fname in tqdm(zip(preds_fnames, label_fnames)):
            print(preds_fname, label_fname)
            preds = load_image(os.path.join(evaluating_config.preds_dir, preds_fname), is_gray=True)
            h, w, _ = preds.shape
            label = load_image(os.path.join(evaluating_config.label_dir, label_fname), is_gray=True, target_size=(h, w))
            _mat = confusion_matrix(label.reshape(-1), preds.reshape(-1), labels=np.arange(n_class))
            mat = mat + _mat
        if evaluating_config.ignore_0:
            mat = mat[1:, 1:]
        avg_metric = compute_global_metrics(mat)

    elif evaluating_config.mode == "per_image":
        avg_metric = {"accuracies_per_class": np.zeros(n_class), "macro_accuracy": 0., "micro_accuracy": 0.,
         "precisions_per_class": np.zeros(n_class), "precision": 0.,
         "recalls_per_class": np.zeros(n_class), "recall": 0.,
         "f1s_pre_class": np.zeros(n_class), "f1": 0.,
         "ious_per_class": np.zeros(n_class), "miou": 0.}
        count = {"accuracies_per_class": np.zeros(n_class), "macro_accuracy": 0, "micro_accuracy": 0,
                 "precisions_per_class": np.zeros(n_class), "precision": 0,
                 "recalls_per_class": np.zeros(n_class), "recall": 0,
                 "f1s_pre_class": np.zeros(n_class), "f1": 0,
                 "ious_per_class": np.zeros(n_class), "miou": 0}
        for preds_fname, label_fname in zip(preds_fnames, label_fnames):
            preds = load_image(os.path.join(evaluating_config.preds_dir, preds_fname), is_gray=True)
            h, w, _ = preds.shape
            label = load_image(os.path.join(evaluating_config.label_dir, label_fname), is_gray=True, target_size=(h, w))

            metric = compute_metrics_per_image(label, preds, n_class)
            for key in metric:
                if not np.isscalar(metric[key]):
                    for i in range(len(metric[key])):
                        if not np.isnan(metric[key][i]):
                            avg_metric[key][i] = avg_metric[key][i] + metric[key][i]
                            count[key][i] += 1
                else:
                    avg_metric[key] = avg_metric[key] + metric[key]
                    count[key] += 1

        for key in avg_metric:
            if not np.isscalar(avg_metric[key]):
                for i in range(len(avg_metric[key])):
                    if not np.isnan(avg_metric[key][i]):
                        avg_metric[key][i] = avg_metric[key][i] / count[key][i]
            else:
                avg_metric[key] = avg_metric[key] / count[key]
    else:
        raise ValueError("Invalid 'mode': %s. Expected to be 'global' or 'per_image'!" % evaluating_config.mode)

    for key in avg_metric:
        print('{:^20s}{}'.format(key, avg_metric[key]))
    return avg_metric


if __name__ == "__main__":
    print(">>>> Evaluating...")
    evaluating_main()