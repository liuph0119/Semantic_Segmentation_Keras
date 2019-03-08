import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, precision_recall_fscore_support


def compute_accuracy(y_true, y_pred, n_class):
    """ compute accuracy for each class and the "macro"&"micro" average accuracies.
    :param y_true: 1-D array or 2-D array.
    :param y_pred: 1-D array or 2-D array.
    :param n_class: int, total number of class of the dataset, for example 21 for VOC2012.
    :return:
        metrics: array, shape=[n_class].
        avg_macro_metric: float, macro average accuracy.
        avg_micro_metric: float, micro average accuracy.
    """
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)

    labels = np.asarray([i for i in range(n_class)])
    _mat = confusion_matrix(y_true, y_pred, labels=labels)
    metrics = np.zeros(n_class)

    for i in range(n_class):
        t_count = np.sum(_mat, axis=1)[i]
        metrics[i] = np.nan if t_count == 0 else _mat[i][i] / t_count

    avg_macro_metric = np.nanmean(metrics)
    avg_micro_metric = np.sum(_mat.diagonal(offset=0, axis1=0, axis2=1)) / np.sum(_mat)
    return metrics, avg_macro_metric, avg_micro_metric


def compute_precision_recall_f1(y_true, y_pred, n_class, avg="weighted"):
    """ compute the precision, recall and f1-score.
    :param y_true: 1-D array or 2-D array.
    :param y_pred: 1-D array or 2-D array.
    :param n_class: int, total number of class of the dataset, for example 21 for VOC2012.
    :param avg: string, average methods, one of ["micro", "macro", "weighted"].

    :return:
        precision_metrics: array, shape=[n_class].
        recall_metrics: array, shape=[n_class].
        f1_metrics: array, shape=[n_class].
        precision: float, average precision.
        recall: float, average recall.
        f1: float, average f1-score.
    """
    precision_metrics = np.zeros(n_class)
    recall_metrics = np.zeros(n_class)
    f1_metrics = np.zeros(n_class)

    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    unique_labels = np.unique(y_true)

    for i in range(n_class):
        _y_true = np.where(y_true==i, 1, 0)
        _y_pred = np.where(y_pred==i, 1, 0)
        precision_metrics[i] = precision_score(_y_true, _y_pred) if i in unique_labels else np.nan
        recall_metrics[i] = recall_score(_y_true, _y_pred) if i in unique_labels else np.nan
        f1_metrics[i] = f1_score(_y_true, _y_pred) if i in unique_labels else np.nan

    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=unique_labels, average=avg)
    return precision_metrics, recall_metrics, f1_metrics, precision, recall, f1



def compute_miou(y_true, y_pred, n_class):
    """ compute mean IoU.
    :param y_true: 1-D array or 2-D array.
    :param y_pred: 1-D array or 2-D array.
    :param n_class: int, total number of class of the dataset, for example 21 for VOC2012.
    :return:
        IoUs: array, shape=[n_class].
        mIoU: float, mean IoU.
    """
    I = np.zeros(n_class)
    U = np.zeros(n_class)
    IoUs = [np.nan for i in range(n_class)]

    unique_labels = np.unique(y_true).astype(np.int)
    for i in unique_labels:
        _y_true = np.where(y_true == i, 1, 0)
        _y_pred = np.where(y_pred == i, 1, 0)

        I[i] = float(np.sum(np.logical_and(_y_true, _y_pred)))
        U[i] = float(np.sum(np.logical_or(_y_true, _y_pred)))
        IoUs[i] = I[i] / U[i] if U[i]>0 else np.nan

    mIoU = np.nanmean(IoUs)
    return IoUs, mIoU


def compute_metrics_per_image(y_true, y_pred, n_class, avg="weighted"):
    """ compute mean IoU.
    :param y_true: 1-D array or 2-D array.
    :param y_pred: 1-D array or 2-D array.
    :param n_class: int, total number of class of the dataset, for example 21 for VOC2012.
    :param avg: string, average methods, one of ["micro", "macro", "weighted"].
    :return:
        dict, consists of multi metrics
    """
    accs, macro_metric, micro_metric = compute_accuracy(y_true, y_pred, n_class=n_class)
    precisions, recalls, f1s, precision, recall, f1 = compute_precision_recall_f1(y_true, y_pred, n_class=n_class, avg=avg)
    ious, mIoU = compute_miou(y_true, y_pred, n_class=n_class)
    return {"accuracies_per_class": accs, "macro_accuracy": macro_metric, "micro_accuracy": micro_metric,
            "precisions_per_class": precisions, "precision": precision,
            "recalls_per_class": recalls, "recall": recall,
            "f1s_pre_class": f1s, "f1": f1,
            "ious_per_class": ious, "miou": mIoU}



def compute_global_metrics(mat):
    """ compute all the metrics according a confusion matrix.
    :param mat: 2-D array, while rows==cols, confusion matrix.
    :return: dict

    for example:
        >> a = np.random.randint(0, 2, (10, 10))
        >> b = np.random.randint(0, 2, (10, 10))
        >> mat = confusion_matrix(a.reshape(-1), b.reshape(-1))
        >> print(compute_global_metrics(mat))
    """
    n_class = mat.shape[0]

    # compute accuracy
    accs = np.zeros(n_class)

    for i in range(n_class):
        t_count = np.sum(mat, axis=1)[i]
        accs[i] = np.nan if t_count == 0 else mat[i][i] / t_count
    avg_macro_acc = np.nanmean(accs)
    avg_micro_acc = np.sum(mat.diagonal(offset=0, axis1=0, axis2=1)) / np.sum(mat)

    # compute precision, recall, f1-score
    precisions = np.diagonal(mat) / np.sum(mat, axis=0)
    precisions[np.sum(mat, axis=1)==0] = 1e-8
    recalls = np.diagonal(mat) / np.sum(mat, axis=1)
    recalls[np.sum(mat, axis=1) == 0] = 1e-8
    f1s = 2*precisions*recalls / (precisions + recalls)
    precision = np.sum(np.sum(mat, axis=1) / np.sum(mat) * precisions)
    recall = np.sum(np.sum(mat, axis=1) / np.sum(mat) * recalls)
    f1 = 2*precision*recall / (precision + recall)
    # compute IoU
    I = np.diag(mat)
    U = np.sum(mat, axis=0) + np.sum(mat, axis=1) - I
    ious = I / U
    ious[np.sum(mat, axis=1) == 0] = np.nan
    miou = np.nanmean(ious)

    return {"macro_avg_acc": avg_macro_acc, "micro_avg_acc": avg_micro_acc,
            "precisions": precisions, "precision": precision,
            "recalls": recalls, "recall": recall,
            "f1s": f1s, "f1": f1,
            "ious": ious, "miou": miou}