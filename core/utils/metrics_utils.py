"""
    Script: metrics_utils.py
    Author: Penghua Liu
    Date: 2019-01-25
    Email: liuphhhh@foxmail.com
    Functions: util function to evaluate metrics
"""
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, average_precision_score, recall_score

def evaluate_average_precision(y_true, y_pred):
    y_true = y_true.reshape(-1).astype(np.uint8)
    y_pred = y_pred.reshape(-1).astype(np.uint8)
    return average_precision_score(y_true, y_pred, average="weighted")


def evaluate_precision_recall_f1(y_true, y_pred):
    y_true = y_true.reshape(-1).astype(np.uint8)
    y_pred = y_pred.reshape(-1).astype(np.uint8)
    pre = precision_score(y_true, y_pred,  average='weighted')
    rec = recall_score(y_true, y_pred, average="weighted")
    f1 = f1_score(y_true, y_pred, average="weighted")
    return pre, rec, f1


def evaluate_iou_per_class(y_true, y_pred, label=1):
    y_true = y_true.reshape(-1).astype(np.uint8)
    y_pred = y_pred.reshape(-1).astype(np.uint8)

    true_i = y_true == label
    pred_i = y_pred == label

    logical_i = float(np.sum(np.logical_and(true_i, pred_i)))
    logical_u = float(np.sum(np.logical_or(true_i, pred_i)))

    return (logical_i+1e-10) / (logical_u+1e-10)



def evaluate_mean_iou(y_true, y_pred):
    unique_labels = np.unique(y_true)
    num_unique_labels = len(unique_labels)

    I = np.zeros(num_unique_labels)
    U = np.zeros(num_unique_labels)

    for index, val in enumerate(unique_labels):
        pred_i = y_pred == val
        label_i = y_true == val

        I[index] = float(np.sum(np.logical_and(label_i, pred_i)))
        U[index] = float(np.sum(np.logical_or(label_i, pred_i)))

    mean_iou = np.mean(I / U)
    return mean_iou


def evaluate_accuracy(y_true, y_pred):
    y_true = y_true.reshape(-1).astype(np.uint8)
    y_pred = y_pred.reshape(-1).astype(np.uint8)
    return accuracy_score(y_true, y_pred)


def evaluate_segmentation(y_true, y_pred):
    acc = evaluate_accuracy(y_true, y_pred)
    pre, rec, f1 = evaluate_precision_recall_f1(y_true, y_pred)
    miou = evaluate_mean_iou(y_true, y_pred)
    iou1 = evaluate_iou_per_class(y_true, y_pred, label=1)

    return acc, pre, rec, f1, miou, iou1