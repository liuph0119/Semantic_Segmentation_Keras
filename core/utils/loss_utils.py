"""
    Script: loss_utils.py
    Author: Penghua Liu
    Date: 2019-01-13
    Email: liuphhhh@foxmail.com
    Functions: some util functions to define metrics and loss functions.

"""

import tensorflow as tf
import numpy as np
import keras.backend as K
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def compute_mean_iou(label, pred):
    """ compute the mean iou of multiple class
    :param label: array, any shape
    :param pred: array, any shape
    :return: the average iou, relatively higher than the positive_iou
    """
    unique_labels = np.unique(label)
    num_unique_labels = len(unique_labels)

    I = np.zeros(num_unique_labels)
    U = np.zeros(num_unique_labels)

    for index, val in enumerate(unique_labels):
        pred_i = pred == val
        label_i = label == val

        I[index] = float(np.sum(np.logical_and(label_i, pred_i)))
        U[index] = float(np.sum(np.logical_or(label_i, pred_i)))

    mean_iou = np.mean(I / U)
    return mean_iou


def compute_positive_iou(label, pred, epsilon=1e-10):
    """ compute the iou of the positive class, which only concerns the positive objects
    :param label: array, any shape
    :param pred: array, any shape
    :param epsilon: 1e-10
    :return: the positive_iou
    """
    I = np.logical_and(label, pred)
    U = np.logical_or(label, pred)
    return (np.sum(I > 0) + epsilon) / (np.sum(U > 0) + epsilon)


def get_batch_positive_iou(label, pred):
    """ compute the positive_iou of a batch
    :param label: 4-d tensor
    :param pred: 4-d tensor
    :return: the average positive_iou of a batch
    """
    batch_size = label.shape[0]
    metrics = []
    for batch in range(batch_size):
        iou = compute_positive_iou(label[batch]>0, pred[batch]>0)
        metrics.append(iou)

    return np.mean(metrics)


def get_batch_iou_vector(label, pred):
    """ compute the positive_iou_vector of a batch
        :param label: 4-d tensor
        :param pred: 4-d tensor
        :return: the average iou vector
        """
    batch_size = label.shape[0]
    metrics = []
    for batch in range(batch_size):
        iou = compute_positive_iou(label[batch] > 0, pred[batch] > 0)
        thresholds = np.arange(0.5, 1, 0.05)
        s = []
        for threshold in thresholds:
            s.append(iou > threshold)
        metrics.append(np.mean(s))

    return np.mean(metrics)


def get_batch_mean_iou(label, pred):
    """ compute the mean_iou of a batch
        :param label: 4-d tensor
        :param pred: 4-d tensor
        :return: the average mean_iou of a batch
        """
    batch_size = label.shape[0]
    metrics = []
    for batch in range(batch_size):
        mean_iou = compute_mean_iou(label[batch]>0, pred[batch]>0)
        metrics.append(mean_iou)

    return np.mean(metrics)


def mIoU_metric(label, pred):
    return tf.py_func(get_batch_mean_iou, [label > 0.5, pred > 0.5], tf.float64)


def positive_iou_metric(label, pred):
    return tf.py_func(get_batch_positive_iou, [label > 0.5, pred > 0.5], tf.float64)


def positive_iou_metric_2(label, pred):
    return tf.py_func(get_batch_iou_vector, [label, pred > 0], tf.float64)


def calAccF1(img_true, img_pred, silence=False):
    """calculate accuracy, f1-score, precision, and recall
        :param img_true: ground truth, 2-dim array, value options: {0, 1}
        :param img_pred: predictions, 2-dim array, value options: {0, 1}
        :return: accuracy and f1-score
        """
    img_true = img_true.reshape(-1).astype(np.int)
    img_pred = img_pred.reshape(-1).astype(np.int)
    acc = accuracy_score(img_true, img_pred)
    f1 = f1_score(img_true, img_pred, average="weighted")
    p = precision_score(img_true, img_pred, average="weighted")
    r = recall_score(img_true, img_true, average="weighted")
    if not silence:
        print("accuracy: {:.6f}\t\tf1-score: {:.6f}".format(acc, f1))
    return (acc, f1, p, r)


# # # =========================================================================
### below are lovasz loss functions

def lovasz_grad(gt_sorted):
    gts = tf.reduce_sum(gt_sorted)
    intersection = gts - tf.cumsum(gt_sorted)
    union = gts + tf.cumsum(1. - gt_sorted)
    jaccard = 1. - intersection / union
    jaccard = tf.concat((jaccard[0:1], jaccard[1:] - jaccard[:-1]), 0)
    return jaccard


def lovasz_hinge(logits, labels, per_image=True, ignore=None):
    """
    Binary Lovasz hinge loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      per_image: compute the loss per image instead of per batch
      ignore: void class id
    """
    if per_image:
        def treat_image(log_lab):
            log, lab = log_lab
            log, lab = tf.expand_dims(log, 0), tf.expand_dims(lab, 0)
            log, lab = flatten_binary_scores(log, lab, ignore)
            return lovasz_hinge_flat(log, lab)
        losses = tf.map_fn(treat_image, (logits, labels), dtype=tf.float32)
        loss = tf.reduce_mean(losses)
    else:
        loss = lovasz_hinge_flat(*flatten_binary_scores(logits, labels, ignore))
    return loss


def lovasz_hinge_flat(logits, labels):
    """
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
      ignore: label to ignore
    """

    def compute_loss():
        labelsf = tf.cast(labels, logits.dtype)
        signs = 2. * labelsf - 1.
        errors = 1. - logits * tf.stop_gradient(signs)
        errors_sorted, perm = tf.nn.top_k(errors, k=tf.shape(errors)[0], name="descending_sort")
        gt_sorted = tf.gather(labelsf, perm)
        grad = lovasz_grad(gt_sorted)
        #loss = tf.tensordot(tf.nn.relu(errors_sorted), tf.stop_gradient(grad), 1, name="loss_non_void")
        loss = tf.tensordot(tf.nn.elu(errors_sorted), tf.stop_gradient(grad), 1, name="loss_non_void")
        return loss

    # deal with the void prediction case (only void pixels)
    loss = tf.cond(tf.equal(tf.shape(logits)[0], 0),
                   lambda: tf.reduce_sum(logits) * 0.,
                   compute_loss,
                   strict=True,
                   name="loss"
                   )
    return loss


def flatten_binary_scores(scores, labels, ignore=None):
    """
    Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    scores = tf.reshape(scores, (-1,))
    labels = tf.reshape(labels, (-1,))
    if ignore is None:
        return scores, labels
    valid = tf.not_equal(labels, ignore)
    vscores = tf.boolean_mask(scores, valid, name='valid_scores')
    vlabels = tf.boolean_mask(labels, valid, name='valid_labels')
    return vscores, vlabels


def lovasz_loss(y_true, y_pred):
    y_true, y_pred = K.cast(K.squeeze(y_true, -1), 'int32'), K.cast(K.squeeze(y_pred, -1), 'float32')
    #logits = K.log(y_pred / (1. - y_pred))
    logits = y_pred #Jiaxin
    loss = lovasz_hinge(logits, y_true, per_image = True, ignore = None)
    return loss
