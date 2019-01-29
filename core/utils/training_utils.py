"""
    The MIT License (MIT)

    Copyright (c) 2017 Penghua Liu

    Script: net_utils.py
    Author: Penghua Liu
    Date: 2019-01-13
    Email: liuphhhh@foxmail.com
    Functions: some util functions for training, including:
        generateTrainData & generateValidData: data generator for training and validation
        plotHistory: plot the logs of training history
        TrainingArgs: training args class

"""

import os, random
import numpy as np
import json
import datetime
from argparse import ArgumentParser

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import RMSprop, Adam, SGD
from keras.utils.vis_utils import plot_model

from ..nets import SemanticSegmentationModel
from .model_utils import load_custom_model
from .boundary import generateBoundaryDist
from .image_utils import load_image
from .loss_utils import lovasz_loss, positive_iou_metric, mIoU_metric



def get_train_val(filepath, val_rate=0.25):
    """ Split training and validation file names
    :param filepath: the root directory, string
    :param val_rate: the rate of validation samples, float, default 0.25
    :return: the training and validation file names
    """
    train_url = []
    train_set = []
    val_set = []
    for pic in os.listdir(filepath + '/image'):
        train_url.append(pic)
    random.shuffle(train_url)
    total_num = len(train_url)
    val_num = int(val_rate * total_num)
    for i in range(len(train_url)):
        if i < val_num:
            val_set.append(train_url[i])
        else:
            train_set.append(train_url[i])
    return train_set, val_set



def generateTrainData(filepath, batch_size, data=[], target_size=None, with_dist=False, dist_bins=10):
    """ Generate data from training file names
    :param filepath: the root directory, string.
    :param batch_size: batch size, int
    :param data: specifically, this is the data file names, namely urls. list of string.
    :param target_size: the target size of images. tuple of int, if set to None, the size will noe be changed.
    :param with_dist: whether to generate distance to boundary. boolean, True or False
    :param dist_bins: distance bins, int.
    :return:
    """
    while True:
        train_data = []
        train_label = []
        train_dist = []
        batch = 0
        np.random.shuffle(data)
        for i in (range(len(data))):
            url = data[i]
            batch += 1
            # HACK: NOTE!!! here values of the image and label are divided by 255.0.
            train_data.append(load_image(filepath + "/image/" + url, grayscale=False, scale=255.0,
                                         target_size=target_size))
            label = np.expand_dims(load_image(filepath + "/label/" + url, grayscale=True, scale=255.0,
                                              target_size=target_size),
                                   axis=-1)
            train_label.append(label)
            if with_dist:
                train_dist.append(generateBoundaryDist(label[:,:,0], K=dist_bins))

            if batch % batch_size == 0:
                train_data = np.array(train_data)
                train_label = np.array(train_label)
                train_dist = np.array(train_dist)
                if with_dist:
                    yield (train_data, [train_label, train_dist])
                else:
                    yield (train_data, train_label)
                train_data = []
                train_label = []
                train_dist = []
                batch = 0



def generateValidData(filepath, batch_size, data=[], target_size=None, with_dist=False, dist_bins = 10):
    """ Generate data from validation file names
    :param filepath: the root directory, string.
    :param batch_size: batch size, int
    :param data: specifically, this is the data file names, namely urls. list of string.
    :param target_size: the target size of images. tuple of int, if set to None, the size will noe be changed.
    :param with_dist: whether to generate distance to boundary. boolean, True or False
    :param dist_bins: distance bins, int.
    :return:
    """
    while True:
        valid_data = []
        valid_label = []
        valid_dist = []
        batch = 0
        start_index = np.random.randint(0, len(data)-batch_size)
        for i in (range(start_index, len(data))):
            url = data[i]
            batch += 1
            valid_data.append(load_image(filepath + "/image/" + url, grayscale=False, scale=255.0,
                                         target_size=target_size))
            label = np.expand_dims(load_image(filepath + "/label/" + url, grayscale=True, scale=255.0,
                                              target_size=target_size),
                                   axis=-1)
            valid_label.append(label)
            if with_dist:
                valid_dist.append(generateBoundaryDist(label[:,:,0], K=dist_bins))

            if batch % batch_size == 0:
                valid_data = np.array(valid_data)
                valid_label = np.array(valid_label)
                valid_dist = np.array(valid_dist)
                if with_dist:
                    yield (valid_data, [valid_label, valid_dist])
                else:
                    yield (valid_data, valid_label)
                valid_data = []
                valid_label = []
                valid_dist = []
                batch = 0


def plotHistory(H, figure_save_path=None, metric_name="my_iou_metric"):
    """ plot figure of training accuracies and losses.
    :param H: the training history instance
    :param figure_save_path: file path to save the figure
    :param metric_name: metric name
    :return: None
    """
    # import matplotlib
    # matplotlib.use("Qt4Agg", warn=False, force=True)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(H.epoch, H.history[metric_name], label="training")
    plt.plot(H.epoch, H.history["val_%s"%metric_name], label="validation")
    plt.title("accuracy max=%.6f" % (np.max(H.history["val_%s"%metric_name])))
    plt.xlabel("epoch")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(H.epoch, H.history["loss"], label="training")
    plt.plot(H.epoch, H.history["val_loss"], label="validation")
    plt.title("loss optimal=%.6f" % (np.min(H.history["val_loss"])))
    plt.xlabel("epoch")
    plt.legend()
    if figure_save_path is not None:
        plt.savefig(figure_save_path)
    #plt.show()


def learning_rate_schedule(epoch, init_lr=0.001):
    if epoch < 25:
        return init_lr
    elif epoch < 50:
        return init_lr * 0.2
    else:
        return init_lr * 0.02



class TrainingArgs(object):
    def __init__(self, config_fname, model_name, dataset_name):
        print("{}>> loading configure file: {}...".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), config_fname))
        args = json.load(open(config_fname, "r"))
        args = args[model_name][dataset_name]
        self.model_name = args["model_name"]
        self.old_model_version = args["old_model_version"]
        self.new_model_version = args["new_model_version"]
        self.training_samples_dir = args["training_samples_dir"]
        self.validation_samples_dir = args["validation_samples_dir"]
        self.validation_size = args["validation_size"]
        self.image_height = args["image_height"]
        self.image_width = args["image_width"]
        self.image_channel = args["image_channel"]
        self.n_class = args["n_class"]
        self.init_learning_rate = args["init_learning_rate"]
        self.optimizer_name = args["optimizer"]
        if self.optimizer_name.lower()=="adam":
            self.optimizer = Adam(self.init_learning_rate)
        elif self.optimizer_name.lower()=="rmsprop":
            self.optimizer = RMSprop(self.init_learning_rate)
        else:
            self.optimizer = SGD(self.init_learning_rate)

        self.verbose = args["verbose"]
        self.batch_size = args["batch_size"]
        self.epoch = args["epoch"]
        self.steps_per_epoch = args["steps_per_epoch"]
        self.steps_per_epoch_val = args["steps_per_epoch_val"]
        self.init_filters = args["init_filters"]
        self.dropout = args["dropout"]
        self.loss = lovasz_loss if args["loss"] == "lovasz_loss" else "binary_crossentropy"

        self.metric_name = args["metric_name"]
        if args["metric_name"] == "mIoU_metric":
            self.metric = mIoU_metric
        elif args["metric_name"] == "positive_iou_metric":
            self.metric = positive_iou_metric
        else:
            self.metric = "acc"
        self.encoder_name = args["encoder_name"]
        self.encoder_weights = args["encoder_weights"]

        if not os.path.exists("{}/models".format(args["workspace"])):
            os.mkdir("{}/models".format(args["workspace"]))
        if not os.path.exists("{}/logs".format(args["workspace"])):
            os.mkdir("{}/logs".format(args["workspace"]))
        if not os.path.exists("{}/figures".format(args["workspace"])):
            os.mkdir("{}/figures".format(args["workspace"]))
        self.load_model_name = "{}/models/{}.h5".format(args["workspace"], self.old_model_version)
        self.save_model_name = "{}/models/{}.h5".format(args["workspace"], self.new_model_version)
        self.save_log_name = "{}/logs/{}.npz".format(args["workspace"], self.new_model_version)
        self.figure_save_path = "{}/figures/Figure_{}.png".format(args["workspace"], self.new_model_version)

        self.callbacks = list()
        self.callbacks.append(ModelCheckpoint(self.save_model_name, monitor='val_%s' % self.metric_name, mode='max',
                                       save_best_only=True, verbose=1))
        # self.callbacks.append(LearningRateScheduler(schedule=learning_rate_schedule))
        if "early_stop" in args["callbacks"]:
            early_stop = EarlyStopping(monitor="val_{}".format(self.metric_name), mode='max',
                                       patience=args["callbacks"]["early_stop"]["patience"], verbose=1)
            self.callbacks.append(early_stop)
        if "reduce_lr" in args["callbacks"]:
            reduce_lr = ReduceLROnPlateau(monitor='val_%s' % self.metric_name, mode='max',
                                          factor=args["callbacks"]["reduce_lr"]["factor"],
                                          patience=args["callbacks"]["reduce_lr"]["patience"],
                                          min_lr=args["callbacks"]["reduce_lr"]["min_lr"], verbose=1)
            self.callbacks.append(reduce_lr)






def parse_params():
    ap = ArgumentParser()
    ap.add_argument("configure_file", help="training configuration file name", type=str)
    ap.add_argument("--model", type=str, default="deeplab_v3p")
    ap.add_argument("--dataset", type=str, default="inria")

    return ap



def training_main(trainargs):
    # load or build model
    if os.path.exists(trainargs.load_model_name):
        print("load model from ", trainargs.load_model_name)
        model = load_custom_model(trainargs.load_model_name)
    else:
        print("build new model: ", trainargs.model_name)
        model = SemanticSegmentationModel(encoder_name=trainargs.encoder_name,
                                          encoder_weights=trainargs.encoder_weights,
                                          model_name=trainargs.model_name,
                                          input_shape=(trainargs.image_height,
                                                       trainargs.image_width,
                                                       trainargs.image_channel),
                                          n_class=trainargs.n_class,
                                          init_filters=trainargs.init_filters)


    # compile the model and then set the callbacks
    model.summary()
    if not os.path.exists(trainargs.load_model_name.replace(".h5", ".png")):
        plot_model(model, trainargs.save_model_name.replace(".h5", ".png"), show_shapes=True)

    model.compile(loss=[trainargs.loss], optimizer=trainargs.optimizer, metrics=[trainargs.metric])


    train_set = os.listdir(trainargs.training_samples_dir + '/image')
    val_set = os.listdir(trainargs.validation_samples_dir + '/image')
    train_numb, valid_numb = len(train_set), len(val_set)
    # if steps are set to 0, all the samples will be used for training and validation
    if (trainargs.steps_per_epoch==0):
        trainargs.steps_per_epoch = train_numb // trainargs.batch_size
        trainargs.steps_per_epoch_val = valid_numb // trainargs.batch_size

    print("+ " * 50)
    print("+    training data size = %d" % train_numb)
    print("+    validation data size = %d" % valid_numb)
    print("+    training iteration/epoch = %d" % trainargs.steps_per_epoch)
    print("+    validation iteration/epoch = %d" % trainargs.steps_per_epoch_val)
    print("+    model save path: %s" % trainargs.save_model_name)
    print("+ " * 50)
    print("%s starting training..." % datetime.datetime.now().strftime("%y-%m-%d %H:%M:%S"))


    H = model.fit_generator(
        generator=generateTrainData(trainargs.training_samples_dir, trainargs.batch_size, train_set,
                                    target_size=(trainargs.image_height, trainargs.image_width), with_dist=False),
        steps_per_epoch=trainargs.steps_per_epoch,
        epochs=trainargs.epoch,
        validation_data=generateValidData(trainargs.validation_samples_dir, trainargs.batch_size, val_set,
                                          target_size=(trainargs.image_height, trainargs.image_width), with_dist=False),
        validation_steps=trainargs.steps_per_epoch_val,
        callbacks=trainargs.callbacks,
        verbose=trainargs.verbose)
    print("%s training success!" % datetime.datetime.now().strftime("%y-%m-%d %H:%M:%S"))

    # save and print logs
    np.savez(trainargs.save_log_name, my_iou_metric=np.array(H.history[trainargs.metric_name]),
             val_my_iou_metric=np.array(H.history['val_%s' % trainargs.metric_name]),
             loss=np.array(H.history["loss"]), val_loss=np.array(H.history["val_loss"]))

    print(H.history[trainargs.metric_name])
    print(H.history['val_%s' % trainargs.metric_name])
    print(H.history["loss"])
    print(H.history["val_loss"])
    plotHistory(H, trainargs.figure_save_path, trainargs.metric_name)
