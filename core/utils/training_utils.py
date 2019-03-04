import os
import pprint
import datetime
import numpy as np
from keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, TensorBoard
from keras.utils.vis_utils import plot_model

from core.configures import *
from core.utils.loss_utils import lovasz_softmax
from core.utils.model_utils import load_custom_model
from core.nets import SemanticSegmentationModel
from core.utils.data_utils.data_generator import ImageDataGenerator


def parse_training_args():
    """ parse args from .configures.py
    :return: a dict， the training args。
    """
    trainingArgs = TRAINING_CONFIG

    def learning_rate_schedule(epoch):
        lr_base = trainingArgs["base_lr"]
        lr_min = trainingArgs["min_lr"]
        epochs = trainingArgs["epoch"]
        lr_power = trainingArgs["lr_power"]
        lr_cycle = trainingArgs["lr_cycle"]
        mode = trainingArgs['lr_mode']
        if mode is 'power_decay':
            # original lr scheduler
            lr = lr_base * ((1 - float(epoch) / epochs) ** lr_power)
        elif mode is 'exp_decay':
            # exponential decay
            lr = (float(lr_base) ** float(lr_power)) ** float(epoch + 1)

        elif mode is 'progressive_drops':
            # drops as progression proceeds, good for sgd
            if epoch > 0.9 * epochs:
                lr = 0.0001
            elif epoch > 0.75 * epochs:
                lr = 0.001
            elif epoch > 0.5 * epochs:
                lr = 0.01
            else:
                lr = 0.1
        elif mode is "cosine_cycle":
            lr = ((lr_base - lr_min) / 2) * (np.cos(2 * np.pi * (epoch % lr_cycle / lr_cycle)) + 1)
        elif mode is "none":
            lr = lr_base
        else:
            raise ValueError("Invalid learning rate schedule mode: {}. Expected 'power_decay', 'exp_decay', 'adam', "
                             "'progressive_drops', 'cosine_cycle'.".format(mode))

        return lr


    losses = {
                "binary_crossentropy": "binary_crossentropy",
                "categorical_crossentropy": "categorical_crossentropy",
                "lovasz_softmax": lovasz_softmax
              }
    metrics = {
                    "acc": "acc"
               }

    trainingArgs["loss"] = losses[trainingArgs.get("loss", "categorical_crossentropy")]
    trainingArgs["metric"] = metrics[trainingArgs.get("metric_name", "acc")]

    if trainingArgs["optimizer_name"].lower()=="adam":
        trainingArgs["optimizer"] = Adam(trainingArgs["base_lr"])
    elif trainingArgs["optimizer_name"].lower()=="rmsprop":
        trainingArgs["optimizer"] = RMSprop(trainingArgs["base_lr"])
    else:
        trainingArgs["optimizer"] = SGD(trainingArgs["base_lr"], momentum=0.9)

    trainingArgs["encoder_name"] = trainingArgs.get("encoder_name", "resnet_v2_101")
    trainingArgs["encoder_weights"] = trainingArgs.get("encoder_weights", None)

    if not os.path.exists("{}/models".format(trainingArgs["workspace"])):
        os.mkdir("{}/models".format(trainingArgs["workspace"]))
    if not os.path.exists("{}/logs".format(trainingArgs["workspace"])):
        os.mkdir("{}/logs".format(trainingArgs["workspace"]))
    if not os.path.exists("{}/figures".format(trainingArgs["workspace"])):
        os.mkdir("{}/figures".format(trainingArgs["workspace"]))
    trainingArgs["load_model_name"] = "{}/models/{}.h5".format(trainingArgs["workspace"], trainingArgs["old_model_version"])
    trainingArgs["save_model_name"] = "{}/models/{}.h5".format(trainingArgs["workspace"], trainingArgs["new_model_version"])

    trainingArgs["callbacks"] = list()
    # TODO: Modify to save weights only
    trainingArgs["callbacks"].append(ModelCheckpoint(trainingArgs["save_model_name"], save_best_only=True, verbose=1))
    trainingArgs["callbacks"].append(LearningRateScheduler(schedule=learning_rate_schedule, verbose=1))
    trainingArgs["callbacks"].append(TensorBoard(log_dir=os.path.join(trainingArgs["workspace"], 'logs')))
    if "early_stop" in trainingArgs["callbacks"]:
        trainingArgs["callbacks"].append(EarlyStopping(patience=trainingArgs["callbacks"]["early_stop"]["patience"], verbose=1))
    return trainingArgs


def training_main(args):
    """ main api to train a model.
    :param args: args, a dict.

    :return: None
    """
    # get training and validation sample names
    with open(args["train_fnames_path"], "r", encoding="utf-8") as f:
        train_base_fnames = [line.strip() for line in f]
    if args["val_fnames_path"] is not None and os.path.exists(args["val_fnames_path"]):
        with open(args["val_fnames_path"], "r", encoding="utf-8") as f:
            val_base_fnames = [line.strip() for line in f]
    else:
        val_base_fnames = []
    n_train, n_val = len(train_base_fnames), len(val_base_fnames)
    # if steps are set to 0, all the samples will be used
    if (args["steps_per_epoch"]==0):
        args["steps_per_epoch"] = n_train // args["batch_size"]
    if (args["steps_per_epoch_val"]==0):
        args["steps_per_epoch_val"] = n_val // args["batch_size"]
    print(">>>> training configurations:")
    pprint.pprint(args)

    # load or build model
    if os.path.exists(args["load_model_name"]):
        print(">>>> load model from ", args["load_model_name"])
        model = load_custom_model(args["load_model_name"])
    else:
        print(">>>> build new model: ", args["save_model_name"])
        model = SemanticSegmentationModel(model_name=args["model_name"],
                                          input_shape=(args["image_height"],
                                                       args["image_width"],
                                                       args["image_channel"]),
                                          n_class=args["n_class"],
                                          encoder_name=args["encoder_name"],
                                          encoder_weights=args["encoder_weights"],
                                          init_filters=INIT_FILTERS,
                                          dropout=DROPOUT,
                                          weight_decay=WEIGHT_DECAY,
                                          kernel_initializer=KERNEL_INITIALIZER,
                                          bn_epsilon=BN_EPSILON,
                                          bn_momentum=BN_MOMENTUM)
        plot_model(model, args["save_model_name"].replace(".h5", ".png"), show_shapes=True)

    model.summary()
    model.compile(loss=args["loss"], optimizer=args["optimizer"], metrics=[args["metric"]])


    print("+ " * 80)
    print("+    training data size = %d" % n_train)
    print("+    validation data size = %d" % n_val)
    print("+    training iteration/epoch = %d" % args["steps_per_epoch"])
    print("+    validation iteration/epoch = %d" % args["steps_per_epoch_val"])
    print("+    model save path: %s" % args["save_model_name"])
    print("+ " * 80)

    train_datagen = ImageDataGenerator(fill_mode='constant',
                                       cval=args["cval"],
                                       label_cval=args["label_cval"],
                                       rescale=args["featurewise_scale"],

                                       channel_shift_range=20.,
                                       horizontal_flip=True
                                        )
    val_datagen = ImageDataGenerator(fill_mode='constant',
                                     cval=args["cval"],
                                     label_cval=args["label_cval"],
                                     rescale=args["featurewise_scale"])

    if n_val==0:
        print("%s starting training without validation..." % datetime.datetime.now().strftime("%y-%m-%d %H:%M:%S"))
        model.fit_generator(
            generator=train_datagen.flow_from_directory(
                base_fnames=train_base_fnames,
                image_dir=args["image_dir"],
                image_suffix=args["image_suffix"],
                image_color_mode=args["image_color_mode"],
                label_dir=args["label_dir"],
                label_suffix=args["label_suffix"],
                n_class=args["n_class"],
                feed_onehot_label=args["feed_onehot_label"],
                ignore_label=args["ignore_label"],
                crop_mode=args["crop_mode"],
                target_size=(args["image_height"], args["image_width"]),
                batch_size=args["batch_size"],
                shuffle=True,
                save_to_dir=args["save_to_dir"],
                save_image_path=args["save_image_path"],
                save_label_path=args["save_label_path"],
                dataset_name=args["dataset_name"]
            ),
            steps_per_epoch=args["steps_per_epoch"],
            validation_steps=args["steps_per_epoch_val"],
            epochs=args["epoch"],
            callbacks=args["callbacks"],
            verbose=args["verbose"]
        )
    else:
        print("%s starting training and validation..." % datetime.datetime.now().strftime("%y-%m-%d %H:%M:%S"))
        model.fit_generator(
            generator=train_datagen.flow_from_directory(
                base_fnames=train_base_fnames,
                image_dir=args["image_dir"],
                image_suffix=args["image_suffix"],
                image_color_mode=args["image_color_mode"],
                label_dir=args["label_dir"],
                label_suffix=args["label_suffix"],
                n_class=args["n_class"],
                feed_onehot_label=args["feed_onehot_label"],
                ignore_label=args["ignore_label"],
                crop_mode=args["crop_mode"],
                target_size=(args["image_height"], args["image_width"]),
                batch_size=args["batch_size"],
                shuffle=True,
                save_to_dir=args["save_to_dir"],
                save_image_path=args["save_image_path"],
                save_label_path=args["save_label_path"],
                dataset_name=args["dataset_name"]
            ),
            validation_data=val_datagen.flow_from_directory(
                base_fnames=train_base_fnames,
                image_dir=args["image_dir"],
                image_suffix=args["image_suffix"],
                image_color_mode=args["image_color_mode"],
                label_dir=args["label_dir"],
                label_suffix=args["label_suffix"],
                n_class=args["n_class"],
                feed_onehot_label=args["feed_onehot_label"],
                ignore_label=args["ignore_label"],
                crop_mode=args["crop_mode"],
                target_size=(args["image_height"], args["image_width"]),
                batch_size=args["batch_size"],
                shuffle=False),
            steps_per_epoch=args["steps_per_epoch"],
            validation_steps=args["steps_per_epoch_val"],
            epochs=args["epoch"],
            callbacks=args["callbacks"],
            verbose=args["verbose"]
        )

    print("%s training success!" % datetime.datetime.now().strftime("%y-%m-%d %H:%M:%S"))