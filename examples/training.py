import os
from pprint import pprint
import datetime
import numpy as np
import sys
sys.path.append('.')
from keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, TensorBoard
from keras.utils.vis_utils import plot_model

from core.configures import training_config, net_config, augment_config
from core.nets import SemanticSegmentationModel
from core.utils.data_utils.data_generator import ImageDataGenerator


def parse_training_args():
    def learning_rate_schedule(epoch):
        lr_base = training_config.base_lr
        lr_min = training_config.min_lr
        epochs = training_config.epoch
        lr_power = training_config.lr_power
        lr_cycle = training_config.lr_cycle
        mode = training_config.lr_mode
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
        elif mode is 'cosine_cycle':
            lr = ((lr_base - lr_min) / 2) * (np.cos(2 * np.pi * (epoch % lr_cycle / lr_cycle)) + 1)
        elif mode is 'none':
            lr = lr_base
        else:
            raise ValueError("Invalid learning rate schedule mode: {}. Expected 'power_decay', 'exp_decay', 'adam', "
                             "'progressive_drops', 'cosine_cycle'.".format(mode))

        return lr

    losses = {
                'binary_crossentropy': 'binary_crossentropy',
                'categorical_crossentropy': 'categorical_crossentropy'
              }
    metrics = {
                    'acc': 'acc'
               }

    training_config.loss = losses[training_config.loss_name]
    training_config.metric = metrics[training_config.metric_name]

    if training_config.optimizer_name.lower() == "adam":
        training_config.optimizer = Adam(training_config.base_lr)
    elif training_config.optimizer_name.lower() == "rmsprop":
        training_config.optimizer = RMSprop(training_config.base_lr)
    else:
        training_config.optimizer = SGD(training_config.base_lr, momentum=0.9)

    if not os.path.exists("{}/models".format(training_config.workspace)):
        os.mkdir("{}/models".format(training_config.workspace))
    if not os.path.exists("{}/logs".format(training_config.workspace)):
        os.mkdir("{}/logs".format(training_config.workspace))
    training_config.load_model_name = "{}/models/{}.h5".format(
        training_config.workspace, training_config.old_model_version)
    training_config.save_model_name = "{}/models/{}.h5".format(
        training_config.workspace, training_config.new_model_version)

    training_config.callbacks = list()
    training_config.callbacks.append(ModelCheckpoint(training_config.save_model_name, save_best_only=True,
                                                     save_weights_only=True, verbose=1))
    training_config.callbacks.append(LearningRateScheduler(schedule=learning_rate_schedule, verbose=1))
    training_config.callbacks.append(TensorBoard(log_dir=os.path.join(training_config.workspace, 'logs')))
    if training_config.early_stop_patience > 0:
        training_config.callbacks.append(EarlyStopping(patience=training_config.early_stop_patience,
                                                       verbose=1))
    return training_config


def training_main():
    """ main api to train a model.
    """
    training_config = parse_training_args()
    # get training and validation sample names
    with open(training_config.train_fnames_path, "r", encoding="utf-8") as f:
        train_base_fnames = [line.strip() for line in f]
    if training_config.val_fnames_path is not None and os.path.exists(training_config.val_fnames_path):
        with open(training_config.val_fnames_path, "r", encoding="utf-8") as f:
            val_base_fnames = [line.strip() for line in f]
    else:
        val_base_fnames = []
    n_train, n_val = len(train_base_fnames), len(val_base_fnames)
    # if steps are set to 0, all the samples will be used
    if training_config.steps_per_epoch == 0:
        training_config.steps_per_epoch = n_train // training_config.batch_size
    if training_config.steps_per_epoch_val == 0:
        training_config.steps_per_epoch_val = n_val // training_config.batch_size
    print(">>>> training configurations:")
    pprint(training_config.__dict__)

    model = SemanticSegmentationModel(model_name=training_config.model_name,
                                      input_shape=(training_config.image_height,
                                                   training_config.image_width,
                                                   training_config.image_channel),
                                      n_class=training_config.n_class,
                                      encoder_name=training_config.encoder_name,
                                      encoder_weights=training_config.encoder_weights,
                                      init_filters=net_config.init_filters,
                                      dropout=net_config.dropout,
                                      weight_decay=net_config.weight_decay,
                                      kernel_initializer=net_config.kernel_initializer,
                                      bn_epsilon=net_config.bn_epsilon,
                                      bn_momentum=net_config.bn_momentum,
                                      upscaling_method=net_config.upsampling_method)
    # load or build model
    if os.path.exists(training_config.load_model_name):
        print(">>>> load model from ", training_config.load_model_name)
        model.load_weights(training_config.load_model_name)
    else:
        print(">>>> build new model: ", training_config.save_model_name)
        plot_model(model, training_config.save_model_name.replace(".h5", ".png"), show_shapes=True)

    if training_config.model_summary:
        model.summary()
    model.compile(loss=training_config.loss, optimizer=training_config.optimizer, metrics=[training_config.metric])

    print("+ " * 80)
    print("+    training data size = %d" % n_train)
    print("+    validation data size = %d" % n_val)
    print("+    training iteration/epoch = %d" % training_config.steps_per_epoch)
    print("+    validation iteration/epoch = %d" % training_config.steps_per_epoch_val)
    print("+    model save path: %s" % training_config.save_model_name)
    print("+ " * 80)

    train_datagen = ImageDataGenerator(channel_shift_range=augment_config.channel_shift_range,
                                       horizontal_flip=augment_config.horizontal_flip,
                                       vertical_flip=augment_config.vertical_flip
                                       # TODO: include all the augmentations here
                                       )
    val_datagen = ImageDataGenerator()

    if n_val == 0:
        print("%s starting training without validation..." % datetime.datetime.now().strftime("%y-%m-%d %H:%M:%S"))
        model.fit_generator(
            generator=train_datagen.flow_from_directory(
                base_fnames=train_base_fnames,
                image_dir=training_config.image_dir,
                image_suffix=training_config.image_suffix,
                image_color_mode=training_config.image_color_mode,
                label_dir=training_config.label_dir,
                label_suffix=training_config.label_suffix,
                n_class=training_config.n_class,
                feed_onehot_label=training_config.feed_onehot_label,
                cval=training_config.cval,
                label_cval=training_config.label_cval,
                crop_mode=training_config.crop_mode,
                target_size=(training_config.image_height, training_config.image_width),
                batch_size=training_config.batch_size,
                shuffle=True,
                debug=training_config.debug,
                dataset_name=training_config.dataset_name
            ),
            steps_per_epoch=training_config.steps_per_epoch,
            validation_steps=training_config.steps_per_epoch_val,
            epochs=training_config.epoch,
            callbacks=training_config.callbacks,
            verbose=training_config.verbose
        )
    else:
        print("%s starting training and validation..." % datetime.datetime.now().strftime("%y-%m-%d %H:%M:%S"))
        model.fit_generator(
            generator=train_datagen.flow_from_directory(
                base_fnames=train_base_fnames,
                image_dir=training_config.image_dir,
                image_suffix=training_config.image_suffix,
                image_color_mode=training_config.image_color_mode,
                label_dir=training_config.label_dir,
                label_suffix=training_config.label_suffix,
                n_class=training_config.n_class,
                feed_onehot_label=training_config.feed_onehot_label,
                cval=training_config.cval,
                label_cval=training_config.label_cval,
                crop_mode=training_config.crop_mode,
                target_size=(training_config.image_height, training_config.image_width),
                batch_size=training_config.batch_size,
                shuffle=True,
                debug=training_config.debug,
                dataset_name=training_config.dataset_name
            ),
            validation_data=val_datagen.flow_from_directory(
                base_fnames=val_base_fnames,
                image_dir=training_config.image_dir,
                image_suffix=training_config.image_suffix,
                image_color_mode=training_config.image_color_mode,
                label_dir=training_config.label_dir,
                label_suffix=training_config.label_suffix,
                n_class=training_config.n_class,
                feed_onehot_label=training_config.feed_onehot_label,
                cval=training_config.cval,
                label_cval=training_config.label_cval,
                crop_mode=training_config.crop_mode,
                target_size=(training_config.image_height, training_config.image_width),
                batch_size=training_config.batch_size,
                shuffle=False),
            steps_per_epoch=training_config.steps_per_epoch,
            validation_steps=training_config.steps_per_epoch_val,
            epochs=training_config.epoch,
            callbacks=training_config.callbacks,
            verbose=training_config.verbose
        )

    print("%s training success!" % datetime.datetime.now().strftime("%y-%m-%d %H:%M:%S"))


if __name__ == "__main__":
    training_main()
