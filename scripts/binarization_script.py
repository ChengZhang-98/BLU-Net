import os
from datetime import datetime

import keras
import keras.callbacks
import pandas as pd
import tensorflow as tf
from keras import callbacks

from data import DataGenerator
from data_augmentation import (HistogramVoodoo, ElasticDeform, GaussianNoise, RandomFlip, RandomRotate,
                               RandomZoomAndShift, DataAugmentation)
from model import (get_compiled_unet, get_compiled_binary_unet)
from residual_binarization import transfer_unet_weights_to_binary_unet
from training_utils import get_validation_plot_callback, train_val_split, get_lr_scheduler


# *: tensorboard --logdir="E:\ED_MS\Semester_3\Codes\MyProject\tensorboard_logs"


def script_fine_tune_binary_unet_based_on_vanilla_unet(name=None, seed=None, train_val_split_ratio=0.8):
    seed = seed
    batch_size_train = 1
    target_size = (256, 256)
    train_use_weight_map = False
    val_use_weight_map = False
    # *: lr rate
    learning_rate = 1e-3

    # *: pretrained binary_unet
    # pretrained_binary_unet_checkpoint_path = "E:/ED_MS/Semester_3/Codes/MyProject/checkpoints/binary_unet.h5"
    pretrained_binary_unet_checkpoint_path = None

    # *: residual binarization gamma
    num_activation_residual_levels = 3
    num_conv_residual_levels = 3

    image_dir = "E:/ED_MS/Semester_3/Dataset/DIC_Set/DIC_Set1_Annotated"
    image_type = "tif"
    mask_dir = "E:/ED_MS/Semester_3/Dataset/DIC_Set/DIC_Set1_Masks"
    mask_type = "tif"
    weight_map_dir = "E:/ED_MS/Semester_3/Dataset/DIC_Set/DIC_Set1_Weights"
    weight_map_type = "npy"
    dataset_name = "DIC"

    logdir = "E:/ED_MS/Semester_3/Codes/MyProject/tensorboard_logs"
    unet_checkpoint_filepath = "E:/ED_MS/Semester_3/Codes/MyProject/checkpoints/vanilla_unet-fine-tuned.h5"
    binary_unet_checkpoint_filepath = "E:/ED_MS/Semester_3/Codes/MyProject/checkpoints/" \
                                      "binary_unet-{}-{}-from_vanilla_unet.h5".format(num_activation_residual_levels,
                                                                                      num_conv_residual_levels)
    # *: continue training
    start_epoch = 0
    end_epoch = 400

    if name is None:
        logdir = os.path.join(logdir, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    else:
        logdir = os.path.join(logdir, datetime.now().strftime("%Y-%m-%d") + "_{}".format(name))

    unet = get_compiled_unet((*target_size, 1), num_levels=5, pretrained_weights=unet_checkpoint_filepath)
    binary_unet = get_compiled_binary_unet((*target_size, 1),
                                           num_activation_residual_levels=num_activation_residual_levels,
                                           num_conv_residual_levels=num_conv_residual_levels,
                                           num_levels=5,
                                           learning_rate=learning_rate)
    binary_unet = transfer_unet_weights_to_binary_unet(unet, binary_unet)
    # *: load pretrained weight
    # binary_unet.load_weights(filepath=pretrained_binary_unet_checkpoint_path)

    data_augmentation_transform = DataAugmentation([HistogramVoodoo(),
                                                    ElasticDeform(sigma=20),
                                                    GaussianNoise(sigma=1 / 255),
                                                    RandomFlip(),
                                                    RandomRotate(),
                                                    RandomZoomAndShift()])
    data_gen_train = DataGenerator(batch_size=batch_size_train, dataset_name=dataset_name, mode="train",
                                   use_weight_map=train_use_weight_map,
                                   image_dir=image_dir, image_type=image_type,
                                   mask_dir=mask_dir, mask_type=mask_type,
                                   weight_map_dir=weight_map_dir, weight_map_type=weight_map_type,
                                   target_size=target_size, data_aug_transform=data_augmentation_transform, seed=seed)
    # *: val_batch_size = 2
    data_gen_train, data_gen_val = train_val_split(data_gen_train, train_val_split_ratio, validation_batch_size=1,
                                                   use_weight_map_val=val_use_weight_map)

    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath=binary_unet_checkpoint_filepath,
                                                                save_weights_only=True,
                                                                verbosde=1,
                                                                monitor='val_binary_IoU',
                                                                mode='max',
                                                                save_best_only=True)
    # *: lr scheduler
    lr_scheduler_callback = keras.callbacks.LearningRateScheduler(
        schedule=get_lr_scheduler((end_epoch - start_epoch) // 2))
    tensorboard_val_image_writer = tf.summary.create_file_writer(logdir + "/val_image")
    tensorboard_callback = callbacks.TensorBoard(log_dir=logdir)
    validation_plot_callback = get_validation_plot_callback(binary_unet, data_gen_val, [0, 1, 2, 3],
                                                            tensorboard_val_image_writer, max_output=4)

    callback_list = [tensorboard_callback, validation_plot_callback,
                     model_checkpoint_callback, lr_scheduler_callback]

    print("dataset seed = {}".format(seed))
    print("Training set contains {} samples with a batch size {},\n"
          "validation set contains {} samples with a batch size {}".format(len(data_gen_train.data_df),
                                                                           data_gen_train.batch_size,
                                                                           len(data_gen_val.data_df),
                                                                           data_gen_val.batch_size))
    print("train_use_weight_map = {}, val_use_weight_map = {}".format(train_use_weight_map, val_use_weight_map))
    print("Training starts (start_epoch = {}, end_epoch = {})".format(start_epoch, end_epoch))
    history = binary_unet.fit(x=data_gen_train, epochs=end_epoch, initial_epoch=start_epoch,
                              validation_data=data_gen_val, shuffle=False,
                              validation_freq=1, callbacks=callback_list,
                              workers=1, use_multiprocessing=False)
    print("Training finished")

    log_df = pd.DataFrame(dict(epoch=history.epoch) | history.history)
    log_df.to_pickle(os.path.join(logdir, "log_" + name + ".pkl"))
    return log_df


def script_train_binary_unet_from_scratch(name=None, seed=None, train_val_split_ratio=0.8):
    seed = seed
    batch_size_train = 1
    target_size = (256, 256)
    train_use_weight_map = False
    val_use_weight_map = False
    # *: lr rate
    learning_rate = 1e-3

    # *: pretrained binary_unet
    # pretrained_binary_unet_checkpoint_path = "E:/ED_MS/Semester_3/Codes/MyProject/checkpoints/binary_unet.h5"
    pretrained_binary_unet_checkpoint_path = None

    # *: residual binarization gamma
    num_activation_residual_levels = 3
    num_conv_residual_levels = 3

    image_dir = "E:/ED_MS/Semester_3/Dataset/DIC_Set/DIC_Set1_Annotated"
    image_type = "tif"
    mask_dir = "E:/ED_MS/Semester_3/Dataset/DIC_Set/DIC_Set1_Masks"
    mask_type = "tif"
    weight_map_dir = "E:/ED_MS/Semester_3/Dataset/DIC_Set/DIC_Set1_Weights"
    weight_map_type = "npy"
    dataset_name = "DIC"

    logdir = "E:/ED_MS/Semester_3/Codes/MyProject/tensorboard_logs"
    binary_unet_checkpoint_filepath = "E:/ED_MS/Semester_3/Codes/MyProject/checkpoints/" \
                                      "binary_unet-{}-{}-trained_from_scratch.h5".format(num_activation_residual_levels,
                                                                                         num_conv_residual_levels)
    # *: continue training
    start_epoch = 0
    end_epoch = 400

    if name is None:
        logdir = os.path.join(logdir, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    else:
        logdir = os.path.join(logdir, datetime.now().strftime("%Y-%m-%d") + "_{}".format(name))

    binary_unet = get_compiled_binary_unet((*target_size, 1),
                                           num_activation_residual_levels=num_activation_residual_levels,
                                           num_conv_residual_levels=num_conv_residual_levels,
                                           num_levels=5,
                                           learning_rate=learning_rate)

    data_augmentation_transform = DataAugmentation([HistogramVoodoo(),
                                                    ElasticDeform(sigma=20),
                                                    GaussianNoise(sigma=1 / 255),
                                                    RandomFlip(),
                                                    RandomRotate(),
                                                    RandomZoomAndShift()])
    data_gen_train = DataGenerator(batch_size=batch_size_train, dataset_name=dataset_name, mode="train",
                                   use_weight_map=train_use_weight_map,
                                   image_dir=image_dir, image_type=image_type,
                                   mask_dir=mask_dir, mask_type=mask_type,
                                   weight_map_dir=weight_map_dir, weight_map_type=weight_map_type,
                                   target_size=target_size, data_aug_transform=data_augmentation_transform, seed=seed)
    # *: val_batch_size = 2
    data_gen_train, data_gen_val = train_val_split(data_gen_train, train_val_split_ratio, validation_batch_size=1,
                                                   use_weight_map_val=val_use_weight_map)

    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath=binary_unet_checkpoint_filepath,
                                                                save_weights_only=True,
                                                                verbosde=1,
                                                                monitor='val_binary_IoU',
                                                                mode='max',
                                                                save_best_only=True)
    # *: lr scheduler
    lr_scheduler_callback = keras.callbacks.LearningRateScheduler(
        schedule=get_lr_scheduler((end_epoch - start_epoch) // 2))
    tensorboard_val_image_writer = tf.summary.create_file_writer(logdir + "/val_image")
    tensorboard_callback = callbacks.TensorBoard(log_dir=logdir)
    validation_plot_callback = get_validation_plot_callback(binary_unet, data_gen_val, [0, 1, 2, 3],
                                                            tensorboard_val_image_writer, max_output=4)

    callback_list = [tensorboard_callback, validation_plot_callback,
                     model_checkpoint_callback, lr_scheduler_callback]

    print("dataset seed = {}".format(seed))
    print("Training set contains {} samples with a batch size {},\n"
          "validation set contains {} samples with a batch size {}".format(len(data_gen_train.data_df),
                                                                           data_gen_train.batch_size,
                                                                           len(data_gen_val.data_df),
                                                                           data_gen_val.batch_size))
    print("train_use_weight_map = {}, val_use_weight_map = {}".format(train_use_weight_map, val_use_weight_map))
    print("Training starts (start_epoch = {}, end_epoch = {})".format(start_epoch, end_epoch))
    history = binary_unet.fit(x=data_gen_train, epochs=end_epoch, initial_epoch=start_epoch,
                              validation_data=data_gen_val, shuffle=False,
                              validation_freq=1, callbacks=callback_list,
                              workers=1, use_multiprocessing=False)
    print("Training finished")

    log_df = pd.DataFrame(dict(epoch=history.epoch) | history.history)
    log_df.to_pickle(os.path.join(logdir, "log_" + name + ".pkl"))
    return log_df


if __name__ == '__main__':
    # binary_unet_fine_tuned_log_df = \
    #     script_fine_tune_binary_unet_based_on_vanilla_unet(name="binary_unet-fine_transferred_from_vanilla",
    #                                                        seed=1,
    #                                                        train_val_split_ratio=0.8)

    binary_unet_trained_from_scratch_df = script_train_binary_unet_from_scratch(name="binary_unet_trained_from_scratch",
                                                                                seed=1,
                                                                                train_val_split_ratio=0.8)

    pass