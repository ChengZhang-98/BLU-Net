import os
from datetime import datetime

import keras
import keras.callbacks
import pandas as pd
import tensorflow as tf
from keras import callbacks

from data_augmentation import (HistogramVoodoo, ElasticDeform, GaussianNoise, RandomFlip, DataAugmentation, RandomRot90)
from knowledge_distillation import KnowledgeDistillation, distill_knowledge
from model import (get_compiled_unet, get_compiled_lightweight_unet,
                   get_compiled_binary_lightweight_unet,
                   get_teacher_vanilla_unet,
                   get_student_lightweight_unet)
from residual_binarization import transfer_lightweight_unet_weights_to_binary_lightweight_unet
from training_utils import (get_validation_plot_callback, train_val_test_split, get_lr_scheduler, append_info_to_notes,
                            get_sleep_callback, CustomModelCheckpointCallBack, CustomLRScheduler)


def _func_get_train_val_test_dataset(target_size=(512, 512), batch_size=1, use_weight_map=False, seed=1,
                                     fold_index=0):
    image_dir = "E:/ED_MS/Semester_3/Dataset/DIC_Set/DIC_Set1_Annotated"
    image_type = "tif"
    mask_dir = "E:/ED_MS/Semester_3/Dataset/DIC_Set/DIC_Set1_Masks"
    mask_type = "tif"
    weight_map_dir = "E:/ED_MS/Semester_3/Dataset/DIC_Set/DIC_Set1_Weights"
    weight_map_type = "npy"
    dataset_name = "DIC"

    data_augmentation_transform = DataAugmentation([HistogramVoodoo(),
                                                    ElasticDeform(sigma=20),
                                                    GaussianNoise(sigma=1 / 255),
                                                    RandomFlip(),
                                                    RandomRot90()])

    train_set, val_set, test_set = train_val_test_split(
        batch_size=batch_size, dataset_name="DIC", use_weight_map=use_weight_map,
        image_dir=image_dir, image_type=image_type, mask_dir=mask_dir, mask_type=mask_type,
        weight_map_dir=weight_map_dir, weight_map_type=weight_map_type,
        target_size=target_size, data_aug_transform=data_augmentation_transform,
        seed=seed, num_folds=5, fold_index=fold_index)
    return train_set, val_set, test_set


def _func_get_callback_list(checkpoint_filepath, monitor, mode, start_epoch, end_epoch, logdir, model, val_set):
    callback_list = []
    # model_checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
    #                                                             save_weights_only=True,
    #                                                             verbose=1,
    #                                                             monitor='val_binary_IoU',
    #                                                             mode='max',
    #                                                             save_best_only=True)
    model_checkpoint_callback = CustomModelCheckpointCallBack(
        ignore=40, filepath=checkpoint_filepath, monitor=monitor, mode=mode,
        logdir=logdir)
    callback_list.append(model_checkpoint_callback)

    lr_scheduler_callback = keras.callbacks.LearningRateScheduler(
        get_lr_scheduler((start_epoch + end_epoch) // 2))
    callback_list.append(lr_scheduler_callback)

    tensorboard_callback = callbacks.TensorBoard(log_dir=logdir)
    callback_list.append(tensorboard_callback)

    tensorboard_val_image_writer = tf.summary.create_file_writer(logdir + "/val_image")
    validation_plot_callback = get_validation_plot_callback(model, val_set, [0, 1, 2, 3],
                                                            tensorboard_val_image_writer, max_output=4)
    callback_list.append(validation_plot_callback)

    callback_list.append(get_sleep_callback(120, 40))

    return callback_list


def _func_print_training_info(name, seed, train_set, val_set, batch_size, use_weight_map, start_epoch,
                              end_epoch):
    print(name)
    print("dataset seed = {}".format(seed))
    print("Training set contains {} samples\n"
          "validation set contains {} samples\n"
          "batch_size = {}".format(len(train_set.data_df),
                                   len(val_set.data_df),
                                   batch_size))

    print("train_use_weight = {}".format(use_weight_map, ))
    print("Training starts (start_epoch = {}, end_epoch = {})".format(start_epoch, end_epoch))

