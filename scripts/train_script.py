import os
from datetime import datetime

import keras.callbacks
import tensorflow as tf
import keras
from keras import callbacks
import pandas as pd

from data import DataGenerator
from model import (get_compiled_unet, get_compiled_lightweight_unet,
                   get_compiled_binary_unet, get_compiled_binary_lightweight_unet)
from data import DataGenerator

from data_augmentation import (HistogramVoodoo, ElasticDeform, GaussianNoise, RandomFlip, RandomRotate,
                               RandomZoomAndShift, DataAugmentation, RandomRot90)
from training_utils import (get_validation_plot_callback, train_val_split, get_lr_scheduler, append_info_to_notes,
                            train_val_test_split, get_sleep_callback, CustomModelCheckpointCallBack)


# *: tensorboard --logdir="E:\ED_MS\Semester_3\Codes\MyProject\tensorboard_logs"

def script_fine_tune_vanilla_unet(name=None, seed=None, train_val_split_ratio=0.8, notes=None):
    seed = seed
    batch_size_train = 1
    target_size = (512, 512)
    train_use_weight_map = True
    val_use_weight_map = False
    pretrained_weight_path = "E:/ED_MS/Semester_3/Codes/MyProject/checkpoints/trained_weights/" \
                             "unet_agarpads_seg_evaluation2.hdf5"
    learning_rate = 1e-5

    image_dir = "E:/ED_MS/Semester_3/Dataset/DIC_Set/DIC_Set1_Annotated"
    image_type = "tif"
    mask_dir = "E:/ED_MS/Semester_3/Dataset/DIC_Set/DIC_Set1_Masks"
    mask_type = "tif"
    weight_map_dir = "E:/ED_MS/Semester_3/Dataset/DIC_Set/DIC_Set1_Weights"
    weight_map_type = "npy"
    dataset_name = "DIC"

    logdir = "E:/ED_MS/Semester_3/Codes/MyProject/tensorboard_logs"
    checkpoint_filepath = "E:/ED_MS/Semester_3/Codes/MyProject/checkpoints/vanilla_unet-fine-tuned.h5"
    start_epoch = 0
    end_epoch = 50

    if name is None:
        logdir = os.path.join(logdir, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    else:
        logdir = os.path.join(logdir, datetime.now().strftime("%Y-%m-%d") + "_{}".format(name))

    if notes is not None:
        with open(os.path.join(logdir, name + "_notes.txt"), "w+") as f:
            f.write(notes)

    unet = get_compiled_unet(input_size=(*target_size, 1),
                             num_levels=5,
                             pretrained_weights=pretrained_weight_path,
                             learning_rate=learning_rate)

    data_augmentation_transform = DataAugmentation([HistogramVoodoo(),
                                                    ElasticDeform(sigma=20),
                                                    GaussianNoise(sigma=1 / 255),
                                                    RandomFlip(),
                                                    RandomRot90(),
                                                    RandomRotate(),
                                                    RandomZoomAndShift()])

    data_gen_train = DataGenerator(batch_size=batch_size_train, dataset_name=dataset_name, mode="train",
                                   use_weight_map=train_use_weight_map,
                                   image_dir=image_dir, image_type=image_type,
                                   mask_dir=mask_dir, mask_type=mask_type,
                                   weight_map_dir=weight_map_dir, weight_map_type=weight_map_type,
                                   target_size=target_size, data_aug_transform=data_augmentation_transform, seed=seed)
    data_gen_train, data_gen_val = train_val_split(data_gen_train, train_val_split_ratio, validation_batch_size=2,
                                                   use_weight_map_val=val_use_weight_map)

    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
                                                                save_weights_only=True,
                                                                verbosde=1,
                                                                monitor='val_binary_IoU',
                                                                mode='max',
                                                                save_best_only=True)
    lr_scheduler_callback = keras.callbacks.LearningRateScheduler(
        get_lr_scheduler((start_epoch + end_epoch) // 2))

    tensorboard_val_image_writer = tf.summary.create_file_writer(logdir + "/val_image")
    tensorboard_callback = callbacks.TensorBoard(log_dir=logdir)
    validation_plot_callback = get_validation_plot_callback(unet, data_gen_val, [0, 1],
                                                            tensorboard_val_image_writer, max_output=4)
    sleep_callback = get_sleep_callback(120, 40)

    callback_list = [tensorboard_callback, validation_plot_callback,
                     model_checkpoint_callback, lr_scheduler_callback, sleep_callback]

    print(name)
    print("dataset seed = {}".format(seed))
    print("Training set contains {} samples with a batch size {},\n"
          "validation set contains {} samples with a batch size {}".format(len(data_gen_train.data_df),
                                                                           data_gen_train.batch_size,
                                                                           len(data_gen_val.data_df),
                                                                           data_gen_val.batch_size))
    print("train_use_weight_map = {}, val_use_weight_map = {}".format(train_use_weight_map, val_use_weight_map))
    print("Training starts (start_epoch = {}, end_epoch = {})".format(start_epoch, end_epoch))
    history = unet.fit(x=data_gen_train, epochs=end_epoch, initial_epoch=start_epoch,
                       validation_data=data_gen_val, shuffle=False,
                       validation_freq=1, callbacks=callback_list,
                       workers=1, use_multiprocessing=False)
    print("Training finished")

    log_df = pd.DataFrame(dict(epoch=history.epoch) | history.history)
    log_df.to_pickle(os.path.join(logdir, "log_" + name + ".pkl"))
    print("maximum val IoU = {:.4f}".format(log_df.loc[:, "val_binary_IoU"].max()))
    return log_df


def script_train_vanilla_unet_from_scratch(name=None, seed=None, train_val_split_ratio=0.8, notes=None):
    seed = seed
    batch_size_train = 1
    target_size = (512, 512)
    train_use_weight_map = False
    val_use_weight_map = False
    learning_rate = 1e-3

    image_dir = "E:/ED_MS/Semester_3/Dataset/DIC_Set/DIC_Set1_Annotated"
    image_type = "tif"
    mask_dir = "E:/ED_MS/Semester_3/Dataset/DIC_Set/DIC_Set1_Masks"
    mask_type = "tif"
    weight_map_dir = "E:/ED_MS/Semester_3/Dataset/DIC_Set/DIC_Set1_Weights"
    weight_map_type = "npy"
    dataset_name = "DIC"

    logdir = "E:/ED_MS/Semester_3/Codes/MyProject/tensorboard_logs"
    checkpoint_filepath = "E:/ED_MS/Semester_3/Codes/MyProject/checkpoints/vanilla_unet-trained_from_scratch.h5"
    start_epoch = 0
    end_epoch = 300

    if name is None:
        logdir = os.path.join(logdir, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    else:
        logdir = os.path.join(logdir, datetime.now().strftime("%Y-%m-%d") + "_{}".format(name))

    if notes is not None:
        with open(os.path.join(logdir, name + "_notes.txt"), "w+") as f:
            f.write(notes)

    # *: levels, learning_rate
    unet = get_compiled_unet(input_size=(*target_size, 1),
                             num_levels=5,
                             pretrained_weights=None,
                             learning_rate=learning_rate)

    data_augmentation_transform = DataAugmentation([HistogramVoodoo(),
                                                    ElasticDeform(sigma=20),
                                                    GaussianNoise(sigma=1 / 255),
                                                    RandomFlip(),
                                                    RandomRot90(),
                                                    RandomRotate(),
                                                    RandomZoomAndShift()])

    data_gen_train = DataGenerator(batch_size=batch_size_train, dataset_name=dataset_name, mode="train",
                                   use_weight_map=train_use_weight_map,
                                   image_dir=image_dir, image_type=image_type,
                                   mask_dir=mask_dir, mask_type=mask_type,
                                   weight_map_dir=weight_map_dir, weight_map_type=weight_map_type,
                                   target_size=target_size, data_aug_transform=data_augmentation_transform, seed=seed)
    data_gen_train, data_gen_val = train_val_split(data_gen_train, train_val_split_ratio, validation_batch_size=2,
                                                   use_weight_map_val=val_use_weight_map)

    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
                                                                save_weights_only=True,
                                                                verbosde=1,
                                                                monitor='val_binary_IoU',
                                                                mode='max',
                                                                save_best_only=True)
    lr_scheduler_callback = keras.callbacks.LearningRateScheduler(
        schedule=get_lr_scheduler((start_epoch + end_epoch) // 2))
    tensorboard_val_image_writer = tf.summary.create_file_writer(logdir + "/val_image")
    tensorboard_callback = callbacks.TensorBoard(log_dir=logdir)
    validation_plot_callback = get_validation_plot_callback(unet, data_gen_val, [0, 1],
                                                            tensorboard_val_image_writer, max_output=4)

    callback_list = [tensorboard_callback, validation_plot_callback,
                     model_checkpoint_callback, lr_scheduler_callback]

    print(name)
    print("dataset seed = {}".format(seed))
    print("Training set contains {} samples with a batch size {},\n"
          "validation set contains {} samples with a batch size {}".format(len(data_gen_train.data_df),
                                                                           data_gen_train.batch_size,
                                                                           len(data_gen_val.data_df),
                                                                           data_gen_val.batch_size))
    print("train_use_weight_map = {}, val_use_weight_map = {}".format(train_use_weight_map, val_use_weight_map))
    print("Training starts (start_epoch = {}, end_epoch = {})".format(start_epoch, end_epoch))
    history = unet.fit(x=data_gen_train, epochs=end_epoch, initial_epoch=start_epoch,
                       validation_data=data_gen_val, shuffle=False,
                       validation_freq=1, callbacks=callback_list,
                       workers=1, use_multiprocessing=False)
    print("Training finished")

    log_df = pd.DataFrame(dict(epoch=history.epoch) | history.history)
    log_df.to_pickle(os.path.join(logdir, "log_" + name + ".pkl"))
    print("maximum val IoU = {:.4f}".format(log_df.loc[:, "val_binary_IoU"].max()))
    return log_df


def script_train_lightweight_unet_from_scratch(name, fold_index, notes, seed=1):
    seed = seed
    batch_size = 1
    target_size = (512, 512)
    learning_rate = 1e-3
    regularizer_factor = 1e-7
    use_weight_map = False

    channel_multiplier = 3

    # *: dataset - DIC
    image_dir = "E:/ED_MS/Semester_3/Dataset/DIC_Set/DIC_Set1_Annotated"
    image_type = "tif"
    mask_dir = "E:/ED_MS/Semester_3/Dataset/DIC_Set/DIC_Set1_Masks"
    mask_type = "tif"
    weight_map_dir = "E:/ED_MS/Semester_3/Dataset/DIC_Set/DIC_Set1_Weights"
    weight_map_type = "npy"
    dataset_name = "DIC"

    logdir = "E:/ED_MS/Semester_3/Codes/MyProject/tensorboard_logs"
    checkpoint_filepath = "E:/ED_MS/Semester_3/Codes/MyProject/checkpoints/" \
                          "lightweight_unet-trained_from_scratch-fold_{}.h5".format(fold_index)
    start_epoch = 0
    end_epoch = 400

    logdir = os.path.join(logdir, datetime.now().strftime("%Y-%m-%d") + "_{}-fold_{}".format(name, fold_index))

    notes = append_info_to_notes(
        notes, fold_index=fold_index, seed=seed, batch_size=batch_size,
        target_size=target_size, use_weight_map=use_weight_map, learning_rate=learning_rate,
        regularizer_factor=regularizer_factor, checkpoint_filepath=checkpoint_filepath,
        start_epoch=start_epoch, end_epoch=end_epoch, channel_multiplier=channel_multiplier)

    lw_unet = get_compiled_lightweight_unet(input_size=(*target_size, 1),
                                            learning_rate=learning_rate,
                                            regularizer_factor=regularizer_factor,
                                            channel_multiplier=channel_multiplier)

    data_augmentation_transform = DataAugmentation([HistogramVoodoo(),
                                                    ElasticDeform(sigma=20),
                                                    GaussianNoise(sigma=1 / 255),
                                                    RandomFlip(),
                                                    RandomRot90(),
                                                    RandomZoomAndShift()])

    train_set, val_set, test_set = train_val_test_split(
        batch_size=batch_size, dataset_name=dataset_name, use_weight_map=use_weight_map,
        image_dir=image_dir, image_type=image_type, mask_dir=mask_dir, mask_type=mask_type,
        weight_map_dir=weight_map_dir, weight_map_type=weight_map_type, target_size=target_size,
        data_aug_transform=data_augmentation_transform, seed=seed, num_folds=5, fold_index=fold_index)

    # *: load checkpoint
    model_checkpoint_callback = CustomModelCheckpointCallBack(ignore=0, filepath=checkpoint_filepath,
                                                              monitor="val_binary_IoU", mode="max",
                                                              checkpoint_log_dir=logdir)
    # *:lr_scheduler
    lr_scheduler_callback = keras.callbacks.LearningRateScheduler(get_lr_scheduler((start_epoch + end_epoch) // 2))
    tensorboard_callback = callbacks.TensorBoard(log_dir=logdir)
    tensorboard_val_image_writer = tf.summary.create_file_writer(logdir + "/val_image")
    validation_plot_callback = get_validation_plot_callback(lw_unet, val_set, [0, 1, 2, 3],
                                                            tensorboard_val_image_writer, max_output=4)
    sleep_callback = get_sleep_callback(120, 40)
    # *: callbacks
    callback_list = [tensorboard_callback, validation_plot_callback,
                     model_checkpoint_callback, lr_scheduler_callback, sleep_callback]

    print(name)
    print("fold = {}, dataset seed = {}".format(fold_index, seed))
    print("batch_size = {}\n"
          "Training set contains {} batches,\n"
          "validation set contains {} batches".format(batch_size, len(train_set), len(val_set)))
    print("use_weight_map = {}".format(use_weight_map))
    print("Training starts (start_epoch = {}, end_epoch = {})".format(start_epoch, end_epoch))
    history = lw_unet.fit(x=train_set, epochs=end_epoch, initial_epoch=start_epoch,
                          validation_data=val_set, shuffle=False,
                          validation_freq=1, callbacks=callback_list)
    print("Training finished")

    with open(os.path.join(logdir, name + "_notes.txt"), "w+") as f:
        f.write(notes)

    log_df = pd.DataFrame(dict(epoch=history.epoch) | history.history)
    log_df.to_pickle(os.path.join(logdir, "log_{}-fold_{}.pkl".format(name, fold_index)))
    print("maximum val IoU = {:.4f}".format(log_df.loc[:, "val_binary_IoU"].max()))
    return log_df


if __name__ == '__main__':
    # log_df_unet = script_fine_tune_vanilla_unet(name="vanilla_unet-fine-tune-weighted_loss-2", seed=1,
    #                                             train_val_split_ratio=0.8)

    # log_df_unet_trained_from_scratch = \
    #     script_train_vanilla_unet_from_scratch(name="vanilla_unet-trained_from_scratch",
    #                                            seed=1, train_val_split_ratio=0.8)

    note_lw_unet = "train lightweight unet from scratch, regularizer added"
    log_df_lw_unet_trained_from_scratch = \
        script_train_lightweight_unet_from_scratch(name="train_lw_unet_from_scratch", seed=1,
                                                   fold_index=0, notes=note_lw_unet)
