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
from training_utils import (get_validation_plot_callback, train_val_test_split, get_lr_scheduler, append_info_to_notes,
                            get_sleep_callback, CustomModelCheckpointCallBack)


def script_func_get_train_val_test_dataset(target_size=(512, 512), batch_size=1, use_weight_map=False, seed=1,
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


def script_func_get_callback_list(checkpoint_filepath, start_epoch, end_epoch, logdir, model, val_set):
    callback_list = []
    # model_checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
    #                                                             save_weights_only=True,
    #                                                             verbose=1,
    #                                                             monitor='val_binary_IoU',
    #                                                             mode='max',
    #                                                             save_best_only=True)
    model_checkpoint_callback = CustomModelCheckpointCallBack(
        ignore=40, filepath=checkpoint_filepath, monitor="val_loss", mode="min",
        checkpoint_log_dir=os.path.join(logdir, "checkpoint_info.txt"))
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


def script_func_print_training_info(name, seed, train_set, val_set, batch_size, use_weight_map, start_epoch,
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


def script_fine_tune_vanilla_unet_with_l2_regularizer(name, fold_index=0, seed=1, notes=None):
    seed = seed
    batch_size = 1
    target_size = (512, 512)
    use_weight_map = False
    pretrained_weight_path = "E:/ED_MS/Semester_3/Codes/MyProject/checkpoints/trained_weights/" \
                             "unet_agarpads_seg_evaluation2.hdf5"
    learning_rate = 1e-4
    regularizer_factor = 1e-7

    logdir = "E:/ED_MS/Semester_3/Codes/MyProject/tensorboard_logs"
    checkpoint_filepath = "E:/ED_MS/Semester_3/Codes/MyProject/checkpoints/" \
                          "vanilla_unet-fine_tuned-fold_{}.h5".format(fold_index)
    start_epoch = 0
    end_epoch = 400

    if name is None:
        logdir = os.path.join(logdir, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    else:
        logdir = os.path.join(logdir, datetime.now().strftime("%Y-%m-%d") + "_{}-fold_{}".format(name, fold_index))

    if notes is not None:
        notes = "automatic generated"
    notes = append_info_to_notes(
        notes, fold_index=fold_index, seed=seed, batch_size=batch_size, target_size=target_size,
        use_weight_map=use_weight_map, pretrained_weight_path=pretrained_weight_path, learning_rate=learning_rate,
        regularizer_factor=regularizer_factor, checkpoint_filepath=checkpoint_filepath, start_epoch=start_epoch,
        end_epoch=end_epoch)

    unet = get_compiled_unet(input_size=(*target_size, 1),
                             num_levels=5,
                             pretrained_weights=pretrained_weight_path,
                             learning_rate=learning_rate,
                             regularizer_factor=regularizer_factor)

    train_set, val_set, test_set = script_func_get_train_val_test_dataset(
        target_size=target_size, batch_size=batch_size, use_weight_map=use_weight_map, seed=seed, fold_index=fold_index)

    callback_list = script_func_get_callback_list(
        checkpoint_filepath=checkpoint_filepath, start_epoch=start_epoch, end_epoch=end_epoch, logdir=logdir,
        model=unet, val_set=val_set)

    script_func_print_training_info(
        name=name, seed=seed, train_set=train_set, val_set=val_set, batch_size=batch_size,
        use_weight_map=use_weight_map, start_epoch=start_epoch, end_epoch=end_epoch)

    history = unet.fit(x=train_set, epochs=end_epoch, initial_epoch=start_epoch,
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
    note_1 = \
        "fine tune a vanilla unet with the weights from Delta 2.0\n"
    df_1 = script_fine_tune_vanilla_unet_with_l2_regularizer(name="fine_tune_vanilla_unet_with_l2", fold_index=2,
                                                             notes=note_1)
