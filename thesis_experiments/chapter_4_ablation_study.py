import os
from datetime import datetime

import keras
import keras.callbacks
import pandas as pd
from keras import callbacks

from data_augmentation import (HistogramVoodoo, ElasticDeform, GaussianNoise, RandomFlip, DataAugmentation, RandomRot90)
from model import (get_compiled_lightweight_unet,
                   get_compiled_binary_lightweight_unet)
from thesis_experiments.chapter_4_accuracy import average_5_folds, _get_test_set
from training_utils import (train_val_test_split, get_lr_scheduler, append_info_to_notes,
                            get_sleepy_callback, CustomModelCheckpointCallBack, CustomLRTrackerCallback,
                            evaluate_on_test_set)


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

    model_checkpoint_callback = CustomModelCheckpointCallBack(
        ignore=40, filepath=checkpoint_filepath, monitor=monitor, mode=mode,
        logdir=logdir)
    callback_list.append(model_checkpoint_callback)

    lr_scheduler_callback = keras.callbacks.LearningRateScheduler(
        get_lr_scheduler((start_epoch + end_epoch) // 2))
    callback_list.append(lr_scheduler_callback)

    tensorboard_callback = callbacks.TensorBoard(log_dir=logdir)
    callback_list.append(tensorboard_callback)

    # tensorboard_val_image_writer = tf.summary.create_file_writer(logdir + "/val_image")
    # validation_plot_callback = get_validation_plot_callback(model, val_set, [0, 1, 2, 3],
    #                                                         tensorboard_val_image_writer, max_output=4)
    # callback_list.append(validation_plot_callback)

    callback_list.append(CustomLRTrackerCallback(logdir))

    callback_list.append(get_sleepy_callback(180, 40))

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


def script_ab_of_knowledge_distillation(name, fold_index, notes, seed=1):
    # *: train a lightweight unet from scratch, rather than train based on the knowledge distillation
    seed = seed
    batch_size = 1
    target_size = (512, 512)
    use_weight_map = False

    learning_rate = 1e-3
    regularizer_factor = 1e-7
    channel_multiplier = 1

    start_epoch = 0
    end_epoch = 400

    logdir = "E:/ED_MS/Semester_3/Codes/MyProject/tensorboard_logs/ablation_study"
    logdir = os.path.join(logdir, datetime.now().strftime("%Y-%m-%d") + "_{}-fold_{}".format(name, fold_index))

    checkpoint_filepath = os.path.join(logdir, "lw_unet-trained_from_scratch-fold_{}.h5".format(fold_index))

    notes = append_info_to_notes(
        notes, fold_index=fold_index, seed=seed, batch_size=batch_size,
        target_size=target_size, use_weight_map=use_weight_map, learning_rate=learning_rate,
        regularizer_factor=regularizer_factor, checkpoint_filepath=checkpoint_filepath,
        start_epoch=start_epoch, end_epoch=end_epoch, channel_multiplier=channel_multiplier)

    train_set, val_set, test_set = _func_get_train_val_test_dataset(
        target_size=target_size, batch_size=batch_size, use_weight_map=use_weight_map, seed=seed, fold_index=fold_index)

    lw_unet = get_compiled_lightweight_unet(input_size=(*target_size, 1),
                                            learning_rate=learning_rate,
                                            regularizer_factor=regularizer_factor,
                                            channel_multiplier=channel_multiplier)

    callback_list = _func_get_callback_list(
        checkpoint_filepath=checkpoint_filepath, monitor="val_loss", mode="min", start_epoch=start_epoch,
        end_epoch=end_epoch, logdir=logdir, model=lw_unet, val_set=val_set)

    _func_print_training_info(
        name=name, seed=seed, train_set=train_set, val_set=val_set, batch_size=batch_size,
        use_weight_map=use_weight_map, start_epoch=start_epoch, end_epoch=end_epoch)

    history = lw_unet.fit(x=train_set, epochs=end_epoch, initial_epoch=start_epoch,
                          validation_data=val_set, shuffle=False,
                          validation_freq=1, callbacks=callback_list)

    print("Training finished")

    with open(os.path.join(logdir, name + "_notes.txt"), "w+") as f:
        f.write(notes)

    log_df = pd.DataFrame(dict(epoch=history.epoch) | history.history)
    log_df.to_pickle(os.path.join(logdir, "log_{}-fold_{}.pkl".format(name, fold_index)))
    return log_df


def script_train_blu_net_from_scratch(name, fold_index, notes, seed=1):
    # *: train a blu-net from scratch. prove that my framework that gradually increases the model sparsity is necessary
    seed = seed
    batch_size = 1
    target_size = (256, 256)
    use_weight_map = False

    num_activation_residual_levels = 3
    num_depthwise_conv_residual_levels = 3
    num_pointwise_conv_residual_levels = 3
    num_conv_residual_levels = 3

    learning_rate = 1e-4

    start_epoch = 0
    end_epoch = 400

    logdir = "E:/ED_MS/Semester_3/Codes/MyProject/tensorboard_logs/ablation_study"
    logdir = os.path.join(logdir, datetime.now().strftime("%Y-%m-%d") + "_{}-fold_{}".format(name, fold_index))

    checkpoint_filepath = os.path.join(logdir, "blu_net-trained_from_scratch-fold_{}.h5".format(fold_index))

    notes = append_info_to_notes(
        notes, fold_index=fold_index, seed=seed, batch_size=batch_size, target_size=target_size,
        use_weight_map=use_weight_map,
        learning_rate=learning_rate, checkpoint_filepath=checkpoint_filepath, start_epoch=start_epoch,
        end_epoch=end_epoch, num_activation_residual_levels=num_activation_residual_levels,
        num_conv_residual_levels=num_conv_residual_levels,
        num_depthwise_conv_residual_levels=num_depthwise_conv_residual_levels,
        num_pointwise_conv_residual_levels=num_pointwise_conv_residual_levels)

    train_set, val_set, test_set = _func_get_train_val_test_dataset(
        target_size=target_size, batch_size=batch_size, use_weight_map=use_weight_map, seed=seed, fold_index=fold_index)

    blu_net = get_compiled_binary_lightweight_unet(
        (*target_size, 1),
        num_activation_residual_levels=num_activation_residual_levels,
        num_depthwise_conv_residual_levels=num_depthwise_conv_residual_levels,
        num_pointwise_conv_residual_levels=num_pointwise_conv_residual_levels,
        num_conv_residual_levels=num_conv_residual_levels,
        learning_rate=learning_rate
    )

    callback_list = _func_get_callback_list(
        checkpoint_filepath=checkpoint_filepath, monitor="val_binary_IoU", mode="max",
        start_epoch=start_epoch, end_epoch=end_epoch, logdir=logdir, model=blu_net, val_set=val_set)

    _func_print_training_info(
        name=name, seed=seed, train_set=train_set, val_set=val_set, batch_size=batch_size,
        use_weight_map=use_weight_map,
        start_epoch=start_epoch, end_epoch=end_epoch)

    history = blu_net.fit(x=train_set, epochs=end_epoch, initial_epoch=start_epoch,
                          validation_data=val_set, shuffle=False, validation_freq=1,
                          callbacks=callback_list)

    print("Training finished")

    with open(os.path.join(logdir, name + "_notes.txt"), "w+") as f:
        f.write(notes)

    log_df = pd.DataFrame(dict(epoch=history.epoch) | history.history)
    log_df.to_pickle(os.path.join(logdir, "log_{}-fold_{}.pkl".format(name, fold_index)))
    print("maximum val IoU = {:.4f}".format(log_df.loc[:, "val_binary_IoU"].max()))
    return log_df


def script_evaluate_lw_unet_trained_from_scratch(fold_index):
    match fold_index:
        case 0:
            weight_path = "E:/ED_MS/Semester_3/Codes/MyProject/tensorboard_logs/ablation_study/" \
                          "2022-08-12_AB_lw_unet_trained_from_scratch-fold_0/" \
                          "lw_unet-trained_from_scratch-fold_0.h5"
        case 1:
            weight_path = "E:/ED_MS/Semester_3/Codes/MyProject/tensorboard_logs/ablation_study/" \
                          "2022-08-12_AB_lw_unet_trained_from_scratch-fold_1/" \
                          "lw_unet-trained_from_scratch-fold_1.h5"
        case 2:
            weight_path = "E:/ED_MS/Semester_3/Codes/MyProject/tensorboard_logs/ablation_study/" \
                          "2022-08-12_AB_lw_unet_trained_from_scratch-fold_2/" \
                          "lw_unet-trained_from_scratch-fold_2.h5"
        case 3:
            weight_path = "E:/ED_MS/Semester_3/Codes/MyProject/tensorboard_logs/ablation_study/" \
                          "2022-08-13_AB_lw_unet_trained_from_scratch-fold_3/" \
                          "lw_unet-trained_from_scratch-fold_3.h5"
        case 4:
            weight_path = "E:/ED_MS/Semester_3/Codes/MyProject/tensorboard_logs/ablation_study/" \
                          "2022-08-13_AB_lw_unet_trained_from_scratch-fold_4/" \
                          "lw_unet-trained_from_scratch-fold_4.h5"
        case _:
            raise RuntimeError("unavailable fold index")

    lw_unet = get_compiled_lightweight_unet((512, 512, 1), pretrained_weight=weight_path)
    test_set = _get_test_set()

    metric_dict = evaluate_on_test_set(lw_unet, test_set)
    return metric_dict


if __name__ == '__main__':
    # *: ablation study 1: knowledge distillation vs training from scratch
    # notes = "ablation study - necessity of knowledge distillation\n" \
    #         "this experiment trains a lw_unet from scratch"
    # log_knowledge_distillation = script_ab_of_knowledge_distillation(
    #     name="AB_lw_unet_trained_from_scratch", fold_index=4, notes=notes)

    # lw_unet-trained_from_scratch, binary_iou = 0.8397658467292786, binary_f1score = 0.9123261570930481
    # average_5_folds("lw_unet-trained_from_scratch", script_evaluate_lw_unet_trained_from_scratch)

    notes = "ablation study - train blu-net from scratch"
    script_train_blu_net_from_scratch(name="blu_net-train-from-scratch", fold_index=2, notes=notes)

    pass
