import os
from datetime import datetime

import keras.callbacks
import tensorflow as tf
import keras
from keras import callbacks
import pandas as pd

from data import DataGenerator
from model import (get_compiled_unet, get_compiled_lightweight_unet,
                   get_compiled_binary_unet, get_compiled_binary_lightweight_unet,
                   get_teacher_vanilla_unet,
                   get_student_lightweight_unet)
from data import DataGenerator

from data_augmentation import (HistogramVoodoo, ElasticDeform, GaussianNoise, RandomFlip, RandomRotate,
                               RandomZoomAndShift, DataAugmentation, RandomRot90)
from training_utils import (get_validation_plot_callback, train_val_test_split, get_lr_scheduler, append_info_to_notes,
                            get_sleep_callback, CustomModelCheckpointCallBack)
from knowledge_distillation import KnowledgeDistillation, distill_knowledge
from residual_binarization import transfer_lightweight_unet_weights_to_binary_lightweight_unet


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


def script_func_get_callback_list(checkpoint_filepath, monitor, mode, start_epoch, end_epoch, logdir, model, val_set):
    callback_list = []
    # model_checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
    #                                                             save_weights_only=True,
    #                                                             verbose=1,
    #                                                             monitor='val_binary_IoU',
    #                                                             mode='max',
    #                                                             save_best_only=True)
    model_checkpoint_callback = CustomModelCheckpointCallBack(
        ignore=40, filepath=checkpoint_filepath, monitor=monitor, mode=mode,
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


def script_fine_tune_vanilla_unet_with_l2_regularizer(name, fold_index, notes, seed=1):
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

    logdir = os.path.join(logdir, datetime.now().strftime("%Y-%m-%d") + "_{}-fold_{}".format(name, fold_index))

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
        checkpoint_filepath=checkpoint_filepath, monitor="val_loss", mode="min", start_epoch=start_epoch,
        end_epoch=end_epoch, logdir=logdir, model=unet, val_set=val_set)

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


def train_lightweight_unet_via_knowledge_distillation(name, fold_index, notes, seed=1):
    seed = seed
    batch_size = 1
    target_size = (512, 512)
    use_weight_map = False

    fine_tuned_unet_weight = "E:/ED_MS/Semester_3/Codes/MyProject/tensorboard_logs/" \
                             "2022-08-06_fine_tune_vanilla_unet_with_l2-fold_0/" \
                             "vanilla_unet-fine-tuned_IoU=0.8880.h5"
    # todo: knowledge distillation - maybe lambda code = 0.1?
    lambda_code = 0
    learning_rate = 1e-3

    logdir = "E:/ED_MS/Semester_3/Codes/MyProject/tensorboard_logs"
    checkpoint_filepath = "E:/ED_MS/Semester_3/Codes/MyProject/checkpoints/" \
                          "knowledge_distillation-lw_unet-fold_{}.h5".format(fold_index)
    start_epoch = 0
    end_epoch = 400

    logdir = os.path.join(logdir, datetime.now().strftime("%Y-%m-%d") + "_{}-fold_{}".format(name, fold_index))
    if not os.path.exists(logdir):
        os.mkdir(logdir)

    teacher = get_teacher_vanilla_unet(input_size=(*target_size, 1), trained_weights=fine_tuned_unet_weight)
    student = get_student_lightweight_unet(input_size=(*target_size, 1))
    # todo: knowledge distillation - load weights
    path = "E:/ED_MS/Semester_3/Codes/MyProject/checkpoints/trained_weights/" \
           "lightweight_unet-trained_from_scratch_IoU=0.838.h5"
    student.load_weights(filepath=path, by_name=True)

    notes = append_info_to_notes(
        notes, fold_index=fold_index, seed=seed, batch_size=batch_size, target_size=target_size,
        use_weight_map=use_weight_map, teacher_weight_path=fine_tuned_unet_weight, learning_rate=learning_rate,
        checkpoint_filepath=checkpoint_filepath, start_epoch=start_epoch, end_epoch=end_epoch)

    knowledge_distillation = KnowledgeDistillation(name="knowledge_distillation", teacher=teacher, student=student)
    knowledge_distillation.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                                   code_loss_fn=tf.keras.losses.MeanAbsoluteError(name="code_loss"),
                                   pred_loss_fn=tf.keras.losses.MeanAbsoluteError(name="pred_loss"),
                                   lambda_code=lambda_code
                                   )

    train_set, val_set, test_set = script_func_get_train_val_test_dataset(
        target_size=target_size, batch_size=batch_size, use_weight_map=use_weight_map, seed=seed, fold_index=fold_index)

    script_func_print_training_info(
        name=name, seed=seed, train_set=train_set, val_set=val_set, batch_size=batch_size,
        use_weight_map=use_weight_map, start_epoch=start_epoch, end_epoch=end_epoch)

    log_df = distill_knowledge(knowledge_distillation=knowledge_distillation, start_epoch=start_epoch,
                               end_epoch=end_epoch,
                               train_set=train_set, val_set=val_set, checkpoint_path=checkpoint_filepath)

    print("Training finished")

    with open(os.path.join(logdir, name + "_notes.txt"), "w+") as f:
        f.write(notes)

    return log_df


def script_binarize_lightweight_unet(name, fold_index, notes, seed=1):
    seed = seed
    batch_size = 1
    target_size = (512, 512)
    use_weight_map = False
    lightweight_unet_weight_path = None  # *: filled up

    num_activation_residual_levels = 3
    num_depthwise_conv_residual_levels = 3
    num_pointwise_conv_residual_levels = 3
    num_conv_residual_levels = 3

    learning_rate = 1e-4

    logdir = "E:/ED_MS/Semester_3/Codes/MyProject/tensorboard_logs"
    logdir = os.path.join(logdir, datetime.now().strftime("%Y-%m-%d") + "_{}-fold_{}".format(name, fold_index))
    checkpoint_filepath = "E:/ED_MS/Semester_3/Codes/MyProject/checkpoints/" \
                          "blu_unet-fold_{}.h5".format(fold_index)
    start_epoch = 0
    end_epoch = 400

    notes = append_info_to_notes(
        notes, fold_index=fold_index, seed=seed, batch_size=batch_size, target_size=target_size,
        use_weight_map=use_weight_map, lightweight_unet_weight_path=lightweight_unet_weight_path,
        learning_rate=learning_rate, checkpoint_filepath=checkpoint_filepath, start_epoch=start_epoch,
        end_epoch=end_epoch, num_activation_residual_levels=num_activation_residual_levels,
        num_conv_residual_levels=num_conv_residual_levels,
        num_depthwise_conv_residual_levels=num_depthwise_conv_residual_levels,
        num_pointwise_conv_residual_levels=num_pointwise_conv_residual_levels)

    lw_unet = get_compiled_lightweight_unet(
        input_size=(*target_size, 1),
        pretrained_weight=lightweight_unet_weight_path)

    blu_net = get_compiled_binary_lightweight_unet(
        input_size=(*target_size, 1),
        num_activation_residual_levels=num_activation_residual_levels,
        num_depthwise_conv_residual_levels=num_depthwise_conv_residual_levels,
        num_pointwise_conv_residual_levels=num_pointwise_conv_residual_levels,
        num_conv_residual_levels=num_conv_residual_levels,
        learning_rate=learning_rate)

    blu_net = transfer_lightweight_unet_weights_to_binary_lightweight_unet(
        lightweight_unet=lw_unet,
        binary_lightweight_unet=blu_net)

    train_set, val_set, test_set = script_func_get_train_val_test_dataset(
        target_size=target_size, batch_size=batch_size, use_weight_map=use_weight_map, seed=seed, fold_index=fold_index)

    callback_list = script_func_get_callback_list(
        checkpoint_filepath=checkpoint_filepath, monitor="val_binary_IoU", mode="max",
        start_epoch=start_epoch, end_epoch=end_epoch, logdir=logdir, model=blu_net, val_set=val_set)

    script_func_print_training_info(
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


if __name__ == '__main__':
    # *: fine tune a vanilla unet with regularizer
    # note_1 = \
    #     "fine tune a vanilla unet with the weights from Delta 2.0\n"
    # df_4 = script_fine_tune_vanilla_unet_with_l2_regularizer(name="fine_tune_vanilla_unet_with_l2", fold_index=4,
    #                                                          notes=note_1)

    # knowledge distillation
    log_df_kd = train_lightweight_unet_via_knowledge_distillation(name="knowledge_distillation", fold_index=0)
    print("done")
