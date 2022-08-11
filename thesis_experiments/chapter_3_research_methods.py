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
                            get_sleep_callback, CustomModelCheckpointCallBack, CustomLRScheduler,
                            CustomLRTrackerCallback)


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


def _func_get_callback_list(checkpoint_filepath, monitor, mode, ignore, start_epoch, end_epoch, logdir, model, val_set):
    callback_list = []

    model_checkpoint_callback = CustomModelCheckpointCallBack(
        ignore=ignore, filepath=checkpoint_filepath, monitor=monitor, mode=mode,
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

    callback_list.append(CustomLRTrackerCallback(logdir))

    # *: sleep period
    callback_list.append(get_sleep_callback(180, 40))

    return callback_list


def _func_print_training_info(name, seed, train_set, val_set, batch_size, use_weight_map, start_epoch,
                              end_epoch, fold_index):
    print(name)
    print("fold_index = {}".format(fold_index))
    print("dataset seed = {}".format(seed))
    print("Training set contains {} samples\n"
          "validation set contains {} samples\n"
          "batch_size = {}".format(len(train_set.data_df),
                                   len(val_set.data_df),
                                   batch_size))

    print("train_use_weight = {}".format(use_weight_map, ))
    print("Training starts (start_epoch = {}, end_epoch = {})".format(start_epoch, end_epoch))


def script_fine_tune_vanilla_unet_with_l2_regularizer(name, fold_index, notes, seed=1):
    # * : 🚩 support 5 folds
    seed = seed
    batch_size = 1
    target_size = (512, 512)
    use_weight_map = False
    pretrained_weight_path = "E:/ED_MS/Semester_3/Codes/MyProject/checkpoints/trained_weights/" \
                             "unet_agarpads_seg_evaluation2.hdf5"
    learning_rate = 1e-4
    regularizer_factor = 1e-7

    start_epoch = 0
    end_epoch = 400

    logdir = "E:/ED_MS/Semester_3/Codes/MyProject/tensorboard_logs"
    logdir = os.path.join(logdir, datetime.now().strftime("%Y-%m-%d") + "_{}-fold_{}".format(name, fold_index))

    checkpoint_filepath = os.path.join(logdir, "vanilla_unet-fine_tuned-fold_{}.h5".format(fold_index))

    notes = append_info_to_notes(
        notes, fold_index=fold_index, seed=seed, batch_size=batch_size, target_size=target_size,
        use_weight_map=use_weight_map, pretrained_weight_path=pretrained_weight_path, learning_rate=learning_rate,
        regularizer_factor=regularizer_factor, checkpoint_filepath=checkpoint_filepath, start_epoch=start_epoch,
        end_epoch=end_epoch)

    train_set, val_set, test_set = _func_get_train_val_test_dataset(
        target_size=target_size, batch_size=batch_size, use_weight_map=use_weight_map, seed=seed, fold_index=fold_index)

    unet = get_compiled_unet(input_size=(*target_size, 1),
                             num_levels=5,
                             pretrained_weights=pretrained_weight_path,
                             learning_rate=learning_rate,
                             regularizer_factor=regularizer_factor)

    callback_list = _func_get_callback_list(
        checkpoint_filepath=checkpoint_filepath, monitor="val_loss", mode="min", ignore=0, start_epoch=start_epoch,
        end_epoch=end_epoch, logdir=logdir, model=unet, val_set=val_set)

    _func_print_training_info(
        name=name, seed=seed, train_set=train_set, val_set=val_set, batch_size=batch_size,
        use_weight_map=use_weight_map, start_epoch=start_epoch, end_epoch=end_epoch, fold_index=fold_index)

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


def script_train_lightweight_unet_via_knowledge_distillation(name, fold_index, notes, seed=1):
    # !： 🚩 not support 5 folds
    seed = seed
    batch_size = 1
    target_size = (256, 256)
    use_weight_map = False

    if fold_index == 0:
        fine_tuned_unet_weight_path = "E:/ED_MS/Semester_3/Codes/MyProject/tensorboard_logs/" \
                                      "2022-08-10_fine_tune_vanilla_unet_with_l2-fold_0/" \
                                      "vanilla_unet-fine_tuned-fold_0.h5"
        if "decoder" in name:
            encoder_weight_path = "E:/ED_MS/Semester_3/Codes/MyProject/tensorboard_logs/" \
                                  "2022-08-11_knowledge_distillation-lw_unet-encoder-fold_0/" \
                                  "knowledge_distillation-lw_unet-fold_0-best.h5"
        else:
            encoder_weight_path = None
    elif fold_index == 1:
        fine_tuned_unet_weight_path = "E:/ED_MS/Semester_3/Codes/MyProject/tensorboard_logs/" \
                                      "2022-08-10_fine_tune_vanilla_unet_with_l2-fold_1/" \
                                      "vanilla_unet-fine_tuned-fold_1.h5"
        if "decoder" in name:
            encoder_weight_path = "E:/ED_MS/Semester_3/Codes/MyProject/tensorboard_logs/" \
                                  "2022-08-11_knowledge_distillation-lw_unet-encoder-fold_1/" \
                                  "knowledge_distillation-lw_unet-encoder-fold_1-best.h5"
        else:
            encoder_weight_path = None

    elif fold_index == 2:
        fine_tuned_unet_weight_path = "E:/ED_MS/Semester_3/Codes/MyProject/tensorboard_logs/" \
                                      "2022-08-10_fine_tune_vanilla_unet_with_l2-fold_2/" \
                                      "vanilla_unet-fine_tuned-fold_2.h5"
    elif fold_index == 3:
        fine_tuned_unet_weight_path = "E:/ED_MS/Semester_3/Codes/MyProject/tensorboard_logs/" \
                                      "2022-08-10_fine_tune_vanilla_unet_with_l2-fold_3/" \
                                      "vanilla_unet-fine_tuned-fold_3.h5"
    elif fold_index == 4:
        fine_tuned_unet_weight_path = "E:/ED_MS/Semester_3/Codes/MyProject/tensorboard_logs/" \
                                      "2022-08-10_fine_tune_vanilla_unet_with_l2-fold_4/" \
                                      "vanilla_unet-fine_tuned-fold_4.h5"
    else:
        raise RuntimeError("unsupported fold index")
    # *: hyper-parameter
    learning_rate = 1e-3
    if "encoder" in name:
        # encoder stack
        features_to_extract = (1, 2, 4, 5, 7, 8, 10, 11, 13, 14)
        calculate_iou = False
    else:
        features_to_extract = (16, 18, 19, 21, 23, 24, 26, 28, 29, 31, 33, 34, 35)
        calculate_iou = False

    logdir = "E:/ED_MS/Semester_3/Codes/MyProject/tensorboard_logs"
    logdir = os.path.join(logdir, datetime.now().strftime("%Y-%m-%d") + "_{}-fold_{}".format(name, fold_index))
    checkpoint_filepath = os.path.join(logdir, "{}-fold_{}.h5".format(name, fold_index))

    start_epoch = 0
    end_epoch = 400

    teacher = get_teacher_vanilla_unet(input_size=(*target_size, 1), trained_weights=fine_tuned_unet_weight_path,
                                       features_to_extract=features_to_extract)
    student = get_student_lightweight_unet(input_size=(*target_size, 1),
                                           features_to_extract=features_to_extract,
                                           trained_weights=encoder_weight_path)

    notes = append_info_to_notes(
        notes, fold_index=fold_index, seed=seed, batch_size=batch_size, target_size=target_size,
        use_weight_map=use_weight_map, teacher_weight_path=fine_tuned_unet_weight_path,
        learning_rate=learning_rate, calculate_iou=calculate_iou,
        checkpoint_filepath=checkpoint_filepath, start_epoch=start_epoch, end_epoch=end_epoch)

    train_set, val_set, test_set = _func_get_train_val_test_dataset(
        target_size=target_size, batch_size=batch_size, use_weight_map=use_weight_map, seed=seed, fold_index=fold_index)

    knowledge_distillation = KnowledgeDistillation(name="knowledge_distillation",
                                                   teacher=teacher,
                                                   student=student,
                                                   calculate_iou=calculate_iou)
    knowledge_distillation.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=CustomLRScheduler(
            initial_learning_rate=learning_rate,
            steps_per_epoch=len(train_set) // batch_size,
            epoch_start_to_decay=(end_epoch - start_epoch) // 2)),
        loss_fn=tf.keras.losses.MeanSquaredError(name="loss")
    )

    _func_print_training_info(
        name=name, seed=seed, train_set=train_set, val_set=val_set, batch_size=batch_size,
        use_weight_map=use_weight_map, start_epoch=start_epoch, end_epoch=end_epoch, fold_index=fold_index)

    log_df = distill_knowledge(knowledge_distillation=knowledge_distillation, start_epoch=start_epoch,
                               end_epoch=end_epoch,
                               train_set=train_set, val_set=val_set, checkpoint_path=checkpoint_filepath,
                               logdir=logdir)

    print("Training finished")

    with open(os.path.join(logdir, name + "_notes.txt"), "w+") as f:
        f.write(notes)

    return log_df


def script_train_lightweight_unet_after_knowledge_distillation(name, fold_index, notes, seed=1):
    # !: 🚩 not support 5 folds
    seed = seed
    seed = seed
    batch_size = 1
    target_size = (512, 512)
    learning_rate = 1e-3
    use_weight_map = False
    channel_multiplier = 1

    if fold_index == 0:
        pretrained_weight_path = "E:/ED_MS/Semester_3/Codes/MyProject/tensorboard_logs/" \
                                 "2022-08-11_knowledge_distillation-lw_unet-decoder-fold_0/" \
                                 "knowledge_distillation-lw_unet-decoder-fold_0-best.h5"
    elif fold_index == 1:
        pretrained_weight_path = "E:/ED_MS/Semester_3/Codes/MyProject/tensorboard_logs/" \
                                 "2022-08-09_knowledge_distillation-fold_1/" \
                                 "knowledge_distillation-lw_unet-fold_1.h5"
    elif fold_index == 2:
        pretrained_weight_path = "E:/ED_MS/Semester_3/Codes/MyProject/tensorboard_logs/" \
                                 "2022-08-09_knowledge_distillation-fold_2/" \
                                 "knowledge_distillation-lw_unet-fold_2.h5"
    elif fold_index == 3:
        pretrained_weight_path = "E:/ED_MS/Semester_3/Codes/MyProject/tensorboard_logs/" \
                                 "2022-08-09_knowledge_distillation-fold_3/" \
                                 "knowledge_distillation-lw_unet-fold_3.h5"
    elif fold_index == 4:
        pretrained_weight_path = "E:/ED_MS/Semester_3/Codes/MyProject/tensorboard_logs/" \
                                 "2022-08-09_knowledge_distillation-fold_4/" \
                                 "knowledge_distillation-lw_unet-fold_4.h5"
    else:
        raise RuntimeError("unsupported fold index")

    start_epoch = 0
    end_epoch = 400

    logdir = "E:/ED_MS/Semester_3/Codes/MyProject/tensorboard_logs"
    logdir = os.path.join(logdir, datetime.now().strftime("%Y-%m-%d") + "_{}-fold_{}".format(name, fold_index))

    checkpoint_filepath = os.path.join(logdir,
                                       "lw_unet_retrained_after_knowledge_distillation-fold_{}.h5".format(fold_index))
    notes = append_info_to_notes(
        notes, fold_index=fold_index, seed=seed, batch_size=batch_size, target_size=target_size,
        use_weight_map=use_weight_map, pretrained_weight_path=pretrained_weight_path, learning_rate=learning_rate,
        checkpoint_filepath=checkpoint_filepath, start_epoch=start_epoch,
        end_epoch=end_epoch)

    train_set, val_set, test_set = _func_get_train_val_test_dataset(
        target_size=target_size, batch_size=batch_size, use_weight_map=use_weight_map, seed=seed, fold_index=fold_index)

    lw_unet = get_compiled_lightweight_unet(input_size=(*target_size, 1),
                                            learning_rate=learning_rate)
    lw_unet.load_weights(pretrained_weight_path, by_name=True)

    callback_list = _func_get_callback_list(
        checkpoint_filepath=checkpoint_filepath, monitor="val_binary_IoU", mode="max", ignore=0,
        start_epoch=start_epoch, end_epoch=end_epoch, logdir=logdir, model=lw_unet, val_set=val_set)

    _func_print_training_info(
        name=name, seed=seed, train_set=train_set, val_set=val_set, batch_size=batch_size,
        use_weight_map=use_weight_map, start_epoch=start_epoch, end_epoch=end_epoch, fold_index=fold_index)

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


# todo: untested
def script_binarize_lightweight_unet(name, fold_index, notes, seed=1):
    # !: not support 5 folds
    seed = seed
    batch_size = 1
    target_size = (256, 256)
    use_weight_map = False

    num_activation_residual_levels = 3
    num_depthwise_conv_residual_levels = 2
    num_pointwise_conv_residual_levels = 2
    num_conv_residual_levels = 3

    learning_rate = 1e-3

    logdir = "E:/ED_MS/Semester_3/Codes/MyProject/tensorboard_logs"
    logdir = os.path.join(logdir,
                          datetime.now().strftime("%Y-%m-%d") + \
                          "_{}-a{}_d{}_p{}_c{}-fold_{}".format(name,
                                                               num_activation_residual_levels,
                                                               num_depthwise_conv_residual_levels,
                                                               num_pointwise_conv_residual_levels,
                                                               num_conv_residual_levels,
                                                               fold_index))
    checkpoint_filepath = os.path.join(logdir,
                                       "blu_unet-a{}_d{}_p{}_c{}-fold_{}.h5".format(num_activation_residual_levels,
                                                                                    num_depthwise_conv_residual_levels,
                                                                                    num_pointwise_conv_residual_levels,
                                                                                    num_conv_residual_levels,
                                                                                    fold_index))

    start_epoch = 0
    end_epoch = 400

    if fold_index == 0:
        lightweight_unet_weight_path = "E:/ED_MS/Semester_3/Codes/MyProject/tensorboard_logs/" \
                                       "2022-08-09_retrain_lw_unet_after_knowledge_distillation-fold_0/" \
                                       "lw_unet_retrained_after_knowledge_distillation-fold_0.h5"
    elif fold_index == 1:
        lightweight_unet_weight_path = "E:/ED_MS/Semester_3/Codes/MyProject/tensorboard_logs/" \
                                       "2022-08-09_retrain_lw_unet_after_knowledge_distillation-fold_1/" \
                                       "lw_unet_retrained_after_knowledge_distillation-fold_1.h5"
    else:
        raise RuntimeError("Unsupported fold index")

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

    train_set, val_set, test_set = _func_get_train_val_test_dataset(
        target_size=target_size, batch_size=batch_size, use_weight_map=use_weight_map, seed=seed, fold_index=fold_index)

    callback_list = _func_get_callback_list(
        checkpoint_filepath=checkpoint_filepath, monitor="val_binary_IoU", mode="max", ignore=0,
        start_epoch=start_epoch, end_epoch=end_epoch, logdir=logdir, model=blu_net, val_set=val_set)

    _func_print_training_info(
        name=name, seed=seed, train_set=train_set, val_set=val_set, batch_size=batch_size,
        use_weight_map=use_weight_map,
        start_epoch=start_epoch, end_epoch=end_epoch, fold_index=fold_index)

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


# * ===========================================================================
def script_fine_tune_blu_net(name, fold_index, notes, seed=1):
    seed = seed
    batch_size = 1
    target_size = (256, 256)
    use_weight_map = False

    num_activation_residual_levels = 3
    num_depthwise_conv_residual_levels = 2
    num_pointwise_conv_residual_levels = 2
    num_conv_residual_levels = 3

    learning_rate = 1e-4

    logdir = "E:/ED_MS/Semester_3/Codes/MyProject/tensorboard_logs"
    logdir = os.path.join(logdir,
                          datetime.now().strftime("%Y-%m-%d") + \
                          "_{}-a{}_d{}_p{}_c{}-fold_{}".format(name,
                                                               num_activation_residual_levels,
                                                               num_depthwise_conv_residual_levels,
                                                               num_pointwise_conv_residual_levels,
                                                               num_conv_residual_levels,
                                                               fold_index))
    checkpoint_filepath = os.path.join(logdir,
                                       "fine_tuned_blu_unet-a{}_d{}_p{}_c{}-fold_{}.h5".format(
                                           num_activation_residual_levels,
                                           num_depthwise_conv_residual_levels,
                                           num_pointwise_conv_residual_levels,
                                           num_conv_residual_levels,
                                           fold_index))
    start_epoch = 0
    end_epoch = 400
    trained_blu_net_path = "E:/ED_MS/Semester_3/Codes/MyProject/tensorboard_logs/" \
                           "2022-08-10_blu_net-residual_binarize_retrained_lw_unet-a3_d2_p2_c3-fold_0/" \
                           "blu_unet-a3_d2_p2_c3-fold_0.h5"

    notes = append_info_to_notes(
        notes, fold_index=fold_index, seed=seed, batch_size=batch_size, target_size=target_size,
        use_weight_map=use_weight_map, trained_blu_net_path=trained_blu_net_path,
        learning_rate=learning_rate, checkpoint_filepath=checkpoint_filepath, start_epoch=start_epoch,
        end_epoch=end_epoch, num_activation_residual_levels=num_activation_residual_levels,
        num_conv_residual_levels=num_conv_residual_levels,
        num_depthwise_conv_residual_levels=num_depthwise_conv_residual_levels,
        num_pointwise_conv_residual_levels=num_pointwise_conv_residual_levels)

    blu_net = get_compiled_binary_lightweight_unet(
        input_size=(*target_size, 1),
        num_activation_residual_levels=num_activation_residual_levels,
        num_depthwise_conv_residual_levels=num_depthwise_conv_residual_levels,
        num_pointwise_conv_residual_levels=num_pointwise_conv_residual_levels,
        num_conv_residual_levels=num_conv_residual_levels,
        learning_rate=learning_rate,
        pretrained_weight=trained_blu_net_path)

    train_set, val_set, test_set = _func_get_train_val_test_dataset(
        target_size=target_size, batch_size=batch_size, use_weight_map=use_weight_map, seed=seed, fold_index=fold_index)

    callback_list = _func_get_callback_list(
        checkpoint_filepath=checkpoint_filepath, monitor="val_binary_IoU", mode="max",
        start_epoch=start_epoch, end_epoch=end_epoch, logdir=logdir, model=blu_net, val_set=val_set)

    _func_print_training_info(
        name=name, seed=seed, train_set=train_set, val_set=val_set, batch_size=batch_size,
        use_weight_map=use_weight_map,
        start_epoch=start_epoch, end_epoch=end_epoch, fold_index=fold_index)

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
    # todo list
    # ! - [ ] Perform knowledge distillation - continue training on fold 2, 3, 4
    # ! - [ ] Perform residual binarization on fold 0, 1, 2, 3, 4, 5

    # *: fine tune a vanilla unet with regularizer
    # note_unet = "fine tune a vanilla unet with the weights from Delta 2.0\n"
    # log_df_unet = script_fine_tune_vanilla_unet_with_l2_regularizer(
    #     name="fine_tune_vanilla_unet_with_l2", fold_index=4,
    #     notes=note_unet)

    # *: knowledge distillation - teacher-student
    note_kd = "train lightweight unet via knowledge distillation"
    # encoder
    # log_df_kd = script_train_lightweight_unet_via_knowledge_distillation(
    #     name="knowledge_distillation-lw_unet-encoder",
    #     fold_index=1,
    #     notes=note_kd)
    # encoder + decoder
    log_df_kd = script_train_lightweight_unet_via_knowledge_distillation(
        name="knowledge_distillation-lw_unet-decoder",
        fold_index=1,
        notes=note_kd)

    # *: knowledge distillation - continue training. Fine tune lw_unet on the basis of knowledge distillation
    # note_retrain = "retrain lw_unet after knowledge distillation"
    # log_df_lw_unet = script_train_lightweight_unet_after_knowledge_distillation(
    #     name="retrain_lw_unet_after_knowledge_distillation",
    #     fold_index=0,
    #     notes=note_retrain)

    # *: residual binarization -> BLU-Net
    # note_blu_net = "binarize retrained lw_unet"
    # log_df_blu_net = script_binarize_lightweight_unet(name="blu_net-residual_binarize_retrained_lw_unet",
    #                                                   fold_index=0,
    #                                                   notes=note_blu_net)

    # *: adjust hyper-parameters adn continue training BLU-Net, encoder stack
    # note_fine_tune_blu_net = "fine_tune_blu_net"
    # log_df_fine_tuned_blu_net = script_fine_tune_blu_net(name="fine_tune_blu_net",
    #                                                      fold_index=0,
    #                                                      notes=note_fine_tune_blu_net)

    pass
