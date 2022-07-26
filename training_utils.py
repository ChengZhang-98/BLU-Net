import os
from collections.abc import Iterable

import numpy as np
import pandas as pd
import tensorflow as tf
from keras import callbacks
from tqdm import tqdm

from data import DataGenerator, postprocess_a_mask_batch
from model import GAN


def _callback_util_get_image_description(image_path_series: pd.Series):
    description = ", ".join(image_path_series.apply(lambda x: os.path.basename(x)))
    description = "**image**: {}".format(description)
    return description


def get_validation_plot_callback(model, validation_set, batch_index_list_to_plot, summary_writer, max_output=4):
    def plot_ground_truth_image_on_validation_set(logs):
        image_batch_list = []
        mask_batch_list = []
        predict_mask_batch_list = []
        image_path_series_list = []
        for index in batch_index_list_to_plot:
            image_batch_i, mask_batch_i = validation_set[index]
            predict_mask_batch_i = model(image_batch_i, training=False)
            image_batch_list.append(image_batch_i)
            mask_batch_list.append(mask_batch_i)
            predict_mask_batch_list.append(predict_mask_batch_i)
            image_path_series_i = validation_set.get_batch_dataframe(index).loc[:, "image"]
            image_path_series_list.append(image_path_series_i)

        images = tf.concat(image_batch_list, axis=0)
        masks = tf.concat(mask_batch_list, axis=0)
        predict_masks = tf.concat(predict_mask_batch_list, axis=0)
        image_path_series = pd.concat(image_path_series_list)
        with summary_writer.as_default():
            image_path_info = _callback_util_get_image_description(image_path_series)
            tf.summary.image("Input_Images", data=images, step=0, max_outputs=max_output,
                             description=image_path_info)
            tf.summary.image("Ground_Truth_Masks", data=masks, step=0, max_outputs=max_output,
                             description=image_path_info)
            tf.summary.image("Before_Training-Predicted_Masks", data=predict_masks, step=0, max_outputs=max_output)

    def plot_predicted_masks_on_validation_set(epoch, logs):
        predicted_mask_list = []
        image_path_series_list = []
        for index in batch_index_list_to_plot:
            image_batch_i, _ = validation_set[index]
            predicted_mask_batch_i = model(image_batch_i, training=False)
            # Except binarization, other methods in postprocessing require parameter adjustment
            # predicted_mask_batch_i = postprocess_a_mask_batch(predicted_mask_batch_i, min_size=5)
            predicted_mask_batch_i = tf.cast(predicted_mask_batch_i > 0.5, dtype=tf.float32)
            predicted_mask_list.append(predicted_mask_batch_i)
            image_path_series_i = validation_set.get_batch_dataframe(index).loc[:, "image"]
            image_path_series_list.append(image_path_series_i)

        predicted_mask_batch = tf.concat(predicted_mask_list, axis=0)
        image_path_series = pd.concat(image_path_series_list)

        with summary_writer.as_default():
            image_path_info = _callback_util_get_image_description(image_path_series)
            description = "**predicted masks** on validation dataset, " + image_path_info
            tf.summary.image("Predicted_Mask_on_Validation_Dataset", data=predicted_mask_batch, step=epoch,
                             max_outputs=max_output, description=description)

    validation_plot_callback = callbacks.LambdaCallback(on_train_begin=plot_ground_truth_image_on_validation_set,
                                                        on_epoch_end=plot_predicted_masks_on_validation_set)
    return validation_plot_callback


def train_val_split(data_gen: DataGenerator, train_ratio: float, validation_batch_size=2, use_weight_map_val=False):
    sample_for_train = int(train_ratio * len(data_gen.data_df))
    data_gen_val = DataGenerator.build_from_dataframe(data_gen.data_df.iloc[sample_for_train:, :],
                                                      batch_size=validation_batch_size,
                                                      mode="validate",
                                                      use_weight_map=use_weight_map_val,
                                                      target_size=data_gen.target_size,
                                                      data_aug_transform=None,
                                                      seed=None)
    data_gen.data_df = data_gen.data_df.iloc[:sample_for_train, :]
    return data_gen, data_gen_val


def get_lr_scheduler(start_epoch=1):
    def lr_scheduler(epoch, lr):
        if epoch < start_epoch:
            return lr
        else:
            return np.exp(-0.1) * lr

    return lr_scheduler


def train_gan(gan: GAN, start_epoch, epochs, training_set, validation_set, tf_summary_writer_val_image=None,
              tf_summary_writer_train_scaler=None, tf_summary_writer_val_scaler=None, epochs_not_train_g=3):
    assert not training_set.use_weight_map, "weight map is not needed for GAN training"
    assert not validation_set.use_weight_map, "weight map is not needed for GAN validation"
    print("Train discriminator but not generator in the first {} epochs".format(epochs_not_train_g))

    if tf_summary_writer_val_image is not None:
        with tf_summary_writer_val_image.as_default():
            image_batch_val, _ = validation_set[0]
            predicted_mask_batch = gan.generator(image_batch_val, training=False)
            predicted_mask_batch = tf.cast(predicted_mask_batch >= 0.5, dtype=tf.float32)
            tf.summary.image("Before_Training-Predicted_Masks", predicted_mask_batch, step=0)

    log_df = pd.DataFrame(dict(epoch=[], train_g_loss=[], train_d_loss=[], val_binary_accuracy=[], val_binary_iou=[]))

    for epoch in np.arange(start_epoch, start_epoch + epochs):
        epoch_log_dict = dict(epoch=[epoch], train_g_loss=[np.NAN], train_d_loss=[np.NAN],
                              val_binary_accuracy=[np.NAN], val_binary_iou=[np.NAN])

        # *: train
        train_g = epoch - start_epoch >= epochs_not_train_g
        train_d = True
        for step, (image_batch_train, mask_batch_train) in tqdm(enumerate(training_set),
                                                                desc="Epoch {}".format(epoch)):
            d_loss, g_loss, generated_masks = gan.train_step(image_batch_train, mask_batch_train,
                                                             train_g=train_g, train_d=train_d)
        # track train log and visualize in Tensorboard
        d_loss = gan.disc_loss_tracker.result()
        epoch_log_dict.update(train_d_loss=[d_loss])
        g_loss = np.NAN
        if train_g:
            g_loss = gan.gen_loss_tracker.result()
            epoch_log_dict.update(train_g_loss=[g_loss])
        if tf_summary_writer_train_scaler is not None:
            with tf_summary_writer_train_scaler.as_default():
                tf.summary.scalar("epoch_loss_discriminator", d_loss, step=epoch)
                if train_g:
                    tf.summary.scalar("epoch_loss_generator", g_loss, step=epoch)
        else:
            print("train_loss-discriminator = {}".format(d_loss))
            if train_g:
                print("train_loss-generator = {}".format(g_loss))

        # *: validate
        max_batch_num = 3
        image_list = []
        ground_truth_mask_list = []
        predicted_mask_list = []
        image_info_list = []
        for val_batch_index, (image_batch_val, mask_batch_val) in enumerate(validation_set):
            predicted_mask_batch = gan.generator(image_batch_val, training=False)
            gan.metric_binary_accuracy.update_state(mask_batch_val, predicted_mask_batch)
            gan.metric_binary_IoU.update_state(mask_batch_val, predicted_mask_batch)
            if val_batch_index <= max_batch_num:
                predicted_mask_list.append(predicted_mask_batch)
                image_path_series = validation_set.get_batch_dataframe(val_batch_index).loc[:, "image"]
                image_info_list.append(_callback_util_get_image_description(image_path_series))
                if epoch == 0:
                    image_list.append(image_batch_val)
                    ground_truth_mask_list.append(mask_batch_val)

        if tf_summary_writer_val_image is not None:
            with tf_summary_writer_val_image.as_default():
                val_predicted_masks = tf.concat(predicted_mask_list, axis=0)
                image_info = ", ".join(image_info_list)
                if epoch == 0:
                    images = tf.concat(image_list, axis=0)
                    ground_truth_masks = tf.concat(ground_truth_mask_list, axis=0)
                    tf.summary.image("Input_Images", images, step=0, description=image_info)
                    tf.summary.image("Ground_Truth_Masks", ground_truth_masks, step=0, description=image_info)
                tf.summary.image("Predicted_Masks", val_predicted_masks, step=epoch)

        val_metric_binary_accuracy = gan.metric_binary_accuracy.result()
        val_metric_binary_iou = gan.metric_binary_IoU.result()
        epoch_log_dict.update(val_binary_accuracy=[val_metric_binary_accuracy], val_binary_iou=[val_metric_binary_iou])
        log_df = pd.concat([log_df, pd.DataFrame(epoch_log_dict)], ignore_index=True)

        if tf_summary_writer_val_scaler is not None:
            with tf_summary_writer_val_scaler.as_default():
                tf.summary.scalar("epoch_binary_accuracy", val_metric_binary_accuracy, step=epoch)
                tf.summary.scalar("epoch_binary_IoU", val_metric_binary_iou, step=epoch)
        else:
            print("val_metric-binary_accuracy = {}".format(val_metric_binary_accuracy))
            print("val_metric-binary_IoU = {}".format(val_metric_binary_accuracy))

        gan.reset_metric_states()

    print("GAN training finished")
    return log_df


def evaluate_model_on_a_dataset(model, dataset, metric_list, postprocessing=True):
    for batch_index, (image_batch, mask_batch) in tqdm(enumerate(dataset)):
        pred_mask_batch = model(image_batch, training=False)
        if postprocessing:
            pred_mask_batch = postprocess_a_mask_batch(pred_mask_batch, binarize_threshold=0.5, remove_min_size=20)
        for metric in metric_list:
            metric.update_state(mask_batch, pred_mask_batch)

    for metric in metric_list:
        print("{}: {}".format(metric.name, metric.result()))

    return metric_list


if __name__ == '__main__':
    from model import get_compiled_unet
    import keras

    seed = None
    batch_size = 2
    target_size = (512, 512)
    # pretrained_weight_path = "E:/ED_MS/Semester_3/Codes/MyProject/checkpoints/trained_weights/" \
    #                          "unet_agarpads_seg_evaluation2.hdf5"
    pretrained_weight_path = "E:/ED_MS/Semester_3/Codes/MyProject/checkpoints/vanilla_unet.h5"

    image_dir = "E:/ED_MS/Semester_3/Dataset/DIC_Set/DIC_Set1_Annotated"
    image_type = "tif"
    mask_dir = "E:/ED_MS/Semester_3/Dataset/DIC_Set/DIC_Set1_Masks"
    mask_type = "tif"
    weight_map_dir = "E:/ED_MS/Semester_3/Dataset/DIC_Set/DIC_Set1_Weights"
    weight_map_type = "npy"
    dataset = "DIC"

    logdir = "E:/ED_MS/Semester_3/Codes/MyProject/tensorboard_logs"
    checkpoint_filepath = "E:/ED_MS/Semester_3/Codes/MyProject/checkpoints/vanilla_unet.h5"
    start_epoch = 0
    end_epoch = 15

    unet = get_compiled_unet(input_size=(*target_size, 1),
                             levels=5,
                             pretrained_weights=pretrained_weight_path,
                             learning_rate=1e-4)
    data_gen_train = DataGenerator(batch_size=batch_size, dataset=dataset, mode="train", use_weight_map=True,
                                   image_dir=image_dir, image_type=image_type,
                                   mask_dir=mask_dir, mask_type=mask_type,
                                   weight_map_dir=weight_map_dir, weight_map_type=weight_map_type,
                                   target_size=target_size, data_aug_transform=None, seed=seed)
    print("Evaluation started...")
    _, data_gen_val = train_val_split(data_gen_train, 0.8, validation_batch_size=2)

    metric_list = [keras.metrics.BinaryAccuracy(name="binary_accuracy", threshold=0.5),
                   keras.metrics.BinaryIoU(target_class_ids=[1], threshold=0.5, name="binary_IoU", )]
    evaluate_model_on_a_dataset(model=unet, dataset=data_gen_val, metric_list=metric_list)
