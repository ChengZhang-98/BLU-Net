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


def get_validation_plot_callback(model, validation_set, batch_index_to_plot, summary_writer, max_output=4):
    def plot_ground_truth_image_on_validation_set(logs):
        if isinstance(batch_index_to_plot, Iterable):
            image_batch_list = []
            mask_batch_list = []
            image_path_series_list = []
            for index in batch_index_to_plot:
                image_batch_i, mask_batch_i = validation_set[index]
                image_batch_list.append(image_batch_i)
                mask_batch_list.append(mask_batch_i)
                image_path_series_i = validation_set.get_batch_dataframe(index).loc[:, "image"]
                image_path_series_list.append(image_path_series_i)

            image_batch = tf.concat(image_batch_list, axis=0)
            mask_batch = tf.concat(mask_batch_list, axis=0)
            image_path_series = pd.concat(image_path_series_list)
        else:
            image_batch, mask_batch = validation_set[batch_index_to_plot]
            image_path_series = validation_set.get_batch_dataframe(batch_index_to_plot).loc[:, "image"]
        with summary_writer.as_default():
            image_path_info = _callback_util_get_image_description(image_path_series)
            description_1 = "**ground truth images** on validation dataset, " + image_path_info
            description_2 = "**ground truth masks** on validation dataset, " + image_path_info
            tf.summary.image("Input_Images", data=image_batch, step=0, max_outputs=max_output,
                             description=description_1)
            tf.summary.image("Ground_Truth_Masks", data=mask_batch, step=0, max_outputs=max_output,
                             description=description_2)

    def plot_predicted_masks_on_validation_set(epoch, logs):
        if isinstance(batch_index_to_plot, Iterable):
            predicted_mask_list = []
            image_path_series_list = []
            for index in batch_index_to_plot:
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
        else:
            image_batch, _ = validation_set[batch_index_to_plot]
            predicted_mask_batch = model.predict_step(image_batch)
            image_path_series = validation_set.get_batch_dataframe(batch_index_to_plot).loc[:, "image"]

        with summary_writer.as_default():
            image_path_info = _callback_util_get_image_description(image_path_series)
            description = "**predicted masks** on validation dataset, " + image_path_info
            tf.summary.image("Predicted_Mask_on_Validation_Dataset", data=predicted_mask_batch, step=epoch,
                             max_outputs=max_output, description=description)

    validation_plot_callback = callbacks.LambdaCallback(on_train_begin=plot_ground_truth_image_on_validation_set,
                                                        on_epoch_end=plot_predicted_masks_on_validation_set)
    return validation_plot_callback


def train_val_split(data_gen: DataGenerator, train_ratio: float, validation_batch_size=2):
    sample_for_train = int(train_ratio * len(data_gen.data_df))
    data_gen_val = DataGenerator.build_from_dataframe(data_gen.data_df.iloc[sample_for_train:, :],
                                                      batch_size=validation_batch_size,
                                                      mode="validate",
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


def train_gan(gan: GAN, epochs, training_set, validation_set, tf_summary_writer_val_image=None,
              tf_summary_writer_train_scaler=None, tf_summary_writer_val_scaler=None, epochs_not_train_g=3):
    print("Train discriminator but not generator in the first {} epochs".format(epochs_not_train_g))

    with tf_summary_writer_val_image.as_default():
        image_batch_val, _ = validation_set[0]
        predicted_mask_batch = gan.generator(image_batch_val, training=False)
        predicted_mask_batch = tf.cast(predicted_mask_batch >= 0.5, dtype=tf.float32)
        tf.summary.image("Before_Training-Predicted_Masks", predicted_mask_batch, step=0)

    log_df = pd.DataFrame(dict(epoch=[], train_g_loss=[], train_d_loss=[], val_binary_accuracy=[], val_binary_iou=[]))

    for epoch in range(epochs):
        epoch_log_dict = dict(epoch=[epoch], train_g_loss=[np.NAN], train_d_loss=[np.NAN],
                              val_binary_accuracy=[np.NAN], val_binary_iou=[np.NAN])

        # *: train
        train_g = epoch >= epochs_not_train_g
        train_d = True
        for step, (image_batch_train, mask_batch_train, weight_map_batch_train) in tqdm(enumerate(training_set),
                                                                                        desc="Epoch {}".format(epoch)):
            d_loss, g_loss, generated_masks = gan.train_step(image_batch_train, mask_batch_train,
                                                             weight_map_batch_train,
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


if __name__ == '__main__':
    a = \
        """
        line1
        line2
        line3"""
    print(a)
