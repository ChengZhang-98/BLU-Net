import os
from collections.abc import Iterable

import pandas as pd
import tensorflow as tf
from keras import callbacks
from tqdm import tqdm

from data import DataGenerator, postprocess_a_mask_batch
from model import GAN


def _callback_util_get_image_description(epoch, image_path_series: pd.Series):
    description = ", ".join(image_path_series.apply(lambda x: os.path.basename(x)))
    description = "**epoch**: {}, **image**: {}".format(epoch, description)
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
            image_path_info = _callback_util_get_image_description(0, image_path_series)
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
                predicted_mask_batch_i = postprocess_a_mask_batch(predicted_mask_batch_i)
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
            image_path_info = _callback_util_get_image_description(epoch, image_path_series)
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


def train_gan(gan: GAN, epochs, training_set, validation_set, tf_summary_writer_val_image=None,
              tf_summary_writer_training_scaler=None, epochs_not_train_g=3):
    print("train discriminator but not generator in the first {} epochs".format(epochs_not_train_g))

    with tf_summary_writer_val_image.as_default():
        image_batch_val, _ = validation_set[0]
        predicted_mask_batch = gan.generator(image_batch_val, training=False)
        tf.summary.image("Before Training - Predicted Masks", predicted_mask_batch, step=0)

    val_metric_df = pd.DataFrame(dict(epoch=[], binary_accuracy=[], binary_iou=[]))

    for epoch in range(epochs):
        print("\nStart of Epoch {}".format(epoch))

        # *: train
        train_g = epoch >= epochs_not_train_g
        train_d = True
        for step, (image_batch_train, mask_batch_train, weight_map_batch_train) in tqdm(enumerate(training_set)):
            d_loss, g_loss, generated_masks = gan.train_step(image_batch_train, mask_batch_train,
                                                             weight_map_batch_train,
                                                             train_g=train_g, train_d=train_d)

        if tf_summary_writer_training_scaler is not None:
            with tf_summary_writer_training_scaler.as_default():
                tf.summary.scalar("train_loss-discriminator", gan.disc_loss_tracker.result(), step=epoch)
                if train_g:
                    tf.summary.scalar("train_loss-generator", gan.gen_loss_tracker.result(), step=epoch)
        else:
            print("train_loss-discriminator = {}".format(gan.disc_loss_tracker.result()))
            if train_g:
                print("train_loss-generator = {}".format(gan.gen_loss_tracker.result()))

        # *: validate
        for val_batch_index, (image_batch_val, mask_batch_val) in enumerate(validation_set):
            predicted_mask_batch = gan.generator(image_batch_val, training=False)
            gan.metric_binary_accuracy.update_state(mask_batch_val, predicted_mask_batch)
            gan.metric_binary_IoU.update_state(mask_batch_val, predicted_mask_batch)

            if tf_summary_writer_val_image is not None:
                with tf_summary_writer_val_image.as_default():
                    image_path_series = validation_set.get_batch_dataframe(val_batch_index).loc[:, "image"]
                    image_path_info = _callback_util_get_image_description(0, image_path_series)
                    max_batch_num = 3 // validation_set.batch_size + 1
                    if epoch == 0 and val_batch_index < max_batch_num:
                        description_1 = "**input images** on validation dataset, " + image_path_info
                        description_2 = "**ground truth masks** on validation dataset, " + image_path_info
                        tf.summary.image("Input Images", image_batch_val, step=0)
                        tf.summary.image("Ground Truth Masks", mask_batch_val, step=0)
                    if val_batch_index < max_batch_num:
                        description = "**predicted masks** on validation dataset, " + image_path_info
                        tf.summary.image("Predicted Masks", predicted_mask_batch, step=epoch)

        val_metric_binary_accuracy = gan.metric_binary_accuracy.result()
        val_metric_binary_iou = gan.metric_binary_IoU.result()
        val_metric_df = pd.concat([val_metric_df, pd.DataFrame(dict(epoch=[epoch],
                                                                    binary_accuracy=[val_metric_binary_accuracy],
                                                                    binary_iou=[val_metric_binary_iou]))],
                                  ignore_index=True)
        if tf_summary_writer_training_scaler is not None:
            with tf_summary_writer_training_scaler.as_default():
                tf.summary.scalar("val_metric-binary_accuracy", val_metric_binary_accuracy, step=epoch)
                tf.summary.scalar("val_metric-binary_IoU", val_metric_binary_iou, step=epoch)
        else:
            print("val_metric-binary_accuracy = {}".format(val_metric_binary_accuracy))
            print("val_metric-binary_IoU = {}".format(val_metric_binary_accuracy))

        gan.reset_metric_states()

    print("GAN training finished")
    # max_ba_index = val_metric_df.loc[:, "binary_accuracy"].idxmax()
    # max_ba_epoch = val_metric_df.iloc[max_ba_index, :].loc["epoch"]
    # max_ba = val_metric_df.iloc[max_ba_epoch, :].loc["binary_accuracy"]
    #
    # max_bi_index = val_metric_df.loc[:, "binary_iou"].idxmax()
    # max_bi_epoch = val_metric_df.iloc[max_bi_index, :].loc["epoch"]
    # max_bi = val_metric_df.iloc[max_bi_index, :].loc["binary_iou"]
    # print("max val-binary_accuracy: epoch = {}, binary_accuracy = {}".format(max_ba_epoch, max_ba))
    # print("max val-binary IoU: epoch = {}, binary_IoU = {}".format(max_bi_epoch, max_bi))


if __name__ == '__main__':
    a = \
        """
        line1
        line2
        line3"""
    print(a)
