import os
from collections.abc import Iterable

import pandas as pd
import tensorflow as tf
from keras import callbacks

from data import DataGenerator


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
                predicted_mask_i = model.predict_step(image_batch_i)
                predicted_mask_list.append(predicted_mask_i)
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
                                                      transforms=None,
                                                      seed=None)
    data_gen.data_df = data_gen.data_df.iloc[:sample_for_train, :]
    return data_gen, data_gen_val


if __name__ == '__main__':
    a = \
        """
        line1
        line2
        line3"""
    print(a)
