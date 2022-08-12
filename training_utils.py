import operator
import os
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from keras import callbacks
from scipy import interpolate
from tqdm import tqdm

from data import DataGenerator, postprocess_a_mask_batch, _get_matched_data_df


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


def _train_val_test_df_split(dataset_name,
                             image_dir, image_type, mask_dir, mask_type, weight_map_dir, weight_map_type,
                             num_train_samples, num_folds, fold_index, seed):
    assert num_folds > fold_index >= 0, "fold_index fails to meet: 0 <= fold_index < {}".format(num_folds)

    data_df = _get_matched_data_df(image_dir=image_dir,
                                   image_type=image_type,
                                   mask_dir=mask_dir,
                                   mask_type=mask_type,
                                   weight_map_dir=weight_map_dir,
                                   weight_map_type=weight_map_type,
                                   dataset_name=dataset_name)

    data_df = data_df.sample(frac=1, random_state=44).reset_index(drop=True)

    train_val_df = data_df.iloc[:num_train_samples, :].sample(frac=1, random_state=seed).reset_index(drop=True)
    test_df = data_df.iloc[num_train_samples:].reset_index(drop=True)

    fold_index_list = np.split(np.arange(num_train_samples), num_folds)
    val_df_indices = fold_index_list.pop(fold_index)
    train_df_indices = np.concatenate(fold_index_list)
    val_df = train_val_df.iloc[val_df_indices, :].copy().reset_index(drop=True)
    train_df = train_val_df.iloc[train_df_indices, :].copy().reset_index(drop=True)
    val_df.loc[:, "fold"] = fold_index
    train_df.loc[:, "fold"] = fold_index

    return train_df, val_df, test_df


def train_val_test_split(batch_size, dataset_name, use_weight_map,
                         image_dir, image_type, mask_dir, mask_type, weight_map_dir, weight_map_type,
                         target_size, data_aug_transform, seed, num_folds=5, fold_index=1):
    train_df, val_df, test_df = _train_val_test_df_split(
        dataset_name=dataset_name,
        image_dir=image_dir, image_type=image_type, mask_dir=mask_dir, mask_type=mask_type,
        weight_map_dir=weight_map_dir, weight_map_type=weight_map_type,
        num_train_samples=30, num_folds=num_folds, fold_index=fold_index, seed=seed)

    train_data_gen = DataGenerator.build_from_dataframe(
        dataframe=train_df, batch_size=batch_size, dataset_name=dataset_name,
        mode="train", use_weight_map=use_weight_map, target_size=target_size, data_aug_transform=data_aug_transform,
        seed=None)

    val_data_gen = DataGenerator.build_from_dataframe(
        dataframe=val_df, batch_size=batch_size, dataset_name=dataset_name,
        mode="validate", use_weight_map=use_weight_map, target_size=target_size, data_aug_transform=None,
        seed=None)

    test_data_gen = DataGenerator.build_from_dataframe(
        dataframe=test_df, batch_size=batch_size, dataset_name=dataset_name,
        mode="test", use_weight_map=use_weight_map, target_size=target_size, data_aug_transform=None,
        seed=None)

    return train_data_gen, val_data_gen, test_data_gen


def train_val_split(data_gen: DataGenerator, train_ratio: float, validation_batch_size=2, use_weight_map_val=False):
    sample_for_train = int(train_ratio * len(data_gen.data_df))
    data_gen_val = DataGenerator.build_from_dataframe(data_gen.data_df.iloc[sample_for_train:, :],
                                                      batch_size=validation_batch_size,
                                                      dataset_name=None,
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
            return 0.95 * lr

    return lr_scheduler


def get_lr_scheduler_v2(epochs, lrs):
    interpolator = interpolate.interp1d(epochs, lrs)

    def lr_scheduler_v2(epoch, lr):
        if epoch < np.max(epochs):
            return interpolator(epoch)
        else:
            return lr * 0.95

    return lr_scheduler_v2


class CustomLRScheduler(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate, steps_per_epoch, epoch_start_to_decay):
        self.current_lr = initial_learning_rate
        self.current_epoch = 0
        self.steps_per_epoch = steps_per_epoch
        self.epoch_start_to_decay = epoch_start_to_decay

    def __call__(self, step):
        current_epoch = step // self.steps_per_epoch
        if self.current_epoch < self.epoch_start_to_decay:
            self.current_epoch = current_epoch
            return self.current_lr
        else:
            if current_epoch > self.current_epoch:
                self.current_lr = 0.95 * self.current_lr
                self.current_epoch = current_epoch
                return self.current_lr
            else:
                return self.current_lr


def evaluate_on_test_set(model, test_set, postprocessing=False):
    iou_list = []
    f1score_list = []

    for double_or_triple in tqdm(test_set, total=len(test_set)):
        if test_set.use_weight_map:
            image_batch, mask_batch, weight_map_batch = double_or_triple
        else:
            image_batch, mask_batch = double_or_triple
        pred_mask_batch = model(image_batch, training=False)

        if postprocessing:
            pred_mask_batch = postprocess_a_mask_batch(pred_mask_batch,
                                                       binarize_threshold=0.5,
                                                       remove_min_size=20)
        batch_f1score = binarize_and_compute_f1score(mask_batch, pred_mask_batch)
        batch_iou = binarize_and_compute_iou(mask_batch, pred_mask_batch)

        f1score_list.append(tf.reduce_mean(batch_f1score))
        iou_list.append(tf.reduce_mean(batch_iou))

    return dict(binary_f1score=np.mean(f1score_list), binary_iou=np.mean(iou_list))


def binarize_and_compute_iou(y_true, y_pred):
    threshold = 0.5
    logical_y_true = y_true > threshold
    logical_y_pred = y_pred > threshold

    m11 = tf.logical_and(logical_y_true, logical_y_pred)
    m10 = tf.logical_and(logical_y_true, tf.logical_not(logical_y_pred))
    m01 = tf.logical_and(tf.logical_not(logical_y_true), logical_y_pred)

    true_positives = tf.reduce_sum(tf.cast(m11, dtype=tf.float32), axis=(1, 2, 3))
    false_negatives = tf.reduce_sum(tf.cast(m10, dtype=tf.float32), axis=(1, 2, 3))
    false_positives = tf.reduce_sum(tf.cast(m01, dtype=tf.float32), axis=(1, 2, 3))

    batch_iou = true_positives / (true_positives + false_negatives + false_positives + 1e-7)

    return batch_iou


def get_custom_metric_iou(name="binary_IoU"):
    def custom_metric_iou(y_true, y_pred):
        return binarize_and_compute_iou(y_true, y_pred)

    custom_metric_iou.__name__ = name
    return custom_metric_iou


def binarize_and_compute_f1score(y_true, y_pred):
    threshold = 0.5
    logical_y_true = y_true > threshold
    logical_y_pred = y_pred > threshold
    m11 = tf.logical_and(logical_y_true, logical_y_pred)
    m10 = tf.logical_and(logical_y_true, tf.logical_not(logical_y_pred))
    m01 = tf.logical_and(tf.logical_not(logical_y_true), logical_y_pred)

    true_positives = tf.reduce_sum(tf.cast(m11, dtype=tf.float32), axis=(1, 2, 3))
    false_negatives = tf.reduce_sum(tf.cast(m10, dtype=tf.float32), axis=(1, 2, 3))
    false_positives = tf.reduce_sum(tf.cast(m01, dtype=tf.float32), axis=(1, 2, 3))

    batch_f1score = true_positives / (1e-7 + true_positives + 0.5 * (false_positives + false_negatives))

    return batch_f1score


def get_custom_metric_f1score(name="binary_F1Score"):
    def custom_metric_f1score(y_true, y_pred):
        return binarize_and_compute_f1score(y_true, y_pred)

    custom_metric_f1score.__name__ = name
    return custom_metric_f1score


def append_info_to_notes(notes=None, **kwargs):
    lines = ["{} = {}".format(k, w) for k, w in kwargs.items()]
    lines = ["-" * 20] + lines + ["-" * 20]
    return notes + "\n" + "\n".join(lines)


def get_sleep_callback(sleep_length=300, per_epoch=50):
    def take_a_break(epoch, logs):
        if epoch != 0 and epoch % per_epoch == 0:
            time.sleep(sleep_length)

    return callbacks.LambdaCallback(on_epoch_end=take_a_break)


class CustomModelCheckpointCallBack(callbacks.Callback):
    def __init__(self, ignore, filepath, monitor, mode, logdir=None):
        super(CustomModelCheckpointCallBack, self).__init__()
        self.ignore = ignore
        self.filepath = filepath
        self.monitor = monitor
        assert mode in ["max", "min"], "parameter `mode` must be either 'max' or 'min'"
        if mode == "max":
            self.compare_current_with_best = operator.ge
        else:
            self.compare_current_with_best = operator.lt
        self.logdir = logdir
        self.summary_writer = tf.summary.create_file_writer(os.path.join(logdir, "train"))
        self.best = None
        self.last = None
        self.best_checkpoint_info = None

    def on_epoch_end(self, epoch, logs=None):
        self.last = logs[self.monitor]
        if epoch == 0:
            self.best = logs[self.monitor]
        elif epoch < self.ignore:
            if self.compare_current_with_best(logs[self.monitor], self.best):
                self.best = logs[self.monitor]
        else:
            if self.compare_current_with_best(logs[self.monitor], self.best):
                self.best = logs[self.monitor]
                self.model.save_weights(self.filepath, overwrite=True, save_format="h5", options=None)
                info = "epoch = {}, best {} = {}, model saved to {}\n\n".format(epoch, self.monitor,
                                                                                self.best,
                                                                                self.filepath)
                self.best_checkpoint_info = info

    def on_train_end(self, logs=None):
        self.model.save_weights(self.filepath.replace(".h5", "-end_epoch.h5"), overwrite=True, save_format='h5')
        info = "training ended, last {} = {}, model saved to {}".format(self.monitor, self.last,
                                                                        self.filepath.replace(".h5",
                                                                                              "-end_epoch.h5"))

        with open(os.path.join(self.logdir, "checkpoint_logs.txt"), "w+") as f:
            if self.best_checkpoint_info is not None:
                f.write(self.best_checkpoint_info)
            f.write(info)
        print(self.best_checkpoint_info)
        print(info)


class CustomLRTrackerCallback(callbacks.Callback):
    def __init__(self, logdir):
        super(CustomLRTrackerCallback, self).__init__()
        self.train_summery_writer = tf.summary.create_file_writer(os.path.join(logdir, "train"))

    def on_epoch_end(self, epoch, logs=None):
        with self.train_summery_writer.as_default():
            tf.summary.scalar(name="learning_rate", data=logs["lr"], step=epoch)


if __name__ == '__main__':
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

    train_df, val_df, test_df = _train_val_test_df_split(
        dataset_name="DIC", image_dir=image_dir, image_type=image_type, mask_dir=mask_dir, mask_type=mask_type,
        weight_map_dir=weight_map_dir, weight_map_type=weight_map_type,
        num_train_samples=30, num_folds=5, fold_index=1, seed=1
    )
