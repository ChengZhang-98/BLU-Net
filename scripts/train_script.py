import os
from datetime import datetime

import keras.callbacks
import tensorflow as tf

import keras
from keras import callbacks

from data import DataGenerator
from model import get_compiled_unet, get_discriminator, GAN
from data import DataGenerator

from data_augmentation import (HistogramVoodoo, ElasticDeform, GaussianNoise, RandomFlip, RandomRotate,
                               RandomZoomAndShift, DataAugmentation)
from training_utils import get_validation_plot_callback, train_val_split, train_gan, get_lr_scheduler


def script_train_unet(name=None):
    # *: tensorboard --logdir="E:\ED_MS\Semester_3\Codes\MyProject\tensorboard_logs"
    seed = None
    batch_size = 2
    target_size = (512, 512)
    pretrained_weight_path = "E:/ED_MS/Semester_3/Codes/MyProject/checkpoints/trained_weights/" \
                             "unet_agarpads_seg_evaluation2.hdf5"

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

    # logdir = os.path.join(logdir, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    if name is None:
        logdir = os.path.join(logdir, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    else:
        logdir = os.path.join(logdir, datetime.now().strftime("%Y-%m-%d") + "_{}".format(name))

    unet = get_compiled_unet(input_size=(*target_size, 1),
                             levels=5,
                             pretrained_weights=pretrained_weight_path,
                             learning_rate=1e-4)

    data_augmentation_transform = DataAugmentation([HistogramVoodoo(),
                                                    ElasticDeform(sigma=20),
                                                    GaussianNoise(sigma=1 / 255),
                                                    RandomFlip(),
                                                    RandomRotate(),
                                                    RandomZoomAndShift()])
    data_gen_train = DataGenerator(batch_size=batch_size, dataset=dataset, mode="train",
                                   image_dir=image_dir, image_type=image_type,
                                   mask_dir=mask_dir, mask_type=mask_type,
                                   weight_map_dir=weight_map_dir, weight_map_type=weight_map_type,
                                   target_size=target_size, data_aug_transform=data_augmentation_transform, seed=seed)
    data_gen_train, data_gen_val = train_val_split(data_gen_train, 0.8, validation_batch_size=2)

    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
                                                                save_weights_only=True,
                                                                verbosde=1,
                                                                monitor='val_binary_IoU',
                                                                mode='max',
                                                                save_best_only=True)
    early_stopping_callback = keras.callbacks.EarlyStopping(monitor="val_binary_IoU", patience=3, mode="max",
                                                            verbose=1, restore_best_weights=True)
    lr_scheduler_callback = keras.callbacks.LearningRateScheduler(schedule=get_lr_scheduler(1))
    tensorboard_val_image_writer = tf.summary.create_file_writer(logdir + "/val_image")
    tensorboard_callback = callbacks.TensorBoard(log_dir=logdir)
    validation_plot_callback = get_validation_plot_callback(unet, data_gen_val, [0, 1],
                                                            tensorboard_val_image_writer, max_output=4)

    callback_list = [tensorboard_callback, validation_plot_callback,
                     model_checkpoint_callback, lr_scheduler_callback]

    print("Training set contains {} samples, validation set contains {} samples".format(len(data_gen_train.data_df),
                                                                                        len(data_gen_val.data_df)))
    print("Training starts...")
    history = unet.fit(x=data_gen_train, epochs=end_epoch, initial_epoch=start_epoch,
                       validation_data=data_gen_val, shuffle=False,
                       validation_freq=1, callbacks=callback_list,
                       workers=1, use_multiprocessing=False)
    print("Training finished")
    return history


def script_train_gan(name=None):
    # *: tensorboard --logdir="E:\ED_MS\Semester_3\Codes\MyProject\tensorboard_logs"
    seed = None
    batch_size = 1
    target_size = (512, 512)

    image_dir = "E:/ED_MS/Semester_3/Dataset/DIC_Set/DIC_Set1_Annotated"
    image_type = "tif"
    mask_dir = "E:/ED_MS/Semester_3/Dataset/DIC_Set/DIC_Set1_Masks"
    mask_type = "tif"
    weight_map_dir = "E:/ED_MS/Semester_3/Dataset/DIC_Set/DIC_Set1_Weights"
    weight_map_type = "npy"
    dataset = "DIC"

    logdir = "E:/ED_MS/Semester_3/Codes/MyProject/tensorboard_logs"
    # todo: load fine-tuned weights
    # checkpoint_filepath = "E:/ED_MS/Semester_3/Codes/MyProject/checkpoints/trained_weights/" \
    #                                "unet_agarpads_seg_evaluation2.hdf5"
    pretrained_g_weight_path = "E:/ED_MS/Semester_3/Codes/MyProject/checkpoints/vanilla_unet.h5"
    start_epoch = 0
    end_epoch = 30
    epochs_not_train_g = 10

    lamda = 1.0
    if name is None:
        logdir = os.path.join(logdir, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    else:
        logdir = os.path.join(logdir, datetime.now().strftime("%Y-%m-%d_") + name)
    tensorboard_train_scaler_writer = tf.summary.create_file_writer(logdir + "/train")
    tensorboard_val_scaler_writer = tf.summary.create_file_writer(logdir + "/validation")
    tensorboard_val_image_writer = tf.summary.create_file_writer(logdir + "/val_image")

    data_augmentation_transform = DataAugmentation([HistogramVoodoo(),
                                                    ElasticDeform(sigma=20),
                                                    GaussianNoise(sigma=1 / 255),
                                                    RandomFlip(),
                                                    RandomRotate(),
                                                    RandomZoomAndShift()])
    data_gen_train = DataGenerator(batch_size=batch_size, dataset=dataset, mode="train",
                                   image_dir=image_dir, image_type=image_type,
                                   mask_dir=mask_dir, mask_type=mask_type,
                                   weight_map_dir=weight_map_dir, weight_map_type=weight_map_type,
                                   target_size=target_size, data_aug_transform=data_augmentation_transform, seed=seed)
    data_gen_train, data_gen_val = train_val_split(data_gen_train, 0.8, validation_batch_size=2)

    generator = get_compiled_unet(input_size=(*target_size, 1),
                                  levels=5,
                                  pretrained_weights=pretrained_g_weight_path)
    discriminator = get_discriminator(input_size=(*target_size, 2))
    gan = GAN(discriminator=discriminator, generator=generator)
    # todo: smaller learning rate
    gan.compile(d_optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5, beta_1=0.5),
                g_optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5, beta_1=0.5),
                loss_fn_adversarial=tf.keras.losses.MeanSquaredError(),
                loss_fn_segmentation=tf.keras.losses.BinaryCrossentropy(),
                lamda=lamda)
    print("Training set contains {} samples, validation set contains {} samples".format(len(data_gen_train.data_df),
                                                                                        len(data_gen_val.data_df)))
    print("Training starts")
    log_df = train_gan(gan, epochs=end_epoch, training_set=data_gen_train, validation_set=data_gen_val,
                       tf_summary_writer_train_scaler=tensorboard_train_scaler_writer,
                       tf_summary_writer_val_scaler=tensorboard_val_scaler_writer,
                       tf_summary_writer_val_image=tensorboard_val_image_writer,
                       epochs_not_train_g=epochs_not_train_g)
    return log_df


if __name__ == '__main__':
    history_unet = script_train_unet(name="unet")
    # log_df_gan = script_train_gan(name="evening-gan")
