import os

import tensorflow as tf


def train_unet():
    from datetime import datetime

    import tensorflow as tf
    from data import DataGenerator
    from model import get_compiled_unet
    from data import DataGenerator
    from keras import callbacks

    from training_utils import get_validation_plot_callback, train_val_split

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

    logdir = "E:/ED_MS/Semester_3/Codes/MyProject/tensorboard_logs/"
    checkpoint_filepath = "E:/ED_MS/Semester_3/Codes/MyProject/checkpoints/vanilla_unet.h5"
    max_epoch = 3

    unet = get_compiled_unet(input_size=(*target_size, 1),
                             levels=5,
                             pretrained_weights=pretrained_weight_path)

    data_gen_train = DataGenerator(batch_size=batch_size, dataset=dataset, mode="train",
                                   image_dir=image_dir, image_type=image_type,
                                   mask_dir=mask_dir, mask_type=mask_type,
                                   weight_map_dir=weight_map_dir, weight_map_type=weight_map_type,
                                   target_size=target_size, transforms=None, seed=seed)
    data_gen_train, data_gen_val = train_val_split(data_gen_train, 0.8, validation_batch_size=2)

    logdir = logdir + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    tensorboard_val_image_writer = tf.summary.create_file_writer(logdir + "/val_image")

    tensorboard_callback = callbacks.TensorBoard(log_dir=logdir)
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
                                                                   save_weights_only=True,
                                                                   verbosde=1,
                                                                   monitor='val_binary_IoU',
                                                                   mode='max',
                                                                   save_best_only=True)
    validation_plot_callback = get_validation_plot_callback(unet, data_gen_val, [0, 1],
                                                            tensorboard_val_image_writer, max_output=4)
    callback_list = [tensorboard_callback, model_checkpoint_callback, validation_plot_callback]

    print("training starts...")
    history = unet.fit(x=data_gen_train, epochs=max_epoch,
                       validation_data=data_gen_val, shuffle=False,
                       validation_freq=1, callbacks=callback_list,
                       workers=4, use_multiprocessing=True)
    print("training finished")


if __name__ == '__main__':
    train_unet()
