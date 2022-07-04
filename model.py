import tensorflow as tf
from keras.losses import BinaryCrossentropy
from keras.layers import (Input,
                          Conv2D,
                          MaxPooling2D,
                          Dropout,
                          UpSampling2D,
                          Concatenate)
from keras.models import Model
from keras.utils import Sequence
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def _get_contracting_block(input_layer, filters, conv2d_params, dropout=0, name="Contracting"):
    pool = MaxPooling2D(pool_size=(2, 2), name=name + "_MaxPooling2D")(input_layer)
    conv1 = Conv2D(filters, 3, **conv2d_params, name=name + "_Conv2D_1")(pool)
    conv2 = Conv2D(filters, 3, **conv2d_params, name=name + "_Conv2D_2")(conv1)

    if dropout == 0:
        return conv2
    else:
        return Dropout(rate=dropout, name=name + "_Dropout")(conv2)


def _get_expanding_block(input_layer, skip_layer, filters, conv2d_params, dropout=0, name="Expanding"):
    up = UpSampling2D(size=(2, 2), name=name + "_UpSampling2D")(input_layer)
    conv1 = Conv2D(filters, 2, **conv2d_params, name=name + "_Conv2D_1")(up)
    merge = Concatenate(axis=3, name=name + "_Concatenate")([skip_layer, conv1])
    conv2 = Conv2D(filters, 3, **conv2d_params, name=name + "_Conv2D_2")(merge)
    conv3 = Conv2D(filters, 3, **conv2d_params, name=name + "_Conv2D_3")(conv2)

    if dropout == 0:
        return conv3
    else:
        return Dropout(rate=dropout, name=name + "_Dropout")(conv3)


def get_uncompiled_unet(input_size, final_activation, output_classes, dropout=0, levels=5):
    conv2d_parameters = {
        "activation": "relu",
        "padding": "same",
        "kernel_initializer": "he_normal",
    }
    inputs = Input(input_size, name="true_input")
    filters = 64

    conv = Conv2D(filters, 3, **conv2d_parameters, name="Level0_Conv2D_1")(inputs)
    conv = Conv2D(filters, 3, **conv2d_parameters, name="Level0_Conv2D_2")(conv)

    level = 0
    contracting_outputs = [conv]
    for level in range(1, levels):
        filters *= 2
        contracting_outputs.append(
            _get_contracting_block(input_layer=contracting_outputs[-1],
                                   filters=filters,
                                   conv2d_params=conv2d_parameters,
                                   dropout=dropout,
                                   name="Level{}_Contracting".format(level)
                                   )
        )

    expanding_output = contracting_outputs.pop()
    while level > 0:
        level -= 1
        filters = int(filters / 2)
        expanding_output = _get_expanding_block(input_layer=expanding_output,
                                                skip_layer=contracting_outputs.pop(),
                                                filters=filters,
                                                conv2d_params=conv2d_parameters,
                                                dropout=dropout,
                                                name="Level{}_Expanding".format(level))

    output = Conv2D(output_classes, 1, activation=final_activation, name="true_output")(expanding_output)

    unet_model = Model(inputs=inputs, outputs=output, name="Uncompiled_Unet")

    return unet_model


def pixel_wise_weighted_binary_crossentropy_loss(y_true, y_pred):
    mask_batch, weight_map_batch = tf.unstack(y_true, axis=-1)
    pixel_wise_bce_loss = BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)(mask_batch, y_pred)
    weighted_pixel_wise_bce_loss = tf.multiply(tf.expand_dims(pixel_wise_bce_loss, -1), weight_map_batch)
    return tf.reduce_mean(weighted_pixel_wise_bce_loss, [-3, -2, -1])


if __name__ == '__main__':
    seed = 1
    batch_size = 4
    target_size = (512, 512)
    weight_path = "./checkpoints/trained_weights/unet_agarpads_seg_evaluation2.hdf5"

    image_dir = "../../Dataset/DIC_Set/DIC_Set1_Annotated"
    image_type = "tif"
    mask_dir = "../../Dataset/DIC_Set/DIC_Set1_Masks"
    mask_type = "tif"
    weight_map_dir = "../../Dataset/DIC_Set/DIC_Set1_Weights"
    weight_map_type = "npy"
    dataset = "DIC"

    unet = get_uncompiled_unet(input_size=(*target_size, 1),
                               final_activation="sigmoid",
                               output_classes=1,
                               levels=5)
    unet.load_weights(filepath=weight_path)

    from data import DataGenerator

    data_gen = DataGenerator(batch_size=batch_size, dataset=dataset,
                             image_dir=image_dir, image_type=image_type,
                             mask_dir=mask_dir, mask_type=mask_type,
                             weight_map_dir=weight_map_dir, weight_map_type=weight_map_type,
                             target_size=target_size, transforms=None, seed=seed)

    from datetime import datetime

    logdir = "E:/ED_MS/Semester_3/Codes/MyProject/tensorboard_logs/" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    tf_board_filewriter = tf.summary.create_file_writer(logdir)

    with tf_board_filewriter.as_default():
        for batch_num, (image_batch, mask_weight_batch) in enumerate(data_gen):
            if batch_num >= 1:
                break
            predicted_mask_batch = unet.predict_step(image_batch)
            mask_batch, weight_batch = tf.unstack(mask_weight_batch, axis=-1)

            tf.summary.image("Input_Images", data=image_batch, step=1, max_outputs=4,
                             description="input images, batch_num = {}".format(batch_num))
            tf.summary.image("Ground_Truth_Masks", data=mask_batch, step=1, max_outputs=4,
                             description="ground truth masks, batch_num = {}".format(batch_num))
            tf.summary.image("Weight_Maps", data=weight_batch, step=1, max_outputs=4,
                             description="weight maps, batch_num = {}".format(batch_num))
            tf.summary.image("Predicted_Masks", data=predicted_mask_batch, step=1, max_outputs=4,
                             description="predicted masks, batch_num = {}".format(batch_num))

    print("done")