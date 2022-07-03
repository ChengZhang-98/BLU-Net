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


def get_contracting_block(input_layer, filters, conv2d_params, dropout=0, name="Contracting"):
    pool = MaxPooling2D(pool_size=(2, 2), name=name + "_MaxPooling2D")(input_layer)
    conv1 = Conv2D(filters, 3, **conv2d_params, name=name + "_Conv2D_1")(pool)
    conv2 = Conv2D(filters, 3, **conv2d_params, name=name + "_Conv2D_2")(conv1)

    if dropout == 0:
        return conv2
    else:
        return Dropout(rate=dropout, name=name + "_Dropout")(conv2)


def get_expanding_block(input_layer, skip_layer, filters, conv2d_params, dropout=0, name="Expanding"):
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
            get_contracting_block(input_layer=contracting_outputs[-1],
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
        expanding_output = get_expanding_block(input_layer=expanding_output,
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
