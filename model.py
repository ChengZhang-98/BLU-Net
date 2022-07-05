import os

import keras.backend
import tensorflow as tf
from keras import metrics
from keras.layers import (Input,
                          Conv2D,
                          MaxPooling2D,
                          Dropout,
                          UpSampling2D,
                          Concatenate)
from keras.losses import BinaryCrossentropy
from keras.metrics import BinaryAccuracy, BinaryIoU
from keras.models import Model
from keras.optimizers import Adam
import tensorflow_addons as tfa

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


def get_compiled_unet(input_size, levels, final_activation="sigmoid", pretrained_weights=None):
    unet_model = get_uncompiled_unet(input_size, final_activation=final_activation, output_classes=1, dropout=0,
                                     levels=levels)
    bce_loss_from_logits = final_activation != "sigmoid"
    unet_model.compile(optimizer=Adam(learning_rate=1e-4),
                       loss=BinaryCrossentropy(name="weighted_binary_crossentropy", from_logits=bce_loss_from_logits),
                       metrics=[BinaryAccuracy(name="binary_accuracy", threshold=0.5),
                                BinaryIoU(target_class_ids=[1], threshold=0.5, name="binary_IoU", )])
    if pretrained_weights is not None:
        unet_model.load_weights(filepath=pretrained_weights)

    return unet_model


if __name__ == '__main__':
    # *: tensorboard --logdir="E:\ED_MS\Semester_3\Codes\MyProject\tensorboard_logs"
    seed = None
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
